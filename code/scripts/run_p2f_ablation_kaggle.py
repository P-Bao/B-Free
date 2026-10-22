#!/usr/bin/env python3
"""Standardized Kaggle GPU runner for P2.F Ablation Grid (R0 -> R5-l).

Guarantees:
  1. Strict offline mode (HF_HUB_OFFLINE=1, TRANSFORMERS_OFFLINE=1, HF_DATASETS_OFFLINE=1).
  2. Absolute Kaggle paths only; fail-fast validation before any GPU compute.
  3. Executes Gate "Step 0 == B-Free" (|Delta logits| < 1e-3) on the real B-Free checkpoint.
  4. Launches `accelerate launch train_lora.py` with exact run hyperparameters (R1..R5-l).
  5. Evaluates the trained checkpoint under Yardstick v2 (harness_version=2), saving:
       - <output_root>/<run_id>/train_log.csv
       - <output_root>/<run_id>/scores.parquet
       - <output_root>/<run_id>/eval_summary.csv
     and prints a ready-to-paste Markdown row for `research/iter-v4-ablation.md`.
"""

import argparse
import glob
import hashlib
import io
import json
import logging
import math
import os
import random
import shutil
import subprocess
import sys
import time
import zlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---- 1. Enforce Offline Environment Immediately ----
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTHONHASHSEED"] = "0"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image, ImageFilter
from scipy.optimize import minimize_scalar
from scipy.special import expit
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, roc_curve

logger = logging.getLogger("p2f_ablation")
logger.setLevel(logging.INFO)
if not logger.handlers:
    _h = logging.StreamHandler(sys.stdout)
    _h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S"))
    logger.addHandler(_h)

# ============================================================================
# Absolute Kaggle Paths (Default Layout)
# ============================================================================
DEFAULT_PATHS = {
    "repo_src": "/kaggle/input/notebooks/baopv051/xla-setup/bfree_src",
    "wheels_dir": "/kaggle/input/notebooks/baopv051/xla-setup/wheels_rtxpro6000",
    "dinov2_sd": "/kaggle/input/notebooks/baopv051/xla-setup/models/vit_base_patch14_reg4_dinov2/model.safetensors",
    "bfree_weights_dir": "/kaggle/input/models/baopv051/bfree-baseline-weights/pytorch/default/1",
    "train_data_root": "/kaggle/input/datasets/baopv051/b-free-training-data",
    "ai_genbench_dir": "/kaggle/input/datasets/baopv051/ai-genbench-v1",
    "wildrf_dir": "/kaggle/input/datasets/ngovanphuong2604/wildrf-dataset",
    "synthwildx_dir": "/kaggle/input/datasets/ngovanphuong2604/synthwildx-dataset",
    "bfree_viral_dir": "/kaggle/input/datasets/ngovanphuong2604/bfree-viral-images",
    "output_root": "/kaggle/working/p2f_runs",
}

SEEN_GENERATORS = {
    "ADM", "DDPM", "DeepFloyd IF", "Palette", "Denoising Diffusion GAN",
    "LaMa", "MAT", "SN-PatchGAN", "StyleGAN1", "VQGAN", "GANformer",
    "CIPS", "FaceSynthetics",
}

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# ============================================================================
# P2.F Ablation Grid Specification (R0 -> R5-l)
# ============================================================================
ABLATION_GRID: Dict[str, Dict] = {
    "R0": {
        "desc": "B-Free official reference in Yardstick v2",
        "train": False,
        "init_from": "bfree",
    },
    "R1": {
        "desc": "Pure recipe: B-Free init + dual-view CE + mix_bfree (no LIB/GSR/DCS)",
        "train": True,
        "init_from": "bfree",
        "use_lib": 0, "use_gsr": 0, "use_dcs": 0,
        "head_mode": "cls", "gsr_zero_init": 1, "lib_beta_init": -6.0,
        "loss_mode": "dual_view", "w_deg": 1.0, "lam_cons": 0.5, "cons_T": 1.0, "p_aug_on_clean": 0.2,
        "lambda_dcs": 0.0, "label_smoothing": 0.1,
        "mix_preset": "mix_bfree", "fmt_match": 0, "res_match": 0,
    },
    "R1b": {
        "desc": "W1 control: B-Free init + clean CE only (legacy) + mix_bfree",
        "train": True,
        "init_from": "bfree",
        "use_lib": 0, "use_gsr": 0, "use_dcs": 0,
        "head_mode": "cls", "gsr_zero_init": 1, "lib_beta_init": -6.0,
        "loss_mode": "legacy", "w_deg": 1.0, "lam_cons": 0.0, "cons_T": 1.0, "p_aug_on_clean": 0.0,
        "lambda_dcs": 0.0, "label_smoothing": 0.1,
        "mix_preset": "mix_bfree", "fmt_match": 0, "res_match": 0,
    },
    "R2": {
        "desc": "Isolated +DCS on top of R1",
        "train": True,
        "init_from": "bfree",
        "use_lib": 0, "use_gsr": 0, "use_dcs": 1,
        "head_mode": "cls", "gsr_zero_init": 1, "lib_beta_init": -6.0,
        "loss_mode": "dual_view", "w_deg": 1.0, "lam_cons": 0.5, "cons_T": 1.0, "p_aug_on_clean": 0.2,
        "lambda_dcs": 0.1, "label_smoothing": 0.1,
        "mix_preset": "mix_bfree", "fmt_match": 0, "res_match": 0,
    },
    "R3": {
        "desc": "Isolated +LIB (beta_init=-6) on top of R1",
        "train": True,
        "init_from": "bfree",
        "use_lib": 1, "use_gsr": 0, "use_dcs": 0,
        "head_mode": "cls", "gsr_zero_init": 1, "lib_beta_init": -6.0,
        "loss_mode": "dual_view", "w_deg": 1.0, "lam_cons": 0.5, "cons_T": 1.0, "p_aug_on_clean": 0.2,
        "lambda_dcs": 0.0, "label_smoothing": 0.1,
        "mix_preset": "mix_bfree", "fmt_match": 0, "res_match": 0,
    },
    "R4": {
        "desc": "Isolated +GSR (zero-init gate) on top of R1",
        "train": True,
        "init_from": "bfree",
        "use_lib": 0, "use_gsr": 1, "use_dcs": 0,
        "head_mode": "cls", "gsr_zero_init": 1, "lib_beta_init": -6.0,
        "loss_mode": "dual_view", "w_deg": 1.0, "lam_cons": 0.5, "cons_T": 1.0, "p_aug_on_clean": 0.2,
        "lambda_dcs": 0.0, "label_smoothing": 0.1,
        "mix_preset": "mix_bfree", "fmt_match": 0, "res_match": 0,
    },
    "R5": {
        "desc": "Full model: B-Free init + LIB + GSR + DCS + dual-view + mix_bfree",
        "train": True,
        "init_from": "bfree",
        "use_lib": 1, "use_gsr": 1, "use_dcs": 1,
        "head_mode": "cls", "gsr_zero_init": 1, "lib_beta_init": -6.0,
        "loss_mode": "dual_view", "w_deg": 1.0, "lam_cons": 0.5, "cons_T": 1.0, "p_aug_on_clean": 0.2,
        "lambda_dcs": 0.1, "label_smoothing": 0.1,
        "mix_preset": "mix_bfree", "fmt_match": 0, "res_match": 0,
    },
    "R5-d": {
        "desc": "DINOv2 init control (R5 with init_from=dinov2)",
        "train": True,
        "init_from": "dinov2",
        "use_lib": 1, "use_gsr": 1, "use_dcs": 1,
        "head_mode": "cls", "gsr_zero_init": 1, "lib_beta_init": -6.0,
        "loss_mode": "dual_view", "w_deg": 1.0, "lam_cons": 0.5, "cons_T": 1.0, "p_aug_on_clean": 0.2,
        "lambda_dcs": 0.1, "label_smoothing": 0.1,
        "mix_preset": "mix_bfree", "fmt_match": 0, "res_match": 0,
    },
    "R5-a": {
        "desc": "AIGB-heavy data mix control (R5 with mix_preset=mix_current)",
        "train": True,
        "init_from": "bfree",
        "use_lib": 1, "use_gsr": 1, "use_dcs": 1,
        "head_mode": "cls", "gsr_zero_init": 1, "lib_beta_init": -6.0,
        "loss_mode": "dual_view", "w_deg": 1.0, "lam_cons": 0.5, "cons_T": 1.0, "p_aug_on_clean": 0.2,
        "lambda_dcs": 0.1, "label_smoothing": 0.1,
        "mix_preset": "mix_current", "fmt_match": 0, "res_match": 0,
    },
    "R5-f": {
        "desc": "Anti-shortcut control (R5 with fmt_match=1, res_match=1)",
        "train": True,
        "init_from": "bfree",
        "use_lib": 1, "use_gsr": 1, "use_dcs": 1,
        "head_mode": "cls", "gsr_zero_init": 1, "lib_beta_init": -6.0,
        "loss_mode": "dual_view", "w_deg": 1.0, "lam_cons": 0.5, "cons_T": 1.0, "p_aug_on_clean": 0.2,
        "lambda_dcs": 0.1, "label_smoothing": 0.1,
        "mix_preset": "mix_bfree", "fmt_match": 1, "res_match": 1,
    },
    "R5-l": {
        "desc": "No label smoothing control (R5 with label_smoothing=0.0 + temp scaling)",
        "train": True,
        "init_from": "bfree",
        "use_lib": 1, "use_gsr": 1, "use_dcs": 1,
        "head_mode": "cls", "gsr_zero_init": 1, "lib_beta_init": -6.0,
        "loss_mode": "dual_view", "w_deg": 1.0, "lam_cons": 0.5, "cons_T": 1.0, "p_aug_on_clean": 0.2,
        "lambda_dcs": 0.1, "label_smoothing": 0.0,
        "mix_preset": "mix_bfree", "fmt_match": 0, "res_match": 0,
    },
}


# ============================================================================
# Fail-Fast Helpers & Provenance
# ============================================================================
def require_dir(p: str | Path, what: str) -> Path:
    path = Path(p)
    if not path.is_absolute():
        raise ValueError(f"FAIL-FAST: path must be absolute ({what}): {path}")
    if not path.is_dir():
        raise FileNotFoundError(
            f"FAIL-FAST (missing directory): {what}\n  expected: {path}\n"
            "  -> Attach the required Kaggle input dataset/notebook output.")
    return path


def require_file(p: str | Path, what: str) -> Path:
    path = Path(p)
    if not path.is_absolute():
        raise ValueError(f"FAIL-FAST: path must be absolute ({what}): {path}")
    if not path.is_file():
        raise FileNotFoundError(
            f"FAIL-FAST (missing file): {what}\n  expected: {path}\n"
            "  -> Attach the required Kaggle input dataset/model.")
    return path


def resolve_bfree_ckpt(weights_dir: Path) -> Path:
    require_dir(weights_dir, "bfree_weights_dir")
    for cand_name in ("model_epoch_best.pth", "bfree_dino2reg4.pth", "model.pth"):
        hits = sorted(weights_dir.rglob(cand_name))
        if hits:
            return hits[0]
    pths = sorted(weights_dir.rglob("*.pth"))
    if pths:
        return pths[0]
    raise FileNotFoundError(f"FAIL-FAST: no .pth file found inside {weights_dir}")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def get_git_sha(repo_dir: Path) -> str:
    try:
        out = subprocess.check_output(
            ["git", "-C", str(repo_dir), "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL, text=True
        ).strip()
        return out or "unknown"
    except Exception:
        return "offline_bundle"


# ============================================================================
# Gate Check: Step 0 == B-Free (|Delta logits| < 1e-3)
# ============================================================================
def verify_step0_gate(code_dir: Path, bfree_ckpt: Path, n_images: int = 16, img_size: int = 504):
    if str(code_dir) not in sys.path:
        sys.path.insert(0, str(code_dir))
    from networks import get_network, load_weights
    from networks.bfree_globalforge_vit import BFreeGlobalForgeViT

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    arch = "vit_base_patch14_reg4_dinov2.lvd142m"
    ref = load_weights(get_network("timm_c5i504_" + arch), str(bfree_ckpt)).to(device).eval()
    ours = BFreeGlobalForgeViT(
        arch=arch, num_classes=2, img_size=img_size, pretrained=False,
        use_lib=True, use_gsr=True, use_dcs=True,
        head_mode="cls", gsr_zero_init=True, lib_beta_init=-6.0,
    )
    rep = ours.load_checkpoint(str(bfree_ckpt), strict=False, require_complete_backbone=True)
    ours = ours.to(device).eval()

    g = torch.Generator().manual_seed(7)
    worst = 0.0
    with torch.no_grad():
        for _ in range(0, n_images, 4):
            bs = min(4, n_images)
            xb = torch.randn(bs, 3, img_size, img_size, generator=g).to(device)
            l_ref = ref(xb)[:, 0]
            lg = ours(xb)["logits"]
            gap = (lg[:, 1] - lg[:, 0] - l_ref).abs().max().item()
            worst = max(worst, gap)
    del ref, ours
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info(f"[GATE P2.B] Step 0 vs B-Free official: max |Delta logits| = {worst:.3e} "
                f"(report={rep['num_missing_backbone']} missing backbone, {rep['num_unexpected']} unexpected)")
    if worst >= 1e-3:
        raise RuntimeError(f"FAIL-FAST Gate P2.B failed: |Delta logits| = {worst:.3e} >= 1e-3")


# ============================================================================
# Yardstick v2 Evaluation Engine (Produces scores.parquet & eval_summary.csv)
# ============================================================================
def fit_temperature_and_threshold(y_true: np.ndarray, logits: np.ndarray) -> Tuple[float, float]:
    """Fit Temperature T > 0 (minimizing balanced NLL) and Youden's J logit threshold z*."""
    from utils.dmetrics import balanced_nll_binary

    y_true = np.asarray(y_true, dtype=int)
    logits = np.asarray(logits, dtype=np.float64)
    if len(np.unique(y_true)) < 2 or len(y_true) == 0:
        return 1.0, 0.0

    def obj(log_t):
        t = math.exp(log_t)
        return float(balanced_nll_binary(y_true, logits / t))

    res = minimize_scalar(obj, bounds=(-2.0, 2.0), method="bounded")
    T_opt = float(math.exp(res.x)) if res.success else 1.0

    fpr, tpr, thr = roc_curve(y_true, logits)
    best_idx = int(np.argmax(tpr - fpr))
    z_star = float(thr[best_idx]) if np.isfinite(thr[best_idx]) else 0.0
    return T_opt, z_star


def compute_yardstick_metrics(
    y_true: np.ndarray,
    logits: np.ndarray,
    T_cal: float = 1.0,
    z_star: float = 0.0,
    n_boot: int = 200,
    seed: int = 42,
) -> Dict[str, float]:
    from utils.dmetrics import (
        balanced_ece_binary,
        balanced_nll_binary,
        calculate_eer2,
        pd_at_far,
    )

    y_true = np.asarray(y_true, dtype=int)
    logits = np.asarray(logits, dtype=np.float64)
    if len(np.unique(y_true)) < 2:
        return {"n": len(y_true), "auc": float("nan"), "bacc_05": float("nan"),
                "bacc_calib": float("nan"), "ece_raw": float("nan"), "ece_cal": float("nan"),
                "nll_cal": float("nan")}

    cal_logits = logits / max(T_cal, 1e-4)
    auc = float(roc_auc_score(y_true, logits))
    bacc_05 = float(balanced_accuracy_score(y_true, logits > 0.0))
    bacc_cal = float(balanced_accuracy_score(y_true, logits > z_star))
    ece_raw = float(balanced_ece_binary(y_true, logits))
    ece_cal = float(balanced_ece_binary(y_true, cal_logits))
    nll_cal = float(balanced_nll_binary(y_true, cal_logits))
    pd_far10 = float(pd_at_far(y_true, logits, 0.10))
    eer = float(calculate_eer2(y_true, logits))

    pos = logits[y_true == 1]
    neg = logits[y_true == 0]
    tpr_05 = float((pos > 0.0).mean()) if len(pos) else float("nan")
    fpr_05 = float((neg > 0.0).mean()) if len(neg) else float("nan")

    # Stratified bootstrap 95% CI for AUC & bAcc@0.5
    rng = np.random.default_rng(seed)
    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]
    boot_auc, boot_bacc = [], []
    if len(pos_idx) > 1 and len(neg_idx) > 1 and n_boot > 0:
        for _ in range(n_boot):
            pi = rng.choice(pos_idx, size=len(pos_idx), replace=True)
            ni = rng.choice(neg_idx, size=len(neg_idx), replace=True)
            idx = np.concatenate([pi, ni])
            boot_auc.append(roc_auc_score(y_true[idx], logits[idx]))
            boot_bacc.append(balanced_accuracy_score(y_true[idx], logits[idx] > 0.0))
    auc_ci_lo, auc_ci_hi = (np.percentile(boot_auc, [2.5, 97.5]) if boot_auc else (auc, auc))
    bacc_ci_lo, bacc_ci_hi = (np.percentile(boot_bacc, [2.5, 97.5]) if boot_bacc else (bacc_05, bacc_05))

    return {
        "n": int(len(y_true)),
        "auc": auc,
        "auc_ci_lo": float(auc_ci_lo),
        "auc_ci_hi": float(auc_ci_hi),
        "bacc_05": bacc_05,
        "bacc_ci_lo": float(bacc_ci_lo),
        "bacc_ci_hi": float(bacc_ci_hi),
        "bacc_calib": bacc_cal,
        "tpr_05": tpr_05,
        "fpr_05": fpr_05,
        "ece_raw": ece_raw,
        "ece_cal": ece_cal,
        "nll_cal": nll_cal,
        "pd_at_far10": pd_far10,
        "eer": eer,
        "temp_T": float(T_cal),
        "thr_logit": float(z_star),
    }


def apply_heldout_degradation(img: Image.Image, cond: str, key: str, seed: int = 42) -> Image.Image:
    """Deterministic per-image held-out degradation (P0.3)."""
    h_seed = zlib.crc32(f"{key}:{cond}:{seed}".encode("utf-8")) & 0xFFFFFFFF
    rng = np.random.default_rng(h_seed)
    img = img.convert("RGB")
    w, h = img.size

    if cond == "heldout_webp":
        q = int(rng.choice([85, 70, 50, 30]))
        buf = io.BytesIO()
        img.save(buf, format="WEBP", quality=q)
        buf.seek(0)
        return Image.open(buf).convert("RGB")
    if cond == "heldout_jpeg_420":
        q = int(rng.choice([85, 65, 45]))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=q, subsampling="4:2:0")
        buf.seek(0)
        return Image.open(buf).convert("RGB")
    if cond == "heldout_resize_lanczos":
        s = float(rng.choice([0.75, 0.50, 0.35]))
        small = img.resize((max(1, int(w * s)), max(1, int(h * s))), Image.LANCZOS)
        return small.resize((w, h), Image.BILINEAR)
    if cond == "heldout_poisson_iso":
        lam = float(rng.choice([100.0, 50.0, 25.0]))
        arr = np.asarray(img).astype(np.float32) / 255.0
        shot = rng.poisson(np.clip(arr, 0.0, 1.0) * lam) / lam
        read = rng.normal(0.0, 0.015, size=arr.shape)
        out = np.clip(shot + read, 0.0, 1.0)
        return Image.fromarray((out * 255.0).astype(np.uint8))
    if cond == "heldout_sharpen":
        pct = int(rng.choice([120, 200, 300]))
        return img.filter(ImageFilter.UnsharpMask(radius=2, percent=pct, threshold=3))
    if cond == "heldout_social_compound":
        # resize(0.75, BILINEAR) -> JPEG(80, 4:2:0) -> resize(0.85, BICUBIC) -> JPEG(70, 4:2:0)
        s1 = img.resize((max(1, int(w * 0.75)), max(1, int(h * 0.75))), Image.BILINEAR).resize((w, h), Image.BILINEAR)
        b1 = io.BytesIO()
        s1.save(b1, format="JPEG", quality=80, subsampling="4:2:0")
        b1.seek(0)
        s2 = Image.open(b1).convert("RGB")
        s3 = s2.resize((max(1, int(w * 0.85)), max(1, int(h * 0.85))), Image.BICUBIC).resize((w, h), Image.BICUBIC)
        b2 = io.BytesIO()
        s3.save(b2, format="JPEG", quality=70, subsampling="4:2:0")
        b2.seek(0)
        return Image.open(b2).convert("RGB")
    raise ValueError(f"Unknown heldout condition: {cond}")


class UnifiedEvaluator:
    """Scores PIL images and returns natural binary logits z = l1 - l0 (or l for 1-logit B-Free)."""

    def __init__(self, model: torch.nn.Module, is_bfree_official: bool, device: str = "cuda:0", img_size: int = 504):
        self.model = model.to(device).eval()
        self.is_bfree_official = is_bfree_official
        self.device = device
        self.img_size = img_size
        self.normalize = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def _five_crops(self, img: Image.Image) -> List[Image.Image]:
        w, h = img.size
        s = self.img_size
        if min(w, h) < s:
            arr = np.asarray(img)
            rep_h = max(1, int(math.ceil(s / h)))
            rep_w = max(1, int(math.ceil(s / w)))
            arr = np.tile(arr, (rep_h, rep_w, 1))[:max(s, h), :max(s, w)]
            img = Image.fromarray(arr)
            w, h = img.size
        if w == s and h == s:
            return [img]
        cx, cy = (w - s) // 2, (h - s) // 2
        coords = [(cx, cy), (0, 0), (0, h - s), (w - s, h - s), (w - s, 0)]
        return [img.crop((x, y, x + s, y + s)) for x, y in coords]

    @torch.no_grad()
    def score_pil(self, img: Image.Image) -> float:
        img = img.convert("RGB")
        if self.is_bfree_official:
            # Official B-Free Wrapper5crops handles 5-crop & replicate_wrap internally on full image
            x = self.normalize(img).unsqueeze(0).to(self.device)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=self.device.startswith("cuda")):
                logit = self.model(x).float().squeeze().item()
            return float(logit)
        crops = self._five_crops(img)
        batch = torch.stack([self.normalize(c) for c in crops]).to(self.device)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=self.device.startswith("cuda")):
            out = self.model(batch)["logits"].float()
            z = (out[:, 1] - out[:, 0]).mean().item()
        return float(z)


def load_trained_ours_for_eval(code_dir: Path, ckpt_path: Path, device: str) -> UnifiedEvaluator:
    if str(code_dir) not in sys.path:
        sys.path.insert(0, str(code_dir))
    from networks.bfree_globalforge_vit import BFreeGlobalForgeViT

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    cfg = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}

    model = BFreeGlobalForgeViT(
        arch=cfg.get("arch", "vit_base_patch14_reg4_dinov2.lvd142m"),
        num_classes=cfg.get("num_classes", 2),
        img_size=cfg.get("img_size", 504),
        pretrained=False,
        use_lib=bool(cfg.get("use_lib", True)),
        use_gsr=bool(cfg.get("use_gsr", True)),
        use_dcs=bool(cfg.get("use_dcs", True)),
        lib_kernel=cfg.get("lib_kernel", 3),
        lib_tau=cfg.get("lib_tau", 0.5),
        gsr_window=cfg.get("gsr_window", 3),
        gsr_mask_prob=cfg.get("gsr_mask_prob", 1.0),
        dcs_tau=cfg.get("dcs_tau", 0.07),
        lambda_dcs=cfg.get("lambda_dcs", 0.1),
        label_smoothing=cfg.get("label_smoothing", 0.1),
        head_mode=cfg.get("head_mode", "cls"),
        gsr_zero_init=bool(cfg.get("gsr_zero_init", False)),
        lib_beta_init=float(cfg.get("lib_beta_init", -0.85)),
    )

    # Merge LoRA weights (W = W + (alpha/r) * B @ A)
    lora_alpha = float(cfg.get("lora_alpha", 32))
    lora_rank = float(cfg.get("lora_rank", 16))
    scaling = lora_alpha / lora_rank

    def _merge_stem(stem: str) -> str:
        prefix_old = "model.base_model.model."
        stem = ("model." + stem[len(prefix_old):]) if stem.startswith(prefix_old) else stem
        return stem.replace(".base_layer.", ".")

    lora_pairs = {}
    for k, v in sd.items():
        if k.endswith(".lora_A.default.weight"):
            lora_pairs.setdefault(_merge_stem(k[:-len(".lora_A.default.weight")]) + ".weight", {})["A"] = v
        elif k.endswith(".lora_B.default.weight"):
            lora_pairs.setdefault(_merge_stem(k[:-len(".lora_B.default.weight")]) + ".weight", {})["B"] = v

    merged = {}
    for k, v in sd.items():
        if ".lora_A.default." in k or ".lora_B.default." in k:
            continue
        k2 = _merge_stem(k)
        if k2 in lora_pairs:
            A, B = lora_pairs[k2]["A"], lora_pairs[k2]["B"]
            merged[k2] = v + scaling * (B.view(B.shape[0], -1) @ A.view(A.shape[0], -1)).view(v.shape)
        else:
            merged[k2] = v

    model.load_state_dict(merged, strict=True)
    return UnifiedEvaluator(model, is_bfree_official=False, device=device)


def load_bfree_official_for_eval(code_dir: Path, bfree_ckpt: Path, device: str) -> UnifiedEvaluator:
    if str(code_dir) not in sys.path:
        sys.path.insert(0, str(code_dir))
    from networks import get_network, load_weights

    arch = "timm_c5i504_vit_base_patch14_reg4_dinov2.lvd142m"
    model = load_weights(get_network(arch), str(bfree_ckpt))
    return UnifiedEvaluator(model, is_bfree_official=True, device=device)


def list_images_in(folder: Path) -> List[Path]:
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.rglob("*") if p.suffix.lower() in IMG_EXTS)


def evaluate_run_yardstick_v2(
    run_id: str,
    evaluator: UnifiedEvaluator,
    paths: Dict[str, Path],
    run_out_dir: Path,
    git_sha: str,
    ckpt_sha: str,
    config_hash: str,
    seed: int = 42,
    max_per_group: int = 200,
) -> Dict[str, float]:
    """Runs Yardstick v2 screening evaluation and writes scores.parquet + eval_summary.csv."""
    records = []

    def _score_and_record(img: Image.Image, image_id: str, benchmark: str, group: str,
                          is_seen: bool, origin: str, condition: str, label: int):
        w, h = img.size
        fmt = (img.format or "RAW").upper()
        if condition != "clean":
            img = apply_heldout_degradation(img, condition, image_id, seed=seed)
        logit = evaluator.score_pil(img)
        prob = float(expit(logit))
        records.append({
            "image_id": str(image_id),
            "benchmark": str(benchmark),
            "group": str(group),
            "is_seen_gen": bool(is_seen),
            "origin_dataset": str(origin),
            "condition": str(condition),
            "label": int(label),
            "logit": np.float32(logit),
            "prob": np.float32(prob),
            "prob_cal": np.float32(prob),  # updated after temperature scaling fit
            "width": int(w),
            "height": int(h),
            "format": str(fmt),
            "run_id": str(run_id),
            "git_sha": str(git_sha),
            "ckpt_sha256": str(ckpt_sha),
            "config_hash": str(config_hash),
            "seed": int(seed),
            "harness_version": 2,
        })

    rng = np.random.default_rng(seed)

    # ---- 1. WildRF (val for calibration + test for evaluation) ----
    wildrf_root = paths.get("wildrf_dir")
    if wildrf_root and wildrf_root.is_dir():
        for split_name, b_tag in [("val", "wildrf_val"), ("test", "wildrf")]:
            for hit in wildrf_root.rglob(split_name):
                if hit.is_dir() and any((p / "0_real").is_dir() for p in hit.iterdir() if p.is_dir()):
                    for plat in sorted(p for p in hit.iterdir() if p.is_dir()):
                        reals = list_images_in(plat / "0_real")
                        fakes = list_images_in(plat / "1_fake")
                        if len(reals) > max_per_group:
                            reals = [reals[i] for i in sorted(rng.choice(len(reals), max_per_group, replace=False))]
                        if len(fakes) > max_per_group:
                            fakes = [fakes[i] for i in sorted(rng.choice(len(fakes), max_per_group, replace=False))]
                        for p in reals:
                            with Image.open(p) as im:
                                _score_and_record(im, str(p.relative_to(wildrf_root)), b_tag, plat.name, False, "wildrf", "clean", 0)
                        for p in fakes:
                            with Image.open(p) as im:
                                _score_and_record(im, str(p.relative_to(wildrf_root)), b_tag, plat.name, False, "wildrf", "clean", 1)
                    break

    # ---- 2. B-Free Viral Images ----
    viral_root = paths.get("bfree_viral_dir")
    if viral_root and viral_root.is_dir():
        real_dirs = [p for p in viral_root.rglob("REAL") if p.is_dir()]
        fake_dirs = [p for p in viral_root.rglob("FAKE") if p.is_dir()]
        if real_dirs and fake_dirs:
            reals = list_images_in(real_dirs[0])
            fakes = list_images_in(fake_dirs[0])
            if len(reals) > max_per_group * 2:
                reals = [reals[i] for i in sorted(rng.choice(len(reals), max_per_group * 2, replace=False))]
            if len(fakes) > max_per_group * 2:
                fakes = [fakes[i] for i in sorted(rng.choice(len(fakes), max_per_group * 2, replace=False))]
            for p in reals:
                with Image.open(p) as im:
                    _score_and_record(im, str(p.relative_to(viral_root)), "bfree_viral_images", "viral", False, "viral", "clean", 0)
            for p in fakes:
                with Image.open(p) as im:
                    _score_and_record(im, str(p.relative_to(viral_root)), "bfree_viral_images", "viral", False, "viral", "clean", 1)

    # ---- 3. AI-GenBench Validation (Seen vs Unseen + RealDeg Held-out) ----
    aigb_root = paths.get("ai_genbench_dir")
    if aigb_root and aigb_root.is_dir():
        from datasets import load_from_disk
        obj = load_from_disk(str(aigb_root))
        ds = obj["validation"] if hasattr(obj, "keys") and "validation" in obj else (obj[list(obj.keys())[0]] if hasattr(obj, "keys") else obj)
        img_col = next(c for c, f in ds.features.items() if f.__class__.__name__ == "Image")
        cols = [c for c in ("label", "generator", "origin_dataset") if c in ds.column_names]
        meta = ds.select_columns(cols).to_pandas()
        label_feat = ds.features.get("label")
        label_names = list(getattr(label_feat, "names", []) or [])
        if label_names:
            fake_ids = {i for i, n in enumerate(label_names) if "fake" in str(n).lower()} or {len(label_names) - 1}
            meta["y"] = meta["label"].apply(lambda v: 1 if int(v) in fake_ids else 0)
        else:
            meta["y"] = (meta["label"].astype(int) != 0).astype(int)
        meta["generator"] = meta.get("generator", pd.Series(["unknown"] * len(meta))).fillna("unknown").astype(str).str.strip()
        meta["origin_dataset"] = meta.get("origin_dataset", pd.Series(["unknown"] * len(meta))).fillna("unknown").astype(str)

        real_idx = meta.index[meta["y"] == 0].to_numpy()
        n_real_eval = min(len(real_idx), max_per_group * 4)
        sel_real = sorted(int(i) for i in rng.choice(real_idx, size=n_real_eval, replace=False))
        for idx in sel_real:
            im = ds[idx][img_col]
            _score_and_record(im, f"aigb_val_{idx}", "ai_genbench", "REAL_POOL", False, meta.loc[idx, "origin_dataset"], "clean", 0)

        per_gen_cap = max(30, max_per_group // 4)
        for gen_name, grp in meta[meta["y"] == 1].groupby("generator"):
            g_idx = grp.index.to_numpy()
            sel_g = sorted(int(i) for i in (rng.choice(g_idx, size=per_gen_cap, replace=False) if len(g_idx) > per_gen_cap else g_idx))
            is_seen = gen_name in SEEN_GENERATORS
            for idx in sel_g:
                im = ds[idx][img_col]
                _score_and_record(im, f"aigb_val_{idx}", "ai_genbench", gen_name, is_seen, meta.loc[idx, "origin_dataset"], "clean", 1)

        # Held-out RealDeg on a balanced subset of AI-GenBench (P0.3 / H-B)
        heldout_conds = [
            "heldout_webp", "heldout_jpeg_420", "heldout_resize_lanczos",
            "heldout_poisson_iso", "heldout_sharpen", "heldout_social_compound",
        ]
        deg_real = sel_real[:min(len(sel_real), 60)]
        unseen_fake_idx = meta.index[(meta["y"] == 1) & (~meta["generator"].isin(SEEN_GENERATORS))].to_numpy()
        deg_fake = sorted(int(i) for i in rng.choice(unseen_fake_idx, size=min(len(unseen_fake_idx), 60), replace=False))
        for cond in heldout_conds:
            for idx in deg_real:
                im = ds[idx][img_col]
                _score_and_record(im, f"aigb_val_{idx}", "realdeg_heldout", "REAL", False, meta.loc[idx, "origin_dataset"], cond, 0)
            for idx in deg_fake:
                im = ds[idx][img_col]
                _score_and_record(im, f"aigb_val_{idx}", "realdeg_heldout", meta.loc[idx, "generator"], False, meta.loc[idx, "origin_dataset"], cond, 1)

    df = pd.DataFrame(records)
    if df.empty:
        raise RuntimeError("No evaluation samples were scored — check dataset paths.")

    # Fit Temperature Scaling T & threshold z* on wildrf_val (or 20% calibration slice)
    cal_df = df[df["benchmark"] == "wildrf_val"]
    if len(cal_df) < 20 or len(cal_df["label"].unique()) < 2:
        cal_df = df[df["condition"] == "clean"].iloc[::5]
    T_cal, z_star = fit_temperature_and_threshold(cal_df["label"].to_numpy(), cal_df["logit"].to_numpy())
    df["prob_cal"] = expit(df["logit"].to_numpy(dtype=np.float64) / max(T_cal, 1e-4)).astype(np.float32)

    # Save per-image scores.parquet (P0.9)
    parquet_path = run_out_dir / "scores.parquet"
    df.to_parquet(parquet_path, index=False)
    logger.info(f"[{run_id}] Saved per-image scores -> {parquet_path} ({len(df)} rows)")

    # Aggregate summary metrics
    summary_rows = []

    def _add_summary(name: str, sub: pd.DataFrame):
        if len(sub) == 0 or len(sub["label"].unique()) < 2:
            return
        m = compute_yardstick_metrics(sub["label"].to_numpy(), sub["logit"].to_numpy(), T_cal=T_cal, z_star=z_star, seed=seed)
        summary_rows.append({
            "run_id": run_id, "subset": name, **m,
            "git_sha": git_sha, "ckpt_sha256": ckpt_sha, "config_hash": config_hash,
            "seed": seed, "harness_version": 2,
        })

    # WildRF + Viral (H-A, H-D, H-E)
    _add_summary("wildrf_test", df[(df["benchmark"] == "wildrf") & (df["condition"] == "clean")])
    _add_summary("bfree_viral", df[(df["benchmark"] == "bfree_viral_images") & (df["condition"] == "clean")])
    _add_summary("wildrf_plus_viral", df[df["benchmark"].isin(["wildrf", "bfree_viral_images"]) & (df["condition"] == "clean")])

    # AIGB Seen vs Unseen (H-C, Guardrail)
    aigb_clean = df[(df["benchmark"] == "ai_genbench") & (df["condition"] == "clean")]
    aigb_real = aigb_clean[aigb_clean["label"] == 0]
    aigb_seen = pd.concat([aigb_real, aigb_clean[(aigb_clean["label"] == 1) & (aigb_clean["is_seen_gen"])]])
    aigb_unseen = pd.concat([aigb_real, aigb_clean[(aigb_clean["label"] == 1) & (~aigb_clean["is_seen_gen"])]])
    _add_summary("aigb_seen", aigb_seen)
    _add_summary("aigb_unseen", aigb_unseen)

    # RealDeg Held-out (H-B)
    _add_summary("realdeg_heldout", df[df["benchmark"] == "realdeg_heldout"])

    sum_df = pd.DataFrame(summary_rows).set_index("subset")
    sum_csv = run_out_dir / "eval_summary.csv"
    sum_df.reset_index().to_csv(sum_csv, index=False)
    logger.info(f"[{run_id}] Saved summary metrics -> {sum_csv}")

    def _get(sub_name: str, col: str) -> float:
        return float(sum_df.loc[sub_name, col]) if sub_name in sum_df.index else float("nan")

    paste_metrics = {
        "wv_auc": _get("wildrf_plus_viral", "auc"),
        "wv_bacc": _get("wildrf_plus_viral", "bacc_05"),
        "wv_ece": _get("wildrf_plus_viral", "ece_cal"),
        "unseen_auc": _get("aigb_unseen", "auc"),
        "unseen_bacc": _get("aigb_unseen", "bacc_05"),
        "unseen_ece": _get("aigb_unseen", "ece_cal"),
        "realdeg_bacc": _get("realdeg_heldout", "bacc_05"),
        "seen_auc": _get("aigb_seen", "auc"),
        "temp_T": T_cal,
    }
    print("\n================ PASTE BACK ROW (" + run_id + ") ================")
    print(
        f"| `{run_id}` | {paste_metrics['wv_auc']:.4f} | {paste_metrics['wv_bacc']:.4f} | "
        f"{paste_metrics['wv_ece']:.4f} | {paste_metrics['unseen_auc']:.4f} | "
        f"{paste_metrics['unseen_bacc']:.4f} | {paste_metrics['unseen_ece']:.4f} | "
        f"{paste_metrics['realdeg_bacc']:.4f} | {paste_metrics['seen_auc']:.4f} | "
        f"{paste_metrics['temp_T']:.3f} | `{ckpt_sha}` |"
    )
    print("=================================================================\n")
    return paste_metrics


# ============================================================================
# Main Runner: Train + Eval per Run ID
# ============================================================================
def run_one(
    run_id: str,
    stage: str,
    paths: Dict[str, Path],
    epochs_override: Optional[int],
    subsample_override: Optional[float],
    batch_size: int,
    seed: int,
    eval_max_per_group: int,
    skip_eval: bool = False,
):
    if run_id not in ABLATION_GRID:
        raise ValueError(f"Unknown run_id={run_id!r}. Valid: {list(ABLATION_GRID)}")
    spec = ABLATION_GRID[run_id]
    run_out_dir = paths["output_root"] / run_id
    run_out_dir.mkdir(parents=True, exist_ok=True)

    code_dir = paths["repo_src"] / "code"
    git_sha = get_git_sha(paths["repo_src"])
    bfree_ckpt = resolve_bfree_ckpt(paths["bfree_weights_dir"])

    epochs = epochs_override if epochs_override is not None else (3 if stage == "screening" else 5)
    subsample = subsample_override if subsample_override is not None else (0.25 if stage == "screening" else 1.0)

    cfg_payload = {**spec, "run_id": run_id, "stage": stage, "epochs": epochs,
                   "train_subsample_frac": subsample, "batch_size": batch_size, "seed": seed}
    config_hash = hashlib.sha256(json.dumps(cfg_payload, sort_keys=True).encode()).hexdigest()[:16]

    logger.info(f"========== START {run_id} ({spec['desc']}) ==========")
    logger.info(f"stage={stage} | epochs={epochs} | subsample={subsample} | seed={seed} | out={run_out_dir}")

    if not spec["train"]:
        # R0: B-Free official reference evaluation only
        ckpt_sha = sha256_file(bfree_ckpt)
        if not skip_eval:
            ev = load_bfree_official_for_eval(code_dir, bfree_ckpt, device="cuda:0")
            evaluate_run_yardstick_v2(
                run_id, ev, paths, run_out_dir, git_sha, ckpt_sha, config_hash,
                seed=seed, max_per_group=eval_max_per_group,
            )
        return

    # 1. Run Step-0 Gate before spending GPU hours if initializing from B-Free
    if spec["init_from"] == "bfree":
        verify_step0_gate(code_dir, bfree_ckpt, n_images=8, img_size=504)

    # 2. Build accelerate launch command for train_lora.py
    cmd = [
        "accelerate", "launch",
        "--num_processes", "1",
        "--num_machines", "1",
        "--mixed_precision", "bf16",
        "--dynamo_backend", "no",
        "train_lora.py",
        "--train_data_root", str(paths["train_data_root"]),
        "--ai_genbench_dir", str(paths["ai_genbench_dir"]),
        "--output_dir", str(run_out_dir),
        "--init_from", str(spec["init_from"]),
        "--epochs", str(epochs),
        "--img_size", "504",
        "--batch_size", str(batch_size),
        "--lr", "1e-4",
        "--weight_decay", "1e-4",
        "--max_grad_norm", "1.0",
        "--scheduler_eta_min", "1e-7",
        "--lambda_dcs", str(spec["lambda_dcs"]),
        "--dcs_tau", "0.07",
        "--label_smoothing", str(spec["label_smoothing"]),
        "--use_lib", str(spec["use_lib"]),
        "--use_gsr", str(spec["use_gsr"]),
        "--use_dcs", str(spec["use_dcs"]),
        "--head_mode", str(spec["head_mode"]),
        "--gsr_zero_init", str(spec["gsr_zero_init"]),
        "--lib_beta_init", str(spec["lib_beta_init"]),
        "--loss_mode", str(spec["loss_mode"]),
        "--w_deg", str(spec["w_deg"]),
        "--lam_cons", str(spec["lam_cons"]),
        "--cons_T", str(spec["cons_T"]),
        "--p_aug_on_clean", str(spec["p_aug_on_clean"]),
        "--mix_preset", str(spec["mix_preset"]),
        "--fmt_match", str(spec["fmt_match"]),
        "--res_match", str(spec["res_match"]),
        "--train_subsample_frac", str(subsample),
        "--select", "last",
        "--seed", str(seed),
    ]
    if spec["init_from"] == "bfree":
        cmd += ["--bfree_ckpt", str(bfree_ckpt)]
    else:
        require_file(paths["dinov2_sd"], "dinov2_sd")
        cmd += ["--dinov2_sd", str(paths["dinov2_sd"])]

    logger.info(f"[{run_id}] Launching training: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=str(code_dir), check=True)

    ckpt_last = run_out_dir / "bfree_globalforge_lora_r16_last.pth"
    require_file(ckpt_last, f"{run_id} trained checkpoint (last)")
    ckpt_sha = sha256_file(ckpt_last)

    if not skip_eval:
        ev = load_trained_ours_for_eval(code_dir, ckpt_last, device="cuda:0")
        evaluate_run_yardstick_v2(
            run_id, ev, paths, run_out_dir, git_sha, ckpt_sha, config_hash,
            seed=seed, max_per_group=eval_max_per_group,
        )


def parse_cli():
    p = argparse.ArgumentParser(description="Kaggle GPU Runner for P2.F Ablation Grid (R0 -> R5-l)")
    p.add_argument("--runs", type=str, required=True,
                   help="Comma-separated run IDs to execute, e.g. 'R0,R1,R1b' or 'R2,R3,R4,R5'")
    p.add_argument("--stage", choices=["screening", "confirm"], default="screening",
                   help="screening: 25%% data, 3 epochs | confirm: 100%% data, 5 epochs")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--train_subsample_frac", type=float, default=None)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval_max_per_group", type=int, default=200)
    p.add_argument("--skip_eval", action="store_true")
    for k, v in DEFAULT_PATHS.items():
        p.add_argument(f"--{k}", type=str, default=v)
    return p.parse_args()


def main():
    args = parse_cli()
    if not torch.cuda.is_available():
        raise RuntimeError("FAIL-FAST: CUDA GPU is not available! Enable GPU accelerator in Kaggle settings.")

    paths = {k: Path(getattr(args, k)) for k in DEFAULT_PATHS}
    require_dir(paths["repo_src"] / "code", "repo_src/code")
    require_dir(paths["bfree_weights_dir"], "bfree_weights_dir")

    run_ids = [r.strip() for r in args.runs.split(",") if r.strip()]
    for r in run_ids:
        if r not in ABLATION_GRID:
            raise ValueError(f"Invalid run ID {r!r}. Choose from {list(ABLATION_GRID)}")
        if ABLATION_GRID[r]["train"]:
            require_dir(paths["train_data_root"], "train_data_root")
            require_dir(paths["ai_genbench_dir"], "ai_genbench_dir")

    for r in run_ids:
        run_one(
            run_id=r,
            stage=args.stage,
            paths=paths,
            epochs_override=args.epochs,
            subsample_override=args.train_subsample_frac,
            batch_size=args.batch_size,
            seed=args.seed,
            eval_max_per_group=args.eval_max_per_group,
            skip_eval=args.skip_eval,
        )


if __name__ == "__main__":
    main()
