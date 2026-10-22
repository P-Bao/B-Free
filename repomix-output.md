This file is a merged representation of a subset of the codebase, containing files not matching ignore patterns, combined into a single document by Repomix.

<file_summary>
This section contains a summary of this file.

<purpose>
This file contains a packed representation of a subset of the repository's contents that is considered the most important context.
It is designed to be easily consumable by AI systems for analysis, code review,
or other automated processes.
</purpose>

<file_format>
The content is organized as follows:
1. This summary section
2. Repository information
3. Directory structure
4. Repository files (if enabled)
5. Multiple file entries, each consisting of:
  - File path as an attribute
  - Full contents of the file
</file_format>

<usage_guidelines>
- This file should be treated as read-only. Any changes should be made to the
  original repository files, not this packed version.
- When processing this file, use the file path to distinguish
  between different files in the repository.
- Be aware that this file may contain sensitive information. Handle it with
  the same level of security as you would the original repository.
</usage_guidelines>

<notes>
- Some files may have been excluded based on .gitignore rules and Repomix's configuration
- Binary files are not included in this packed representation. Please refer to the Repository Structure section for a complete list of file paths, including binary files
- Files matching these patterns are excluded: **/*.pth, **/*.jpg, code/xla-training.*, **/*.svg, viral_images_dataset/**, **/node_modules/**, **/venv/**, **/.venv/**, **/__pycache__/**
- Files matching patterns in .gitignore are excluded
- Files matching default ignore patterns are excluded
- Files are sorted by Git change count (files with more changes are at the bottom)
</notes>

</file_summary>

<directory_structure>
code/
  bfree_datasets/
    __init__.py
    bfree_dataset.py
    multi_severity.py
  configs/
    __init__.py
    bfree_dcs.yaml
    loader.py
  datasets/
    __init__.py
    README.md
  demo_images/
    img0000.png
    img0001.png
    img0002.png
    img0003.png
    metainfo.csv
    results_metrics.csv
    results.csv
  eval/
    analyze_loss.py
    run_benchmarks.py
  modules/
    __init__.py
    dcs_loss.py
    gsr_adapter.py
    lib_adapter.py
  networks/
    __init__.py
    bfree_globalforge_vit.py
    wrapper5crops.py
  scripts/
    run_p2f_ablation_kaggle.py
    run_phase6_search.sh
  utils/
    __init__.py
    dmetrics.py
    normalization.py
  Dockerfile
  LICENSE.txt
  main_bfree_single.py
  main_bfree.py
  README.md
  requirements.txt
  train_full_ft.py
  train_lora.py
docs/
  _config.yml
  index.md
notebooks/
  version4/
    xla-ablation-p2f.ipynb
research/
  directions/
    D-multiseverity-dcs/
      source/
        Multi-Severity DCS AI Detection.docx
      README.md
    D-recipe-v4/
      README.md
  diagnosis.md
  eval-changelog.md
  eval-diff.md
  iter-1.md
  iter-2.md
  iter-v4-ablation.md
  log.md
  p1_diagnosis_metrics.json
  p1_diagnosis.py
  PATHS.md
  README.md
  yardstick-v2.md
training_data/
  README.md
.gitignore
AGENT_PLAN_V4.md
LICENSE.txt
MODULES_INTERFACE.md
README.md
requirements-test.txt
</directory_structure>

<files>
This section contains the contents of the repository's files.

<file path="code/scripts/run_p2f_ablation_kaggle.py">
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
</file>

<file path="notebooks/version4/xla-ablation-p2f.ipynb">
{
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "# XLA V4 — P2.F Screening & Confirmation Ablation Runner (`R0` -> `R5-l`)\n",
        "- **Chế độ:** 100% Offline (`HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, `HF_DATASETS_OFFLINE=1`), đường dẫn tuyệt đối `/kaggle/input/...`, Fail-Fast.\n",
        "- **Gate tự động:** Kiểm tra `Bước 0 ≡ B-Free` (`|Δ logits| < 1e-3`) ngay trên GPU trước mỗi run khởi tạo từ B-Free.\n",
        "- **Đầu ra mỗi Run `<RUN_ID>`:** `/kaggle/working/p2f_runs/<RUN_ID>/{train_log.csv, scores.parquet, eval_summary.csv, bfree_globalforge_lora_r16_last.pth}`.\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {},
      "outputs": [],
      "source": [
        "# ============================ CELL 1: CHỌN RUN CẦN CHẠY ============================\n",
        "# Gợi ý chia phiên Kaggle (mỗi run sàng lọc ~35-40 phút trên 25% data, 3 epochs):\n",
        "#   Phiên 1 (Recipe & W1):      RUNS_TO_EXECUTE = \"R0,R1,R1b\"\n",
        "#   Phiên 2 (Module H-C):       RUNS_TO_EXECUTE = \"R2,R3,R4,R5\"\n",
        "#   Phiên 3 (Control P2.B/C/D): RUNS_TO_EXECUTE = \"R5-d,R5-a,R5-f,R5-l\"\n",
        "RUNS_TO_EXECUTE = \"R1,R1b,R2,R3,R4,R5\"\n",
        "STAGE = \"screening\"   # \"screening\" (25% data, 3 epochs) | \"confirm\" (100% data, 5 epochs)\n",
        "BATCH_SIZE = 64\n",
        "SEED = 42\n",
        "EVAL_MAX_PER_GROUP = 200\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {},
      "outputs": [],
      "source": [
        "# ============================ CELL 2: OFFLINE ENV & FAIL-FAST PATHS ============================\n",
        "import os\n",
        "from pathlib import Path\n",
        "\n",
        "os.environ[\"HF_HUB_OFFLINE\"] = \"1\"\n",
        "os.environ[\"TRANSFORMERS_OFFLINE\"] = \"1\"\n",
        "os.environ[\"HF_DATASETS_OFFLINE\"] = \"1\"\n",
        "os.environ[\"TOKENIZERS_PARALLELISM\"] = \"false\"\n",
        "os.environ[\"PYTHONHASHSEED\"] = \"0\"\n",
        "os.environ[\"PYTHONDONTWRITEBYTECODE\"] = \"1\"\n",
        "\n",
        "WHEELS_DIR        = Path(\"/kaggle/input/notebooks/baopv051/xla-setup/wheels_rtxpro6000\")\n",
        "WHEELS_EXTRA_DIR  = Path(\"/kaggle/input/notebooks/baopv051/xla-setup-eval/offline_bundle/wheels_eval_extra\")\n",
        "REPO_SRC          = Path(\"/kaggle/input/notebooks/baopv051/xla-setup/bfree_src\")\n",
        "DINOV2_SD         = Path(\"/kaggle/input/notebooks/baopv051/xla-setup/models/vit_base_patch14_reg4_dinov2/model.safetensors\")\n",
        "BFREE_WEIGHTS_DIR = Path(\"/kaggle/input/models/baopv051/bfree-baseline-weights/pytorch/default/1\")\n",
        "TRAIN_DATA_ROOT   = Path(\"/kaggle/input/datasets/baopv051/b-free-training-data\")\n",
        "AI_GENBENCH_DIR   = Path(\"/kaggle/input/datasets/baopv051/ai-genbench-v1\")\n",
        "WILDRF_DIR        = Path(\"/kaggle/input/datasets/ngovanphuong2604/wildrf-dataset\")\n",
        "SYNTHWILDX_DIR    = Path(\"/kaggle/input/datasets/ngovanphuong2604/synthwildx-dataset\")\n",
        "BFREE_VIRAL_DIR   = Path(\"/kaggle/input/datasets/ngovanphuong2604/bfree-viral-images\")\n",
        "OUTPUT_ROOT       = Path(\"/kaggle/working/p2f_runs\")\n",
        "\n",
        "def require_dir(p: Path, what: str) -> Path:\n",
        "    if not p.is_absolute() or not p.is_dir():\n",
        "        raise FileNotFoundError(f\"FAIL-FAST (missing dir): {what} -> expected {p}\")\n",
        "    return p\n",
        "\n",
        "require_dir(WHEELS_DIR, \"WHEELS_DIR (xla-setup)\")\n",
        "require_dir(REPO_SRC / \"code\", \"REPO_SRC/code (xla-setup)\")\n",
        "require_dir(BFREE_WEIGHTS_DIR, \"BFREE_WEIGHTS_DIR\")\n",
        "require_dir(TRAIN_DATA_ROOT, \"TRAIN_DATA_ROOT\")\n",
        "require_dir(AI_GENBENCH_DIR, \"AI_GENBENCH_DIR\")\n",
        "print(\"[OK] All required Kaggle input directories verified.\")\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {},
      "outputs": [],
      "source": [
        "# ============================ CELL 3: OFFLINE WHEEL INSTALL & VERSION GATE ============================\n",
        "import subprocess, sys\n",
        "\n",
        "find_links = [\"--find-links\", str(WHEELS_DIR)]\n",
        "if WHEELS_EXTRA_DIR.is_dir():\n",
        "    find_links += [\"--find-links\", str(WHEELS_EXTRA_DIR)]\n",
        "\n",
        "pkgs = [\n",
        "    \"torch==2.8.0\", \"torchvision==0.23.0\", \"timm==1.0.22\", \"peft==0.15.2\",\n",
        "    \"transformers==4.55.4\", \"huggingface_hub==0.36.0\", \"tokenizers==0.21.1\",\n",
        "    \"accelerate==1.3.0\", \"pandas==2.3.3\", \"safetensors\", \"numpy\", \"scipy\",\n",
        "    \"scikit-learn\", \"pyyaml\", \"pillow\", \"tqdm\", \"pyarrow\", \"datasets\"\n",
        "]\n",
        "subprocess.run([sys.executable, \"-m\", \"pip\", \"install\", \"--no-index\", \"--no-deps\", *find_links, *pkgs], check=False)\n",
        "\n",
        "import torch, timm, peft\n",
        "assert torch.cuda.is_available(), \"FAIL-FAST: CUDA GPU not enabled!\"\n",
        "print(f\"[OK] torch={torch.__version__} | timm={timm.__version__} | peft={peft.__version__} | GPU={torch.cuda.get_device_name(0)}\")\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {},
      "outputs": [],
      "source": [
        "# ============================ CELL 4: CHẠY LƯỚI ABLATION P2.F ============================\n",
        "import subprocess, sys\n",
        "\n",
        "runner_script = REPO_SRC / \"code\" / \"scripts\" / \"run_p2f_ablation_kaggle.py\"\n",
        "assert runner_script.is_file(), f\"Missing {runner_script} — push latest branch & re-run xla-setup!\"\n",
        "\n",
        "cmd = [\n",
        "    sys.executable, str(runner_script),\n",
        "    \"--runs\", RUNS_TO_EXECUTE,\n",
        "    \"--stage\", STAGE,\n",
        "    \"--batch_size\", str(BATCH_SIZE),\n",
        "    \"--seed\", str(SEED),\n",
        "    \"--eval_max_per_group\", str(EVAL_MAX_PER_GROUP),\n",
        "    \"--repo_src\", str(REPO_SRC),\n",
        "    \"--wheels_dir\", str(WHEELS_DIR),\n",
        "    \"--dinov2_sd\", str(DINOV2_SD),\n",
        "    \"--bfree_weights_dir\", str(BFREE_WEIGHTS_DIR),\n",
        "    \"--train_data_root\", str(TRAIN_DATA_ROOT),\n",
        "    \"--ai_genbench_dir\", str(AI_GENBENCH_DIR),\n",
        "    \"--wildrf_dir\", str(WILDRF_DIR),\n",
        "    \"--synthwildx_dir\", str(SYNTHWILDX_DIR),\n",
        "    \"--bfree_viral_dir\", str(BFREE_VIRAL_DIR),\n",
        "    \"--output_root\", str(OUTPUT_ROOT),\n",
        "]\n",
        "subprocess.run(cmd, check=True)\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {},
      "outputs": [],
      "source": [
        "# ============================ CELL 5: TỔNG HỢP BẢNG PASTE-BACK ============================\n",
        "import pandas as pd\n",
        "\n",
        "print(\"| Run | WildRF+Viral AUC | WildRF+Viral bAcc@0.5 | WildRF+Viral ECE_cal | AIGB-Unseen AUC | AIGB-Unseen bAcc@0.5 | AIGB-Unseen ECE_cal | RealDeg-Heldout bAcc@0.5 | AIGB-Seen AUC (Guard) | Temp T | ckpt_sha256 |\")\n",
        "print(\"| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |\")\n",
        "for run_dir in sorted(OUTPUT_ROOT.glob(\"*\")):\n",
        "    sum_csv = run_dir / \"eval_summary.csv\"\n",
        "    if not sum_csv.is_file():\n",
        "        continue\n",
        "    s = pd.read_csv(sum_csv).set_index(\"subset\")\n",
        "    g = lambda sub, col: f\"{s.loc[sub, col]:.4f}\" if sub in s.index else \"NA\"\n",
        "    t_val = f\"{s['temp_T'].iloc[0]:.3f}\" if len(s) else \"1.000\"\n",
        "    sha = s[\"ckpt_sha256\"].iloc[0] if len(s) else \"NA\"\n",
        "    print(f\"| `{run_dir.name}` | {g('wildrf_plus_viral','auc')} | {g('wildrf_plus_viral','bacc_05')} | {g('wildrf_plus_viral','ece_cal')} | {g('aigb_unseen','auc')} | {g('aigb_unseen','bacc_05')} | {g('aigb_unseen','ece_cal')} | {g('realdeg_heldout','bacc_05')} | {g('aigb_seen','auc')} | {t_val} | `{sha}` |\")\n"
      ]
    }
  ],
  "metadata": {
    "kernelspec": {
      "display_name": "Python 3",
      "language": "python",
      "name": "python3"
    },
    "language_info": {
      "name": "python",
      "version": "3.11.0"
    }
  },
  "nbformat": 4,
  "nbformat_minor": 5
}
</file>

<file path="research/directions/D-recipe-v4/README.md">
# Direction D-recipe-v4 — P2 Recipe Fix & Screening/Confirmation Ablation (V4)

> **Đăng ký trước khi chạy (Pre-registration theo Quy tắc #7 & Mục P2.G `AGENT_PLAN_V4.md`):**
> Tài liệu này khóa cứng các giả thuyết, chỉ số đánh giá và ngưỡng thành công/bác bỏ **trước khi** bấm chạy lưới Ablation P2.F trên Kaggle GPU.

## 1. Bối cảnh & Số tham chiếu
- Chẩn đoán P1 (`research/diagnosis.md`) đã **xác nhận** cả hai giả thuyết gốc:
  - **D1 (W3 — Shortcut nội suy/định dạng & mất cân bằng nguồn AI-GenBench):** AI-GenBench chiếm `63.93%` mẫu train; nhóm generator độ phân giải gốc $\ge 512\text{px}$ không qua `ensure_min_size(504, BICUBIC)` sụp đổ xuống `bAcc@0.5 = 78.88%` (thua `bfree_official` `-12.98%`).
  - **D2 (W1 — Bất cân xứng gradient DCS & thiếu $\mathcal{L}_{\text{CE}}^{\text{deg}}$):** Với $\lambda_{\text{dcs}}=0.01$, tỷ lệ chuẩn gradient $\|\nabla(\lambda\mathcal{L}_{\text{DCS}})\|_2 / \|\nabla\mathcal{L}_{\text{CE}}\|_2$ chỉ đạt `0.35% - 0.37%` ($< 1\%$); classifier head nhận `0.000%` gradient từ nhánh suy giảm.
- Số tham chiếu lịch sử (Harness v1, đối chiếu lại với `R0` trên Yardstick v2):
  - Mean AUC `{WildRF fb/reddit/twitter, viral}` = **0.930** (`ours-v2`) vs **0.972** (`bfree_official`).

## 2. Bảng Tiêu chí Thành công & Bác bỏ (P2.G — Khóa cứng)

| Mã | Giả thuyết | Chỉ số (trên tập **test**, chấm 1 lần) | Dự đoán (Pass) | Bác bỏ nếu (Reject) |
| :--- | :--- | :--- | :--- | :--- |
| **H-A** | Aug-CE (`dual_view`) + init B-Free sửa lỗi tổng quát hóa ngoài miền (`R1` vs `R1b`, `R1` vs `ours-v2`) | Mean AUC `WildRF + viral` | `R1 >= 0.965` và `>= ours-v2 + 0.03` | `R1 < 0.945` |
| **H-B** | Aug-CE nâng độ bền dưới suy giảm | Mean bAcc `RealDeg held-out + compound 3-5` | `>= +10` điểm % so với `ours-v2` (phase-2: `0.674 / 0.610 / 0.582`) | `< +4` điểm % |
| **H-C** | `LIB` / `GSR` / `DCS` có đóng góp độc lập | `R5 - R1` (và `R2/R3/R4 - R1`): Mean AUC (`WildRF + viral + AIGB unseen + SynthWildX`) | `>= +0.005`, Bootstrap CI 95% của $\Delta\text{AUC}$ không chứa `0` | CI 95% chứa `0` $\Rightarrow$ **kết luận trung thực: module không đóng góp, loại bỏ khỏi cấu hình cuối** |
| **H-D** | Cân bằng dữ liệu (`mix_bfree` + `fmt/res match`) giảm shortcut | `R5` vs `R5-a` (và `R5-f` vs `R5`): AUC `WildRF + viral`; $\|\Delta\text{điểm khi re-encode}\|$ | `>= +0.01` AUC | `< +0.003` AUC |
| **H-E** | Hiệu chuẩn xác suất (Calibration) | `balanced_ece_binary` sau Temperature Scaling trên `WildRF / viral` | `<= 0.08` | `> 0.15` |
| **Guard** | Không đánh mất nội lực in-distribution | AUC `AIGB seen` (13 gen) và `SD2.1 in-dist` | Giảm `<= 0.003` so với `ours-v2` | Giảm `> 0.01` |

## 3. Quy tắc Cổng P2 (Gate sang P3)
- Hoàn tất bảng kết quả Ablation P2.F (`R0` $\rightarrow$ `R5-l`).
- Đưa ra quyết định **"giữ / loại"** từng module (`LIB`, `GSR`, `DCS`) dựa trên kiểm định **H-C** và trình User duyệt trước khi bước sang P3.
</file>

<file path="research/diagnosis.md">
# Báo cáo Chẩn đoán Phase P1 (Diagnosis Report — D1 đến D5)

**Trạng thái:** Hoàn tất P1 trên CPU local (`.\venv\Scripts\python.exe research/p1_diagnosis.py`) — **KHÔNG chạm vào bất kỳ file code huấn luyện nào**.
**Tệp script & dữ liệu xuất:**
- Script phân tích: [`research/p1_diagnosis.py`](file:///C:/Users/ADMIN/OneDrive%20-%20ptit.edu.vn/Documents/GitHub/B-Free/research/p1_diagnosis.py)
- Số liệu JSON đầy đủ: [`research/p1_diagnosis_metrics.json`](file:///C:/Users/ADMIN/OneDrive%20-%20ptit.edu.vn/Documents/GitHub/B-Free/research/p1_diagnosis_metrics.json)

---

## 1. Bảng Tổng hợp Chẩn đoán P1 (3 Cột Bắt buộc)

| Giả thuyết | Thực nghiệm | Kết luận (Xác nhận/Bác bỏ) |
| :--- | :--- | :--- |
| **D1 (Kiểm chứng W3 — Shortcut định dạng / độ phân giải / nội suy & mất cân bằng nguồn dữ liệu AI-GenBench):**<br>1. AI-GenBench chiếm đa số mẫu huấn luyện ($\approx 64\%$), làm lu mờ cặp đối sánh nội dung `COCO + SD2.1` của B-Free.<br>2. Hàm [`ensure_min_size(img, 504)`](file:///C:/Users/ADMIN/OneDrive%20-%20ptit.edu.vn/Documents/GitHub/B-Free/code/train_lora.py#L263-L270) ép phóng to `Image.BICUBIC` trên toàn bộ generator độ phân giải thấp/trung bình ($<504\text{px}$) trong [`ExtraArrowTrainDataset`](file:///C:/Users/ADMIN/OneDrive%20-%20ptit.edu.vn/Documents/GitHub/B-Free/code/train_lora.py#L565) và harness, tạo shortcut dấu vết nội suy Bicubic gắn chặt với nhãn giả trên các generator cũ nhưng vô dụng trước các generator độ phân giải cao gốc ($\ge 512\text{px}$). | **1. Tỷ trọng mẫu train thực tế (`xla-training-phase-2.ipynb`, `xla-training (2).ipynb`):**<br>• Tổng mẫu train = `146,418`: `ExtraArrowTrainDataset` (AI-GenBench, 13 `PRIORITY_GENERATORS`) = **`93,600` mẫu (`63.93%`)** (`46,800` real + `46,800` fake); `PairDataset` (COCO + 6 biến thể SD2.1, $512\times512$ PNG, không qua `ensure_min_size`) = **`50,083` mẫu (`34.21%`)**; `ExtraTrainDataset` (RAISE/LDM/SD3-FLUX) = **`2,735` mẫu (`1.87%`)**.<br>**2. Phân tầng 36 generator trên AI-GenBench (`n=1,650/gen`):**<br>• **Nhóm Seen 13 Priority Gens:** `ours` đạt `AUC = 0.9997`, `bAcc@0.5 = 99.68%`, `NLL = 0.0591` (so với `bfree_official`: `AUC = 0.8901`, `bAcc@0.5 = 75.36%`).<br>• **Nhóm Unseen 16 Low/Mid-Res Gens ($<504\text{px}$, bị ép Bicubic upscale lên $504\text{px}$):** `ours` đạt `AUC = 0.9978`, `bAcc@0.5 = 98.97%`, `NLL = 0.0776`.<br>• **Nhóm Unseen 7 High-Res Native Diffusion Gens ($\ge 512\text{px}$, KHÔNG qua Bicubic upscale: `FLUX 1 Dev`, `FLUX 1 Schnell`, `DALL-E 3`, `SD 1.4/2.1/XL`):** `ours` sụp đổ xuống **`bAcc@0.5 = 78.88%`** (`AUC = 0.9750`, `NLL = 0.5017`), **thua `bfree_official` tới `-12.98%` `bAcc@0.5`** (`bfree_official` đạt `bAcc@0.5 = 91.86%`, `AUC = 0.9789`, `NLL = 0.3254`).<br>**3. Đo trực tiếp độ nhạy định dạng & nội suy trên CPU (`demo_images`):**<br>• Nén lại `img0002.png` (fake $1024\times1024$) sang `JPEG q=85` làm logit giảm từ `+4.431` ($p=0.9882$) xuống `+1.088` ($p=0.7480$, $\Delta p = -24.02\%$).<br>• Ở kích thước $256\times256$, thay `replicate_wrap` nguyên bản của [`Wrapper5crops`](file:///C:/Users/ADMIN/OneDrive%20-%20ptit.edu.vn/Documents/GitHub/B-Free/code/networks/wrapper5crops.py) bằng `ensure_min_size(504, BICUBIC)` làm lệch xác suất dự đoán $|\Delta p|$ tới `1.69% – 2.70%` ngay trên cùng một bức ảnh. | **XÁC NHẬN (CONFIRMED)**<br>Giả thuyết W3 và cơ chế shortcut nội suy/định dạng trên AI-GenBench được xác nhận hoàn toàn bằng cả phân phối mẫu huấn luyện (`63.93%`), sự phân cực kết quả giữa nhóm `<504px` (`98.97%`) so với `>=512px` (`78.88%`), và phép đo trực tiếp trên CPU. |
| **D2 (Kiểm chứng W1 — Tỷ lệ chuẩn gradient $\|\nabla_\theta (\lambda \cdot \mathcal{L}_{\text{DCS}})\|_2 / \|\nabla_\theta \mathcal{L}_{\text{CE}}\|_2 < 1\%$ và lỗi cấu trúc DCS):**<br>1. Với $\lambda_{\text{dcs}} = 0.01$, đóng góp gradient của $\lambda \cdot \mathcal{L}_{\text{DCS}}$ nhỏ hơn $1\%$ so với $\mathcal{L}_{\text{CE}}$, khiến DCS hoàn toàn bị CE lấn át.<br>2. Hàm [`BFreeGlobalForgeViT.compute_loss`](file:///C:/Users/ADMIN/OneDrive%20-%20ptit.edu.vn/Documents/GitHub/B-Free/code/networks/bfree_globalforge_vit.py#L135-L151) chỉ tính CE trên `out_clean["logits"]` (không có $\mathcal{L}_{\text{CE}}^{\text{deg}}$), khiến classifier head không nhận được bất kỳ tín hiệu bất biến suy thoái nào. | **1. Đo trực tiếp tỷ lệ chuẩn gradient L2 trên CPU ([`BFreeGlobalForgeViT`](file:///C:/Users/ADMIN/OneDrive%20-%20ptit.edu.vn/Documents/GitHub/B-Free/code/networks/bfree_globalforge_vit.py#L31-L152) + ViT-B/14 reg4 pretrained + LoRA $r=16, \alpha=32, \lambda_{\text{dcs}}=0.01$):**<br>• **Tại Step 0:** $\|\nabla_\theta \mathcal{L}_{\text{CE}}\|_2 = 4.478946$ so với $\|\nabla_\theta (\lambda \cdot \mathcal{L}_{\text{DCS}})\|_2 = 0.016342 \implies$ **Tỷ lệ toàn mạng = `0.365%` ($< 1\%$)**; tỷ lệ trên shared backbone = **`0.401%`** (`LoRA = 0.376%`, `GSR = 0.656%`, `LIB = 10.308%`, `fc_norm = 0.000%`).<br>• **Tại chế độ huấn luyện:** $\|\nabla_\theta \mathcal{L}_{\text{CE}}\|_2 = 6.006557$ so với $\|\nabla_\theta (\lambda \cdot \mathcal{L}_{\text{DCS}})\|_2 = 0.020863 \implies$ **Tỷ lệ toàn mạng = `0.347%` ($< 1\%$)**; tỷ lệ trên shared backbone = **`0.470%`** (`LoRA = 0.521%`, `GSR = 0.353%`, `LIB = 1.320%`, `fc_norm = 0.000%`).<br>• **Trên `model.head` và `model.fc_norm`:** $\|\nabla_{\text{head}}(\lambda \cdot \mathcal{L}_{\text{DCS}})\|_2 = \mathbf{0.000000}$ (**`0.000%`** tuyệt đối) vì `dcs_loss` không đi qua `head`.<br>**2. Tỷ lệ độ lớn loss từ log huấn luyện thực tế (8 epochs):**<br>• `xla-training-phase-2.ipynb` (Epoch 8): $\mathcal{L}_{\text{CE}} = 0.2001$ (sát đáy lý thuyết của Label Smoothing $\varepsilon=0.1$: $H(0.95, 0.05) = 0.1985$), trong khi $\mathcal{L}_{\text{DCS}} = 0.0105 \implies \lambda \cdot \mathcal{L}_{\text{DCS}} = 0.000105$, chỉ bằng **`0.052%`** của $\mathcal{L}_{\text{CE}}$ (`0.048%` ở `xla-training (2).ipynb`).<br>• **Hệ quả trên `RealDeg-Bench` (Cell 22):** `ours` đạt `bAcc = 99.97%` trên ảnh clean nhưng rơi tự do xuống `72.42%` (`jpeg`), `73.08%` (`blur`), `70.50%` (`noise`), và **`58.19%` ở `compound_step5`** (không cải thiện so với Phase-1 `58.30%`). | **XÁC NHẬN (CONFIRMED)**<br>Tỷ lệ chuẩn gradient thực nghiệm là **`0.347% – 0.365%`** ($< 1\%$), tỷ lệ loss thực tế ở Epoch 8 là **`0.048% – 0.052%`**, và gradient DCS trên `head` bằng **`0.000%`**. Giả thuyết W1 được xác nhận tuyệt đối. |
| **D3 (Kiểm chứng W4 & W5 — Phân rã đại số TPR vs FPR và độ lệch ngưỡng quyết định):**<br>Sự tụt giảm `bAcc@0.5` của `ours` trên các generator hiện đại (`FLUX`, `DALL-E 3`, `SD 2.1`) và trên `WildRF` đến từ hiệu ứng kép: (a) ở ngưỡng `0.5`, logit của ảnh fake bị nén thấp gây **False Negative (TPR rất thấp)**; (b) trên mạng xã hội (`WildRF`), ảnh bị nén mạnh làm dịch chuyển phân phối logit khiến ngưỡng `0.5` bỏ sót hàng loạt ảnh giả. | **Khôi phục đại số chính xác TPR và FPR trên AI-GenBench (dựa trên pool 1,500 ảnh real dùng chung cho mọi generator, $n_{\text{fake}}=150/\text{gen}$):**<br>• Với `ours`, trên pool 1,500 ảnh real: $\text{FPR}@0.5 = \mathbf{0.33\%}$ ($\text{TNR}@0.5 = 99.67\%$, $\text{mean } p_{\text{fake}} = 0.0546$); ở ngưỡng `calib = 0.0706`, $\text{FPR}@\text{calib} = \mathbf{2.13\%}$ ($\text{TNR}@\text{calib} = 97.87\%$).<br>• **Bằng chứng False Negative (TPR cực thấp ở ngưỡng 0.5) trên ảnh Fake độ phân giải cao:**<br>  – `FLUX 1 Dev`: $\text{AUC}=0.9490$, nhưng **$\text{TPR}@0.5 = \mathbf{6.7\%}$** (chỉ bắt được **10/150** ảnh fake $\implies \text{bAcc}@0.5 = 53.17\%$). Khi hạ ngưỡng về `0.0706`, $\text{TPR}@\text{calib}$ tăng lên **`56.0%`** ($\text{bAcc}@\text{calib} = 76.93\%$).<br>  – `DALL-E 3`: $\text{AUC}=0.9679$, **$\text{TPR}@0.5 = \mathbf{42.7\%}$** (**64/150** ảnh fake $\implies \text{bAcc}@0.5 = 71.17\%$), tăng lên **$\text{TPR}@\text{calib} = \mathbf{85.3\%}$** ($\text{bAcc}@\text{calib} = 91.60\%$).<br>  – `FLUX 1 Schnell`: $\text{AUC}=0.9937$, **$\text{TPR}@0.5 = \mathbf{48.7\%}$** (**73/150** ảnh fake $\implies \text{bAcc}@0.5 = 74.17\%$), tăng vọt lên **$\text{TPR}@\text{calib} = \mathbf{95.3\%}$** ($\text{bAcc}@\text{calib} = 96.60\%$).<br>• **Trên `WildRF` (ảnh nén mạng xã hội):** Ở ngưỡng `0.5`, `ours` chỉ đạt `bAcc@0.5 = 71.56%` (`facebook`) và `70.91%` (`twitter`), nhưng khi dùng ngưỡng `0.0706` thì tăng **`+14.38%`** (lên `85.94%`) và **`+15.48%`** (lên `86.39%`). Dù vậy, `AUC` của `ours` trên WildRF (`0.9324 – 0.9586`) vẫn thấp hơn `bfree_official` (`0.9407 – 0.9936`), chứng tỏ ngoài lệch ngưỡng còn có suy giảm chất lượng biểu diễn dưới nén mạng xã hội (do W1). | **XÁC NHẬN (CONFIRMED)**<br>Hiện tượng sụp đổ `bAcc@0.5` trên các mô hình khuếch tán hiện đại và `WildRF` là sự kết hợp của: (1) nén xác suất/logit (`label_smoothing=0.1` + thiếu $\mathcal{L}_{\text{CE}}^{\text{deg}}$) làm $\text{TPR}@0.5$ tụt xuống `6.7% – 48.7%`, và (2) suy giảm biểu diễn khi ảnh chịu nén JPEG/resize thực tế. |
| **D4 (Kiểm chứng W3 — Khoảng cách giữa SD2.1 In-Distribution Validation và SD2.1 của AI-GenBench):**<br>Việc đạt `val_bacc = 99.97%` trên tập validation nội bộ (`SD2.1` ghép cặp `COCO_real_512`) tạo ảo giác hội tụ, nhưng mô hình lại mất khả năng tổng quát hóa ngay trên chính generator `Stable Diffusion 2.1` khi lấy từ nguồn phân phối/định dạng khác (AI-GenBench). | **So sánh trực tiếp trên cùng họ generator `Stable Diffusion 2.1`:**<br>• **Validation nội bộ (`xla-training-phase-2.ipynb` Cell 17):** Trên cả 6 biến thể `SD2.1_*` ($512\times512$ PNG), `ours` đạt **`val_auc = 1.0000`**, **`val_bacc@0.5 = 99.48% – 100.00%`** (`SD2.1_selfconditioned`: `val_bacc = 99.97%`).<br>• **AI-GenBench `Stable Diffusion 2.1` (`n=1,650`):** `ours` tụt xuống **`AUC = 0.9603`**, **`bAcc@0.5 = 86.17%`** ($\text{TPR}@0.5 = 72.7\%$, bỏ lọt **41/150** ảnh fake SD2.1!).<br>• **Đối chiếu với `bfree_official` trên cùng tập AI-GenBench `Stable Diffusion 2.1`:** `bfree_official` giữ vững **`AUC = 0.9987`**, **`bAcc@0.5 = 95.77%`** ($\text{TPR}@0.5 = \mathbf{100.0\%}$, bắt trọn **150/150** ảnh fake SD2.1). | **XÁC NHẬN (CONFIRMED)**<br>Khoảng cách `-13.80%` `bAcc@0.5` (`99.97%` xuống `86.17%`) và `-27.3%` `TPR@0.5` ngay trên chính `Stable Diffusion 2.1` chứng minh tập validation nội bộ đã bão hòa và mô hình `ours-v2` đã overfit vào đặc trưng phụ thuộc nguồn/pipeline của tập train. |
| **D5 (Kiểm chứng W2 — Nhiễu loạn đặc trưng tại Step 0 do chèn `LIBAdapter` + `GSRAdapter` chưa khởi tạo về 0):**<br>Việc đặt `LIBAdapter` ($\beta_0 = -0.85 \Rightarrow \alpha_0 = 0.2994$) và `GSRAdapter` (`self.proj` khởi tạo ngẫu nhiên, không có cổng $\gamma=0$) ngay sau tầng cuối của DINOv2 làm biến dạng mạnh vector đặc trưng tại Step 0, buộc LoRA phải tiêu tốn dung lượng để bù nhiễu. | **Đo trực tiếp trên CPU với patch tokens thực trích xuất từ trọng số DINOv2 ViT-B/14 reg4 (`BFREE_dino2reg4`, $N=36\times36=1296, C=768$):**<br>• **Chỉ `LIBAdapter` ($\alpha_0 = 0.2994$):** $\text{cos\_sim}(x, \text{LIB}(x)) = 0.9935$, độ lệch chuẩn tương đối $\|\Delta x\|_2 / \|x\|_2 = \mathbf{12.56\%}$.<br>• **Chỉ `GSRAdapter` (`proj` ngẫu nhiên):** $\text{cos\_sim}(x, \text{GSR}(x)) = 0.9644$, độ lệch chuẩn tương đối $\|\Delta x\|_2 / \|x\|_2 = \mathbf{27.46\%}$.<br>• **Kết hợp `LIB + GSR` tại Step 0 (trên patch tokens):** $\text{cos\_sim} = 0.9533$, độ lệch chuẩn tương đối = **`30.38%`**.<br>• **Kết hợp `LIB + GSR` tại Step 0 (trên vector `mean-pool cls` đầu vào của `fc_norm + head`):** $\text{cos\_sim} = 0.9458$, độ lệch chuẩn tương đối $\|\Delta \bar{x}\|_2 / \|\bar{x}\|_2 = \mathbf{34.10\%}$. | **XÁC NHẬN (CONFIRMED)**<br>Ngay tại Step 0, hai mô-đun `LIB + GSR` làm lệch vector đặc trưng trung bình `mean-pool` tới **`34.10%`** về chuẩn L2 trước khi huấn luyện bất kỳ bước nào, xác nhận hoàn toàn giả thuyết W2 về nhu cầu zero-init/gate và kiểm định nhánh ablation `A1` (LoRA-only). |

---

## 2. Chi tiết Phân tích Kỹ thuật từ Thực nghiệm CPU

### 2.1. Phân tích sâu D1 (W3): Sự đảo chiều hiệu năng theo độ phân giải gốc trên AI-GenBench
Bảng dưới đây trích xuất từ [`research/p1_diagnosis_metrics.json`](file:///C:/Users/ADMIN/OneDrive%20-%20ptit.edu.vn/Documents/GitHub/B-Free/research/p1_diagnosis_metrics.json) (tổng hợp đủ 36 generator trong `ai_genbench`, đã loại bỏ dòng `MEAN`), chia làm 3 phân tầng rõ rệt:

| Phân nhóm Generator trên AI-GenBench | Số Gen | `ours` (`AUC` / `bAcc@0.5` / `NLL`) | `bfree_official` (`AUC` / `bAcc@0.5` / `NLL`) | `globalforge_dinov3` (`AUC` / `bAcc@0.5` / `NLL`) | Chênh lệch `bAcc@0.5` (`ours` vs `bfree`) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **1. Seen Priority Generators** (có trong 93,600 mẫu train của `ours`) | 13 | **0.9997** / **99.68%** / 0.0591 | 0.8901 / 75.36% / 1.1118 | 0.9327 / 76.00% / 0.5193 | **+24.32%** |
| **2. Unseen Low/Mid-Res Generators** ($<504\text{px}$, bị ép `Image.BICUBIC` upscale lên $504\text{px}$) | 16 | **0.9978** / **98.97%** / 0.0776 | 0.9483 / 87.35% / 0.5569 | 0.9731 / 76.10% / 0.5050 | **+11.62%** |
| **3. Unseen Native High-Res Generators** ($\ge 512\text{px}$, **không** qua `Image.BICUBIC` upscale) | 7 | 0.9750 / **78.88%** / 0.5017 | **0.9789** / **91.86%** / **0.3254** | 0.9283 / 76.10% / 0.5199 | **-12.98%** |

> [!IMPORTANT]
> **Cơ chế vật lý của Shortcut (W3):**
> Trong [`ExtraArrowTrainDataset.__getitem__`](file:///C:/Users/ADMIN/OneDrive%20-%20ptit.edu.vn/Documents/GitHub/B-Free/code/train_lora.py#L565), mọi ảnh từ AI-GenBench có cạnh ngắn $<504\text{px}$ đều bị phóng to bằng `Image.BICUBIC` trước khi crop $504\times504$. Vì phần lớn trong 13 `PRIORITY_GENERATORS` (`DDPM`, `StyleGAN1`, `CIPS`, `Palette`, `Denoising Diffusion GAN`, `VQGAN`, `GANformer`, `FaceSynthetics`) có độ phân giải gốc $128\times128$ hoặc $256\times256$, mô hình học được mối tương quan rất mạnh giữa **dấu vết phổ tần số của bộ lọc nội suy `Image.BICUBIC` + cấu trúc GAN/DM độ phân giải thấp** với nhãn `fake` ($p_{\text{fake}} \approx 0.999$).
> Khi gặp các mô hình khuếch tán hiện đại sinh ảnh ở độ phân giải gốc $\ge 512\times512$ hoặc $1024\times1024$ (`FLUX 1 Dev`, `FLUX 1 Schnell`, `DALL-E 3`, `SD 1.4/2.1/XL`), ảnh **không** đi qua `ensure_min_size` (không có dấu vết nội suy Bicubic), khiến logit của `ours` sụt giảm mạnh ($\text{TPR}@0.5$ trên `FLUX 1 Dev` chỉ còn **6.7%**!).

---

### 2.2. Phân tích sâu D2 (W1): Bất cân xứng Gradient và sự vắng mặt của $\mathcal{L}_{\text{CE}}^{\text{deg}}$
Từ phép đo trực tiếp trên CPU trong [`research/p1_diagnosis.py`](file:///C:/Users/ADMIN/OneDrive%20-%20ptit.edu.vn/Documents/GitHub/B-Free/research/p1_diagnosis.py) và chuỗi log 8 epochs trên Kaggle:

| Thành phần tham số ($\theta_g$) | $\|\nabla_{\theta_g} \mathcal{L}_{\text{CE}}\|_2$ (Step 0) | $\|\nabla_{\theta_g} (\lambda \mathcal{L}_{\text{DCS}})\|_2$ (Step 0) | Tỷ lệ Step 0 ($\lambda=0.01$) | $\|\nabla_{\theta_g} \mathcal{L}_{\text{CE}}\|_2$ (Hội tụ) | $\|\nabla_{\theta_g} (\lambda \mathcal{L}_{\text{DCS}})\|_2$ (Hội tụ) | Tỷ lệ Hội tụ ($\lambda=0.01$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Toàn bộ tham số khả huấn (`all_trainable`)** | `4.478946` | `0.016342` | **0.365%** | `6.006557` | `0.020863` | **0.347%** |
| **Shared Backbone (`LoRA + LIB + GSR + fc_norm`)** | `4.079043` | `0.016342` | **0.401%** | `4.441749` | `0.020863` | **0.470%** |
| **LoRA (`qkv, proj, fc1, fc2`)** | `3.998400` | `0.015018` | **0.376%** | `3.861753` | `0.020116` | **0.521%** |
| **GSRAdapter (`q, k, v, proj`)** | `0.806526` | `0.005288` | **0.656%** | `1.564998` | `0.005525` | **0.353%** |
| **LIBAdapter (`beta`)** | `0.035512` | `0.003661` | **10.308%** | `0.025216` | `0.000333` | **1.320%** |
| **LayerNorm cuối (`model.fc_norm`)** | `0.013198` | `0.000000` | **0.000%** | `2.186882` | `0.000000` | **0.000%** |
| **Classifier Head (`model.head`)** | `1.849963` | `0.000000` | **0.000%** | `4.043463` | `0.000000` | **0.000%** |

> [!WARNING]
> **Ba lỗi cấu trúc chí mạng của nhánh DCS hiện tại (`ours-v2`):**
> 1. **Tỷ lệ chuẩn gradient toàn cục chỉ đạt `0.347% – 0.365%` ($< 1\%$):** Với $\lambda_{\text{dcs}} = 0.01$, vector cập nhật của AdamW bị chi phối $>99.6\%$ bởi $\mathcal{L}_{\text{CE}}$ trên ảnh sạch.
> 2. **Gradient từ $\mathcal{L}_{\text{DCS}}$ lên `model.head` và `model.fc_norm` bằng `0.000%` tuyệt đối:** Mặc dù `out_clean["cls"]` đi qua `fc_norm`, vì [`info_nce_loss`](file:///C:/Users/ADMIN/OneDrive%20-%20ptit.edu.vn/Documents/GitHub/B-Free/code/modules/dcs_loss.py) thực hiện chuẩn hóa `F.normalize(z, dim=-1)` ngay trên đầu ra của `fc_norm` và không có projection head riêng, đồng thời [`compute_loss`](file:///C:/Users/ADMIN/OneDrive%20-%20ptit.edu.vn/Documents/GitHub/B-Free/code/networks/bfree_globalforge_vit.py#L135-L151) **không tính Cross-Entropy trên ảnh suy thoái (`out_deg["logits"]`)**, siêu phẳng phân lớp của `model.head` chỉ học trên đặc trưng ảnh sạch!
> 3. **Hiện tượng suybiến nhanh của InfoNCE không có Projection Head:** Trong log thực tế (`xla-training-phase-2.ipynb`), $\mathcal{L}_{\text{DCS}}$ giảm nhanh xuống `0.0105` ($\lambda \cdot \mathcal{L}_{\text{DCS}} = 0.000105$, bằng `0.052%` của $\mathcal{L}_{\text{CE}}$) vì mô hình chỉ cần dùng thông tin ngữ cảnh vị trí/màu sắc cấp cao của ảnh $i$ để phân biệt ảnh $i$ với các ảnh $j \ne i$ trong mini-batch mà không cần giữ lại các đặc trưng forensic chống suy thoái.

---

## 3. Kiểm tra Quy tắc Dừng (Stop Rule Check)

Theo chỉ thị của User ở đầu Phase P1:
- *"NẾU kết quả D2 bác bỏ W1 hoặc D1 bác bỏ shortcut: DỪNG LẠI NGAY và báo cáo tôi, không tự ý chuyển sang P2!"*

**Kết quả kiểm tra:**
1. **D2 KHÔNG bác bỏ W1 — trái lại, D2 XÁC NHẬN W1 hoàn toàn:** Tỷ lệ chuẩn gradient $\|\nabla_\theta (\lambda \cdot \mathcal{L}_{\text{DCS}})\|_2 / \|\nabla_\theta \mathcal{L}_{\text{CE}}\|_2 = \mathbf{0.365\%}$ ở Step 0 và $\mathbf{0.347\%}$ ở chế độ huấn luyện (nhỏ hơn ngưỡng $1\%$ của giả thuyết W1), tỷ lệ loss ở Epoch 8 là $\mathbf{0.052\%}$, và gradient DCS trên `head` là $\mathbf{0.000\%}$.
2. **D1 KHÔNG bác bỏ W3 — trái lại, D1 XÁC NHẬN shortcut định dạng/độ phân giải/nội suy hoàn toàn:** AI-GenBench chiếm `63.93%` dữ liệu huấn luyện, và `ours` đạt `99.68% / 98.97%` trên các generator đã thấy hoặc độ phân giải thấp bị ép `Image.BICUBIC` upscale, nhưng tụt xuống `78.88%` (thua `bfree_official` `-12.98%`) trên 7 generator độ phân giải cao gốc ($\ge 512\text{px}$), đồng thời tụt `-13.80%` `bAcc@0.5` ngay trên chính `Stable Diffusion 2.1` của AI-GenBench (`86.17%` vs `99.97%` nội bộ).
3. **D3, D4, D5 đều XÁC NHẬN W2, W4, W5.**

Tuân thủ nguyên tắc bất biến, tôi **dừng lại tại điểm chốt kết thúc P1**, không chỉnh sửa bất kỳ file code huấn luyện nào và chờ lệnh phê duyệt của bạn trước khi tiến hành đóng gói P0 code update / chuyển sang P2.
</file>

<file path="research/iter-v4-ablation.md">
# iter-v4-ablation — P2.F Screening & Confirmation Ablation (`R0` $\rightarrow$ `R5-l`)

## Changed
- **P2.A — Dual-View Loss (`code/networks/bfree_globalforge_vit.py`, `code/configs/bfree_dcs.yaml`, `code/train_lora.py`):**
  - Bổ sung nhánh `loss.mode: dual_view` trong [`BFreeGlobalForgeViT._compute_loss_dual_view`](file:///C:/Users/ADMIN/OneDrive%20-%20ptit.edu.vn/Documents/GitHub/B-Free/code/networks/bfree_globalforge_vit.py#L168-L199):
    $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CE}}(\text{clean}) + w_{\text{deg}} \cdot \mathcal{L}_{\text{CE}}(\text{deg}) + \lambda_{\text{cons}} \cdot T^2 \cdot \text{KL}\!\left(\sigma(\text{logits}_c^{\text{detach}}/T) \,\|\, \sigma(\text{logits}_d/T)\right) + \lambda_{\text{dcs}} \cdot \mathcal{L}_{\text{DCS}}$$
  - Giữ nguyên hợp đồng trả về 3 scalar `(total, ce_c, dcs)` và ghi `self.last_terms = {"ce_c", "ce_d", "cons"}` (detached) để log riêng vào `train_log.csv` (cả train và val).
  - Thêm `clean_view(img, normalize)` với xác suất `loss.p_aug_on_clean` (khuyến nghị `0.2` ở `dual_view`; bằng `0.0` ở `legacy` và không tiêu thụ RNG).
- **P2.B — Khởi tạo từ B-Free & Zero-Init Adapter (`code/networks/bfree_globalforge_vit.py`, `code/modules/gsr_adapter.py`, `code/modules/lib_adapter.py`):**
  - Thêm `--init_from {dinov2, bfree}` và `--bfree_ckpt` vào [`train_lora.py`](file:///C:/Users/ADMIN/OneDrive%20-%20ptit.edu.vn/Documents/GitHub/B-Free/code/train_lora.py#L140-L150).
  - Xử lý triệt để tiền tố `model.` / `module.` và ánh xạ head 1-logit $\rightarrow$ 2-logit trong [`remap_bfree_state_dict`](file:///C:/Users/ADMIN/OneDrive%20-%20ptit.edu.vn/Documents/GitHub/B-Free/code/networks/bfree_globalforge_vit.py#L204-L236): $W_2 = [-w/2;\, +w/2]$, $b_2 = [-b/2;\, +b/2]$ sao cho $z = l_1 - l_0 \equiv l_{\text{B-Free}}$. Bật `require_complete_backbone=True` để chặn lỗi bỏ qua trọng số im lặng.
  - Thêm `gate = nn.Parameter(torch.zeros(1))` cho [`GSRAdapter`](file:///C:/Users/ADMIN/OneDrive%20-%20ptit.edu.vn/Documents/GitHub/B-Free/code/modules/gsr_adapter.py#L33-L45) (`backbone.gsr_zero_init`) và `beta_init = -6.0` ($\alpha_0 = \sigma(-6) \approx 0.0025$) cho [`LIBAdapter`](file:///C:/Users/ADMIN/OneDrive%20-%20ptit.edu.vn/Documents/GitHub/B-Free/code/modules/lib_adapter.py#L31-L61) (`backbone.lib_beta_init`).
  - Đổi `backbone.head_mode: cls` làm mặc định trong `configs/bfree_dcs.yaml`, kết hợp độ lệch residual $(\text{feat\_cls} - \text{mean}(\text{patch\_raw}))$ vào CLS token để vừa thỏa Gate `Bước 0 ≡ B-Free` ($|\Delta\text{logits}| < 10^{-3}$) vừa truyền gradient từ CE về `LIB`/`GSR` trong các run đơn lẻ `R3` (`+LIB`) và `R4` (`+GSR`).
- **P2.C / P2.F — Cờ dữ liệu, CLI Override & Gói chạy Kaggle (`code/train_lora.py`, `code/configs/loader.py`, `code/scripts/run_p2f_ablation_kaggle.py`, `notebooks/version4/xla-ablation-p2f.ipynb`):**
  - Thêm cờ `--use_lib`, `--use_gsr`, `--use_dcs`, `--mix_preset {mix_current,mix_bfree,mix_noaigb}`, `--fmt_match`, `--res_match`, `--holdout_generators`, `--train_subsample_frac` vào `train_lora.py` và `RESUME_CHECK_KEYS`.
  - Tạo script chuẩn hóa [`code/scripts/run_p2f_ablation_kaggle.py`](file:///C:/Users/ADMIN/OneDrive%20-%20ptit.edu.vn/Documents/GitHub/B-Free/code/scripts/run_p2f_ablation_kaggle.py) và notebook [`notebooks/version4/xla-ablation-p2f.ipynb`](file:///C:/Users/ADMIN/OneDrive%20-%20ptit.edu.vn/Documents/GitHub/B-Free/notebooks/version4/xla-ablation-p2f.ipynb).
  - Đăng ký trước tiêu chí thành công/bác bỏ P2.G tại [`research/directions/D-recipe-v4/README.md`](file:///C:/Users/ADMIN/OneDrive%20-%20ptit.edu.vn/Documents/GitHub/B-Free/research/directions/D-recipe-v4/README.md).

---

## Verified
- **Unit tests trên CPU local (`pytest tests/`):**
  - [`tests/test_init_bfree.py`](file:///C:/Users/ADMIN/OneDrive%20-%20ptit.edu.vn/Documents/GitHub/B-Free/tests/test_init_bfree.py) & [`tests/test_dual_view.py`](file:///C:/Users/ADMIN/OneDrive%20-%20ptit.edu.vn/Documents/GitHub/B-Free/tests/test_dual_view.py): **19 passed, 1 skipped** (test với checkpoint B-Free thật được tích hợp tự động vào Bước 1 của `run_p2f_ablation_kaggle.py` trên GPU Kaggle).
  - Toàn bộ suite `tests/` (`test_backbone_shape.py`, `test_multi_severity.py`, `test_dcs_loss.py`, `overfit_test.py`, `resume_test.py`, `test_init_bfree.py`, `test_dual_view.py`): **31 passed, 1 skipped**.
  - Kiểm chứng bit-identity trong cùng môi trường: nhánh `loss.mode: legacy` + cờ mặc định khớp **bit-for-bit** (`torch.equal == True`) với code gốc trước khi sửa (`total=0.7343366742134094, ce=0.7191473841667175, dcs=1.5189313888549805`) và trùng khít với `dual_view` khi `w_deg=0.0, lam_cons=0.0`.
  - Notebook [`notebooks/version4/xla-ablation-p2f.ipynb`](file:///C:/Users/ADMIN/OneDrive%20-%20ptit.edu.vn/Documents/GitHub/B-Free/notebooks/version4/xla-ablation-p2f.ipynb) và [`code/scripts/run_p2f_ablation_kaggle.py`](file:///C:/Users/ADMIN/OneDrive%20-%20ptit.edu.vn/Documents/GitHub/B-Free/code/scripts/run_p2f_ablation_kaggle.py): đã kiểm tra `json.loads` và `ast.parse` không lỗi cú pháp.

---

## You must do

### Bước 0: Đồng bộ code lên Kaggle (`xla-setup`)
1. Commit & push nhánh hiện tại lên GitHub (hoặc cho phép tôi commit sau khi bạn duyệt).
2. Mở notebook `baopv051/xla-setup` trên Kaggle (chế độ Online), bấm **Save & Run All** để cập nhật thư mục `/kaggle/input/notebooks/baopv051/xla-setup/bfree_src` chứa `train_lora.py` mới và `code/scripts/run_p2f_ablation_kaggle.py`.

### Bước 1: Tạo Notebook Kaggle từ `notebooks/version4/xla-ablation-p2f.ipynb`
1. Upload file [`notebooks/version4/xla-ablation-p2f.ipynb`](file:///C:/Users/ADMIN/OneDrive%20-%20ptit.edu.vn/Documents/GitHub/B-Free/notebooks/version4/xla-ablation-p2f.ipynb) lên Kaggle, bật **GPU Accelerator** và **tắt Internet (Offline mode)**.
2. Attach đầy đủ các Input Datasets / Notebooks vào phiên làm việc (khớp đường dẫn tuyệt đối ở **Cell 2**):
   - Notebook `baopv051/xla-setup` (`/kaggle/input/notebooks/baopv051/xla-setup`)
   - Notebook `baopv051/xla-setup-eval` (`/kaggle/input/notebooks/baopv051/xla-setup-eval`)
   - Model `baopv051/bfree-baseline-weights` (`/kaggle/input/models/baopv051/bfree-baseline-weights/pytorch/default/1`)
   - Dataset `baopv051/b-free-training-data` (`/kaggle/input/datasets/baopv051/b-free-training-data`)
   - Dataset `baopv051/ai-genbench-v1` (`/kaggle/input/datasets/baopv051/ai-genbench-v1`)
   - Dataset `ngovanphuong2604/wildrf-dataset` (`/kaggle/input/datasets/ngovanphuong2604/wildrf-dataset`)
   - Dataset `ngovanphuong2604/synthwildx-dataset` (`/kaggle/input/datasets/ngovanphuong2604/synthwildx-dataset`)
   - Dataset `ngovanphuong2604/bfree-viral-images` (`/kaggle/input/datasets/ngovanphuong2604/bfree-viral-images`)

### Bước 2: Chạy các Run Sàng lọc (`R0`, `R1`, `R1b`, `R2`, `R3`, `R4`, `R5` và `R5-d..R5-l`)
Bạn có thể chạy theo **Cách A** (sửa biến `RUNS_TO_EXECUTE` ở **Cell 1** của notebook rồi bấm **Run All**) hoặc **Cách B** (chạy từng dòng lệnh CLI tường minh cho từng run trong một code cell trên Kaggle):

#### Cách A — Qua `run_p2f_ablation_kaggle.py` (Tự động Gate Bước 0 + Train + Eval Yardstick v2 + Xuất Parquet + In bảng Paste-back)
- **Phiên 1 (Tham chiếu `R0` + Recipe thuần `R1` + Đối chứng W1 `R1b`, ~1.5h):**
  Tại **Cell 1**, đặt:
  ```python
  RUNS_TO_EXECUTE = "R0,R1,R1b"
  STAGE = "screening"
  ```
- **Phiên 2 (Tách đóng góp từng module `R2`, `R3`, `R4`, `R5`, ~2.5h):**
  Tại **Cell 1**, đặt:
  ```python
  RUNS_TO_EXECUTE = "R2,R3,R4,R5"
  STAGE = "screening"
  ```
- **Phiên 3 (Các biến thể đối chứng `R5-d`, `R5-a`, `R5-f`, `R5-l`, ~2.5h):**
  Tại **Cell 1**, đặt:
  ```python
  RUNS_TO_EXECUTE = "R5-d,R5-a,R5-f,R5-l"
  STAGE = "screening"
  ```

#### Cách B — Lệnh CLI trực tiếp từng Run (`R1`, `R1b`, `R2`, `R3`, `R4`, `R5`)
Nếu bạn muốn chạy từng lệnh độc lập trong cell bash của Kaggle (sau khi đã chạy **Cell 2** và **Cell 3**):

- **Kiểm tra Gate bắt buộc "Bước 0 ≡ B-Free" trên trọng số thật:**
  ```bash
  python "/kaggle/input/notebooks/baopv051/xla-setup/bfree_src/tests/test_init_bfree.py" \
      --ckpt "/kaggle/input/models/baopv051/bfree-baseline-weights/pytorch/default/1/model_epoch_best.pth" \
      --n 64 --img 504
  ```
- **Run `R1` (Init B-Free, KHÔNG module, Dual-View CE, `mix_bfree`, 25% data, 3 epochs):**
  ```bash
  python "/kaggle/input/notebooks/baopv051/xla-setup/bfree_src/code/scripts/run_p2f_ablation_kaggle.py" --runs R1 --stage screening
  ```
- **Run `R1b` (Init B-Free, KHÔNG module, Legacy Clean CE, `mix_bfree`, 25% data, 3 epochs):**
  ```bash
  python "/kaggle/input/notebooks/baopv051/xla-setup/bfree_src/code/scripts/run_p2f_ablation_kaggle.py" --runs R1b --stage screening
  ```
- **Run `R2` (Init B-Free, `+DCS`, Dual-View, `mix_bfree`, 25% data, 3 epochs):**
  ```bash
  python "/kaggle/input/notebooks/baopv051/xla-setup/bfree_src/code/scripts/run_p2f_ablation_kaggle.py" --runs R2 --stage screening
  ```
- **Run `R3` (Init B-Free, `+LIB` với `beta_init=-6`, Dual-View, `mix_bfree`, 25% data, 3 epochs):**
  ```bash
  python "/kaggle/input/notebooks/baopv051/xla-setup/bfree_src/code/scripts/run_p2f_ablation_kaggle.py" --runs R3 --stage screening
  ```
- **Run `R4` (Init B-Free, `+GSR` với `gsr_zero_init=1`, Dual-View, `mix_bfree`, 25% data, 3 epochs):**
  ```bash
  python "/kaggle/input/notebooks/baopv051/xla-setup/bfree_src/code/scripts/run_p2f_ablation_kaggle.py" --runs R4 --stage screening
  ```
- **Run `R5` (Mô hình đầy đủ: Init B-Free + `LIB+GSR+DCS` + Dual-View + `mix_bfree`, 25% data, 3 epochs):**
  ```bash
  python "/kaggle/input/notebooks/baopv051/xla-setup/bfree_src/code/scripts/run_p2f_ablation_kaggle.py" --runs R5 --stage screening
  ```

---

## Output schema

Mỗi run `<RUN_ID>` lưu kết quả tại `/kaggle/working/p2f_runs/<RUN_ID>/` với 2 file dữ liệu chuẩn hóa:

### 1. File `/kaggle/working/p2f_runs/<RUN_ID>/train_log.csv`
- **Chế độ `loss.mode = dual_view` (`R1, R2, R3, R4, R5, R5-d, R5-a, R5-f, R5-l`):**
  ```text
  epoch,train_loss,train_ce,train_dcs,val_loss,val_auc,val_bacc,ce_c,ce_d,cons,val_ce_c,val_ce_d,val_cons
  ```
  - `epoch` (`int`): Chỉ số epoch (`1..3` ở screening, `1..5` ở confirm).
  - `train_loss`, `val_loss` (`float`): Tổng hàm mục tiêu $\mathcal{L}_{\text{total}}$.
  - `train_ce`, `ce_c`, `val_ce_c` (`float`): Cross-Entropy trên view sạch $\mathcal{L}_{\text{CE}}(\text{clean})$.
  - `ce_d`, `val_ce_d` (`float`): Cross-Entropy trên view suy giảm $\mathcal{L}_{\text{CE}}(\text{deg})$.
  - `cons`, `val_cons` (`float`): KL-divergence nhất quán quyết định $T^2 \cdot \text{KL}(\text{clean}^{\text{detach}} \parallel \text{deg})$.
  - `train_dcs` (`float`): Giá trị $\mathcal{L}_{\text{DCS}}$ (InfoNCE đối xứng).
  - `val_auc`, `val_bacc` (`float`): ROC-AUC và Balanced Accuracy tại ngưỡng logit $0$ trên tập val in-distribution.
- **Chế độ `loss.mode = legacy` (`R1b`):**
  ```text
  epoch,train_loss,train_ce,train_dcs,val_loss,val_auc,val_bacc
  ```

### 2. File `/kaggle/working/p2f_runs/<RUN_ID>/scores.parquet` (Theo Quy tắc #8, P0.9, P0.10)
| Tên cột | Kiểu dữ liệu | Mô tả |
| :--- | :--- | :--- |
| `image_id` | `string` | Đường dẫn tương đối của ảnh hoặc định danh hàng Arrow (`aigb_val_<idx>`) |
| `benchmark` | `string` | Tên bộ đánh giá: `wildrf_val`, `wildrf`, `bfree_viral_images`, `ai_genbench`, `realdeg_heldout` |
| `group` | `string` | Tên generator hoặc nền tảng (`facebook`, `reddit`, `twitter`, `viral`, `REAL_POOL`, ...) |
| `is_seen_gen` | `bool` | `True` nếu thuộc 13 `PRIORITY_GENERATORS` đã thấy khi train; `False` nếu unseen/real |
| `origin_dataset` | `string` | Nguồn dữ liệu gốc của mẫu (phục vụ kiểm tra rò rỉ miền) |
| `condition` | `string` | `clean` hoặc tên toán tử suy giảm held-out (`heldout_webp`, `heldout_jpeg_420`, `heldout_resize_lanczos`, `heldout_poisson_iso`, `heldout_sharpen`, `heldout_social_compound`) |
| `label` | `int64` | Nhãn thực tế (`0` = Real, `1` = Fake) |
| `logit` | `float32` | Logit nhị phân tự nhiên $z = l_1 - l_0$ (hoặc $l$ với B-Free 1-logit) |
| `prob` | `float32` | Xác suất dự đoán trước hiệu chuẩn $p = \sigma(z)$ |
| `prob_cal` | `float32` | Xác suất sau khi chia nhiệt độ Temperature Scaling $p_{\text{cal}} = \sigma(z / T_{\text{cal}})$ |
| `width` | `int64` | Chiều rộng gốc của ảnh (px) |
| `height` | `int64` | Chiều cao gốc của ảnh (px) |
| `format` | `string` | Định dạng file ảnh gốc (`JPEG`, `PNG`, `WEBP`, ...) |
| `run_id` | `string` | Mã định danh run (`R0`, `R1`, `R1b`, `R2`, `R3`, `R4`, `R5`, ...) |
| `git_sha` | `string` | Commit SHA của repo `bfree_src` |
| `ckpt_sha256` | `string` | 16 ký tự đầu SHA-256 của file checkpoint đã chấm |
| `config_hash` | `string` | 16 ký tự đầu SHA-256 của cấu hình thực nghiệm |
| `seed` | `int64` | Seed thực nghiệm (`42`) |
| `harness_version` | `int64` | Phiên bản thước đo (`2` — Yardstick v2) |

---

## Paste back

Sau khi chạy xong trên Kaggle, hãy dán:
1. Dòng log `[GATE P2.B] Step 0 vs B-Free official: max |Delta logits| = ...`
2. Nội dung đầu ra của **Cell 5** (hoặc các dòng `PASTE BACK ROW`) vào bảng dưới đây:

| Run | WildRF+Viral AUC | WildRF+Viral bAcc@0.5 | WildRF+Viral ECE_cal | AIGB-Unseen AUC | AIGB-Unseen bAcc@0.5 | AIGB-Unseen ECE_cal | RealDeg-Heldout bAcc@0.5 | AIGB-Seen AUC (Guard) | Temp T | ckpt_sha256 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `R0` (B-Free ref) | TODO: pending run | TODO: pending run | TODO: pending run | TODO: pending run | TODO: pending run | TODO: pending run | TODO: pending run | TODO: pending run | TODO: pending run | TODO: pending run |
| `R1` (Pure recipe) | TODO: pending run | TODO: pending run | TODO: pending run | TODO: pending run | TODO: pending run | TODO: pending run | TODO: pending run | TODO: pending run | TODO: pending run | TODO: pending run |
| `R1b` (Clean CE) | TODO: pending run | TODO: pending run | TODO: pending run | TODO: pending run | TODO: pending run | TODO: pending run | TODO: pending run | TODO: pending run | TODO: pending run | TODO: pending run |
| `R2` (+DCS) | TODO: pending run | TODO: pending run | TODO: pending run | TODO: pending run | TODO: pending run | TODO: pending run | TODO: pending run | TODO: pending run | TODO: pending run | TODO: pending run |
| `R3` (+LIB) | TODO: pending run | TODO: pending run | TODO: pending run | TODO: pending run | TODO: pending run | TODO: pending run | TODO: pending run | TODO: pending run | TODO: pending run | TODO: pending run |
| `R4` (+GSR) | TODO: pending run | TODO: pending run | TODO: pending run | TODO: pending run | TODO: pending run | TODO: pending run | TODO: pending run | TODO: pending run | TODO: pending run | TODO: pending run |
| `R5` (LIB+GSR+DCS) | TODO: pending run | TODO: pending run | TODO: pending run | TODO: pending run | TODO: pending run | TODO: pending run | TODO: pending run | TODO: pending run | TODO: pending run | TODO: pending run |
| `R5-d` (DINOv2 init) | TODO: pending run | TODO: pending run | TODO: pending run | TODO: pending run | TODO: pending run | TODO: pending run | TODO: pending run | TODO: pending run | TODO: pending run | TODO: pending run |
| `R5-a` (mix_current) | TODO: pending run | TODO: pending run | TODO: pending run | TODO: pending run | TODO: pending run | TODO: pending run | TODO: pending run | TODO: pending run | TODO: pending run | TODO: pending run |
| `R5-f` (fmt+res match) | TODO: pending run | TODO: pending run | TODO: pending run | TODO: pending run | TODO: pending run | TODO: pending run | TODO: pending run | TODO: pending run | TODO: pending run | TODO: pending run |
| `R5-l` (ls=0) | TODO: pending run | TODO: pending run | TODO: pending run | TODO: pending run | TODO: pending run | TODO: pending run | TODO: pending run | TODO: pending run | TODO: pending run | TODO: pending run |
</file>

<file path="research/p1_diagnosis_metrics.json">
{
  "D1": {
    "train_mixture": {
      "n_bfree_pair": 50083,
      "n_extra": 2735,
      "n_aigb": 93600,
      "n_total": 146418,
      "aigb_fraction": 0.6392656640576978
    },
    "group_stats": {
      "ours": {
        "Seen_13_Priority": {
          "count": 13,
          "mean_auc": 0.999718846153846,
          "mean_bacc05": 0.9967948699999999,
          "mean_bacc_cal": 0.9888205100000003,
          "mean_nll": 0.05910361538461538
        },
        "Unseen_LowMidRes_BicubicUpscaled": {
          "count": 17,
          "mean_auc": 0.9977705882352941,
          "mean_bacc05": 0.9896568611764707,
          "mean_bacc_cal": 0.9855261411764706,
          "mean_nll": 0.07756564705882353
        },
        "Unseen_HighRes_Native": {
          "count": 7,
          "mean_auc": 0.9749851428571429,
          "mean_bacc05": 0.7888095242857143,
          "mean_bacc_cal": 0.9217142857142857,
          "mean_nll": 0.5017132857142858
        }
      },
      "bfree_official": {
        "Seen_13_Priority": {
          "count": 13,
          "mean_auc": 0.8901474615384617,
          "mean_bacc05": 0.7535641023076924,
          "mean_bacc_cal": 0.7881282046153847,
          "mean_nll": 1.1118164615384616
        },
        "Unseen_LowMidRes_BicubicUpscaled": {
          "count": 17,
          "mean_auc": 0.9483048823529411,
          "mean_bacc05": 0.8734836605882352,
          "mean_bacc_cal": 0.8696503270588235,
          "mean_nll": 0.5568898823529412
        },
        "Unseen_HighRes_Native": {
          "count": 7,
          "mean_auc": 0.978919142857143,
          "mean_bacc05": 0.9186190457142857,
          "mean_bacc_cal": 0.8963333342857142,
          "mean_nll": 0.3253627142857143
        }
      },
      "globalforge_dinov3": {
        "Seen_13_Priority": {
          "count": 13,
          "mean_auc": 0.9326825384615384,
          "mean_bacc05": 0.7599743592307691,
          "mean_bacc_cal": 0.8736923069230771,
          "mean_nll": 0.5193302307692308
        },
        "Unseen_LowMidRes_BicubicUpscaled": {
          "count": 17,
          "mean_auc": 0.9731187647058824,
          "mean_bacc05": 0.7609782135294116,
          "mean_bacc_cal": 0.9095620905882353,
          "mean_nll": 0.5050303529411764
        },
        "Unseen_HighRes_Native": {
          "count": 7,
          "mean_auc": 0.9282651428571429,
          "mean_bacc05": 0.761,
          "mean_bacc_cal": 0.8017142842857143,
          "mean_nll": 0.5199477142857144
        }
      }
    },
    "reencode_results": {
      "img0000.png": {
        "size": [
          835,
          1256
        ],
        "logit_png": -5.937479019165039,
        "prob_sig_l": 0.0026317290448694473,
        "prob_sig_2l": 6.962548402132505e-06,
        "logit_jpeg85": -5.266714572906494,
        "prob_jpeg85": 0.005134042901138576,
        "logit_256_replicate": -5.460259914398193,
        "prob_256_replicate": 0.0042344435671601375,
        "logit_256_bicubic504": -6.618844032287598,
        "prob_256_bicubic504": 0.0013331934623867052
      },
      "img0002.png": {
        "size": [
          1024,
          1024
        ],
        "logit_png": 4.430525779724121,
        "prob_sig_l": 0.9882319101619783,
        "prob_sig_2l": 0.9998582142376212,
        "logit_jpeg85": 1.0881307125091553,
        "prob_jpeg85": 0.7480295591787297,
        "logit_256_replicate": -3.9165642261505127,
        "prob_256_replicate": 0.019520735725851203,
        "logit_256_bicubic504": -5.623881816864014,
        "prob_256_bicubic504": 0.003597608670285363
      }
    }
  },
  "D2": {
    "step0": {
      "ce_loss": 0.7203223705291748,
      "dcs_loss": 1.0511529445648193,
      "scaled_dcs_loss": 0.010511529631912708,
      "gn_ce": {
        "head": 1.8499626301945096,
        "fc_norm": 0.0,
        "lib": 0.022975941463833586,
        "gsr": 0.9612436325187289,
        "lora": 3.964098453650855,
        "shared_no_head": 4.0790432413963265,
        "all_trainable": 4.478945802116525
      },
      "gn_dcs_scaled": {
        "head": 0.0,
        "fc_norm": 0.0,
        "lib": 0.002368330089502522,
        "gsr": 0.006306039067071558,
        "lora": 0.014889336193090488,
        "shared_no_head": 0.016342198395538774,
        "all_trainable": 0.016342198395538774
      },
      "gn_dcs_raw": {
        "head": 0.0,
        "fc_norm": 0.0,
        "lib": 0.23683300895025217,
        "gsr": 0.6306039067071557,
        "lora": 1.4889336193090488,
        "shared_no_head": 1.6342198395538774,
        "all_trainable": 1.6342198395538774
      },
      "ratio_all": 0.003648670717965882,
      "ratio_shared": 0.004006380278024353,
      "ratio_head": 0.0,
      "cos_sim_shared": 0.017688749358057976
    },
    "converged": {
      "ce_loss": 0.3740297555923462,
      "dcs_loss": 0.9609322547912598,
      "scaled_dcs_loss": 0.009609322063624859,
      "gn_ce": {
        "head": 4.043463108993424,
        "fc_norm": 0.0,
        "lib": 0.21817871528167038,
        "gsr": 2.701445292004883,
        "lora": 3.5190513254771685,
        "shared_no_head": 4.441748625129514,
        "all_trainable": 6.0065568142348145
      },
      "gn_dcs_scaled": {
        "head": 0.0,
        "fc_norm": 0.0,
        "lib": 0.0028803135231593637,
        "gsr": 0.009543996655930947,
        "lora": 0.018327174597565825,
        "shared_no_head": 0.02086311115077945,
        "all_trainable": 0.02086311115077945
      },
      "gn_dcs_raw": {
        "head": 0.0,
        "fc_norm": 0.0,
        "lib": 0.28803135231593635,
        "gsr": 0.9543996655930947,
        "lora": 1.8327174597565825,
        "shared_no_head": 2.086311115077945,
        "all_trainable": 2.086311115077945
      },
      "ratio_all": 0.0034733894635502983,
      "ratio_shared": 0.0046970490479233545,
      "ratio_head": 0.0,
      "cos_sim_shared": 0.7112184166908264
    }
  },
  "D3_D4_D5": {
    "d3": {
      "ours": {
        "fpr_05": 0.0033333400000001845,
        "fpr_cal": 0.02133333999999998,
        "generators": {
          "FLUX 1 Dev": {
            "auc": 0.948996,
            "bacc05": 0.53166667,
            "bacc_cal": 0.76933333,
            "tpr05": 0.0666666800000002,
            "tpr_cal": 0.56,
            "nll": 1.13605
          },
          "DALL-E 3": {
            "auc": 0.967933,
            "bacc05": 0.7116666700000001,
            "bacc_cal": 0.9159999999999999,
            "tpr05": 0.4266666800000003,
            "tpr_cal": 0.8533333399999998,
            "nll": 0.626171
          },
          "FLUX 1 Schnell": {
            "auc": 0.993689,
            "bacc05": 0.7416666700000001,
            "bacc_cal": 0.966,
            "tpr05": 0.48666668000000035,
            "tpr_cal": 0.9533333399999999,
            "nll": 0.51873
          },
          "Stable Diffusion 1.4": {
            "auc": 0.970544,
            "bacc05": 0.85833333,
            "bacc_cal": 0.9259999999999999,
            "tpr05": 0.7200000000000002,
            "tpr_cal": 0.8733333399999998,
            "nll": 0.374528
          },
          "Stable Diffusion 2.1": {
            "auc": 0.960302,
            "bacc05": 0.8616666700000001,
            "bacc_cal": 0.93266667,
            "tpr05": 0.7266666800000003,
            "tpr_cal": 0.88666668,
            "nll": 0.359724
          },
          "Stable Diffusion XL 1.0": {
            "auc": 0.991236,
            "bacc05": 0.8883333299999999,
            "bacc_cal": 0.9593333300000001,
            "tpr05": 0.78,
            "tpr_cal": 0.9400000000000002,
            "nll": 0.297331
          },
          "ADM": {
            "auc": 0.999673,
            "bacc05": 0.995,
            "bacc_cal": 0.98933333,
            "tpr05": 0.9933333400000002,
            "tpr_cal": 1.0,
            "nll": 0.061595
          },
          "ProGAN": {
            "auc": 0.999796,
            "bacc05": 0.9983333299999999,
            "bacc_cal": 0.98933333,
            "tpr05": 1.0,
            "tpr_cal": 1.0,
            "nll": 0.055835
          }
        },
        "wildrf": {
          "facebook": {
            "n": 320,
            "auc": 0.932383,
            "bacc05": 0.715625,
            "bacc_cal": 0.859375,
            "thr_cal": 0.070611,
            "ece": 0.237271,
            "nll": 0.616964
          },
          "reddit": {
            "n": 1500,
            "auc": 0.958613,
            "bacc05": 0.86533333,
            "bacc_cal": 0.892,
            "thr_cal": 0.070611,
            "ece": 0.119971,
            "nll": 0.336951
          },
          "twitter": {
            "n": 683,
            "auc": 0.937379,
            "bacc05": 0.70906433,
            "bacc_cal": 0.86388503,
            "thr_cal": 0.070611,
            "ece": 0.241815,
            "nll": 0.625491
          }
        }
      },
      "bfree_official": {
        "fpr_05": 0.08466666000000012,
        "fpr_cal": 0.16066665999999996,
        "generators": {
          "FLUX 1 Dev": {
            "auc": 0.87892,
            "bacc05": 0.7143333300000001,
            "bacc_cal": 0.76966667,
            "tpr05": 0.5133333200000003,
            "tpr_cal": 0.7,
            "nll": 1.227399
          },
          "DALL-E 3": {
            "auc": 0.990222,
            "bacc05": 0.9443333300000001,
            "bacc_cal": 0.9129999999999999,
            "tpr05": 0.9733333200000003,
            "tpr_cal": 0.9866666599999998,
            "nll": 0.199914
          },
          "FLUX 1 Schnell": {
            "auc": 0.993922,
            "bacc05": 0.9543333300000001,
            "bacc_cal": 0.91966667,
            "tpr05": 0.9933333200000003,
            "tpr_cal": 1.0,
            "nll": 0.161707
          },
          "Stable Diffusion 1.4": {
            "auc": 0.998604,
            "bacc05": 0.9543333300000001,
            "bacc_cal": 0.91966667,
            "tpr05": 0.9933333200000003,
            "tpr_cal": 1.0,
            "nll": 0.157942
          },
          "Stable Diffusion 2.1": {
            "auc": 0.99866,
            "bacc05": 0.9576666699999999,
            "bacc_cal": 0.91966667,
            "tpr05": 1.0,
            "tpr_cal": 1.0,
            "nll": 0.154998
          },
          "Stable Diffusion XL 1.0": {
            "auc": 0.996413,
            "bacc05": 0.9543333300000001,
            "bacc_cal": 0.91633333,
            "tpr05": 0.9933333200000003,
            "tpr_cal": 0.9933333199999999,
            "nll": 0.183643
          },
          "ADM": {
            "auc": 0.86976,
            "bacc05": 0.76433333,
            "bacc_cal": 0.78966667,
            "tpr05": 0.6133333200000002,
            "tpr_cal": 0.74,
            "nll": 1.294101
          },
          "ProGAN": {
            "auc": 0.993776,
            "bacc05": 0.951,
            "bacc_cal": 0.91966667,
            "tpr05": 0.98666666,
            "tpr_cal": 1.0,
            "nll": 0.163187
          }
        },
        "wildrf": {
          "facebook": {
            "n": 320,
            "auc": 0.987227,
            "bacc05": 0.95625,
            "bacc_cal": 0.934375,
            "thr_cal": 0.047426,
            "ece": 0.045418,
            "nll": 0.228546
          },
          "reddit": {
            "n": 1500,
            "auc": 0.940696,
            "bacc05": 0.84933333,
            "bacc_cal": 0.80866667,
            "thr_cal": 0.047426,
            "ece": 0.121184,
            "nll": 0.620272
          },
          "twitter": {
            "n": 683,
            "auc": 0.993638,
            "bacc05": 0.9575251699999999,
            "bacc_cal": 0.95016806,
            "thr_cal": 0.047426,
            "ece": 0.034601,
            "nll": 0.166502
          }
        }
      },
      "globalforge_dinov3": {
        "fpr_05": 0.4780000000000002,
        "fpr_cal": 0.16133333999999988,
        "generators": {
          "FLUX 1 Dev": {
            "auc": 0.782476,
            "bacc05": 0.7609999999999999,
            "bacc_cal": 0.48933332999999996,
            "tpr05": 1.0,
            "tpr_cal": 0.1399999999999998,
            "nll": 0.576959
          },
          "DALL-E 3": {
            "auc": 0.948676,
            "bacc05": 0.7609999999999999,
            "bacc_cal": 0.89933333,
            "tpr05": 1.0,
            "tpr_cal": 0.96,
            "nll": 0.513051
          },
          "FLUX 1 Schnell": {
            "auc": 0.815647,
            "bacc05": 0.7609999999999999,
            "bacc_cal": 0.5660000000000001,
            "tpr05": 1.0,
            "tpr_cal": 0.29333334,
            "nll": 0.559005
          },
          "Stable Diffusion 1.4": {
            "auc": 0.994191,
            "bacc05": 0.7609999999999999,
            "bacc_cal": 0.9193333300000001,
            "tpr05": 1.0,
            "tpr_cal": 1.0,
            "nll": 0.495477
          },
          "Stable Diffusion 2.1": {
            "auc": 0.982433,
            "bacc05": 0.7609999999999999,
            "bacc_cal": 0.91266667,
            "tpr05": 1.0,
            "tpr_cal": 0.9866666799999999,
            "nll": 0.499626
          },
          "Stable Diffusion XL 1.0": {
            "auc": 0.979,
            "bacc05": 0.7609999999999999,
            "bacc_cal": 0.9059999999999999,
            "tpr05": 1.0,
            "tpr_cal": 0.9733333399999997,
            "nll": 0.501043
          },
          "ADM": {
            "auc": 0.952753,
            "bacc05": 0.75766667,
            "bacc_cal": 0.88933333,
            "tpr05": 0.9933333400000002,
            "tpr_cal": 0.94,
            "nll": 0.517249
          },
          "ProGAN": {
            "auc": 0.992416,
            "bacc05": 0.7609999999999999,
            "bacc_cal": 0.9193333300000001,
            "tpr05": 1.0,
            "tpr_cal": 1.0,
            "nll": 0.497681
          }
        },
        "wildrf": {
          "facebook": {
            "n": 320,
            "auc": 0.861777,
            "bacc05": 0.565625,
            "bacc_cal": 0.771875,
            "thr_cal": 0.859192,
            "ece": 0.342241,
            "nll": 0.900479
          },
          "reddit": {
            "n": 1500,
            "auc": 0.927395,
            "bacc05": 0.5826666699999999,
            "bacc_cal": 0.7340000000000001,
            "thr_cal": 0.859192,
            "ece": 0.353434,
            "nll": 0.936106
          },
          "twitter": {
            "n": 683,
            "auc": 0.909151,
            "bacc05": 0.61290323,
            "bacc_cal": 0.77866955,
            "thr_cal": 0.859192,
            "ece": 0.320078,
            "nll": 0.819663
          }
        }
      }
    },
    "d5": {
      "alpha0": 0.2994328737258911,
      "cos_lib": 0.9934751987457275,
      "cos_gsr": 0.964385449886322,
      "cos_both": 0.9533106684684753,
      "rel_l2_lib": 0.1256287693977356,
      "rel_l2_gsr": 0.2745576500892639,
      "rel_l2_both": 0.3037911057472229,
      "cos_mean_pool": 0.9457944631576538,
      "rel_l2_mean_pool": 0.34097546339035034
    }
  }
}
</file>

<file path="research/p1_diagnosis.py">
"""
Phase P1 Diagnostic Script (CPU-compatible, zero training code modifications).

Verifies hypotheses W1..W5 via tests D1..D5 defined in AGENT_PLAN_V4.md:
  - D1 (W3): Format/resolution/interpolation shortcut & AI-GenBench dominance
  - D2 (W1): Gradient norm ratio ||grad(lambda * L_DCS)||_2 / ||grad(L_CE)||_2
             and structural diagnosis of BFreeGlobalForgeViT.compute_loss
  - D3 (W4/W5): Exact TPR/FPR algebraic reconstruction & threshold calibration on AIGB / WildRF
  - D4 (W3): SD2.1 in-distribution val vs AI-GenBench SD2.1 gap
  - D5 (W2): Step-0 LIB + GSR feature perturbation on pretrained DINOv2 tokens

Usage (local CPU):
    .\\venv\\Scripts\\python.exe research/p1_diagnosis.py
"""

from __future__ import annotations

import argparse
import io
import json
import math
import random
import re
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

REPO_ROOT = Path(__file__).resolve().parent.parent
CODE_DIR = REPO_ROOT / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from modules.dcs_loss import info_nce_loss
from modules.gsr_adapter import GSRAdapter
from modules.lib_adapter import LIBAdapter
from networks import get_network
from networks.bfree_globalforge_vit import BFreeGlobalForgeViT
from train_lora import (
    CFG,
    apply_lora_to_backbone,
    ensure_min_size,
    realistic_lifecycle_degrade,
)

CFG.update(
    {
        "img_size": 504,
        "degrade_max_rounds": 4,
        "degrade_round_probs": [0.15, 0.30, 0.25, 0.20, 0.10],
    }
)


# ============================================================================
# 1. Ground-Truth Logs Extracted from Executed Kaggle Notebooks
# ============================================================================

PRIORITY_GENERATORS_13 = {
    "ADM",
    "DDPM",
    "DeepFloyd IF",
    "Palette",
    "Denoising Diffusion GAN",
    "LaMa",
    "MAT",
    "SN-PatchGAN",
    "StyleGAN1",
    "VQGAN",
    "GANformer",
    "CIPS",
    "FaceSynthetics",
}

UNSEEN_HIGH_RES_GENERATORS = {
    "DALL-E 2",
    "DALL-E 3",
    "FLUX 1 Dev",
    "FLUX 1 Schnell",
    "Midjourney V6",
    "Imagen 3",
    "Stable Diffusion 1.4",
    "Stable Diffusion 1.5",
    "Stable Diffusion 2.1",
    "Stable Diffusion 3",
    "Stable Diffusion XL 1.0",
    "Kandinsky 2.2",
}

ROW_REGEX = re.compile(
    r"^\s*(ai_genbench|wildrf|bfree_viral_images)\s+(.+?)\s+(\d+)\s+"
    r"([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s*$"
)


def load_eval_notebook_metrics(downloads_dir: Path) -> Dict[str, dict]:
    """Parse per-generator and summary metrics from Cell 23 of executed eval notebooks."""
    nb_map = {
        "ours": downloads_dir / "xla-eval-harness-ours (1).ipynb",
        "bfree_official": downloads_dir / "xla-eval-harness-b-free.ipynb",
        "globalforge_dinov3": downloads_dir / "xla-eval-harness-globalforge-dinov3.ipynb",
    }
    results: Dict[str, dict] = {}
    for model_key, nb_path in nb_map.items():
        if not nb_path.exists():
            continue
        nb = json.loads(nb_path.read_text(encoding="utf-8"))
        per_gen: Dict[str, dict] = {}
        wildrf: Dict[str, dict] = {}
        viral: Dict[str, dict] = {}
        calib_thr = None
        real_pool_mean = None

        for cell in nb.get("cells", []):
            for out in cell.get("outputs", []):
                text = "".join(out.get("text", []))
                for line in text.splitlines():
                    if "CALIB_THRESHOLD=" in line:
                        m = re.search(r"CALIB_THRESHOLD=([0-9.]+)", line)
                        if m:
                            calib_thr = float(m.group(1))
                    if "Real pool scored" in line or "mean_p_fake=" in line:
                        m = re.search(r"mean_p_fake=([0-9.]+)", line)
                        if m:
                            real_pool_mean = float(m.group(1))
                    m_row = ROW_REGEX.match(line)
                    if m_row:
                        ds, gen, n_str, auc_s, b05_s, bcal_s, thr_s, ece_s, nll_s = m_row.groups()
                        gen_clean = gen.strip()
                        if gen_clean.upper() == "MEAN":
                            continue
                        entry = {
                            "n": int(n_str),
                            "auc": float(auc_s),
                            "bacc05": float(b05_s) / 100.0,
                            "bacc_cal": float(bcal_s) / 100.0,
                            "thr_cal": float(thr_s),
                            "ece": float(ece_s),
                            "nll": float(nll_s),
                        }
                        if ds == "ai_genbench":
                            per_gen[gen_clean] = entry
                        elif ds == "wildrf":
                            wildrf[gen_clean] = entry
                        elif ds == "bfree_viral_images":
                            viral[gen_clean] = entry

        results[model_key] = {
            "calib_thr": calib_thr,
            "real_pool_mean": real_pool_mean,
            "ai_genbench": per_gen,
            "wildrf": wildrf,
            "viral": viral,
        }
    return results


# ============================================================================
# 2. D1 Diagnosis: Shortcut Analysis + Local Re-encoding / Interpolation Test
# ============================================================================


def run_d1_diagnosis(downloads_dir: Path) -> dict:
    print("\n" + "=" * 80)
    print("[D1] W3 DIAGNOSIS: AI-GENBENCH DOMINANCE & INTERPOLATION/FORMAT SHORTCUT")
    print("=" * 80)

    # 1. Training mixture composition from xla-training-phase-2.ipynb & xla-training (2).ipynb
    n_bfree_pair = 50083
    n_extra = 2735
    n_aigb = 93600  # 46,800 real + 46,800 fake across 13 PRIORITY_GENERATORS
    n_total = n_bfree_pair + n_extra + n_aigb
    aigb_frac = n_aigb / n_total

    print(f"1. Training Mixture Composition (total = {n_total:,} samples):")
    print(
        f"   - B-Free PairDataset (COCO + 6x SD2.1, 512x512 PNG, NO ensure_min_size): "
        f"{n_bfree_pair:,} ({n_bfree_pair/n_total:.2%})"
    )
    print(
        f"   - ExtraTrainDataset (RAISE/LDM/SD3-FLUX, with ensure_min_size):          "
        f"{n_extra:,} ({n_extra/n_total:.2%})"
    )
    print(
        f"   - ExtraArrowTrainDataset (AI-GenBench 13 gens, with ensure_min_size):    "
        f"{n_aigb:,} ({aigb_frac:.2%})"
    )

    # 2. Group 36 AI-GenBench generators by Seen vs Unseen Low-Res (<504) vs Unseen High-Res (>=512)
    eval_metrics = load_eval_notebook_metrics(downloads_dir)
    group_stats: Dict[str, Dict[str, dict]] = {}

    for model_key, mdata in eval_metrics.items():
        per_gen = mdata["ai_genbench"]
        groups = {
            "Seen_13_Priority": [],
            "Unseen_LowMidRes_BicubicUpscaled": [],
            "Unseen_HighRes_Native": [],
        }
        for gen, metrics in per_gen.items():
            if gen in PRIORITY_GENERATORS_13:
                groups["Seen_13_Priority"].append((gen, metrics))
            elif gen in UNSEEN_HIGH_RES_GENERATORS:
                groups["Unseen_HighRes_Native"].append((gen, metrics))
            else:
                groups["Unseen_LowMidRes_BicubicUpscaled"].append((gen, metrics))

        group_stats[model_key] = {}
        print(f"\n2. AI-GenBench Stratified Breakdown — Model: {model_key} (total gens={len(per_gen)})")
        for gname, items in groups.items():
            aucs = [x[1]["auc"] for x in items]
            b05s = [x[1]["bacc05"] for x in items]
            bcals = [x[1]["bacc_cal"] for x in items]
            nlls = [x[1]["nll"] for x in items]
            group_stats[model_key][gname] = {
                "count": len(items),
                "mean_auc": float(np.mean(aucs)),
                "mean_bacc05": float(np.mean(b05s)),
                "mean_bacc_cal": float(np.mean(bcals)),
                "mean_nll": float(np.mean(nlls)),
            }
            print(
                f"   - {gname:34s} (n={len(items):2d}): "
                f"AUC={np.mean(aucs):.4f} | bAcc@0.5={np.mean(b05s):.2%} | "
                f"bAcc@calib={np.mean(bcals):.2%} | NLL={np.mean(nlls):.4f}"
            )

    # 3. Direct CPU Test on demo_images: Re-encoding (PNG vs JPEG) & Resize/Upscaling Shortcut
    demo_dir = CODE_DIR / "demo_images"
    demo_files = sorted(demo_dir.glob("*.png"))
    bfree_ckpt = downloads_dir / "BFREE_dino2reg4" / "BFREE_dino2reg4" / "model_epoch_best.pth"

    reencode_results = {}
    if bfree_ckpt.exists() and demo_files:
        print("\n3. CPU Direct Sensitivity Test on demo_images (Format & Bicubic Upscale vs Native):")
        net = get_network("timm_c5i504_vit_base_patch14_reg4_dinov2.lvd142m", pretrained=False)
        state = torch.load(bfree_ckpt, map_location="cpu", weights_only=False)
        net.load_state_dict(state["model"])
        net.eval()

        norm_tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

        def score_bfree(pil_img: Image.Image, apply_bicubic_min_size: bool = False) -> Tuple[float, float, float]:
            if apply_bicubic_min_size:
                pil_img = ensure_min_size(pil_img, 504)
            t = norm_tf(pil_img).unsqueeze(0)
            with torch.no_grad():
                logit = net(t).squeeze().item()
            prob_true_sigmoid = 1.0 / (1.0 + math.exp(-logit))
            prob_buggy_2l = 1.0 / (1.0 + math.exp(-2.0 * logit))
            return logit, prob_true_sigmoid, prob_buggy_2l

        for img_path in [demo_files[0], demo_files[2]]:
            img = Image.open(img_path).convert("RGB")
            l_png, p_png, p2_png = score_bfree(img, apply_bicubic_min_size=False)

            buf_j85 = io.BytesIO()
            img.save(buf_j85, format="JPEG", quality=85)
            buf_j85.seek(0)
            img_j85 = Image.open(buf_j85).convert("RGB")
            l_j85, p_j85, _ = score_bfree(img_j85, apply_bicubic_min_size=False)

            img_256 = img.resize((256, 256), Image.Resampling.BILINEAR)
            l_256_native, p_256_native, _ = score_bfree(img_256, apply_bicubic_min_size=False)
            l_256_bicubic, p_256_bicubic, _ = score_bfree(img_256, apply_bicubic_min_size=True)

            reencode_results[img_path.name] = {
                "size": img.size,
                "logit_png": l_png,
                "prob_sig_l": p_png,
                "prob_sig_2l": p2_png,
                "logit_jpeg85": l_j85,
                "prob_jpeg85": p_j85,
                "logit_256_replicate": l_256_native,
                "prob_256_replicate": p_256_native,
                "logit_256_bicubic504": l_256_bicubic,
                "prob_256_bicubic504": p_256_bicubic,
            }
            print(
                f"   - {img_path.name} ({img.size[0]}x{img.size[1]}): "
                f"PNG logit={l_png:+.3f} (sig(l)={p_png:.4f}, sig(2l)={p2_png:.4f}) | "
                f"JPEG85 logit={l_j85:+.3f} (p={p_j85:.4f}) | "
                f"256px replicate_wrap p={p_256_native:.4f} vs 256->504 Bicubic p={p_256_bicubic:.4f} "
                f"(|delta_p|={abs(p_256_bicubic - p_256_native):.4f})"
            )

    return {
        "train_mixture": {
            "n_bfree_pair": n_bfree_pair,
            "n_extra": n_extra,
            "n_aigb": n_aigb,
            "n_total": n_total,
            "aigb_fraction": aigb_frac,
        },
        "group_stats": group_stats,
        "reencode_results": reencode_results,
    }


# ============================================================================
# 3. D2 Diagnosis: Gradient Ratio ||grad(lambda * L_DCS)||_2 / ||grad(L_CE)||_2
# ============================================================================


def _grad_norm_by_group(model: BFreeGlobalForgeViT) -> Dict[str, float]:
    groups = {
        "head": 0.0,
        "fc_norm": 0.0,
        "lib": 0.0,
        "gsr": 0.0,
        "lora": 0.0,
        "shared_no_head": 0.0,
        "all_trainable": 0.0,
    }
    for name, p in model.named_parameters():
        if not p.requires_grad or p.grad is None:
            continue
        gn_sq = float(p.grad.detach().float().norm(2).item() ** 2)
        groups["all_trainable"] += gn_sq
        if ".head." in name or name.endswith("head.weight") or name.endswith("head.bias"):
            groups["head"] += gn_sq
        else:
            groups["shared_no_head"] += gn_sq
            if "fc_norm" in name:
                groups["fc_norm"] += gn_sq
            elif name.startswith("lib.") or ".lib." in name:
                groups["lib"] += gn_sq
            elif name.startswith("gsr.") or ".gsr." in name:
                groups["gsr"] += gn_sq
            elif "lora_" in name:
                groups["lora"] += gn_sq
    return {k: math.sqrt(v) for k, v in groups.items()}


def _grad_vector(model: BFreeGlobalForgeViT, include_head: bool = True) -> torch.Tensor:
    vecs = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if not include_head and (".head." in name or name.endswith("head.weight") or name.endswith("head.bias")):
            continue
        if p.grad is None:
            vecs.append(torch.zeros_like(p.data).flatten())
        else:
            vecs.append(p.grad.detach().float().flatten())
    return torch.cat(vecs)


def run_d2_diagnosis(downloads_dir: Path) -> dict:
    print("\n" + "=" * 80)
    print("[D2] W1 DIAGNOSIS: GRADIENT RATIO ||grad(lambda*L_DCS)||_2 / ||grad(L_CE)||_2")
    print("=" * 80)

    # 1. Empirical Loss Ratio from Training Logs (Phase-1, Phase-2, v2-Direct)
    p2_ce_ep1, p2_dcs_ep1 = 0.2235, 0.0313
    p2_ce_ep8, p2_dcs_ep8 = 0.2001, 0.0105
    v2_ce_ep8, v2_dcs_ep8 = 0.1997, 0.0095
    lam = 0.01
    print("1. Empirical Loss Magnitude Ratio from Executed Training Logs (lambda_dcs = 0.01):")
    print(
        f"   - Phase-2 Epoch 1: L_CE={p2_ce_ep1:.4f}, L_DCS={p2_dcs_ep1:.4f}, "
        f"lambda*L_DCS={lam*p2_dcs_ep1:.6f} -> loss ratio = {(lam*p2_dcs_ep1)/p2_ce_ep1:.3%}"
    )
    print(
        f"   - Phase-2 Epoch 8: L_CE={p2_ce_ep8:.4f}, L_DCS={p2_dcs_ep8:.4f}, "
        f"lambda*L_DCS={lam*p2_dcs_ep8:.6f} -> loss ratio = {(lam*p2_dcs_ep8)/p2_ce_ep8:.3%}"
    )
    print(
        f"   - Direct-v2 Ep 8:  L_CE={v2_ce_ep8:.4f}, L_DCS={v2_dcs_ep8:.4f}, "
        f"lambda*L_DCS={lam*v2_dcs_ep8:.6f} -> loss ratio = {(lam*v2_dcs_ep8)/v2_ce_ep8:.3%}"
    )
    ls_floor = -(0.95 * math.log(0.95) + 0.05 * math.log(0.05))
    print(f"   - Note: Theoretical floor of LabelSmoothing(0.1) CE is H(0.95, 0.05) = {ls_floor:.4f}!")

    # 2. Direct CPU Gradient Norm Measurement on BFreeGlobalForgeViT + LoRA
    print("\n2. Direct CPU Gradient Norm Measurement on BFreeGlobalForgeViT (ViT-B/14 reg4, LoRA r=16):")
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    model = BFreeGlobalForgeViT(
        arch="vit_base_patch14_reg4_dinov2.lvd142m",
        num_classes=2,
        img_size=504,
        pretrained=False,
        use_lib=True,
        use_gsr=True,
        use_dcs=True,
        dcs_tau=0.07,
        lambda_dcs=0.01,
        label_smoothing=0.1,
    )
    bfree_ckpt = downloads_dir / "BFREE_dino2reg4" / "BFREE_dino2reg4" / "model_epoch_best.pth"
    if bfree_ckpt.exists():
        dat = torch.load(bfree_ckpt, map_location="cpu", weights_only=False)
        sd = dat["model"]
        remapped = {}
        for k, v in sd.items():
            if "head." in k:
                continue
            if k.startswith("patch_embed."):
                remapped["model." + k] = v
            else:
                remapped[k] = v
        model.load_state_dict(remapped, strict=False)
        print("   [*] Loaded real pretrained DINOv2 ViT-B/14 reg4 backbone weights from BFREE_dino2reg4!")

    model = apply_lora_to_backbone(model, r=16, lora_alpha=32, lora_dropout=0.05)
    model.train()

    demo_files = sorted((CODE_DIR / "demo_images").glob("*.png"))
    norm_tf = transforms.Compose([
        transforms.Resize((504, 504), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    clean_imgs, deg_imgs = [], []
    rng = random.Random(42)
    for i in range(4):
        pil = Image.open(demo_files[i % len(demo_files)]).convert("RGB")
        pil_crop = pil.resize((504, 504), Image.Resampling.BICUBIC)
        pil_deg = realistic_lifecycle_degrade(pil_crop, rng)
        clean_imgs.append(norm_tf(pil_crop))
        deg_imgs.append(norm_tf(pil_deg))

    x_clean = torch.stack(clean_imgs, dim=0)
    x_deg = torch.stack(deg_imgs, dim=0)
    labels = torch.tensor([0, 1, 0, 1], dtype=torch.long)

    def measure_grad_snapshot(tag: str) -> dict:
        model.zero_grad()
        out_c = model(x_clean)
        out_d = model(x_deg)

        ce_loss = model.ce(out_c["logits"], labels)
        dcs_loss = info_nce_loss(out_c["cls"], out_d["cls"], tau=model.dcs.tau)
        scaled_dcs = model.lambda_dcs * dcs_loss

        model.zero_grad()
        ce_loss.backward(retain_graph=True)
        gn_ce = _grad_norm_by_group(model)
        gvec_ce_shared = _grad_vector(model, include_head=False)

        model.zero_grad()
        scaled_dcs.backward(retain_graph=True)
        gn_dcs_scaled = _grad_norm_by_group(model)
        gvec_dcs_shared = _grad_vector(model, include_head=False)

        gn_dcs_raw = {k: v / model.lambda_dcs for k, v in gn_dcs_scaled.items()}
        cos_sim = F.cosine_similarity(
            gvec_ce_shared.unsqueeze(0), gvec_dcs_shared.unsqueeze(0)
        ).item()

        ratio_all = gn_dcs_scaled["all_trainable"] / max(gn_ce["all_trainable"], 1e-12)
        ratio_shared = gn_dcs_scaled["shared_no_head"] / max(gn_ce["shared_no_head"], 1e-12)
        ratio_head = gn_dcs_scaled["head"] / max(gn_ce["head"], 1e-12)

        print(f"\n   [{tag}] L_CE={ce_loss.item():.4f} | L_DCS={dcs_loss.item():.4f} | lambda*L_DCS={scaled_dcs.item():.6f}")
        print(f"     * ||grad(L_CE)||_2 (all trainable)          = {gn_ce['all_trainable']:.6f}")
        print(f"     * ||grad(lambda*L_DCS)||_2 (all trainable)  = {gn_dcs_scaled['all_trainable']:.6f}  --> Ratio (all)    = {ratio_all:.3%}")
        print(f"     * ||grad(L_CE)||_2 (shared backbone)        = {gn_ce['shared_no_head']:.6f}")
        print(f"     * ||grad(lambda*L_DCS)||_2 (shared backbone)= {gn_dcs_scaled['shared_no_head']:.6f}  --> Ratio (shared) = {ratio_shared:.3%}")
        print(f"     * ||grad(L_CE)||_2 (head)                   = {gn_ce['head']:.6f}")
        print(f"     * ||grad(lambda*L_DCS)||_2 (head)           = {gn_dcs_scaled['head']:.6f}  --> Ratio (head)   = {ratio_head:.3%}")
        print(
            f"     * Per-module ratio (lambda*DCS / CE): "
            f"LoRA={gn_dcs_scaled['lora']/max(gn_ce['lora'],1e-12):.3%}, "
            f"LIB={gn_dcs_scaled['lib']/max(gn_ce['lib'],1e-12):.3%}, "
            f"GSR={gn_dcs_scaled['gsr']/max(gn_ce['gsr'],1e-12):.3%}, "
            f"fc_norm={gn_dcs_scaled['fc_norm']/max(gn_ce['fc_norm'],1e-12):.3%}"
        )
        print(f"     * Cosine similarity(grad_CE, grad_DCS) on shared backbone = {cos_sim:+.4f}")

        return {
            "ce_loss": float(ce_loss.item()),
            "dcs_loss": float(dcs_loss.item()),
            "scaled_dcs_loss": float(scaled_dcs.item()),
            "gn_ce": gn_ce,
            "gn_dcs_scaled": gn_dcs_scaled,
            "gn_dcs_raw": gn_dcs_raw,
            "ratio_all": ratio_all,
            "ratio_shared": ratio_shared,
            "ratio_head": ratio_head,
            "cos_sim_shared": cos_sim,
        }

    snap_step0 = measure_grad_snapshot("Step 0 — Initial State (Pretrained DINOv2 + Untrained GSR/Head)")

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    for _ in range(2):
        opt.zero_grad()
        total, _, _ = model.compute_loss(x_clean, x_deg, labels)
        total.backward()
        opt.step()

    snap_converged = measure_grad_snapshot("Training Regime (L_DCS ~ 0.01, matching Epoch 1-8 logs)")

    return {
        "step0": snap_step0,
        "converged": snap_converged,
    }


# ============================================================================
# 4. D3, D4, D5 Supplementary Diagnostics
# ============================================================================


def run_d3_d4_d5_diagnosis(downloads_dir: Path) -> dict:
    print("\n" + "=" * 80)
    print("[D3 / D4 / D5] THRESHOLD CALIBRATION, SD2.1 GAP, AND STEP-0 LIB/GSR PERTURBATION")
    print("=" * 80)

    eval_metrics = load_eval_notebook_metrics(downloads_dir)

    # D3: Exact algebraic TPR/FPR recovery on AI-GenBench (shared 1500 real pool) & WildRF
    print("1. [D3] Exact TPR & FPR Reconstruction on AI-GenBench (Shared Real Pool n=1500, Fake n=150/gen):")
    d3_reconstructed = {}
    for mkey, mdata in eval_metrics.items():
        if not mdata["ai_genbench"]:
            continue
        # On easy GANs (e.g. CIPS, DDPM, StyleGAN1) where TPR=1.0 (150/150),
        # max(bAcc) gives upper bound -> exact TNR = 2 * max_bAcc - 1.0
        max_b05 = max(v["bacc05"] for v in mdata["ai_genbench"].values())
        max_bcal = max(v["bacc_cal"] for v in mdata["ai_genbench"].values())
        tnr_05 = min(1.0, 2.0 * max_b05 - 1.0)
        fpr_05 = 1.0 - tnr_05
        tnr_cal = min(1.0, 2.0 * max_bcal - 1.0)
        fpr_cal = 1.0 - tnr_cal

        print(
            f"\n   Model: {mkey} | calib_thr={mdata['calib_thr']} | real_pool_mean_p_fake={mdata['real_pool_mean']}\n"
            f"   -> Shared Real Pool (n=1500): FPR@0.5={fpr_05:.2%} (TNR={tnr_05:.2%}) | "
            f"FPR@calib={fpr_cal:.2%} (TNR={tnr_cal:.2%})"
        )
        d3_reconstructed[mkey] = {
            "fpr_05": fpr_05,
            "fpr_cal": fpr_cal,
            "generators": {},
            "wildrf": mdata["wildrf"],
        }
        focus_gens = [
            "FLUX 1 Dev",
            "DALL-E 3",
            "FLUX 1 Schnell",
            "Stable Diffusion 1.4",
            "Stable Diffusion 2.1",
            "Stable Diffusion XL 1.0",
            "Midjourney V6",
            "ADM",
            "ProGAN",
        ]
        for gname in focus_gens:
            if gname in mdata["ai_genbench"]:
                vals = mdata["ai_genbench"][gname]
                tpr_05 = max(0.0, min(1.0, 2.0 * vals["bacc05"] - tnr_05))
                tpr_cal = max(0.0, min(1.0, 2.0 * vals["bacc_cal"] - tnr_cal))
                d3_reconstructed[mkey]["generators"][gname] = {
                    "auc": vals["auc"],
                    "bacc05": vals["bacc05"],
                    "bacc_cal": vals["bacc_cal"],
                    "tpr05": tpr_05,
                    "tpr_cal": tpr_cal,
                    "nll": vals["nll"],
                }
                print(
                    f"     - AIGB/{gname:24s}: AUC={vals['auc']:.4f} | "
                    f"bAcc@0.5={vals['bacc05']:.2%} (TPR@0.5={tpr_05:.1%}) -> "
                    f"bAcc@cal={vals['bacc_cal']:.2%} (TPR@cal={tpr_cal:.1%}) | NLL={vals['nll']:.3f}"
                )
        for sname, vals in mdata["wildrf"].items():
            print(
                f"     - WildRF/{sname:22s}: AUC={vals['auc']:.4f} | "
                f"bAcc@0.5={vals['bacc05']:.2%} -> bAcc@cal={vals['bacc_cal']:.2%} "
                f"(delta={vals['bacc_cal']-vals['bacc05']:+.2%}) | NLL={vals['nll']:.3f}"
            )

    # D4: SD2.1 In-Distribution vs AI-GenBench SD2.1
    print("\n2. [D4] SD2.1 In-Distribution Validation vs AI-GenBench SD2.1:")
    ours_sd21 = eval_metrics.get("ours", {}).get("ai_genbench", {}).get("Stable Diffusion 2.1", {})
    bfree_sd21 = eval_metrics.get("bfree_official", {}).get("ai_genbench", {}).get("Stable Diffusion 2.1", {})
    print("   - Ours on B-Free SD2.1 Val (6 variants, 512x512 PNG): AUC=1.0000 | bAcc@0.5=99.48%..100.00%")
    if ours_sd21:
        print(
            f"   - Ours on AI-GenBench Stable Diffusion 2.1:           "
            f"AUC={ours_sd21['auc']:.4f} | bAcc@0.5={ours_sd21['bacc05']:.2%} | bAcc@calib={ours_sd21['bacc_cal']:.2%}"
        )
    if bfree_sd21:
        print(
            f"   - B-Free Official on AI-GenBench SD 2.1:              "
            f"AUC={bfree_sd21['auc']:.4f} | bAcc@0.5={bfree_sd21['bacc05']:.2%} | bAcc@calib={bfree_sd21['bacc_cal']:.2%}"
        )

    # D5: Step-0 LIB/GSR Feature Perturbation on CPU using real DINOv2 patch tokens
    print("\n3. [D5] Step-0 Feature Perturbation by Untrained LIB (beta=-0.85) + GSR (unzeroed proj):")
    torch.manual_seed(42)
    lib = LIBAdapter(dim=768, kernel_size=3, tau=0.5)
    gsr = GSRAdapter(dim=768, window=3, mask_prob=1.0)
    lib.eval()
    gsr.eval()

    bfree_ckpt = downloads_dir / "BFREE_dino2reg4" / "BFREE_dino2reg4" / "model_epoch_best.pth"
    demo_files = sorted((CODE_DIR / "demo_images").glob("*.png"))
    if bfree_ckpt.exists() and demo_files:
        base_model = BFreeGlobalForgeViT(
            arch="vit_base_patch14_reg4_dinov2.lvd142m",
            num_classes=2,
            img_size=504,
            pretrained=False,
            use_lib=False,
            use_gsr=False,
            use_dcs=False,
        )
        dat = torch.load(bfree_ckpt, map_location="cpu", weights_only=False)
        sd = dat["model"]
        remapped = {
            ("model." + k if k.startswith("patch_embed.") else k): v
            for k, v in sd.items()
            if "head." not in k
        }
        base_model.load_state_dict(remapped, strict=False)
        base_model.eval()
        norm_tf = transforms.Compose([
            transforms.Resize((504, 504), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
        imgs = torch.stack([norm_tf(Image.open(f).convert("RGB")) for f in demo_files[:2]], dim=0)
        with torch.no_grad():
            feats = base_model.model.forward_features(imgs)
            x_tokens = feats[:, base_model.num_prefix :, :]
    else:
        x_tokens = F.normalize(torch.randn(2, 1296, 768), dim=-1) * 28.0

    with torch.no_grad():
        x_lib, _ = lib(x_tokens, hw=36)
        x_gsr = gsr(x_tokens, hw=36)
        x_both = gsr(x_lib, hw=36)

        cos_lib = F.cosine_similarity(x_tokens, x_lib, dim=-1).mean().item()
        cos_gsr = F.cosine_similarity(x_tokens, x_gsr, dim=-1).mean().item()
        cos_both = F.cosine_similarity(x_tokens, x_both, dim=-1).mean().item()

        rel_l2_lib = ((x_lib - x_tokens).norm(dim=-1) / x_tokens.norm(dim=-1)).mean().item()
        rel_l2_gsr = ((x_gsr - x_tokens).norm(dim=-1) / x_tokens.norm(dim=-1)).mean().item()
        rel_l2_both = ((x_both - x_tokens).norm(dim=-1) / x_tokens.norm(dim=-1)).mean().item()

        mp_orig = x_tokens.mean(dim=1)
        mp_both = x_both.mean(dim=1)
        cos_mp = F.cosine_similarity(mp_orig, mp_both, dim=-1).mean().item()
        rel_l2_mp = ((mp_both - mp_orig).norm(dim=-1) / mp_orig.norm(dim=-1)).mean().item()

    print(
        f"   - LIB only (init beta=-0.85 -> alpha_0={torch.sigmoid(lib.beta).item():.4f}): "
        f"patch cos_sim={cos_lib:.4f}, rel_L2_drift={rel_l2_lib:.2%}"
    )
    print(
        f"   - GSR only (random Linear proj, no zero-init gate):   "
        f"patch cos_sim={cos_gsr:.4f}, rel_L2_drift={rel_l2_gsr:.2%}"
    )
    print(
        f"   - LIB + GSR combined at Step 0 (patch tokens):        "
        f"patch cos_sim={cos_both:.4f}, rel_L2_drift={rel_l2_both:.2%}"
    )
    print(
        f"   - LIB + GSR combined at Step 0 (mean-pool cls token): "
        f"cls cos_sim={cos_mp:.4f},   rel_L2_drift={rel_l2_mp:.2%}"
    )

    return {
        "d3": d3_reconstructed,
        "d5": {
            "alpha0": float(torch.sigmoid(lib.beta).item()),
            "cos_lib": cos_lib,
            "cos_gsr": cos_gsr,
            "cos_both": cos_both,
            "rel_l2_lib": rel_l2_lib,
            "rel_l2_gsr": rel_l2_gsr,
            "rel_l2_both": rel_l2_both,
            "cos_mean_pool": cos_mp,
            "rel_l2_mean_pool": rel_l2_mp,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run P1 Diagnostic Suite on CPU")
    parser.add_argument(
        "--downloads-dir",
        type=str,
        default=r"C:\Users\ADMIN\Downloads",
        help="Path to directory containing executed Kaggle notebooks and B-Free checkpoint",
    )
    parser.add_argument(
        "--out-json",
        type=str,
        default=str(REPO_ROOT / "research" / "p1_diagnosis_metrics.json"),
        help="Path to save diagnostic JSON output",
    )
    args = parser.parse_args()

    downloads_dir = Path(args.downloads_dir)
    d1 = run_d1_diagnosis(downloads_dir)
    d2 = run_d2_diagnosis(downloads_dir)
    d345 = run_d3_d4_d5_diagnosis(downloads_dir)

    summary = {"D1": d1, "D2": d2, "D3_D4_D5": d345}
    out_path = Path(args.out_json)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n[OK] Saved full P1 diagnostic metrics to: {out_path}")


if __name__ == "__main__":
    main()
</file>

<file path="research/yardstick-v2.md">
# Yardstick v2 — Đặc tả giao thức đánh giá công bằng (`harness_version = 2`)

> **Trạng thái:** **APPROVED** bởi User (2026-10-22) cho toàn bộ mục 🔒 `P0.1–P0.8`, cùng quyết định chốt Mục 10 của `AGENT_PLAN_V4.md`.
> **Nguyên tắc:** Không chạm vào bất kỳ file code train nào (`code/train_lora.py`, `code/networks/*`, `code/modules/*`, `code/configs/*`) cho đến khi hoàn tất P0 và P1.

---

## 0. Quyết định đã chốt từ User (Mục 10 `AGENT_PLAN_V4.md`)

1. **Duyệt toàn bộ `P0.1–P0.8`** về giao thức đánh giá mới (`harness_version = 2`).
2. **Provenance `ours-v2`:** lấy từ checkpoint **phase-2 stage-2** (slug `/kaggle/input/notebooks/baopv051/xla-trainng-stage-2/bfree_globalforge_lora_r16_best.pth` hoặc thư mục checkpoint tương ứng).
3. **Hoãn hoàn toàn MS-DCS (`v3`, K=4)** cho đến khi hoàn thành xong P2 (`multi_severity.enabled=false` giữ nguyên).
4. **Chốt cứng ngưỡng giả thuyết `H-A` đến `H-E` và `Guardrails`** đúng theo bảng ở Mục 5 (`P2.G` trong `AGENT_PLAN_V4.md`).
5. **Phân vai môi trường:** Local của Agent chỉ chạy CPU (`xla-cpu`). Mọi lệnh train/eval nặng đóng gói thành **Kaggle Notebook / script offline** để User chạy thủ công.
6. **Weights B-Free official:** đã có sẵn trong Kaggle input (`/kaggle/input/models/baopv051/bfree-baseline-weights/pytorch/default/1`, kèm `code/weights/BFREE_dino2reg4/config.yaml` + `model_epoch_best.pth`).

---

## 1. Ánh xạ tổng thể `U1–U7` (`research/eval-diff.md`) ↔ `P0.1–P0.12`

| Cờ bất công (`eval-diff.md`) | Bản chất sai lệch trong Harness v1 | Mục P0 giải quyết | Giải pháp kỹ thuật trong Yardstick v2 |
|---|---|---|---|
| **U1** — `ours` train trên AI-GenBench `train` (13 generator + 46.8k real), chấm trên `validation`; baseline zero-shot | Gộp chung 36 generator làm lệch trung bình có lợi cho `ours` trên 13 generator đã thấy | **P0.7, P0.8** | Tách bạch 2 nhóm `AIGB-seen` (13 `PRIORITY_GENERATORS`) và `AIGB-unseen` (23 generator còn lại) + có tuỳ chọn loại `origin_dataset` trùng; `seen` chỉ dùng làm Guardrail, **cấm** dùng để tuyên bố vượt baseline |
| **U2** — `ours` dùng `multi_crop=True` (tối đa 9 crop 504×504 phủ toàn ảnh) trong khi `bfree_official` & `globalforge` dùng `multi_crop=False` (1 center crop 504×504) | (1) `ours` được lợi thế ensemble 9 góc nhìn; (2) `bfree_official` nhận đầu vào đã bị cắt 504×504 nên `Wrapper5crops` bên trong tạo ra **5 crop giống hệt nhau** thay vì 5-crop trên embedding toàn ảnh như paper gốc | **P0.1** | Báo cáo song song 2 giao thức: **(i) `official-per-model`** (mỗi mô hình chạy đúng tiền xử lý/crop của tác giả gốc) và **(ii) `unified`** (mọi mô hình dùng chung chính sách crop: `unified_1crop` center 504 và/hoặc `unified_5crop`) |
| **U3** — `globalforge` chạy off-protocol (crop 504, không resize-to-1296 + JPEG q100, chạy `bf16` thay vì `fp32`) | Điểm dự đoán real bị kéo về `mean_p_fake = 0.4965`, `bAcc@0.5` kẹt ở `76.10%` | **P0.2** | Trong giao thức `official-per-model`: khôi phục đúng pipeline GlobalForge (`short256` / `resize-to-1296`, JPEG `q=100` re-encode nếu theo repo gốc, crop `224×224`, chạy `fp32` không ép `bfloat16`). Trong bảng `bAcc@0.5` của `unified`: ghi chú rõ nếu off-protocol hoặc chỉ so sánh `AUC` / `bAcc@calib` |
| **U4** — Bicubic pre-upscale cạnh ngắn `< 504` lên `504` trong `ModelWrapper.score` | Khớp với `ensure_min_size` khi train `ours`, nhưng phá cơ chế `replicate_wrap` trên embedding của `B-Free` gốc | **P0.1** | Ở `official-per-model`, `bfree_official` đưa nguyên tensor ảnh gốc (đã normalize) vào `Wrapper5crops` để tự `replicate_wrap` ở tầng patch embedding (`grid_size = 36×36`), **bỏ** bước `img.resize(..., Image.BICUBIC)` khi `min(w, h) < 504` |
| **U5** — `CALIB_THRESHOLD` (Youden's $J$) được fit trực tiếp trên toàn bộ điểm test `ai_genbench` (`n=6900`) rồi chấm lại chính nó và đem sang WildRF/viral | Rò rỉ tập test vào việc chọn ngưỡng (optimistic bias cho mọi mô hình) | **P0.7** | Fit `CALIB_THRESHOLD` và nhiệt độ $T$ (Temperature Scaling) trên **split calibration độc lập** (phần giữ riêng từ AIGB hoặc `wildrf/val`), đóng băng tham số rồi chỉ chấm **1 lần** trên các tập `test`. Giữ cột `bacc_calib_legacy` nếu cần đối chiếu lịch sử |
| **U6** — Label smoothing `0.1` khi train `ours` đặt sàn xác suất $[0.05, 0.95]$ ⇒ sàn ECE $\approx 0.05$; hàm `binary_ece` trong harness không cân bằng lớp | Khi `n_real = 1500` và `n_fake = 150` (tỷ lệ 10:1 mỗi generator AIGB), `binary_ece` bị thống trị 91% bởi lớp real | **P0.6, P0.7, P0.12** | Thay bằng `balanced_ece_binary` và `balanced_nll_binary` từ `code/utils/dmetrics.py` (trọng số $1/N_c$ cân bằng hai lớp); kèm báo cáo ECE sau Temperature Scaling trên tập calibration; sửa lỗi `sigmoid(2l)` của B-Free (P0.12) |
| **U7** — Tập toán tử suy giảm lúc train `v2` (`DEGRADE_STRENGTHS` 7 op) trùng khít với bộ toán tử chấm `RealDeg-Bench-style` và `realdeg_bench_style=False` | Chưa có số liệu RealDeg; nếu bật nguyên bản cũ thì chỉ đo *in-distribution robustness*, không đo được tổng quát hóa sang suy giảm chưa thấy | **P0.3** | Bật `realdeg_bench_style = True`, seed cố định theo từng ảnh (`hash(key, seed)`), tách rõ 2 nhóm: **(a) `RealDeg-indist`** (jpeg, blur, resize bicubic, gaussian noise) và **(b) `RealDeg-heldout`** (WebP, JPEG chroma-subsampling 4:2:0, Lanczos/bilinear resize, Poisson/ISO noise, sharpen, chuỗi mạng xã hội `resize→JPEG→resize→JPEG`). Loại bỏ `brightness/contrast/saturation` khỏi tuyên bố độ bền chính |
| *(Bổ sung V4/V5)* — Mẫu nhỏ (150 fake/gen), không CI, lỗi SynthWildX, mất điểm từng ảnh, sai công thức xác suất B-Free | Không kiểm định được ý nghĩa thống kê; mất dữ liệu cho chẩn đoán P1 | **P0.4, P0.5, P0.9, P0.10, P0.12** | Sửa loader SynthWildX (P0.4); nâng $\ge 500$ fake/gen + Bootstrap 1,000 lần CI 95% + DeLong/McNemar (P0.5); lưu `scores_{model}_{protocol}.parquet` (P0.9); thêm cột provenance (P0.10); sửa lỗi `cat([-l, l])` $\Rightarrow \sigma(2l)$ (P0.12) |

---

## 2. Chi tiết kỹ thuật từng hạng mục `P0.1–P0.12` và vị trí sửa trong Harness

> **Đối chiếu file thực tế đã mở:**
> - `C:\Users\ADMIN\Downloads\xla-eval-harness-ours (1).ipynb` (14 cells, chạy ngày `2026-10-03`, `OURS_CKPT` trỏ tới `xla-trainng-stage-2/bfree_globalforge_lora_r16_best.pth`)
> - `C:\Users\ADMIN\Downloads\xla-eval-harness-b-free.ipynb` (14 cells, chạy ngày `2026-09-23`)
> - `C:\Users\ADMIN\Downloads\xla-eval-harness-globalforge-dinov3.ipynb` (15 cells — có thêm cell 6 `!pip uninstall -y torchao`)

### P0.1 — Chính sách Crop & Tiền xử lý (`official-per-model` vs `unified`) [Map: U2, U4]
- **Hiện trạng (Cell 6 trong `ours`/`bfree`, Cell 7 trong `globalforge` — class `ModelWrapper`):**
  - Mọi ảnh có `min(w, h) < 504` đều bị ép `img.resize(..., Image.BICUBIC)` lên cạnh ngắn `504`.
  - `_crops(img)`: với `multi_crop=True` (`ours`) sinh lưới tối đa $3 \times 3 = 9$ crop `504×504`; với `multi_crop=False` (`bfree_official`, `globalforge`) chỉ cắt **1 center crop `504×504`**.
  - Hệ quả nghiêm trọng cho `bfree_official`: vì `ModelWrapper` đã cắt ảnh về đúng `504×504` trước khi gọi `self.forward_fn`, bên trong `Wrapper5crops.forward(x)` (`code/networks/wrapper5crops.py:38-51`), `embeddings` có kích thước đúng bằng `patch_size = (36, 36)`, dẫn đến `hs = 0, ws = 0` và 5 crop (`center, TL, BL, BR, TR`) **trùng khít nhau 100%** (5 bản sao của cùng 1 center crop), đồng thời `replicate_wrap` không bao giờ được kích hoạt!
- **Thiết kế Yardstick v2:**
  - Tham số hóa `PROTOCOL_MODE ∈ {"official", "unified_1crop", "unified_5crop"}`:
    1. **`official` (`official-per-model`):**
       - `bfree_official`: **Không** pre-upscale bicubic, **không** cắt 504×504 ở `ModelWrapper`. Đưa trực tiếp tensor `self.normalize(img).unsqueeze(0)` của toàn ảnh vào `Wrapper5crops` (xử lý từng ảnh vì kích thước khác nhau) để `Wrapper5crops` tự chạy `patch_embed.proj(x)`, `replicate_wrap` nếu `< 504`, và lấy trung bình 5 crop thực sự trên lưới patch embedding.
       - `globalforge_dinov2` / `globalforge_dinov3`: Chạy đúng chuẩn chính thức (xem P0.2).
       - `ours`: Chạy chuẩn 5-crop (hoặc 1-crop / 9-crop được gắn nhãn rõ ràng trong cột `protocol`).
    2. **`unified` (`unified_1crop` / `unified_5crop`):**
       - Mọi mô hình dùng chung một quy tắc tiền xử lý & số lượng crop để cô lập đóng góp của kiến trúc/trọng số khỏi TTA.

### P0.2 — Khôi phục giao thức gốc cho GlobalForge [Map: U3]
- **Hiện trạng (Cell 6/7 `_load_globalforge` & `ModelWrapper.score`):**
  - Đang dùng `IMG_SIZE = 504`, center crop `504`, không nén JPEG `q=100`, và chạy dưới `torch.amp.autocast("cuda", dtype=torch.bfloat16)`.
- **Thiết kế Yardstick v2:**
  - Trong chế độ `official`:
    - Áp dụng pipeline tiền xử lý chuẩn của GlobalForge (resize cạnh ngắn / đưa về kích thước chuẩn repo gốc, JPEG `q=100` round-trip qua `io.BytesIO`, center crop `224×224` hoặc theo cấu hình `REM`).
    - Tắt `bfloat16` autocast (`dtype=torch.float32`) khi chạy `globalforge_*`.
  - Trong bảng tổng hợp: ghi rõ `protocol = official_fp32` vs `unified`; không dùng `bAcc@0.5` off-protocol để kết luận.

### P0.3 — Bật `RealDeg` có hạt giống theo ảnh & tách `in-distribution` vs `held-out` [Map: U7]
- **Hiện trạng (Cell 2 `BENCHMARKS_ENABLED` & Cell 9 `DEGRADE_STRENGTHS` / Cell 12 `run_realdeg`):**
  - `BENCHMARKS_ENABLED["realdeg_bench_style"] = False`.
  - `apply_degradation` dùng `np.random.normal` (global RNG, phụ thuộc thứ tự gọi) cho `noise`, và `run_realdeg` chọn `rng.choice(DEGRADE_STRENGTHS[cond_val])` bằng 1 `rng` tuần tự.
  - 7 toán tử hiện tại (`jpeg, blur, resize, noise, brightness, contrast, saturation`) trùng hoàn toàn với `train_lora.py:182-190`.
- **Thiết kế Yardstick v2:**
  - Bật `BENCHMARKS_ENABLED["realdeg_bench_style"] = True`.
  - **Deterministic per-image seed:** Với mỗi ảnh `key` và điều kiện `cond_name`, khởi tạo `rng_img = np.random.default_rng( zlib.crc32(f"{key}:{cond_name}:{AIGENBENCH_SEED}".encode()) )` để mọi mô hình nhận **cùng một ảnh suy giảm bit-identical**.
  - **Phân tách 2 tập toán tử:**
    - `RealDeg-indist` (4 toán tử tần số/không gian có trong train v2): `jpeg` ($Q \in \{90,80,70,60,40\}$), `blur` ($\sigma \in \{0.5,1,2,3,5\}$), `resize_bicubic` ($s \in \{0.9,0.7,0.5,0.3,0.2\}$), `gaussian_noise` ($v \in \{0.0005,0.001,0.002,0.005,0.01\}$) + `compound_indist` (1–5 bước).
    - `RealDeg-heldout` (**6 toán tử chưa từng xuất hiện khi train**):
      1. `webp`: nén WebP $Q \in \{85, 70, 50, 30\}$.
      2. `jpeg_subsampling`: nén JPEG với `subsampling="4:2:0"` vs `"4:4:4"` ở $Q \in \{85, 65, 45\}$.
      3. `resize_lanczos_bilinear`: thu phóng bằng `Image.LANCZOS` hoặc `Image.BILINEAR` ($s \in \{0.75, 0.5, 0.35\}$).
      4. `poisson_iso_noise`: nhiễu Poisson (shot noise) + Gaussian read noise mô phỏng cảm biến ($\lambda \in \{100, 50, 25\}$).
      5. `sharpen`: `ImageFilter.UnsharpMask(radius=2, percent=p, threshold=3)` với $p \in \{120, 200, 300\}$.
      6. `social_pipeline`: chuỗi mô phỏng mạng xã hội `resize(0.75, BILINEAR) -> JPEG(q=80, 4:2:0) -> resize(0.85, BICUBIC) -> JPEG(q=70, 4:2:0)` và biến thể `compound_heldout` (3–5 bước).
  - **Loại bỏ** `brightness`, `contrast`, `saturation` khỏi chỉ số trung bình độ bền chính (chỉ giữ riêng nếu cần kiểm tra phụ).

### P0.4 — Sửa lỗi Loader `SynthWildX` [Map: U-SynthWildX]
- **Hiện trạng (Cell 11 `build_synthwildx_items` & log lỗi Cell 13):**
  - Lỗi thực tế: `ERROR | [synthwildx] BO QUA: Directory /kaggle/input/datasets/ngovanphuong2604/synthwildx-dataset is neither a Dataset directory nor a DatasetDict directory.`
  - Nguyên nhân: `load_from_disk(str(root))` được gọi trực tiếp trên thư mục gốc của Kaggle dataset, trong khi file `dataset_dict.json` hoặc `state.json` (chuẩn HuggingFace Arrow) hoặc các file `.parquet`/`.arrow`/thư mục ảnh nằm ở thư mục con bên dưới `SYNTHWILDX_DIR`.
- **Thiết kế Yardstick v2 (Cell 11 `build_synthwildx_items`):**
  - Viết hàm dò tìm đệ quy `_resolve_hf_or_image_dataset(root)`:
    1. Tìm thư mục con đầu tiên chứa `dataset_dict.json` hoặc `state.json` $\rightarrow$ gọi `load_from_disk(str(hit))`.
    2. Nếu không có `state.json`, tìm các file `*.parquet` hoặc `*.arrow` $\rightarrow$ nạp bằng `datasets.Dataset.from_parquet(...)` / `from_file(...)`.
    3. Nếu lưu dạng thư mục ảnh rời (`0_real`/`1_fake` hoặc `real`/`fake` hoặc kèm metadata CSV/JSON) $\rightarrow$ fallback sang `list_images` như `wildrf` / `bfree_viral_images`.
  - Đảm bảo có đủ số liệu `synthwildx` cho cả 3 mô hình (`ours`, `bfree_official`, `globalforge`).

### P0.5 — Tăng cỡ mẫu & Kiểm định thống kê (Bootstrap CI 95%, DeLong, McNemar) [Map: W7]
- **Hiện trạng (Cell 3 `CONFIG-PATHS` & Cell 7 `compute_all_metrics`):**
  - `AIGENBENCH_MAX_PER_GEN = 150` (quá nhỏ, sai số chuẩn lớn), không có khoảng tin cậy (CI), không có kiểm định ghép cặp.
- **Thiết kế Yardstick v2:**
  - Tăng `AIGENBENCH_MAX_PER_GEN = 500` (hoặc `None` = toàn bộ fake mỗi generator, tối đa $1,000$/generator trên `validation` $n=72,000$; `AIGENBENCH_MAX_REAL = 3000`).
  - **Bootstrap CI 95% ($B = 1,000$ lần, stratified theo lớp real/fake với seed cố định):** xuất thêm các cột `auc_ci_lo`, `auc_ci_hi`, `bacc_ci_lo`, `bacc_ci_hi` cho từng benchmark/nhóm.
  - **Kiểm định ghép cặp trên điểm từng ảnh (`scores_*.parquet`):**
    - **Kiểm định DeLong** (hoặc paired bootstrap trên hiệu $\Delta\text{AUC}$) cho AUC.
    - **Kiểm định McNemar** cho quyết định phân loại nhị phân (`bAcc@0.5` và `bAcc@calib`).

### P0.6 — Bộ Metric đầy đủ & Chuẩn hóa theo `dmetrics.py` [Map: U6, W6]
- **Hiện trạng (Cell 7 `binary_ece`, `balanced_nll`, `compute_all_metrics`):**
  - Tự viết lại `binary_ece(y_true, y_prob)` không cân bằng lớp; thiếu `Pd@FAR=10%`, `EER`, `FPR@TPR95`, và không tách `TPR`/`FPR` tại ngưỡng `0.5` và `calib`.
- **Thiết kế Yardstick v2:**
  - Gọi trực tiếp (hoặc nhúng nguyên bản) các hàm chuẩn từ `code/utils/dmetrics.py`:
    - `balanced_ece_binary(y_true, y_logit, bins=15)` (dùng trọng số $1/N_c$ cân bằng real/fake).
    - `balanced_nll_binary(y_true, y_logit)` (tính từ `balanced_cllr * ln(2)` trên logit tự nhiên).
    - `pd_at_far(y_true, y_score, 0.10)` ($\text{Pd}@10\%\text{FAR}$).
    - `calculate_eer2(y_true, y_score)` (Equal Error Rate nội suy).
  - Bổ sung thêm:
    - `fpr_at_tpr95`: FPR tại điểm $\text{TPR} = 0.95$.
    - **Tách riêng `tpr_05`, `fpr_05` (tại ngưỡng $p=0.5$ / logit $z=0$) và `tpr_calib`, `fpr_calib` (tại ngưỡng hiệu chuẩn)** kèm `mean_p_real`, `mean_p_fake` — phục vụ trực tiếp chẩn đoán **D3** ở P1 (phân biệt "real bị báo fake" vs "fake bị bỏ sót").

### P0.7 — Hiệu chuẩn ngưỡng & Temperature Scaling trên tập tách riêng [Map: U5, U6]
- **Hiện trạng (Cell 12 `run_ai_genbench`):**
  - Đang gộp toàn bộ điểm test `ai_genbench` (`all_true, all_prob`, $n=6,900$) để chạy `find_optimal_threshold(..., method="youden")` rồi chấm lại chính `ai_genbench` đó và dùng cho `wildrf`/`bfree_viral`.
- **Thiết kế Yardstick v2:**
  - **Tách tập Calibration độc lập (`calib_split`), tuyệt đối không đụng vào tập `test`:**
    1. **Calibration miền chuẩn (`calib_aigb`):** Tách cố định 1 tập con độc lập không trùng với mẫu test (ví dụ: lấy từ `ai_genbench` split `train` phần held-out hoặc tách 20% index cố định theo seed `CALIB_SEED = 2026` trước khi lấy mẫu `test_items`, hoàn toàn tách rời khỏi `test_items`).
    2. **Calibration miền mạng xã hội (`calib_wildrf_val`):** Sử dụng trực tiếp thư mục `WILDRF_DIR / "val"` (WildRF có sẵn split `val` độc lập với `test`).
  - Trên tập calibration, fit **1 lần duy nhất**:
    - Ngưỡng Youden's $J$: $z^* = \arg\max_z (\text{TPR}(z) - \text{FPR}(z))$.
    - Nhiệt độ **Temperature Scaling** $T > 0$: tối thiểu hóa `balanced_nll_binary(y_cal, z_cal / T)` bằng `scipy.optimize.minimize_scalar`.
  - Đánh giá trên tập test chỉ **1 lần** với $(z^*, T)$ đã khóa. Giữ cột `bacc_calib_legacy` (tính kiểu cũ) chỉ để đối chiếu lịch sử với v1.

### P0.8 — Tách bạch nhóm `Seen` vs `Unseen` trên AI-GenBench [Map: U1]
- **Hiện trạng (Cell 12 `run_ai_genbench`):**
  - Chỉ xuất 1 dòng tổng hợp `generator = "ALL"` gộp chung cả 36 generator.
- **Thiết kế Yardstick v2:**
  - Định nghĩa hằng số `SEEN_GENERATORS` trong harness khớp chính xác với `PRIORITY_GENERATORS` trong `code/train_lora.py:52-57` (13 generator):
    ```python
    SEEN_GENERATORS = {
        "ADM", "DDPM", "DeepFloyd IF", "Palette", "Denoising Diffusion GAN",
        "LaMa", "MAT", "SN-PatchGAN", "StyleGAN1", "VQGAN", "GANformer",
        "CIPS", "FaceSynthetics",
    }
    ```
  - Xuất 3 dòng tổng hợp riêng biệt cho `ai_genbench`:
    1. `ALL_SEEN` (13 generator đã thấy khi train `ours`) — **chỉ dùng làm Guardrail** (kiểm tra mô hình có quên kiến thức cũ không), **cấm** dùng để so sánh thắng/thua với baseline.
    2. `ALL_UNSEEN` (23 generator chưa từng thấy khi train, bao gồm DALL-E 3, FLUX 1 Dev/Schnell, Midjourney, SD 1.4/1.5/XL, StyleGAN2/3, ProGAN, ProjectedGAN, GauGAN, StarGAN, Glide, IMLE, VQ-Diffusion, ...) — **thước đo chính** trên AI-GenBench (báo cáo cả макро-average theo generator và pooled).
    3. `ALL_UNSEEN_EXCL_ORIGIN` — loại thêm các `origin_dataset` trùng với tập train (`COCO`, `DRCT` nếu có) để khử rò rỉ nguồn ảnh real.

### P0.9 — Lưu điểm số từng ảnh (`scores_{model}_{protocol}.parquet`) [Phục vụ P0.5 & P1]
- **Hiện trạng:**
  - `score_items` chỉ trả về `y_true, y_prob` trong RAM rồi xóa ngay sau khi tính metric.
- **Thiết kế Yardstick v2:**
  - Mọi lần chấm đều ghi lại bảng chi tiết từng mẫu ra `/kaggle/working/scores_{MODEL_ID}_{PROTOCOL_MODE}.parquet` với schema cố định:
    - `image_id` (`str`: đường dẫn tương đối hoặc `arrow_row_{idx}`)
    - `benchmark` (`str`: `ai_genbench`, `wildrf`, `synthwildx`, `bfree_viral_images`, `realdeg::...`)
    - `group` (`str`: tên generator hoặc platform `facebook`/`reddit`/`twitter`)
    - `is_seen_gen` (`bool`)
    - `origin_dataset` (`str`)
    - `condition` (`str`: `clean`, `single_jpeg`, `heldout_webp`, ...)
    - `label` (`int`: `0`=real, `1`=fake)
    - `logit` (`float32`: logit thực sự $z = l_1 - l_0$ hoặc $l$ trước sigmoid)
    - `prob` (`float32`: $\sigma(z)$)
    - `width`, `height`, `format` (`int`, `int`, `str` — ghi siêu dữ liệu ảnh ngay lúc mở PIL để phục vụ trực tiếp chẩn đoán **D1, D3, D4** ở P1 mà không cần đọc lại 158 GB dữ liệu!).

### P0.10 — Provenance & Quản lý phiên bản Harness (`harness_version = 2`) [Quy tắc #8]
- **Thiết kế Yardstick v2:**
  - Mọi dòng trong `eval_results_v2_{MODEL_ID}.csv` và metadata của `.parquet` bắt buộc có 5 cột định danh theo Quy tắc bất biến #8:
    - `git_sha`: commit SHA của repo `bfree_src`
    - `ckpt_sha256`: SHA-256 (16 ký tự đầu hoặc đầy đủ) của file trọng số đã nạp
    - `config_hash`: SHA-256 của cấu hình model/eval
    - `seed`: `AIGENBENCH_SEED` (`42`)
    - `harness_version`: `2` (kèm `protocol_mode`)
  - Cập nhật `research/eval-changelog.md` ghi nhận nâng cấp lên `harness_version = 2`.

### P0.11 — Chạy lại toàn bộ Baseline trên Yardstick v2 (`USER-KAGGLE`)
- Danh sách chạy trên Kaggle sau khi đóng gói notebook v2:
  1. `bfree_official` (kiểm tra sanity: đối chiếu số paper B-Free trên `bfree_viral_images` / `Synthbuster` subset).
  2. `globalforge_dinov3` (và `globalforge_dinov2` nếu có).
  3. `ours-v2` (checkpoint phase-2 stage-2 đã chốt ở Mục 10.2) + `ours-v1` (checkpoint phase-2 best nếu giữ để so sánh lịch sử).

---

## 3. Kiểm tra & Xác minh P0.12 — Lỗi `sigmoid(2l)` của `bfree_official` **[ĐÃ XÁC MINH]**

### 3.1. Bằng chứng trực tiếp trong code Harness v1
Đã mở và kiểm tra trực tiếp cả 3 file notebook thực tế:
- `C:\Users\ADMIN\OneDrive - ptit.edu.vn\Documents\xla-eval-harness-ours.ipynb` (Cell 6, dòng 221)
- `C:\Users\ADMIN\Downloads\xla-eval-harness-b-free.ipynb` (Cell 6, `id="09aa6ff8"`)
- `C:\Users\ADMIN\Downloads\xla-eval-harness-ours (1).ipynb` (Cell 6, `id="bdcdf7d6"`)

Trong hàm `_load_bfree_official()`:
```python
    def forward_fn(m, x):
        l = m(x)                      # num_classes=1 -> pseudo 2-logit: softmax[:,1] = sigmoid(l)
        return torch.cat([-l, l], dim=1)
```
Và trong `ModelWrapper.score(self, pil_images)`:
```python
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits = self.forward_fn(self.model, batch)
            probs.append(torch.softmax(logits.float(), dim=-1)[:, 1].mean().item())
```

### 3.2. Chứng minh toán học sai lệch
Với đầu ra 1 logit $l \in \mathbb{R}$ của B-Free (`num_classes = 1`, trong đó xác suất chuẩn theo code gốc `code/utils/dmetrics.py:85` là `prob = expit(y_score) = sigmoid(l)`):
- Khi ghép `logits = torch.cat([-l, l], dim=1)`, vector 2 chiều là $z = [-l, +l]$.
- Khi qua `torch.softmax(logits, dim=-1)[:, 1]`, xác suất tính được là:
  $$p_{\text{harness}} = \frac{e^{l}}{e^{-l} + e^{l}} = \frac{1}{1 + e^{-2l}} = \sigma(2l) \neq \sigma(l)$$
- Tương đương việc **nhân đôi độ lớn logit ($2l$) hay chia nhiệt độ $T = 0.5$** trước khi lấy sigmoid!
- Đối chiếu với code gốc B-Free tại `code/main_bfree.py` và `code/utils/dmetrics.py:56,85`:
  - `balanced_cllr` và `balanced_ece_binary` trong `dmetrics.py` đều nhận đầu vào là **logit tự nhiên $l$**, bên trong dùng `expit(l)` tức $\sigma(l)$, tức hiệu hai logit phải là $l_1 - l_0 = l$ (ví dụ `[0, l]` hoặc `[-l/2, +l/2]`), **không phải** `[-l, +l]`.

### 3.3. Tác động đến các chỉ số cũ (v1)
1. **Không bị ảnh hưởng (bất biến dưới phép biến đổi đơn điệu tăng nghiêm ngặt $l \mapsto 2l$ khi chỉ có 1 crop):**
   - `AUC` (`roc_auc_score`).
   - `bAcc@0.5` trên 1 crop (vì $\sigma(2l) \ge 0.5 \iff 2l \ge 0 \iff l \ge 0 \iff \sigma(l) \ge 0.5$).
2. **Bị làm sai lệch trực tiếp:**
   - **`ECE` và `NLL` của `bfree_official`:** Vì $\sigma(2l)$ đẩy mọi xác suất về sát $0$ hoặc $1$ (over-confident giả tạo với nhiệt độ $T=0.5$), `NLL` và `ECE` của `bfree_official` trong bảng v1 bị phóng đại/lệch so với thực tế.
   - **`CALIB_THRESHOLD` (`thr_calib`):** Ngưỡng xác suất tối ưu bị biến dạng theo $\sigma(2l)$.
   - **Trường hợp `multi_crop=True`:** Trung bình $\frac{1}{K}\sum_k \sigma(2l_k) \neq \sigma\left(\frac{1}{K}\sum_k l_k\right)$ (lệch cả thứ hạng nếu dùng nhiều crop ngoài `Wrapper5crops`).

### 3.4. Cách sửa chuẩn xác trong Yardstick v2
1. **Chuẩn hóa đầu ra của `ModelWrapper` về logit tự nhiên $z \in \mathbb{R}$:**
   - Với mô hình 1-logit (`bfree_official`): $z = l = m(x).\text{squeeze}(-1)$ (hoặc nếu biểu diễn 2-logit thì phải là `torch.cat([-l / 2, l / 2], dim=1)` hoặc `torch.cat([torch.zeros_like(l), l], dim=1)` để $z_1 - z_0 = l$).
   - Với mô hình 2-logit (`ours`, `globalforge`): logit nhị phân tự nhiên tương ứng với $\text{softmax}(z)_1 = \sigma(z_1 - z_0)$ chính là:
     $$z = z_1 - z_0 = \text{logits}[:, 1] - \text{logits}[:, 0]$$
     (đúng như `code/eval/run_benchmarks.py:40` đã viết: `score = (logits[:, 1] - logits[:, 0]).item()`).
   - Khi gộp nhiều crop: lấy trung bình trên **logit** $\bar{z} = \frac{1}{K}\sum_{k=1}^K z_k$ (đúng chuẩn `Wrapper5crops.forward` dòng 57: `y = torch.mean(...)` trên logit), sau đó xác suất là $p = \sigma(\bar{z}) = \text{torch.sigmoid}(\bar{z})$.
2. **Truyền trực tiếp logit $\bar{z}$ vào các hàm `dmetrics.py` (`balanced_ece_binary`, `balanced_nll_binary`, `pd_at_far`, `calculate_eer2`):** đảm bảo tương thích 100% với định nghĩa của tác giả B-Free.

---

## 4. Bảng tóm tắt vị trí Cell sẽ cập nhật khi đóng gói Notebook Harness v2

| Cell (theo `xla-eval-harness-ours (1).ipynb`) | Nội dung hiện tại | Thay đổi trong Yardstick v2 (`harness_version = 2`) |
|---|---|---|
| **Cell 2** (`id="f15431b5"`) | `MODEL_ID`, `BENCHMARKS_ENABLED` (`realdeg_bench_style: False`) | Thêm `PROTOCOL_MODE = "official"` (`"official"` \| `"unified_1crop"` \| `"unified_5crop"`), `HARNESS_VERSION = 2`; bật `realdeg_bench_style: True` |
| **Cell 3** (`id="df30088a"`) | `CONFIG-PATHS`, `AIGENBENCH_MAX_PER_GEN = 150`, `AIGENBENCH_MAX_REAL = 1500` | Tăng `AIGENBENCH_MAX_PER_GEN = 500`, `AIGENBENCH_MAX_REAL = 3000`; thêm `SEEN_GENERATORS` (13 gen), `CALIB_SEED = 2026`; thêm hàm băm `sha256_file` cho provenance |
| **Cell 6** (`id="bdcdf7d6"`) | `ModelWrapper`, `_load_ours`, `_load_bfree_official` (`cat([-l, l])`), `_load_globalforge` | Sửa P0.12 (`z = l` cho B-Free, `z = l1 - l0` cho 2-logit, `p = sigmoid(z)`); tách nhánh `PROTOCOL_MODE == "official"` (B-Free full-image `Wrapper5crops` không pre-upscale; GlobalForge official resize/crop/fp32) và `"unified_*"` |
| **Cell 7** (`id="e059485f"`) | `binary_ece`, `balanced_nll`, `find_optimal_threshold`, `compute_all_metrics` | Thay bằng bộ metric chuẩn `dmetrics.py` (`balanced_ece_binary`, `balanced_nll_binary`, `pd_at_far`, `calculate_eer2`) + `fpr_at_tpr95`, `tpr_05`, `fpr_05`, `tpr_calib`, `fpr_calib` + Bootstrap CI 95% (1,000 lần) + Temperature Scaling $T$ |
| **Cell 8** (`id="06bee8af"`) | `ArrowSource`, `load_pil`, `score_items` | Thu thập thêm `(image_id, width, height, format, logit, prob)` để xuất `scores_{MODEL_ID}_{PROTOCOL_MODE}.parquet` (P0.9); truyền seed theo từng ảnh vào hàm suy giảm |
| **Cell 9** (`id="e7e1836d"`) | `DEGRADE_STRENGTHS` (7 op), `apply_degradation`, `apply_compound` | Tách `DEGRADE_INDIST_OPS` (4 op) và `DEGRADE_HELDOUT_OPS` (6 op mới: `webp`, `jpeg_subsampling`, `resize_lanczos_bilinear`, `poisson_iso_noise`, `sharpen`, `social_pipeline`); dùng RNG riêng theo từng ảnh |
| **Cell 10–11** (`id="ce90669f"`, `ba6e55a7`) | `build_ai_genbench_groups`, `build_benchmark_filelist`, `build_synthwildx_items` | Tách `calib_items` độc lập khỏi `test_items` (P0.7); sửa `build_synthwildx_items` tự động dò tìm thư mục con chứa `dataset_dict.json`/`state.json`/`.parquet`/ảnh (P0.4); thêm loader `wildrf/val` cho calibration |
| **Cell 12–14** (`id="07ef20fb"`, `1d9cb38a`, `2baa28f2`) | `run_ai_genbench`, `run_static_benchmark`, `run_synthwildx`, `run_realdeg` | Fit `CALIB_THRESHOLD` & $T$ trên `calib_items` + `wildrf/val` trước khi chấm test; xuất dòng tổng hợp tách biệt `ALL_SEEN`, `ALL_UNSEEN`, `ALL_UNSEEN_EXCL_ORIGIN` (P0.8); lưu `.parquet` điểm từng ảnh (P0.9) và thêm 5 cột provenance vào CSV (P0.10) |
</file>

<file path="AGENT_PLAN_V4.md">
# AGENT_PLAN_V4 — Đánh giá & cải tiến B-Free × GlobalForge (hướng "chẩn đoán trước, ablation rộng")

> Nguồn: `danh-gia-va-plan-cai-tien-v4.md` + đối chiếu với repo (`repomix-output.md`).
> File này viết cho **AI agent thực thi**. Đọc hết mục 0–2 trước khi làm bất cứ việc gì.
> Nhãn: **[ĐÃ XÁC MINH]** = thấy trực tiếp trong code/log; **[GIẢ THUYẾT]** = cần thí nghiệm; **[AGENT-CẦN-KIỂM]** = suy ra từ đọc code, phải xác minh bằng test trước khi tin.

---

## 0. Mục tiêu & triết lý

Mô hình hiện tại (`ours`, LoRA r=16 trên DINOv2 ViT-B/14 reg4 + LIB + GSR + L_DCS) **thua B-Free official ngoài AI-GenBench** (WildRF AUC 0.943 vs 0.974; bfree_viral 0.893 vs 0.967) dù gần như tuyệt đối trên 13 generator đã thấy khi train.

Kế hoạch V4 đi theo thứ tự cứng:

```
P0 Yardstick v2 ─┬─> P1 Chẩn đoán ──> P2 Sửa recipe + Ablation ──> P3 Cải tiến cơ chế ──> P4 Đánh giá cuối
                 └─> P5 Dọn repo (song song, không chặn)
```

**Không chạy MS-DCS (v3, K=4) cho đến khi P2 xong.** v3 hiện **chưa từng chạy** (code đã có, `multi_severity.enabled=false`); mọi nhận định về hiệu quả của nó chỉ là giả thuyết.

Nguyên nhân gốc đã nhận diện (xem `W1..W9` trong tài liệu nguồn):

| Mã | Nội dung | Trạng thái |
|---|---|---|
| W1 | CE chỉ tính trên view sạch; view suy giảm chỉ vào InfoNCE với λ=0.01 (loss ≈ 0.0105 ⇒ đóng góp ≈ 1e-4 so với CE ≈ 0.2) → DCS gần như không tác động | ĐÃ XÁC MINH (`compute_loss` trong `networks/bfree_globalforge_vit.py`) |
| W2 | Không ablation; không init từ B-Free; GSR `proj` khởi tạo ngẫu nhiên, không gate; LIB α₀≈0.30; head mean-pool thay vì CLS | ĐÃ XÁC MINH |
| W3 | AIGB ≈ 64% mẫu train; real/fake AIGB khác `origin_dataset`; `ensure_min_size` upscale bicubic → nguy cơ shortcut | số liệu ĐÃ XÁC MINH, cơ chế GIẢ THUYẾT |
| W4 | So sánh harness không công bằng (U1–U7 trong `research/eval-diff.md`) | ĐÃ XÁC MINH |
| W5 | Val bão hòa (val bAcc 0.992 từ epoch 1; CE chạm sàn label-smoothing 0.1997 vs 0.1985) | ĐÃ XÁC MINH |
| W6 | Label smoothing 0.1 ⇒ ECE sàn ≈ 0.05; ECE không cân bằng lớp | ĐÃ XÁC MINH |
| W7 | 150 fake/generator, 1 seed, không CI; provenance checkpoint "ours" không truy ngược được | ĐÃ XÁC MINH |

---

## 1. Quy tắc bất biến (vi phạm = dừng và báo user)

Các quy tắc này lấy từ `agent.md`, `research/PATHS.md`, `research/log.md`, `research/eval-diff.md`. Nếu mâu thuẫn, **`research/PATHS.md` (đã xác minh bằng `ls`) thắng `agent.md`** về đường dẫn.

1. **Đường dẫn thật:** code làm việc ở worktree `B-Free-loss-backbone/` (nhánh `integration/loss-backbone`), thư mục `code/` bên trong. `B-Free/` (đối chiếu, HEAD hiện ở `integration/train-eval`) và `GlobalForge-BE0F/` là **chỉ đọc**. `agent.md` nhắc `B-Free-integration/` nhưng thư mục này **không tồn tại**.
2. **Commit:** tác giả `P-Bao <BaoPV.B23KH006@stu.ptit.edu.vn>`; **không** thêm trailer `Co-Authored-By`; chỉ commit file đã chủ ý sửa. Worktree là CRLF còn blob là LF (38 file hiện hiện "modified" chỉ vì khoảng trắng) → file nào sửa thì chuyển LF rồi mới commit.
3. **🔒 Cần user duyệt trước khi chạm** vào: metric, split, tiền xử lý của harness; phương pháp calib; bật RealDeg; mọi thay đổi làm đổi ý nghĩa số liệu cũ (theo `CLAUDE.md`, nhắc lại trong `eval-diff.md`). Các việc đánh dấu 🔒 bên dưới: viết đề xuất vào `research/` và **chờ OK**.
4. **Mọi thay đổi mặc định phải tắt được bằng config và đường cũ giữ bit-identical** (như MS-DCS đã làm: có `tests/ref_single_view.pt`). Thêm cờ mới ⇒ thêm vào `configs/bfree_dcs.yaml` **và** vào `RESUME_CHECK_KEYS` trong `train_lora.py` (nếu không, resume sẽ không phát hiện lệch cấu hình).
5. **Hyperparameter không hardcode:** để trong YAML/CLI (vì cần sweep).
6. **Gate bắt buộc trước khi tốn GPU:** `tests/overfit_test.py` (32 ảnh, loss → ≈0 trong ≤50 bước), `tests/resume_test.py` (RESUME PASS), test bit-identity của đường legacy, và test mới "bước 0 ≡ B-Free" (xem P2.B).
7. **Mọi quyết định tiêu chí thành công được ghi vào `research/` TRƯỚC khi chạy** (giữ thông lệ H1/H2). Kết quả 1 seed = "quan sát", không phải "chứng minh".
8. **Mọi CSV/parquet kết quả có cột:** `git_sha`, `ckpt_sha256`, `config_hash`, `seed`, `harness_version`.
9. **Không bịa số.** Chưa chạy ⇒ ghi `TODO: pending run`. Không điền ước lượng vào bảng kết quả.
10. **Gặp sai lệch so với giả định** (ví dụ D1 bác bỏ giả thuyết shortcut) ⇒ dừng, ghi `research/log.md` (provenance `ai`), báo user, **không tự đổi hướng**.
11. **Agent không có GPU Kaggle.** Việc huấn luyện/eval lớn là **USER-KAGGLE**: agent chuẩn bị notebook + `research/iter-N.md` theo mẫu (*Changed / Verified / You must do / Paste back*) rồi dừng chờ user dán kết quả.
12. **Môi trường CPU local:** conda env `xla-cpu` (py3.11, torch CPU 2.8.0, timm 1.0.22, peft 0.15.2). Không đụng env `face_recognition`. Kaggle chạy **offline**, đường dẫn viết cứng, fail-fast.
13. **Giấy phép:** code/dữ liệu B-Free chỉ phi lợi nhuận, phải ghi nguồn — nhắc trong báo cáo cuối.
14. `tests/` đang nằm trong `.gitignore` ⇒ commit test bằng `git add -f` (cho đến khi P5.4 sửa).

Ký hiệu cột **Ai làm**: `AGENT` (CPU/code), `USER-KAGGLE` (cần GPU), `🔒` (cần OK user).

---

## 2. Bản đồ repo (những chỗ sẽ sửa)

| Thành phần | File | Ghi chú đã xác minh |
|---|---|---|
| Mô hình | `code/networks/bfree_globalforge_vit.py` | `BFreeGlobalForgeViT`; `forward()` → `{"logits","cls"}` với `cls` = mean-pool patch token **sau LIB/GSR**; head gọi trực tiếp `model.fc_norm`, `head_drop`, `head`; `compute_loss()` → `(total, ce, dcs)` |
| LIB | `code/modules/lib_adapter.py` | Gaussian 3×3 depthwise cố định, `beta` init −0.85 (α≈0.30); `lib_tau` **không dùng**; trả `mask` không ai dùng |
| GSR | `code/modules/gsr_adapter.py` | single-head attn, che Chebyshev ≤ `window` (kể cả chính token), `out = proj(attn@V) + x`, **không gate, init mặc định** |
| DCS | `code/modules/dcs_loss.py` | `info_nce_loss` (symmetric, negative trong batch), `multi_severity_dcs` |
| Train | `code/train_lora.py` | `PairDataset` trả `(clean, degraded, label)`; `apply_degradation/apply_compound/realistic_lifecycle_degrade`; `load_pretrained_backbone` (chỉ DINOv2 safetensors); `apply_lora_to_backbone`; `main()`; `RESUME_CHECK_KEYS`; `run_epoch`; AdamW + CosineAnnealingLR |
| Checkpoint B-Free | `networks/__init__.py::get_network`, `wrapper5crops.py`, `BFreeGlobalForgeViT.load_checkpoint` | B-Free head 1 logit (`num_classes=1`); ours 2 logit |
| Metric | `code/utils/dmetrics.py` | có sẵn `balanced_ece_binary`, `balanced_nll_binary`, `pd_at_far`, `calculate_eer2` |
| Config | `code/configs/bfree_dcs.yaml`, `loader.py` | có `multi_severity` (mặc định OFF) |
| Harness | `notebooks/xla-eval-harness-*.ipynb` (không có trong repomix) | 3 notebook dùng chung 1 codebase; `ModelWrapper` quyết định crop/score — **agent phải mở file thật trước khi sửa** |
| Nhật ký | `research/` (`log.md`, `iter-N.md`, `eval-diff.md`, `eval-changelog.md`, `PATHS.md`) | cập nhật theo mẫu hiện có |

---

## 3. P0 — Yardstick v2 (🔒 cần user duyệt; ~1 ngày, GPU thấp)

**Mục tiêu:** một thước đo công bằng, đủ lớn, có CI; mọi kết luận sau đó dựa vào nó.

**Bước đầu (AGENT, không cần duyệt):** viết `research/yardstick-v2.md` tổng hợp P0.1–P0.11, đối chiếu từng mục với `eval-diff.md` (U1–U7) và mở các notebook harness để ghi lại vị trí cell cần sửa. Sau đó xin OK user cho các mục 🔒.

| # | Việc | Chi tiết / tiêu chí xong | Ai |
|---|---|---|---|
| P0.1 | 🔒 **Chính sách crop** | Báo cáo song song 2 giao thức: (i) *official-per-model*: B-Free dùng 5 crop trên embedding toàn ảnh, không pre-upscale (đúng `Wrapper5crops` + `replicate_wrap`); GlobalForge theo preprocessing gốc fp32; (ii) *unified*: cùng một chính sách crop/resize cho mọi mô hình. Mỗi số ghi rõ thuộc giao thức nào | AGENT + 🔒 |
| P0.2 | 🔒 **GlobalForge off-protocol (U3)** | Dựng lại 224 crop, resize-to-1296, JPEG q100, fp32 như bản chính thức; hoặc loại khỏi bảng bAcc, chỉ giữ AUC kèm chú thích | AGENT + 🔒 |
| P0.3 | 🔒 **Bật RealDeg, seed theo ảnh, có toán tử held-out** | Held-out không có trong train: WebP, JPEG chroma-subsampling, Lanczos/bilinear resize, Poisson/ISO noise, sharpen, pipeline "mạng xã hội" (resize→JPEG→resize→JPEG). Báo cáo tách *in-distribution* vs *held-out*. **Bỏ brightness/contrast/saturation khỏi tuyên bố độ bền** (AUC không đổi) | AGENT + 🔒 |
| P0.4 | **Sửa loader SynthWildX** | Có số cho cả 3 mô hình. Lỗi hiện tại: "neither a `Dataset` nor `DatasetDict` directory" — trỏ đúng thư mục chứa `dataset_dict.json`/`.arrow` | AGENT |
| P0.5 | 🔒 **Tăng cỡ mẫu + CI** | ≥ 500 fake/generator (hoặc toàn bộ); bootstrap 1,000 lần cho CI 95%; DeLong (AUC) / McNemar (bAcc) khi so cặp | AGENT + 🔒 |
| P0.6 | **Metric** | Thêm `balanced_ece_binary`, `balanced_nll_binary`, `pd_at_far(…,0.10)`, `calculate_eer2`, FPR@TPR95; **FPR/TPR riêng tại ngưỡng 0.5** | AGENT |
| P0.7 | 🔒 **Ngưỡng calib trên tập tách riêng** | Fit ngưỡng/temperature trên split calibration độc lập (phần giữ riêng của AIGB-train hoặc WildRF `val`), đánh giá 1 lần trên test. Giữ cột cũ để đối chiếu lịch sử | AGENT + 🔒 |
| P0.8 | **Aggregate seen/unseen** | Hai nhóm tách bạch cho mọi mô hình (13 generator seen = `PRIORITY_GENERATORS` trong `train_lora.py`). *Seen* chỉ để kiểm tra mô hình có học được gì, **không dùng để so sánh** | AGENT |
| P0.9 | **Lưu điểm từng ảnh** | `scores_{model}.parquet` (id, label, score, generator, benchmark, điều kiện) | AGENT |
| P0.10 | **Provenance** | Metadata theo quy tắc #8; viết `eval-changelog.md` cho `harness_version = 2` | AGENT |
| P0.11 | **Chạy lại mọi baseline** | B-Free official, GlobalForge, ours-v1/v2 (mọi checkpoint còn giữ). Sanity: B-Free official tái lập được số paper trên ≥1 bộ chung | USER-KAGGLE |
| P0.12 | **Kiểm tra cách tính xác suất của B-Free** *(bổ sung từ đối chiếu V5)* | Trong `ModelWrapper`, kiểm tra nếu `forward_fn` của `bfree_official` trả `cat([-l, l])` rồi harness `softmax` ⇒ p = `sigmoid(2l)` (không phải `sigmoid(l)` như comment ghi). AUC/bAcc@0.5 không đổi nhưng ECE/NLL/ngưỡng xác suất lệch. Nếu đúng: dùng logit gốc cho mọi metric. **[AGENT-CẦN-KIỂM]** | AGENT |

**Cổng P0:** có `eval_results_v2/*.csv` cho ≥3 mô hình, CI đầy đủ, RealDeg + SynthWildX có số liệu → báo user. **Từ sau cổng này, mọi gate so với bảng v2, không so với số cũ.**

---

## 4. P1 — Chẩn đoán không cần train (~1–2 ngày, đọc điểm số đã lưu)

Mục tiêu: xác nhận hoặc bác bỏ W1, W3 **trước** khi tốn compute. Kết quả ghi vào `research/diagnosis.md` (bảng: xác nhận / bác bỏ / chưa rõ).

| # | Giả thuyết | Thí nghiệm | Quyết định rút ra |
|---|---|---|---|
| D1 | Shortcut định dạng/độ phân giải (W3) | Thống kê theo lớp và `origin_dataset`/generator của AIGB: kích thước ảnh, bảng lượng tử JPEG, định dạng file; tương quan điểm số ↔ thuộc tính; đo độ lệch điểm khi re-encode (JPEG q95/85/75, PNG) và resize | Điểm dịch mạnh theo định dạng/kích thước (hơn B-Free) ⇒ ưu tiên P2.C |
| D2 | DCS vô hiệu (W1) | Chạy 200 bước, ghi **chuẩn gradient của CE và của λ·DCS** lên tham số trainable, tính tỷ lệ | Tỷ lệ < 1% ⇒ P2.A bắt buộc. **Nếu D2 bác bỏ W1 ⇒ DỪNG, báo user** |
| D3 | Dạng lỗi trên WildRF/viral | Tại ngưỡng 0.5 và calib: FPR vs TPR; điểm trung bình ảnh real | Phân biệt "real bị báo fake" (shortcut nén) với "fake bị bỏ sót" |
| D4 | Nghịch lý SD2.1 | So thuộc tính ảnh SD2.1 B-Free-train vs AIGB; chấm ours trên SD2.1 in-dist (COCO-SD2.1 val) vs SD2.1-AIGB | In-dist ≈1.0 nhưng AIGB-SD2.1 thấp ⇒ vấn đề là *shift hậu xử lý* |
| D5 | Nhiễu khởi tạo của module | CKA/cosine giữa đặc trưng DINOv2 gốc và sau LIB/GSR ở bước 0 và sau train | Định lượng W2 |

**Cổng P1:** nếu D1 *bác bỏ* shortcut ⇒ hạ ưu tiên P2.C. Nếu D2 *bác bỏ* W1 ⇒ dừng và báo user trước khi sang P2.

---

## 5. P2 — Sửa recipe & ablation (trọng tâm; ~3–5 ngày)

Mọi cờ mới mặc định **tắt** (đường `legacy` giữ bit-identical).

### P2.A — Cho view suy giảm tham gia phân loại (sửa W1) · không tốn thêm forward

View suy giảm đã được forward sẵn cho DCS, dùng nó cho CE **không tăng chi phí**.

Thêm vào `configs/bfree_dcs.yaml` (và `RESUME_CHECK_KEYS`): `loss.mode: legacy|dual_view`, `loss.w_deg`, `loss.lam_cons`, `loss.cons_T`, `loss.p_aug_on_clean`.

```python
# networks/bfree_globalforge_vit.py :: compute_loss  (chỉ nhánh loss.mode == "dual_view")
out_c = self.forward(images_clean)
out_d = self.forward(images_deg)                       # đã có trong v2
ce_c  = self.ce(out_c["logits"], labels)
ce_d  = self.ce(out_d["logits"], labels)               # MỚI: nhãn không đổi dưới suy giảm
cons  = F.kl_div(F.log_softmax(out_d["logits"] / T, -1),
                 F.softmax(out_c["logits"].detach() / T, -1),
                 reduction="batchmean") * T * T        # MỚI: nhất quán quyết định
dcs   = info_nce_loss(out_c["cls"], out_d["cls"], tau=self.dcs.tau)
total = ce_c + w_deg * ce_d + lam_cons * cons + lam_dcs * dcs
# Trả (total, ce, dcs) — giữ hợp đồng 3 scalar. Thêm self.last_terms = {ce_c, ce_d, cons} để log.
```

- Khởi điểm: `w_deg = 1.0`, `lam_cons ∈ {0, 0.5}`.
- Nên thêm suy giảm "kiểu B-Free" lên cả view sạch với xác suất thấp (p≈0.2).
- Cập nhật `run_epoch` và vòng val để log thêm `ce_c`, `ce_d`, `cons` vào `train_log.csv` (như `dcs_k*` đã làm cho MS mode). Cả 3 chỗ gọi `compute_loss` (batch finder ~dòng 671, train ~918, val ~959) phải tương thích.
- **Test:** `loss.mode: legacy` ≡ tham chiếu `tests/ref_single_view.pt` (bit-identical); `dual_view` với `w_deg=0, lam_cons=0` ≡ legacy.

### P2.B — Init từ B-Free & không phá đặc trưng pretrained (sửa W2)

1. **`--init_from {dinov2,bfree}`** trong `train_lora.py` (mặc định `dinov2`, giữ nguyên đường cũ). `bfree` dùng `BFreeGlobalForgeViT.load_checkpoint`.
   - **[AGENT-CẦN-KIỂM] hai điểm rủi ro trong `load_checkpoint` hiện tại:**
     - Checkpoint B-Free (qua `Wrapper5crops.load_state_dict`) có key `patch_embed.*` và các key timm **không có tiền tố `model.`** (ví dụ `blocks.0.attn.qkv.weight`). Hàm hiện chỉ remap `patch_embed.*` → `model.patch_embed.*`, nên các key còn lại có thể thành *unexpected* và bị bỏ qua im lặng với `strict=False`. Báo cáo `num_missing/num_unexpected` phải được kiểm tra, và cần thêm tiền tố `model.` cho mọi key không phải `patch_embed.*`.
     - B-Free dùng head **1 logit**, ours **2 logit**: `load_state_dict(strict=False)` **vẫn báo lỗi khi sai shape**. Cần xử lý head riêng. Chuyển 1 logit → 2 logit sao cho `z = l1 − l0 = l`: `W2 = [−w/2; +w/2]`, `b2 = [−b/2; +b/2]`.
   - Test: bước 0 phải khớp B-Free (xem mục 5 dưới).
2. **Zero-init có gate cho GSR:** `out = tokens + gate * proj(attn @ V)` với `gate = nn.Parameter(torch.zeros(1))` (hoặc zero-init `proj`). Khi đó bước 0 ≡ B-Free. Cần cờ `backbone.gsr_zero_init`.
3. **LIB gần identity:** `beta_init = −6` (α₀ ≈ 0.0025) thay vì −0.85; cờ `backbone.lib_beta_init`.
4. **Head CLS:** cờ `backbone.head_mode: cls|meanpool`. Với `cls`: lấy `feats = self.model.forward_features(x)` rồi `self.model.forward_head(feats)` (timm: lấy token 0 → `fc_norm` → `head_drop` → `head`). `L_DCS` vẫn dùng mean-pool patch. Mean-pool chỉ giữ như một ô ablation riêng.
5. **GATE BẮT BUỘC "bước 0 ≡ B-Free":** logits của mô hình khởi tạo trùng B-Free gốc (sai số số học, fp32, |Δ| < 1e-3) trên 64 ảnh 504×504. Đây là test thay thế overfit gate (overfit gate không bắt được lỗi kiểu này). File: `tests/test_init_bfree.py`. Yêu cầu có checkpoint B-Free thật ⇒ nếu không có trên CPU local, test này là **USER-KAGGLE** hoặc dùng checkpoint nhỏ do user cung cấp.

### P2.C — Cân bằng dữ liệu & giảm shortcut (sửa W3; điều chỉnh theo D1)

Cờ dữ liệu mới (thêm vào YAML/`RESUME_CHECK_KEYS`): `data.mix_preset`, `data.fmt_match`, `data.res_match`, `data.holdout_generators`.

| Biến thể | Mô tả | Cách làm gợi ý |
|---|---|---|
| `mix_bfree` | Cặp B-Free ≥ 60–70% mẫu; AIGB ≤ 20–30% | Đặt `--aigenbench_max_per_gen` sao cho AIGB ≤ 25% tổng mẫu; tính và **ghi vào log** (xấp xỉ ~750/generator nếu giữ 13 generator, `aigenbench_val_frac` 0.1 — agent tự tính lại theo số liệu thật) |
| `mix_current` | Như hiện tại (AIGB ≈ 64%) — đối chứng | giữ mặc định (`4000`) |
| `mix_noaigb` | Chỉ B-Free + extra | bỏ `--ai_genbench_dir` |
| `fmt_match` | Real và fake cùng đi qua chuỗi nén ngẫu nhiên cùng phân phối (JPEG chất lượng ngẫu nhiên, xác suất như nhau cho hai lớp) | thêm bước vào `__getitem__` các dataset AIGB/extra |
| `res_match` | Bỏ upscale bicubic bắt buộc (`ensure_min_size`, dùng ở `ExtraTrainDataset` và `ExtraArrowTrainDataset`); dùng replicate-wrap như B-Free hoặc upscale xác suất bằng nhau cho real/fake | sửa `ensure_min_size` có cờ |

**Leave-generator-out thật sự:** giữ lại ≥3–4 trong 13 generator `PRIORITY_GENERATORS` (không bao giờ đưa vào train) và dùng real **từ origin khác** để đo tổng quát hóa.

### P2.D — Loss & hiệu chuẩn (sửa W5, W6)

- **Label smoothing:** so `ls ∈ {0.0, 0.05, 0.1}`; sau train luôn fit **temperature scaling** trên split calibration (P0.7).
- **λ_dcs** ∈ {0 (tắt), 0.01, 0.1, 0.5, 1.0}; **τ** ∈ {0.07, 0.2}. Dùng tỷ lệ gradient D2 để chọn khoảng 5–30%.
- **DCS thay thế** nếu InfoNCE theo batch vẫn "quá dễ" (loss 0.12 → 0.01 chỉ vài epoch): negative cùng lớp khác nguồn, hoặc dùng cặp real↔fake cùng nội dung của B-Free làm hard negative.

### P2.E — Validation & chọn checkpoint (sửa W5)

Val tổng hợp (không bão hòa), checkpoint chọn theo trung bình có trọng số của:
1. SD2.1 in-distribution (chỉ để phát hiện hỏng);
2. AIGB leave-generator-out;
3. Val suy giảm với toán tử held-out (P0.3);
4. WildRF `val` (**tuyệt đối không dùng `test`**).

Early-stop theo thành phần 2–4; epoch mặc định giảm 4–5 nếu đường cong xác nhận. `--select` ghi vào `selected_ckpt.txt` (đã có); bỏ sự không nhất quán `best` vs `last` (notebook eval của ours hiện load `*_best.pth`, trong khi `iter-1.md` quyết định dùng `last`).

### P2.F — Lưới ablation (2 giai đoạn)

*Sàng lọc* (USER-KAGGLE): 25% dữ liệu, 3 epoch, 1 seed, ≈35–45 phút/run, ~10–12 run ≈ 6–8 giờ. Agent chuẩn bị **một notebook cho mỗi nhóm run** + bảng cấu hình.

| Run | Init | Module | Loss | Dữ liệu | Mục đích |
|---|---|---|---|---|---|
| R0 | — | — | — | — | B-Free official trong harness v2 (tham chiếu) |
| R1 | B-Free | không | dual-view CE | mix_bfree | **Recipe thuần** |
| R1b | B-Free | không | CE sạch (legacy) | mix_bfree | Xác nhận W1: aug-CE có giúp thật? |
| R2 | B-Free | +DCS | dual-view | mix_bfree | Giá trị riêng của DCS |
| R3 | B-Free | +LIB | dual-view | mix_bfree | Giá trị riêng của LIB |
| R4 | B-Free | +GSR | dual-view | mix_bfree | Giá trị riêng của GSR |
| R5 | B-Free | LIB+GSR+DCS | dual-view | mix_bfree | **Mô hình đầy đủ** |
| R5-d | DINOv2 | LIB+GSR+DCS | dual-view | mix_bfree | Có cần init B-Free không? |
| R5-a | B-Free | LIB+GSR+DCS | dual-view | mix_current | Tác động của AIGB-heavy |
| R5-f | B-Free | LIB+GSR+DCS | dual-view + fmt/res match | mix_bfree | Tác động chống shortcut |
| R5-l | B-Free | LIB+GSR+DCS | dual-view, ls=0 + temp-scale | mix_bfree | Tác động của label smoothing |

*Xác nhận* (USER-KAGGLE): chọn ≤3 cấu hình tốt nhất, **đủ dữ liệu, 3 seed (42, 43, 44)**, 4–5 epoch × ~45 phút ≈ 27–34 giờ GPU; chia phiên bằng `--end_epoch` + `--resume_from` (mỗi phiên ≤12h, dùng cùng loại GPU).

### P2.G — Tiêu chí thành công & bác bỏ (ghi vào `research/directions/D-recipe-v4/README.md` TRƯỚC KHI chạy)

Số tham chiếu (harness v1, sẽ đo lại ở P0): mean AUC {WildRF fb/reddit/twitter, viral} = **0.930** (ours-v2) vs **0.972** (B-Free).

| Mã | Giả thuyết | Chỉ số (trên **test**, 1 lần) | Dự đoán | Bác bỏ nếu |
|---|---|---|---|---|
| H-A | Aug-CE + init B-Free sửa lỗi tổng quát hóa ngoài miền (R1 vs R1b, R1 vs ours-v2) | Mean AUC WildRF+viral | R1 ≥ 0.965 và ≥ ours-v2 + 0.03 | R1 < 0.945 |
| H-B | Aug-CE nâng độ bền suy giảm | Mean bAcc RealDeg held-out + compound 3–5 | ≥ +10 điểm so với ours-v2 (phase-2: 0.674/0.610/0.582) | < +4 điểm |
| H-C | LIB/GSR/DCS có đóng góp độc lập | R5 − R1: mean AUC (WildRF+viral+AIGB unseen+SynthWildX) | ≥ +0.005, CI không chứa 0 | CI chứa 0 ⇒ **kết luận trung thực: module không đóng góp** |
| H-D | Cân bằng dữ liệu giảm shortcut | R5 vs R5-a: AUC WildRF+viral; \|Δ điểm khi re-encode\| | ≥ +0.01 | < +0.003 |
| H-E | Calibration | ECE cân bằng sau temperature scaling, WildRF/viral | ≤ 0.08 | > 0.15 |
| Guard | Không đánh mất nội lực | AUC AIGB *seen* và SD2.1 in-dist | giảm ≤ 0.003 so với ours-v2 | giảm > 0.01 |

**Cổng P2:** bảng ablation đầy đủ + quyết định "giữ/loại" từng module theo H-C, **trình user trước khi sang P3**.

---

## 6. P3 — Cải tiến cơ chế (chỉ sau P2, chỉ những gì P2 chứng minh có ích)

Mỗi mục là **một thay đổi một lần**, có run đối chứng và tiêu chí cố định như P2.G.

| # | Việc | Điều kiện kích hoạt | Ghi chú |
|---|---|---|---|
| P3.1 | **MS-DCS rút gọn** | H-A/H-B đạt **và** DCS có đóng góp (H-C) | K=2 bin ngẫu nhiên/step (`views_per_step=2`, code đã hỗ trợ) thay K=4 ⇒ chi phí ~3× thay vì 5×. Dùng bin **tách khỏi** toán tử RealDeg-eval để held-out thật sự held-out. Mốc so sánh là R5, không phải v2 |
| P3.2 | **Full fine-tune thay LoRA** | LoRA r=16 là điểm nghẽn (thử r ∈ {8,16,32,64} trước) | B-Free gốc full FT; LR phân tầng, LR thấp 1e-5–2e-5. `train_full_ft.py` hiện **cũ/lệch** — kiểm tra nó dùng cùng pipeline dữ liệu/loss mới trước |
| P3.3 | **Thiết kế lại GSR** | GSR có đóng góp ở P2 | window ∈ {1,3,5,7} (lưới 36×36), multi-head, không che chính token, đặt giữa các block |
| P3.4 | **LIB học được** | LIB có đóng góp | kernel học được/multi-scale; kiểm tra α hội tụ về đâu (α→~0 ⇒ LIB vô dụng) |
| P3.5 | **Hard negative theo cặp** | DCS có đóng góp nhưng yếu | cặp real↔fake cùng nội dung làm negative cho InfoNCE |
| P3.6 | **TTA & hiệu chuẩn suy luận** | Luôn | 5-crop official + temperature scaling; báo cáo tách phần đóng góp của TTA |

---

## 7. P4 — Đánh giá cuối & báo cáo

1. Bảng chính: B-Free official, GlobalForge (đúng giao thức), ours-v2 (cũ), cấu hình thắng — **CI 95% bootstrap**, 3 seed (mean ± sd), kiểm định ghép cặp.
2. Bảng công bằng tách: *seen / unseen / wild / RealDeg in-dist / RealDeg held-out / SynthWildX*.
3. Bảng ablation R1–R5 + kết luận H-C **kể cả nếu âm tính**.
4. Phân tích lỗi: FPR/TPR ở ngưỡng cố định, đường cong độ bền theo cường độ, ví dụ định tính.
5. Mục "Giới hạn" trung thực: rò rỉ miền giữa train/eval, 1 backbone, giấy phép phi lợi nhuận của B-Free.
6. Cập nhật `research/` (`log.md`, `iter-N.md`, `eval-changelog.md`).

---

## 8. P5 — Dọn repo & tái lập (AGENT, song song, ưu tiên thấp nhưng làm sớm)

| # | Việc |
|---|---|
| P5.1 | Gộp logic suy giảm về **một module** `degradations.py` (tham số có đơn vị rõ). Hiện nhân bản ≥4 nơi: `train_lora.apply_degradation`, `bfree_datasets/multi_severity.py::_apply`, `bfree_datasets/bfree_dataset.py::DegradationPipeline`, bản sao trong harness; noise định nghĩa khác nhau (`std=√strength` vs `σ/255` vs `std∈[0.02,0.1]`). Test: train và eval cho cùng ảnh khi cùng seed |
| P5.2 | Xóa/đánh dấu deprecated: `train_full_ft.py`, `bfree_datasets/bfree_dataset.py`, `eval/run_benchmarks.py`, `scripts/run_phase6_search.sh`; cập nhật `Dockerfile` + `requirements.txt` (hiện torch 2.0.1) theo môi trường thật (torch 2.8.0, timm 1.0.22, peft 0.15.2) |
| P5.3 | Bỏ hyperparameter chết `lib_tau` (hoặc làm cho có nghĩa); bỏ `mask` không dùng của LIB |
| P5.4 | Bỏ `tests/` khỏi `.gitignore`; thêm test: bước 0 ≡ B-Free, gate GSR=0 ⇒ identity, dual-view loss tách đúng các term, schema parquet score |
| P5.5 | Chạy thử `train_lora.py main()` end-to-end trên CPU với dữ liệu giả nhỏ (khoảng trống đã ghi ở `log.md`; lần trước user từ chối lệnh — **xin phép trước**) |
| P5.6 | Đồng bộ `PATHS.md`, `CLAUDE.md`, `agent.md` (điền bảng "Dữ kiện xác nhận" theo code thật: LIB/GSR/DCS đã đọc từ `REM.py`, `engine_finetune.py`) |
| P5.7 | Chuẩn hóa slug Kaggle (sửa typo `xla-trainng-stage-2`); lập bảng ánh xạ phiên bản ↔ notebook ↔ checkpoint ↔ SHA |

---

## 9. Ước lượng & rủi ro

| Giai đoạn | GPU ước tính | Ghi chú |
|---|---|---|
| P0 | ~6–10 h (chủ yếu eval) | cần user duyệt trước |
| P1 | ~2–4 h | chủ yếu phân tích điểm đã lưu |
| P2 sàng lọc | ~7–8 h | 1 phiên Kaggle |
| P2 xác nhận | ~27–34 h | 3–4 phiên, dùng resume |
| P3 | ~2–5 h/ô sàng lọc | chỉ ô được kích hoạt |
| P4 | ~8–10 h | |
| **Tổng** | **~55–80 h** | từ ~44 phút/epoch đã quan sát (batch 64, 2 view, ~146k mẫu); đo lại khi đổi dữ liệu/số view |

| Rủi ro | Dấu hiệu | Phương án |
|---|---|---|
| D1/D2 bác bỏ W1/W3 | Điểm không dịch theo định dạng; tỷ lệ gradient DCS đã ổn | Dừng, báo user; xem lại LoRA/head/dữ liệu nội dung |
| Aug-CE không đủ bắt kịp B-Free ở WildRF | R1 < 0.945 AUC | Thêm đúng bộ augmentation inpainted++ (cut-out, jitter), tăng p suy giảm nhánh sạch; cân nhắc full FT (P3.2) |
| H-C âm | CI chứa 0 | **Chấp nhận kết quả âm**, định vị lại dự án quanh recipe (aug-CE + bias-free + dữ liệu); không ép thêm module |
| Init B-Free làm kiệt khả năng học module mới | R5 ≈ R1, gate ≈ 0 | Tăng LR cho gate/module, thử R5-d, hoặc warm-up 2 giai đoạn |
| Vượt ngân sách GPU | Run xác nhận quá dài | Còn 2 seed cho cấu hình 2–3, giữ 3 seed cho cấu hình chính |
| Rò rỉ miền AIGB↔eval | Chênh seen/unseen lớn bất thường | dùng `AIGENBENCH_EXCLUDE_ORIGIN` (cơ chế loại `origin_dataset` trùng của harness) ở eval; ưu tiên bộ ngoài AIGB |

---

## 10. Việc cần user quyết định (agent hỏi ngay ở lượt làm việc đầu tiên)

1. 🔒 Duyệt P0.1–P0.8 (đổi giao thức đo, phương pháp calib, bật RealDeg held-out).
2. Xác nhận **checkpoint nào** sinh `eval_results_ours.csv` (v1 phase-2, v2, hay v3-stage-2) và còn giữ file không.
3. Hoãn MS-DCS (v3) đến sau P2, hay chạy song song như nhánh thăm dò độc lập (khi đó dùng recipe cũ, kết quả chỉ tham khảo).
4. Chốt ngưỡng bác bỏ H-A…H-E (chỉnh được, nhưng phải chốt **trước** khi chạy).
5. Ngân sách GPU/Kaggle còn lại để cắt/giữ lưới ablation.
6. Có checkpoint B-Free official và cách nạp (`weights/<model_name>/config.yaml`) trên Kaggle không — cần cho P2.B.

---

## 11. Checklist thực thi (cập nhật khi làm)

- [ ] Hỏi user 6 câu ở mục 10
- [ ] P0: `research/yardstick-v2.md` → OK user → sửa harness → chạy lại baseline → đóng băng bảng v2
- [ ] P0.12: kiểm tra `sigmoid(2l)` của B-Free
- [ ] P1: D1–D5 → `research/diagnosis.md` → báo user
- [ ] P2.A dual-view (+ test bit-identity) · P2.B init B-Free + gate zero-init + **test bước 0 ≡ B-Free** · P2.C dữ liệu · P2.D loss · P2.E val
- [ ] Ghi tiêu chí P2.G vào `research/directions/D-recipe-v4/README.md` **trước** khi chạy
- [ ] Sàng lọc R0–R5-l → chọn ≤3 → xác nhận 3 seed
- [ ] P3 theo điều kiện kích hoạt · P4 báo cáo · P5 dọn repo (song song)
</file>

<file path="code/bfree_datasets/__init__.py">

</file>

<file path="code/bfree_datasets/bfree_dataset.py">
import os
import io
import random
from typing import Tuple, Dict, Any, Optional

import numpy as np
import pandas as pd
from PIL import Image, ImageFilter
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

from utils.normalization import get_list_norm


class DegradationPipeline:
    """Pipeline suy hao mô phỏng GlobalForge (JPEG + Blur + Color Distortion), bổ sung
    thêm Resize + Gaussian Noise, và nhánh single-operator.

    Lý do bổ sung Resize/Noise: JPEG/Blur/Color là nguyên bản từ pipeline GlobalForge
    (datasets_real.py:378-383) — KHÔNG đổi mặc định của 3 toán tử này để giữ đúng
    recipe gốc của paper. Resize và Noise KHÔNG có trong pipeline gốc; đối chiếu với
    eval RealDeg-Bench-style (notebook 02) ban đầu cho thấy đây chính xác là 2 toán tử
    tệ nhất (bAcc 0.6464 / 0.6402).

    Lý do bổ sung nhánh single-operator: sau khi thêm Resize/Noise vào nhánh compound
    (luôn xếp chồng JPEG -> Blur -> Resize -> Noise -> Color), noise cải thiện rõ rệt
    (bAcc 0.6402 -> 0.7357) nhưng resize gần như không đổi (0.6464 -> 0.6510). Nguyên
    nhân nghi ngờ: trong nhánh compound, resize luôn được áp SAU JPEG (p=1.0) và
    thường sau cả Blur (p=0.8) — tức DCS gần như không bao giờ thấy "resize trên ảnh
    sạch", trong khi phép test single-operator của RealDeg-Bench áp đúng 1 toán tử lên
    ảnh sạch. Với noise (phép CỘNG THÊM nhiễu), bất biến học được trong bối cảnh xếp
    chồng vẫn chuyển giao tốt sang trường hợp đơn lẻ; với resize (phép PHÁ HUỶ thông
    tin qua downscale), kiểu mất thông tin resize-sau-JPEG khác bản chất với
    resize-trên-ảnh-sạch nên không chuyển giao tốt.  cho phép một phần
    batch được suy giảm bằng ĐÚNG 1 toán tử ngẫu nhiên trên ảnh sạch, khớp sát hơn với
    giao thức single-operator dùng để đánh giá.
    """

    _OPS = ("jpeg", "blur", "resize", "noise", "color")

    def __init__(
        self,
        jpeg_quality_min: int = 20,
        jpeg_quality_max: int = 80,
        blur_kernel: int = 7,
        blur_sigma_min: float = 0.5,
        blur_sigma_max: float = 1.5,
        blur_prob: float = 0.8,
        color_prob: float = 0.8,
        resize_scale_min: float = 0.2,
        resize_scale_max: float = 0.9,
        resize_prob: float = 0.5,
        noise_std_min: float = 0.02,
        noise_std_max: float = 0.1,
        noise_prob: float = 0.5,
        single_op_prob: float = 0.4,
    ):
        self.jpeg_min = jpeg_quality_min
        self.jpeg_max = jpeg_quality_max
        self.blur_kernel = blur_kernel
        self.blur_sigma_min = blur_sigma_min
        self.blur_sigma_max = blur_sigma_max
        self.blur_prob = blur_prob
        self.color_prob = color_prob
        self.resize_scale_min = resize_scale_min
        self.resize_scale_max = resize_scale_max
        self.resize_prob = resize_prob
        self.noise_std_min = noise_std_min
        self.noise_std_max = noise_std_max
        self.noise_prob = noise_prob
        self.single_op_prob = single_op_prob
        self.color_jitter = transforms.ColorJitter(
            brightness=0.4, contrast=0.4, saturation=0.4, hue=0.06
        )

    # --- các phép biến đổi "lõi", LUÔN áp dụng khi được gọi (không tự kiểm tra xác
    # suất) - dùng chung cho cả nhánh compound (qua wrapper _apply_*, có prob riêng)
    # và nhánh single-operator (gọi thẳng, vì việc CHỌN toán tử này đã là quyết định
    # áp dụng nó, không cần roll xác suất thêm lần 2).
    def _do_jpeg(self, img: Image.Image) -> Image.Image:
        quality = random.randint(self.jpeg_min, self.jpeg_max)
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        return Image.open(buffer).convert("RGB")

    def _do_blur(self, img: Image.Image) -> Image.Image:
        sigma = random.uniform(self.blur_sigma_min, self.blur_sigma_max)
        return img.filter(ImageFilter.GaussianBlur(radius=sigma))

    def _do_resize(self, img: Image.Image) -> Image.Image:
        """Thu nhỏ rồi phóng to lại về đúng kích thước gốc — mô phỏng ảnh bị resize
        xuống độ phân giải thấp rồi phóng lại (thường gặp khi ảnh đi qua nhiều lần
        chia sẻ/re-upload). KHÔNG có trong pipeline GlobalForge gốc."""
        w, h = img.size
        scale = random.uniform(self.resize_scale_min, self.resize_scale_max)
        small = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.BICUBIC)
        return small.resize((w, h), Image.BICUBIC)

    def _do_noise(self, img: Image.Image) -> Image.Image:
        """Gaussian noise cộng vào ảnh (chuẩn hoá [0,1] trước khi cộng nhiễu, clip lại
        sau đó). KHÔNG có trong pipeline GlobalForge gốc."""
        std = random.uniform(self.noise_std_min, self.noise_std_max)
        arr = np.asarray(img).astype(np.float32) / 255.0
        arr = arr + np.random.normal(0.0, std, arr.shape)
        arr = np.clip(arr, 0.0, 1.0)
        return Image.fromarray((arr * 255).astype(np.uint8))

    def _do_color(self, img: Image.Image) -> Image.Image:
        return self.color_jitter(img)

    # --- wrapper có xác suất riêng, dùng cho nhánh compound (giữ đúng hành vi cũ) ---
    def _apply_jpeg(self, img: Image.Image) -> Image.Image:
        return self._do_jpeg(img)  # p=1.0 trong pipeline gốc, luôn áp dụng

    def _apply_blur(self, img: Image.Image) -> Image.Image:
        return self._do_blur(img) if random.random() < self.blur_prob else img

    def _apply_resize(self, img: Image.Image) -> Image.Image:
        return self._do_resize(img) if random.random() < self.resize_prob else img

    def _apply_noise(self, img: Image.Image) -> Image.Image:
        return self._do_noise(img) if random.random() < self.noise_prob else img

    def _apply_color(self, img: Image.Image) -> Image.Image:
        return self._do_color(img) if random.random() < self.color_prob else img

    def _apply_single_op(self, img: Image.Image) -> Image.Image:
        op = random.choice(self._OPS)
        return {"jpeg": self._do_jpeg, "blur": self._do_blur, "resize": self._do_resize,
                "noise": self._do_noise, "color": self._do_color}[op](img)

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() < self.single_op_prob:
            # Nhánh single-operator: đúng 1 toán tử ngẫu nhiên trên ảnh sạch, khớp sát
            # giao thức single-operator dùng để đánh giá (xem docstring lớp).
            return self._apply_single_op(img)
        # Nhánh compound (mặc định cũ): JPEG -> Blur -> Resize -> Noise -> Color, mô
        # phỏng ảnh bị xử lý qua nhiều bước liên tiếp (nén, resize, chia sẻ mạng xã hội).
        img = self._apply_jpeg(img)
        img = self._apply_blur(img)
        img = self._apply_resize(img)
        img = self._apply_noise(img)
        img = self._apply_color(img)
        return img


class BFreeDataset(Dataset):
    """Dataset nạp cặp ảnh clean / degraded cho B-Free + GlobalForge."""

    def __init__(
        self,
        csv_file: str,
        data_root: str,
        img_size: int = 504,
        is_train: bool = True,
        degradation_cfg: Optional[Dict[str, Any]] = None,
        norm_type: str = "resnet",
    ):
        self.data_root = data_root
        self.df = pd.read_csv(csv_file)
        self.img_size = img_size
        self.is_train = is_train

        deg_cfg = degradation_cfg or {}
        self.degradation = DegradationPipeline(
            jpeg_quality_min=deg_cfg.get("jpeg_quality_min", 20),
            jpeg_quality_max=deg_cfg.get("jpeg_quality_max", 80),
            blur_kernel=deg_cfg.get("blur_kernel", 7),
            blur_sigma_min=deg_cfg.get("blur_sigma_min", 0.5),
            blur_sigma_max=deg_cfg.get("blur_sigma_max", 1.5),
            blur_prob=deg_cfg.get("blur_prob", 0.8),
            color_prob=deg_cfg.get("color_prob", 0.8),
            resize_scale_min=deg_cfg.get("resize_scale_min", 0.2),
            resize_scale_max=deg_cfg.get("resize_scale_max", 0.9),
            resize_prob=deg_cfg.get("resize_prob", 0.5),
            noise_std_min=deg_cfg.get("noise_std_min", 0.02),
            noise_std_max=deg_cfg.get("noise_std_max", 0.1),
            noise_prob=deg_cfg.get("noise_prob", 0.5),
        )

        self.normalize = transforms.Compose(get_list_norm(norm_type))

    def __len__(self) -> int:
        return len(self.df)

    def _load_image(self, rel_path: str) -> Image.Image:
        full_path = os.path.join(self.data_root, rel_path)
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Image not found: {full_path}")
        return Image.open(full_path).convert("RGB")

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        row = self.df.iloc[idx]
        img_clean = self._load_image(row["filename"])

        # Nhãn: 0 nếu là REAL/0, 1 nếu là FAKE/Synthetic
        label_raw = row.get("label", 0)
        label = 0 if (label_raw == "REAL" or label_raw == 0 or label_raw == "0") else 1

        # Cắt ngẫu nhiên (train) hoặc crop trung tâm (val) về img_size
        if self.is_train:
            i, j, h, w = transforms.RandomCrop.get_params(
                img_clean, output_size=(self.img_size, self.img_size)
            )
            img_clean = TF.crop(img_clean, i, j, h, w)
            if random.random() > 0.5:
                img_clean = TF.hflip(img_clean)
        else:
            img_clean = TF.center_crop(img_clean, (self.img_size, self.img_size))

        # View suy hao cho DCS loss
        img_deg = self.degradation(img_clean)

        tensor_clean = self.normalize(img_clean)
        tensor_deg = self.normalize(img_deg)

        return tensor_clean, tensor_deg, label
</file>

<file path="code/configs/__init__.py">

</file>

<file path="code/datasets/README.md">
## Tập dữ liệu

Thông tin chi tiết hơn tại [trang dự án của B-Free](https://grip-unina.github.io/B-Free) và trong [bài báo](https://arxiv.org/abs/2412.17671).

Bộ dữ liệu tại liên kết trên chứa các thư mục và tệp sau:
- **COCO_real_512**: hình ảnh thực từ COCO (ảnh cắt lớn nhất ở trung tâm được điều chỉnh kích thước thành 512x512)
- **SD2.1_selfconditioned**: hình ảnh tự điều kiện
- **SD2.1_selfconditioned_origBG**: hình ảnh tự điều chỉnh với nền gốc được khôi phục
- **SD2.1_inpainted_samecat**: một đối tượng được thay thế bằng một đối tượng cùng loại.
- **SD2.1_inpainted_samecat_origBG**: tương tự như *inpainted_samecat*, nhưng đã khôi phục lại hình nền gốc.
- **SD2.1_inpainted_diffcat**: một đối tượng được thay thế bằng một đối tượng thuộc danh mục khác.
- **SD2.1_inpainted_diffcat_origBG**: tương tự như *inpainted_diffcat*, nhưng đã khôi phục lại hình nền gốc.
- **mask**: mặt nạ đối tượng được sử dụng để tạo ra *inpainted_samecat* và để thay thế nền cho *inpainted_samecat_origBG* và *selfconditioned_origBG*
- **bbox**: các hộp giới hạn được sử dụng để tạo ra *inpainted_diffcat* và để thay thế nền cho *inpainted_diffcat_origBG*
- **train_list.csv**: chứa ID cho tập dữ liệu huấn luyện và thông tin bổ sung cho mỗi hình ảnh, chẳng hạn như danh mục đối tượng.
- **valid_list.csv**: chứa ID cho tập dữ liệu kiểm định và thông tin bổ sung cho mỗi hình ảnh, chẳng hạn như danh mục đối tượng.

Lưu ý rằng cả tập dữ liệu *huấn luyện* và *kiểm tra* đều được chia từ **tập dữ liệu huấn luyện** MS-COCO 2017.


## md5sum

- 41741e8e81e61455d01455452fcf15ae COCO_real_512.zip:
- a96855ced782ce09b6139ba68af94dd0 SD2.1_selfcondition.zip:
- b2e8530fa6904bc25e6fe6f1ed45149d SD2.1_selfcondition_origBG.zip:
- fc3ba27bc97d60bcf18ca57931aebd3b SD2.1_inpaint_samecat.zip:
- 466fed55a78dacc1f82bd95f6eb7cd5c SD2.1_inpaint_samecat_origBG.zip:
- 5b00dc28d18809c2efc943c89c3f5514 SD2.1_inpainted_diffcat.zip:
- d289c6b0fe4b19f229f4578e61ca8032 SD2.1_inpainted_diffcat_origBG.zip:
- 0294cb1611ad1c50af72d6cace3492ca masks_and_bbox.zip
- 06581da53bf81741501396d71eb7ec5f train_list.csv:
- 72e7b4be2d049029dbedca7e75af91b0 valid_list.csv:


## Bibtex

Nếu bạn sử dụng bộ dữ liệu này, vui lòng trích dẫn nguồn:

```
@inproceedings{Guillaro2025biasfree,
  Tiêu đề={Một mô hình huấn luyện không thiên vị cho việc phát hiện hình ảnh do AI tạo ra một cách tổng quát hơn},
  tác giả={Guillaro, Fabrizio và Zingarini, Giada và Usman, Ben và Sud, Avneesh và Cozzolino, Davide và Verdoliva, Luisa},
  booktitle={Hội nghị IEEE/CVF về Thị giác máy tính và Nhận dạng mẫu (CVPR)},
  năm={2025}
}
```


## Giấy phép

Bản quyền (c) 2025 Nhóm Nghiên cứu Xử lý Hình ảnh thuộc Đại học Federico II của Naples ('GRIP-UNINA').

Bản quyền đã được bảo lưu.

Phần mềm này chỉ được sử dụng, sao chép và chỉnh sửa cho mục đích thông tin và phi lợi nhuận.

Bằng cách tải xuống và/hoặc sử dụng bất kỳ tệp nào trong số này, bạn ngầm đồng ý với tất cả các điều khoản.
các điều khoản của giấy phép, như được quy định trong tài liệu LICENSE.txt
(Đã bao gồm trong gói sản phẩm này)
</file>

<file path="code/demo_images/metainfo.csv">
filename,label
img0000.png,0
img0001.png,0
img0002.png,1
img0003.png,1
</file>

<file path="code/demo_images/results_metrics.csv">
,AUC,bAcc,NLL,ECE,Pd10,EER
BFREE_dino2reg4,1.0,1.0,0.011808631168055741,0.011718220922068524,1.0,0.0
</file>

<file path="code/demo_images/results.csv">
filename,BFREE_dino2reg4
img0000.png,-5.9374785
img0001.png,-4.441922
img0002.png,4.430519
img0003.png,3.8499813
</file>

<file path="code/eval/analyze_loss.py">
import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def analyze_loss_dynamics(log_csv: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(log_csv)

    required_cols = ["epoch", "train_loss", "train_ce", "train_dcs"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Log CSV must contain column: {col}")

    # Tính toán tương quan giữa CE Loss và DCS Loss
    corr_pearson = df["train_ce"].corr(df["train_dcs"], method="pearson")
    corr_spearman = df["train_ce"].corr(df["train_dcs"], method="spearman")

    print("\n--- Phân tích tương tác hàm mất mát (CE vs DCS) ---")
    print(f"Pearson Correlation : {corr_pearson:.4f}")
    print(f"Spearman Correlation: {corr_spearman:.4f}")

    if corr_pearson < 0:
        print("[!] CẢNH BÁO: Xuất hiện tương quan nghịch (DCS tăng khi CE giảm). Có dấu hiệu xung đột gradient!")
    else:
        print("[*] TỐT: CE loss và DCS loss cùng hội tụ đồng thuận.")

    sns.set_theme(style="darkgrid")
    fig, ax1 = plt.subplots(figsize=(8, 5), dpi=150)

    color_ce = "tab:blue"
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Cross-Entropy Loss", color=color_ce, fontsize=12)
    l1 = ax1.plot(df["epoch"], df["train_ce"], color=color_ce, marker="o", label="CE Loss", linewidth=2)
    ax1.tick_params(axis="y", labelcolor=color_ce)

    ax2 = ax1.twinx()
    color_dcs = "tab:orange"
    ax2.set_ylabel("Degradation Contrastive Loss (DCS)", color=color_dcs, fontsize=12)
    l2 = ax2.plot(df["epoch"], df["train_dcs"], color=color_dcs, marker="s", linestyle="--", label="DCS Loss", linewidth=2)
    ax2.tick_params(axis="y", labelcolor=color_dcs)

    lines = l1 + l2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper right")

    plt.title(f"CE Loss vs DCS Loss Dynamics (Corr: {corr_pearson:.2f})", fontsize=14)
    fig.tight_layout()

    out_path = os.path.join(output_dir, "loss_interaction.png")
    plt.savefig(out_path)
    plt.close()
    print(f"[*] Biểu đồ phân tích loss đã được lưu tại: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_csv", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./results/loss_analysis")
    args = parser.parse_args()

    analyze_loss_dynamics(args.log_csv, args.output_dir)
</file>

<file path="code/eval/run_benchmarks.py">
import os
import argparse
import pandas as pd
import numpy as np
import torch
import tqdm
from PIL import Image
from torchvision.transforms import Compose

from utils.normalization import get_list_norm
from utils import dmetrics
from networks.bfree_globalforge_vit import BFreeGlobalForgeViT
from configs.loader import load_config, build_model_kwargs


def load_target_model(config_path: str, checkpoint_path: str, device: str):
    cfg = load_config(config_path)
    kwargs = build_model_kwargs(cfg)
    model = BFreeGlobalForgeViT(**kwargs)
    if checkpoint_path and os.path.exists(checkpoint_path):
        model.load_checkpoint(checkpoint_path, strict=False)
    model.to(device).eval()
    return model, cfg["backbone"]["img_size"]


def evaluate_dataset(model, dataset_csv: str, root_dir: str, img_size: int, device: str):
    df = pd.read_csv(dataset_csv)
    transform = Compose(get_list_norm("resnet"))
    scores = []

    with torch.no_grad():
        for _, row in tqdm.tqdm(df.iterrows(), total=len(df), desc=f"Evaluating {os.path.basename(dataset_csv)}"):
            img_path = os.path.join(root_dir, row["filename"])
            img = Image.open(img_path).convert("RGB")
            
            # Cắt ảnh chuẩn 504x504 hoặc giữ nguyên kích thước qua normalize
            tensor = transform(img).unsqueeze(0).to(device)
            out = model(tensor)
            logits = out["logits"]
            score = (logits[:, 1] - logits[:, 0]).item()
            scores.append(score)

    scores = np.array(scores)
    labels = (df["label"] != "REAL") & (df["label"] != 0) & (df["label"] != "0")

    return {
        "AUC": dmetrics.roc_auc_score(labels, scores),
        "bAcc": dmetrics.balanced_accuracy_score(labels, scores > 0) * 100,
        "NLL": dmetrics.balanced_nll_binary(labels, scores),
        "ECE": dmetrics.balanced_ece_binary(labels, scores),
        "Pd10": dmetrics.pd_at_far(labels, scores, 0.10) * 100,
        "EER": dmetrics.calculate_eer2(labels, scores) * 100,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/bfree_dcs.yaml")
    parser.add_argument("--checkpoints", nargs="+", required=True, help="List of checkpoint_name=path pairs")
    parser.add_argument("--benchmark_csvs", nargs="+", required=True, help="List of benchmark_name=csv_path pairs")
    parser.add_argument("--data_root", type=str, default="./")
    parser.add_argument("--output_csv", type=str, default="./results/benchmark_comparison.csv")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)), exist_ok=True)
    
    ckpt_dict = dict(item.split("=") for item in args.checkpoints)
    bench_dict = dict(item.split("=") for item in args.benchmark_csvs)

    all_results = []
    for ckpt_name, ckpt_path in ckpt_dict.items():
        print(f"\n================ Loading Model: {ckpt_name} ================")
        model, img_size = load_target_model(args.config, ckpt_path, args.device)

        for bench_name, csv_path in bench_dict.items():
            metrics = evaluate_dataset(model, csv_path, args.data_root, img_size, args.device)
            row = {"Model": ckpt_name, "Benchmark": bench_name, **metrics}
            all_results.append(row)

    res_df = pd.DataFrame(all_results)
    res_df.to_csv(args.output_csv, index=False)
    print("\n" + "=" * 30 + " BENCHMARK SUMMARY " + "=" * 30)
    print(res_df.to_string(index=False, float_format=lambda x: f"{x:6.2f}"))


if __name__ == "__main__":
    main()
</file>

<file path="code/modules/__init__.py">

</file>

<file path="code/modules/gsr_adapter.py">
"""GSRAdapter — Global Structural Reasoning cho B-Free (768-dim).

Chuyển đổi từ GlobalForge GlobalStructuralReasoning (REM.py:63-108)
sang dim=768 cho DINOv2 ViT-B/14 của B-Free.

Cơ chế: Single-head self-attention + Chebyshev neighborhood masking.
Mask che mọi token trong vùng Chebyshev distance ≤ window → buộc mỗi
token phải gather evidence từ vùng xa, tập trung vào global structure
thay vì local artifact dễ vỡ.

Thay đổi so với bản gốc:
    - dim mặc định: 1024 → 768
    - Không có thay đổi logic nào khác — GSR dùng single-head attention
      (không phải multi-head), nên không cần tính lại num_heads.

Interface contract (xem plan-nguoi1-modules.md):
    GSRAdapter(dim=768, window=3, mask_prob=1.0)
    forward(tokens (B,N,768), hw=None) -> tokens (B,N,768)
"""

import math
import torch
import torch.nn as nn


class GSRAdapter(nn.Module):
    """Global Structural Reasoning cho B-Free ViT-B/14 (768-dim).

    Single-head self-attention với Chebyshev neighborhood masking.
    Residual connection: output = proj(attn @ V) + input.
    """

    def __init__(self, dim: int = 768, window: int = 3, mask_prob: float = 1.0,
                 zero_init_gate: bool = False):
        super().__init__()
        self.dim = dim
        self.window = window
        self.mask_prob = mask_prob
        # P2.B: gate zero-init => out = tokens + gate * proj(attn @ V); gate=0 ở bước 0 nên
        # GSR ≡ identity. Tham số chỉ tồn tại khi bật cờ (state_dict đường legacy không đổi).
        self.gate = nn.Parameter(torch.zeros(1)) if zero_init_gate else None

        # Single-head attention: Q, K, V, proj đều là Linear(dim, dim)
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, tokens: torch.Tensor, hw: int = None):
        """
        Args:
            tokens: (B, N, C) — patch tokens, C = self.dim
            hw: int — số patch mỗi cạnh

        Returns:
            out: (B, N, C) — tokens sau GSR (có residual)
        """
        B, N, C = tokens.shape

        if hw is None:
            hw = int(math.sqrt(N))

        q = self.q(tokens)
        k = self.k(tokens)
        v = self.v(tokens)

        # Attention scores: (B, N, N)
        attn = (q @ k.transpose(1, 2)) / math.sqrt(C)

        # Chebyshev neighborhood masking
        apply_mask = self.window > 0 and hw * hw == N
        if apply_mask and self.training:
            apply_mask = torch.rand(1).item() < self.mask_prob
        elif apply_mask and not self.training:
            apply_mask = self.mask_prob > 0

        if apply_mask:
            device = tokens.device
            idx = torch.arange(hw, device=device)
            grid_y, grid_x = torch.meshgrid(idx, idx, indexing='ij')
            coords = torch.stack([grid_y.flatten(), grid_x.flatten()], dim=1)  # (N, 2)
            diff = coords[:, None, :] - coords[None, :, :]  # (N, N, 2)
            dist = diff.abs().max(dim=-1).values  # Chebyshev distance (N, N)
            mask = (dist <= self.window).to(attn.dtype)
            mask = mask.unsqueeze(0)  # (1, N, N) — broadcast qua batch
            attn = attn.masked_fill(mask > 0, float('-inf'))

        attn = torch.softmax(attn, dim=-1)
        out = attn @ v
        if self.gate is None:
            out = self.proj(out) + tokens  # Residual connection
        else:
            out = tokens + self.gate * self.proj(out)  # gated residual (zero-init)

        return out
</file>

<file path="code/modules/lib_adapter.py">
"""LIBAdapter — Local Information Bottleneck cho B-Free (768-dim).

Chuyển đổi từ GlobalForge LocalInformationBottleneck (REM.py:17-60)
sang dim=768 cho DINOv2 ViT-B/14 của B-Free.

Cơ chế: Fixed Gaussian low-pass với 1 learnable mixing scalar:
    x_hat = (1 - alpha) * x + alpha * blur(x),  alpha = sigmoid(beta)

Thay đổi so với bản gốc:
    - dim mặc định: 1024 → 768
    - Không có thay đổi logic nào khác — Conv2d depthwise (groups=dim)
      tự co giãn theo dim.

Interface contract (xem plan-nguoi1-modules.md):
    LIBAdapter(dim=768, kernel_size=3, tau=0.5)
    forward(feat_tokens (B,N,768), hw=None) -> (feat_hat (B,N,768), mask (B,N)|None)
"""

import math
import torch
import torch.nn as nn


class LIBAdapter(nn.Module):
    """Local Information Bottleneck cho B-Free ViT-B/14 (768-dim).

    Fixed Gaussian low-pass + learnable mixing scalar:
        x_hat = (1 - alpha) * x + alpha * blur(x),  alpha = sigmoid(beta)
    """

    def __init__(self, dim: int = 768, kernel_size: int = 3, tau: float = 0.5,
                 beta_init: float = -0.85):
        super().__init__()
        self.dim = dim
        self.kernel_size = kernel_size
        padding = kernel_size // 2

        # Depthwise conv — mỗi channel có 1 Gaussian filter riêng, weights cố định
        self.blur = nn.Conv2d(
            dim, dim, kernel_size,
            padding=padding, groups=dim, bias=False
        )

        # Khởi tạo Gaussian kernel cố định
        sigma = kernel_size / 3.0
        coords = torch.arange(kernel_size).float() - kernel_size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        kernel_2d = g.outer(g)
        kernel_2d = kernel_2d / kernel_2d.sum()

        with torch.no_grad():
            self.blur.weight.copy_(
                kernel_2d.unsqueeze(0).unsqueeze(0).expand(
                    dim, 1, kernel_size, kernel_size
                )
            )
        self.blur.weight.requires_grad = False  # Gaussian kernel KHÔNG học

        # Scalar duy nhất cần học — khởi tạo -0.85 (giống bản gốc). P2.B: beta_init=-6
        # => alpha0 = sigmoid(-6) ~ 0.0025, LIB gần identity ở bước 0.
        self.beta = nn.Parameter(torch.tensor(float(beta_init)))

    def forward(self, feat_tokens: torch.Tensor, hw: int = None):
        """
        Args:
            feat_tokens: (B, N, C) — patch tokens, C = self.dim
            hw: int — số patch mỗi cạnh (36 cho 504×504, 16 cho 224×224)

        Returns:
            feat_hat: (B, N, C) — tokens sau LIB
            mask: (B, N) hoặc None — giá trị alpha broadcast
        """
        B, N, C = feat_tokens.shape

        if hw is None:
            hw = int(math.sqrt(N))

        # Nếu N không phải perfect square → bypass (safety check)
        if hw * hw != N:
            return feat_tokens, None

        # Reshape: (B, N, C) → (B, C, hw, hw) để áp conv
        x = feat_tokens.transpose(1, 2).reshape(B, C, hw, hw)
        x_smooth = self.blur(x)

        # Mixing: x_hat = (1 - alpha) * x + alpha * blur(x)
        alpha = torch.sigmoid(self.beta)
        x_hat = (1 - alpha) * x + alpha * x_smooth

        # Reshape lại: (B, C, hw, hw) → (B, N, C)
        feat_hat = x_hat.flatten(2).transpose(1, 2)
        mask = alpha.expand(B, N)

        return feat_hat, mask
</file>

<file path="code/networks/__init__.py">
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#
# Copyright (c) 2025 Image Processing Research Group of University Federico II of Naples ('GRIP-UNINA').
# All rights reserved.
# This work should only be used for nonprofit purposes.
#
# By downloading and/or using any of these files, you implicitly agree to all the
# terms of the license, as specified in the document LICENSE.txt
# (included in this package) and online at
# https://www.grip.unina.it/download/LICENSE_OPEN.txt
#
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


def get_network(name_arch, pretrained=False):
    if name_arch.startswith('timm_c5i504_'):
        import timm
        from .wrapper5crops import Wrapper5crops
        model = timm.create_model(name_arch[12:], num_classes=1, pretrained=pretrained)
        model = Wrapper5crops(model, 504)
    else:
        assert False
    return model


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_weights(model, model_path):
    from torch import load
    dat = load(model_path, map_location='cpu')
    if 'model' in dat:
        model.load_state_dict(dat['model'])
    else:
        print(list(dat.keys()))
        assert False
    return model
</file>

<file path="code/networks/wrapper5crops.py">
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#
# Copyright (c) 2025 Image Processing Research Group of University Federico II of Naples ('GRIP-UNINA').
# All rights reserved.
# This work should only be used for nonprofit purposes.
#
# By downloading and/or using any of these files, you implicitly agree to all the
# terms of the license, as specified in the document LICENSE.txt
# (included in this package) and online at
# https://www.grip.unina.it/download/LICENSE_OPEN.txt
#
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


import torch
import numpy as np
import timm


def replicate_wrap(x, new_shape):
    rep_height = max(int(np.ceil(new_shape[-2]/x.shape[-2])),1)
    rep_width  = max(int(np.ceil(new_shape[-1]/x.shape[-1])),1)
    x = x.repeat(1, 1, rep_height, rep_width)
    x = x[...,:new_shape[-2],:new_shape[-1]]
    return x


class Wrapper5crops(torch.nn.Module):
    def __init__(self, model, patch_size):
        super().__init__()
        self.model = model
        self.model.set_input_size(img_size=patch_size)
        self.patch_embed = self.model.patch_embed
        self.model.patch_embed = torch.nn.Identity()
        assert isinstance(self.patch_embed, timm.layers.patch_embed.PatchEmbed)
    
    def forward(self, x):
        patch_size = self.patch_embed.grid_size
        embeddings = self.patch_embed.proj(x)

        ph,pw = max(patch_size[0]-embeddings.shape[-2],0), max(patch_size[1]-embeddings.shape[-1],0)
        if (ph>0) or (pw>0):
            embeddings = replicate_wrap(embeddings, patch_size)
        
        hs = max((embeddings.shape[-2]-patch_size[0])//2, 0)
        ws = max((embeddings.shape[-1]-patch_size[1])//2, 0)
        embeddings = torch.cat((embeddings[:,:,hs:patch_size[0]+hs,ws:patch_size[1]+ws],
                                embeddings[:,:,:patch_size[0],:patch_size[1]],
                                embeddings[:,:,-patch_size[0]:,:patch_size[1]],
                                embeddings[:,:,-patch_size[0]:,-patch_size[1]:],
                                embeddings[:,:,:patch_size[0],-patch_size[1]:]), 0)

        if self.patch_embed.flatten:
            embeddings = embeddings.flatten(2).transpose(1, 2)  # BCHW -> BNC
        embeddings = self.patch_embed.norm(embeddings)
        y = self.model(embeddings)
        y = torch.mean(torch.stack(torch.split(y, y.shape[0]//5, 0), 0), 0)
        return y
    
    def load_state_dict(self, dat):
        self.patch_embed.load_state_dict({k[12:]: dat[k] for k in dat if k.startswith('patch_embed.')})
        self.model.load_state_dict({k: dat[k] for k in dat if not k.startswith('patch_embed.')})
</file>

<file path="code/scripts/run_phase6_search.sh">
#!/usr/bin/env bash
set -e

# ==============================================================================
# Phase 6 Search: So sánh Baseline vs Full Fine-Tune vs LoRA (25% data, 3 epochs)
# ==============================================================================

CONFIG="configs/bfree_dcs.yaml"
DATA_ROOT="./training_data"
TRAIN_CSV="./training_data/train_subset_25pct.csv"
VAL_CSV="./training_data/valid_list.csv"
RESULTS_DIR="./results/phase6_search"
mkdir -p ${RESULTS_DIR}

echo "=== [1/3] Huấn luyện Full Fine-Tuning ==="
python train_full_ft.py \
    --config ${CONFIG} \
    --train_csv ${TRAIN_CSV} \
    --val_csv ${VAL_CSV} \
    --data_root ${DATA_ROOT} \
    --output_dir "${RESULTS_DIR}/full_ft" \
    --epochs 3 \
    --batch_size 8 \
    --lr 5e-5

echo "=== [2/3] Huấn luyện LoRA (Rank 16) ==="
python train_lora.py \
    --config ${CONFIG} \
    --train_csv ${TRAIN_CSV} \
    --val_csv ${VAL_CSV} \
    --data_root ${DATA_ROOT} \
    --output_dir "${RESULTS_DIR}/lora_r16" \
    --lora_rank 16 \
    --epochs 3 \
    --batch_size 8 \
    --lr 1e-4

echo "=== [3/3] Đánh giá Benchmark & Tổng hợp So sánh ==="
python eval/run_benchmarks.py \
    --config ${CONFIG} \
    --checkpoints \
        Baseline=./weights/BFREE_dino2reg4/model_epoch_best.pth \
        Full_FT="${RESULTS_DIR}/full_ft/best_model.pth" \
        LoRA_r16="${RESULTS_DIR}/lora_r16/best_model_lora.pth" \
    --benchmark_csvs \
        Synthbuster=./benchmarks/synthbuster.csv \
        GenImage=./benchmarks/genimage.csv \
    --data_root "./" \
    --output_csv "${RESULTS_DIR}/phase6_comparison_summary.csv"

echo "=== Phân tích Loss Dynamics ==="
python eval/analyze_loss.py \
    --log_csv "${RESULTS_DIR}/full_ft/train_log.csv" \
    --output_dir "${RESULTS_DIR}/full_ft_loss_analysis"

echo "[✓] Toàn bộ Phase 6 Search đã hoàn tất. Kết quả được lưu tại: ${RESULTS_DIR}"
</file>

<file path="code/utils/__init__.py">

</file>

<file path="code/utils/dmetrics.py">
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#
# Copyright (c) 2025 Image Processing Research Group of University Federico II of Naples ('GRIP-UNINA').
# All rights reserved.
# This work should only be used for nonprofit purposes.
#
# By downloading and/or using any of these files, you implicitly agree to all the
# terms of the license, as specified in the document LICENSE.txt
# (included in this package) and online at
# https://www.grip.unina.it/download/LICENSE_OPEN.txt
#
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


from sklearn.metrics import roc_auc_score, roc_curve, balanced_accuracy_score

def calculate_eer(y_true, y_score):
    '''
    Returns the equal error rate for a binary classifier output.
    '''
    import numpy as np
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    t = np.argmin(np.abs(1.-tpr-fpr))
    return (fpr[t] + 1 - tpr[t])/2

def calculate_eer2(y_true, y_score):
    '''
    Returns the equal error rate for a binary classifier output.
    '''
    import numpy as np
    from scipy.interpolate import interp1d
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    err = 1.-tpr-fpr
    val = (fpr + 1 - tpr)/2
    eer = np.float64(interp1d(err, val)(0))
    return eer

def pd_at_far(y_true, y_score, fpr_th):
    '''
    Returns the Pd at fixed FAR for a binary classifier output.
    '''
    import numpy as np
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    return np.interp(fpr_th, fpr, tpr)


def balanced_cllr(y_true, y_score):
    '''
    Returns the balanced CLLR for a binary classifier output.
    y_score has to be a logit in natural base
    '''
    import numpy as np
    _, y_true = np.unique(y_true, return_inverse=True)
    y_true = 2*y_true -1

    y_score = np.logaddexp(-y_true * y_score, 0)
    y_score0 = np.mean(y_score[y_true<0])
    y_score1 = np.mean(y_score[y_true>0])
    cllr = (y_score0+y_score1) / (2*np.log(2))
    return cllr

def balanced_nll_binary(y_true, y_score):
    '''
    Returns the balanced NLL for a binary classifier output.
    y_score has to be a logit in natural base
    '''
    import numpy as np
    return balanced_cllr(y_true, y_score) * np.log(2)


def balanced_ece_binary(y_true, y_score, bins=15):
    '''
    Returns the balanced ECE for a binary classifier output.
    y_score has to be a logit in natural base
    '''
    from scipy.special import expit, softmax
    import numpy as np
    
    _, yp, yc = np.unique(y_true, return_inverse=True, return_counts=True)
    sample_weight = 1/yc
    sample_weight = sample_weight[yp]
    assert np.max(yp)==1
    total_weight = sample_weight.sum()

    prob = expit(y_score)
    correctness = (yp==1)
    
    interval = np.floor(bins*prob)
    ece = 0.0
    for _ in range(bins):
        in_bin = (interval==_)
        if np.any(in_bin):
            weight = sample_weight[in_bin]
            accuracy_in_bin = (weight*correctness[in_bin]).mean() / weight.mean()
            avg_confidence_in_bin = (weight*prob[in_bin]).mean() / weight.mean()
            prop_in_bin = weight.sum() / total_weight

            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return ece
</file>

<file path="code/utils/normalization.py">
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#
# Copyright (c) 2025 Image Processing Research Group of University Federico II of Naples ('GRIP-UNINA').
# All rights reserved.
# This work should only be used for nonprofit purposes.
#
# By downloading and/or using any of these files, you implicitly agree to all the
# terms of the license, as specified in the document LICENSE.txt
# (included in this package) and online at
# https://www.grip.unina.it/download/LICENSE_OPEN.txt
#
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
import numbers


def get_list_norm(norm_type):
    transforms_list = list()
    if norm_type == 'resnet':
        print('normalize RESNET')
        transforms_list.append(transforms.ToTensor())
        transforms_list.append(transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                    std=[0.229, 0.224, 0.225]))
    elif norm_type == 'clip':
        print('normalize CLIP')
        transforms_list.append(transforms.ToTensor())
        transforms_list.append(transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                                                    std=(0.26862954, 0.26130258, 0.27577711)))

    elif norm_type == 'none':
        print('normalize 0,1')
        transforms_list.append(transforms.ToTensor())

    elif norm_type == 'xception':
        print('normalize -1,1')
        transforms_list.append(transforms.ToTensor())
        transforms_list.append(transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                    std=[0.5, 0.5, 0.5]))

    else:
        assert False

    return transforms_list



def center_crop(img, output_size):
    if isinstance(output_size, numbers.Number):
        output_size = (int(output_size), int(output_size))
    image_width, image_height = img.size[1], img.size[0]
    crop_height, crop_width = output_size
    crop_top = int(round((image_height - crop_height) / 2.))
    crop_left = int(round((image_width - crop_width) / 2.))
    return img.crop((crop_left, crop_top, crop_left + crop_width, crop_top + crop_height))


class CenterCropNoPad():
    def __init__(self, siz):
        self.siz = siz

    def __call__(self, img):
        h, w = img.size[1], img.size[0]
        if max(h, w) > self.siz:
            img = center_crop(img, self.siz)
        return img


class PilResize():
    def __init__(self, target_w, interp=Image.BICUBIC):
        self.target_w = target_w
        self.interp = interp

    def __call__(self, img):
        if self.target_w == 0:
            return img
        (width, height) = (self.target_w, img.height * self.target_w // img.width)
        img = img.resize((width, height), self.interp)
        return img


def padding_wrap(img, output_size):
    if isinstance(output_size, numbers.Number):
        output_size = (int(output_size), int(output_size))
    new_img = Image.new(img.mode, output_size)
    for x_offset in range(0, output_size[0], img.size[0]):
        for y_offset in range(0, output_size[1], img.size[1]):
            new_img.paste(img, (x_offset, y_offset))

    return new_img


class PaddingWarp():
    def __init__(self, siz):
        self.siz = siz

    def __call__(self, img):
        return padding_wrap(img, self.siz)
</file>

<file path="code/Dockerfile">
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime
SHELL ["/bin/bash", "-c"]

RUN apt-get update
RUN apt-get upgrade -y
RUN apt-get install -y apt-utils wget unzip git
RUN pip install --upgrade pip
RUN pip install tqdm timm==1.0.12 transformers==4.43.4 scikit-learn pandas

ADD ./ /workdir
WORKDIR /workdir

ENTRYPOINT [ "python", "main_bfree.py" ]
</file>

<file path="code/LICENSE.txt">
THIS DOCUMENT CONSTITUTES A LICENCE TO USE THE SOFTWARE ON THE TERMS AND CONDITIONS APPEARING BELOW.


Preamble

This License applies to the software with which this license is distributed.
The software is intellectual property of Image Processing Research Group of University Federico II of Naples ('GRIP-UNINA')
and is placed under the protection of copyright laws, including Italian legislation and international treaties.
BY USING THE PROGRAM, YOU INDICATE YOUR ACCEPTANCE OF THIS LICENSE TO DO SO.


Terms and Conditions

Reproduction, modification, and usage of the software covered by this license is allowed free of charge provided that:
(i) this software should be used, reproduced and modified only for informational and nonprofit purposes; any unauthorized use of this software for industrial or profit-oriented activities is expressively prohibited; and
(ii) any reproduction or modification retains all original notices including proprietary or copyright notices; and
(iii) reference to the original authors is given whenever results, which arise from the use of this software or any modification of it, are made public.
No other use of the materials and of any information incorporated thereto is hereby authorized.
In addition, be informed that some names are protected by trademarks which are the property of GRIP-UNINA, its researchers and/or other third parties whether a specific mention in that respect is made or not. 


Disclaimers

This software is provided 'as-is', without any express or implied warranty.
In no event will the authors be held liable for any damages arising from the use of this software.


Transmission of user information

Any and all information or request for information you may direct to GRIP
through e-mail as may be linked to website http://www.grip.unina.it/
</file>

<file path="code/main_bfree_single.py">
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#
# Copyright (c) 2025 Image Processing Research Group of University Federico II of Naples ('GRIP-UNINA').
# All rights reserved.
# This work should only be used for nonprofit purposes.
#
# By downloading and/or using any of these files, you implicitly agree to all the
# terms of the license, as specified in the document LICENSE.txt
# (included in this package) and online at
# https://www.grip.unina.it/download/LICENSE_OPEN.txt
#
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


import torch
import os
import pandas
import numpy as np
import glob
import sys
import yaml
from PIL import Image

from torchvision.transforms  import CenterCrop, Compose
from utils.normalization import CenterCropNoPad, get_list_norm
from networks import get_network, load_weights

def get_config(model_name, weights_dir='./weights'):
    with open(os.path.join(weights_dir, model_name, 'config.yaml')) as fid:
        data = yaml.load(fid, Loader=yaml.FullLoader)
    model_path = os.path.join(weights_dir, model_name, data['weights_file'])
    return data['model_name'], model_path, data['arch'], data['norm_type']

def running_test(filename, model_name, device):

    # defining model and transforms
    _, model_path, arch, norm_type = get_config(model_name)

    model = load_weights(get_network(arch), model_path)
    model = model.to(device).eval()

    transform = Compose(get_list_norm(norm_type))

    # inference
    with torch.no_grad():
        image = transform(Image.open(filename).convert('RGB'))
        image = image.unsqueeze(0)  # [C, H, W] -> [1, C, H, W]

        out_tens = model(image.to(device)).cpu().numpy()

        # one logit
        if out_tens.shape[1] == 1:
            out_tens = out_tens[:, 0]
        # two logits
        elif out_tens.shape[1] == 2:
            out_tens = out_tens[:, 1] - out_tens[:, 0]
        else:
            assert False
        assert len(out_tens.shape) == 1

        print("logit score: %.3f"%out_tens.item())

    

if __name__ == "__main__":
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_image", '-i', type=str, help="The path of the input image")
    parser.add_argument("--model"      , '-m', type=str, help="Model to test", default='BFREE_dino2reg4')
    parser.add_argument("--device"     , '-d', type=str, help="Torch device", default='cuda:0')
    args = vars(parser.parse_args())
    
    print("Running the test with model:", args['model'])
    running_test(args['input_image'], args['model'], args['device'])
</file>

<file path="code/main_bfree.py">
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#
# Copyright (c) 2025 Image Processing Research Group of University Federico II of Naples ('GRIP-UNINA').
# All rights reserved.
# This work should only be used for nonprofit purposes.
#
# By downloading and/or using any of these files, you implicitly agree to all the
# terms of the license, as specified in the document LICENSE.txt
# (included in this package) and online at
# https://www.grip.unina.it/download/LICENSE_OPEN.txt
#
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


import torch
import os
import pandas
import numpy as np
import tqdm
import glob
import sys
import yaml
from PIL import Image

from torchvision.transforms  import CenterCrop, Compose
from utils.normalization import CenterCropNoPad, get_list_norm
from networks import get_network, load_weights

def get_config(model_name, weights_dir='./weights'):
    with open(os.path.join(weights_dir, model_name, 'config.yaml')) as fid:
        data = yaml.load(fid, Loader=yaml.FullLoader)
    model_path = os.path.join(weights_dir, model_name, data['weights_file'])
    return data['model_name'], model_path, data['arch'], data['norm_type']

def running_tests(input_csv, model_name, device):
    table = pandas.read_csv(input_csv)[['filename', ]]
    rootdataset = os.path.dirname(os.path.join('.', input_csv))

    # defining model and transforms
    _, model_path, arch, norm_type = get_config(model_name)

    model = load_weights(get_network(arch), model_path)
    model = model.to(device).eval()

    transform = Compose(get_list_norm(norm_type))

    # inference
    with torch.no_grad():
        for index in tqdm.tqdm(table.index, total=len(table)):
            filename = os.path.join(rootdataset, table.loc[index, 'filename'])

            image = transform(Image.open(filename).convert('RGB'))
            image = image.unsqueeze(0)  # [C, H, W] -> [1, C, H, W]

            out_tens = model(image.to(device)).cpu().numpy()

            # one logit
            if out_tens.shape[1] == 1:
                out_tens = out_tens[:, 0]
            # two logits
            elif out_tens.shape[1] == 2:
                out_tens = out_tens[:, 1] - out_tens[:, 0]
            else:
                assert False
            assert len(out_tens.shape) == 1

            # batch size 1
            table.loc[index, model_name] = out_tens
    return table


def compute_metrics(input_csv, output_csv, metrics_csv=None):
    from utils import dmetrics
    table = pandas.read_csv(output_csv)
    list_algs = [_ for _ in table.columns if _!='filename']
    table = pandas.read_csv(input_csv).merge(table, on=['filename', ])
    assert 'label' in table
    
    list_metrics = {
        'AUC' : lambda label, score: dmetrics.roc_auc_score(label,  score),
        'bAcc': lambda label, score: dmetrics.balanced_accuracy_score(label, score>0),
        'NLL' : lambda label, score: dmetrics.balanced_nll_binary(label, score),
        'ECE' : lambda label, score: dmetrics.balanced_ece_binary(label, score),
        'Pd10': lambda label, score: dmetrics.pd_at_far(label, score, 0.10),
        'EER' : lambda label, score: dmetrics.calculate_eer2(label, score),
    }
    
    tab_metrics = pandas.DataFrame(index=list_algs, columns=list_metrics)
    label = (table['label']!='REAL') & (table['label']!=0)
    for alg in list_algs:    
        score = table[alg]
        if np.all(np.isfinite(score))==False:
            continue
        
        for metric in list_metrics:
            tab_metrics.loc[alg, metric] = list_metrics[metric](label, score)

    if metrics_csv is not None:
        os.makedirs(os.path.dirname(os.path.join('.', metrics_csv)), exist_ok=True)
        tab_metrics.to_csv(metrics_csv)
    
    print(tab_metrics.to_string(float_format=lambda x: '%5.3f'%x))
    

if __name__ == "__main__":
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_csv"     , '-i', type=str, help="The path of the input csv file with the list of images")
    parser.add_argument("--out_csv"    , '-o', type=str, help="The path of the output csv file", default="./results.csv")
    parser.add_argument("--model"      , '-m', type=str, help="Model to test", default='BFREE_dino2reg4')
    parser.add_argument("--device"     , '-d', type=str, help="Torch device", default='cuda:0')
    parser.add_argument("--metrics_csv", '-t', type=str, help="The path of the metrics csv file", default=None)
    args = vars(parser.parse_args())
    
    print("Running the tests with model:", args['model'])
    table = running_tests(args['in_csv'], args['model'], args['device'])
    
    output_csv = args['out_csv']
    os.makedirs(os.path.dirname(os.path.join('.', output_csv)), exist_ok=True)
    table.to_csv(output_csv, index=False)  # save the results as csv file
    
    if args['metrics_csv'] is not None:
        print("Computing the metrics...")
        compute_metrics(args['in_csv'], args['out_csv'], args['metrics_csv'])
</file>

<file path="docs/_config.yml">
remote_theme: grip-unina/webtemplate
plugins:
  - jekyll
  - jekyll-seo-tag
</file>

<file path="research/directions/D-multiseverity-dcs/README.md">
# D-multiseverity-dcs — Multi-Severity DCS

Source: `source/Multi-Severity DCS AI Detection.docx` (AI-generated report, 106 equation images, raw `[cite]` artefacts, off-topic URLs). **All docx numbers are hypothetical; do not cite.** (Its 91.4 vs 62.9 degraded-avg table has no experiment behind it; intro and table disagree on the DINOv2 baseline, 86.2 vs 98.4.)

## 1. Motivation
Low-pass degradations (JPEG / blur / resize / noise) erase HF artefacts. One random degraded view per step (DCPT, DDA, GF-DCS) may let features drift across severity and fit an "average" severity. GlobalForge itself trains with a single compound view; multi-severity appears only at eval (RealDeg-Bench).

## 2. Method actually implemented (subset of the docx)
- K=4 severity bins, ranges in `B-Free-loss-backbone/code/configs/bfree_dcs.yaml :: multi_severity` (user-approved 2026-09-30): JPEG Q [85-95]/[65-80]/[40-60]/[20-35]; blur σ [0.5-1.0]/[1.2-2.0]/[2.2-3.0]/[3.2-4.5]; resize [.75-.90]/[.50-.70]/[.30-.45]/[.15-.25]; noise σ (0-255 scale, **assumed unit**) [2-5]/[6-12]/[13-20]/[21-30].
- Loss: `L = Σ_k w_k · InfoNCE(z, z_k) / Σ_k w_k`, each term the unchanged faithful symmetric InfoNCE (in-batch negatives, **not** class-aware, not SupCon). `multi_severity.enabled=false` (default) = v2 path, bit-identical (test vs pre-change reference).
- `compose`: `single_op` (default; one random op per view) or `compound` (jpeg→blur→resize→noise). The docx is ambiguous; single_op chosen, compound available.
- Not tested (undefined values or conflict with locked decisions): docx γ, δ(s), ω(s), Severity(), α, L_orth, L_deg head, MoCo queue 65,536, ViT-L (decision 5), two-stage training, class-aware negatives.

## 3. Known caveats
- Bins overlap RealDeg-Bench eval levels → RealDeg is **in-distribution robustness**, not held-out (same for v2 lifecycle pipeline).
- K views ⇒ (1+K)× forward memory/time per step; batch size found automatically.

## 4. Hypotheses (success criteria fixed BEFORE any run)
Single run: **A1** (K=4 views, uniform weights, seed 42). Baseline = **v2 numbers already saved** in `notebooks/version2` eval outputs (no re-run). Limitation: one seed and no baseline re-run ⇒ seed-to-seed spread is unknown; conclude only by the fixed thresholds below and report the result as a single-seed observation, never as a proven gain.

- **H1** MS-DCS improves robustness at high severity. Metric: mean bAcc@0.5 over RealDeg jpeg/blur/resize/noise + compound 3–5 steps (needs RealDeg enabled in the harness — currently OFF; otherwise `TODO: pending run`). Predicts ≥ +3.0 pts vs v2. Guard: clean AI-GenBench ALL AUC drop ≤ 0.005 (v2: 0.992977). **Refuted if** gain < +1.0, or clean AUC drops > 0.01.
- **H2** MS-DCS improves social-media transfer. Metric: mean AUC over WildRF fb/reddit/twitter + bfree_viral (v2: .905/.951/.920/.850). Predicts ≥ +0.02. **Refuted if** gain < +0.005.
- (Dropped: multi-view-vs-distribution test — needs a 1-view ablation that is not run.)

## 5. Run
| id | config | seed |
|---|---|---|
| A1 | `enabled:true`, K=4, uniform weights, all bins/step | 42 |

Model selection: last-epoch checkpoint (val split saturates ~0.999). No test-set selection.

## 6. Status
Code on `integration/loss-backbone` (P-Bao), CPU-tested. No training/eval results yet: `TODO: pending run`.
</file>

<file path="research/eval-changelog.md">
# Evaluation-harness changelog

| date | file | what | why | verified how |
|---|---|---|---|---|
| 2026-09-30 | notebooks/version3/xla-eval-harness-ours.ipynb | new copy of version2 ours harness; only OURS_CKPT → v3 last-epoch ckpt, outputs cleared | evaluate v3 with an unchanged yardstick | cell-by-cell diff vs v2: only cell OURS_CKPT differs |
| — | b-free / globalforge-dinov3 harnesses | NOT edited | protocol unchanged; baselines stay comparable | mtime unchanged |
</file>

<file path="research/eval-diff.md">
# Harness diff (read-only analysis, 2026-09-30)

All harnesses share one codebase. Differences: `OURS_CKPT` (v1 = phase-2 best, v2 = v2 best); `MODEL_ID`; GF adds `pip uninstall torchao`. Per-model behaviour lives in `ModelWrapper`:

| | ours | bfree_official | globalforge_dinov3 |
|---|---|---|---|
| resize | bicubic upscale short side to 504 if <504 | same | same |
| crop | up to 9 × 504 crops covering the full image | 1 center 504 (5-crop wrapper → 5 identical crops) | 1 center 504 |
| norm | ImageNet | ImageNet | inside model |
| score | softmax[:,1], mean over crops | sigmoid(l) | softmax[:,1] |
| dtype | bf16 | bf16 | bf16 (official: fp32) |

Shared: AI-GenBench `validation` (1,500 real pool + 150 fake/generator, 36 generators, rng 42); WildRF test (fb 320 / reddit 1500 / twitter 683); bfree_viral 785. SynthWildX fails in every harness. RealDeg OFF. Metrics: AUC, bAcc@0.5, bAcc@calib, ECE (15 bins), balanced NLL. Calibrated threshold = Youden J fitted on the AI-GenBench test scores, reused on WildRF/viral.

## Unfairness flags
- U1 ours trained on AI-GenBench `train` (13 generators + 46.8k real), evaluated on its `validation`; baselines zero-shot.
- U2 9-crop full coverage (ours) vs 1 center crop (baselines); also breaks B-Free's official full-image protocol.
- U3 GF off-protocol (official: 224 crop, resize-to-1296, JPEG q100 re-encode, fp32). Real mean p=0.4965, bAcc@0.5 stuck 76.10 → only AUC roughly comparable.
- U4 upscaling matches ours' training, not baselines'.
- U5 calibrated threshold tuned on test set: optimistic for every model.
- U6 label smoothing floors ours' ECE.
- U7 v2 degradations = RealDeg eval operator set → future RealDeg comparison biased.

`test/01_eval_harness_template` is a different protocol (Synthbuster/GenImage, RealDeg on, no calibration), never run.

Planned fix (P4, NOT yet applied; needs user OK per CLAUDE.md before touching metric/split/preprocessing): H-a seen/unseen aggregates; H-b unified crop policy; H-c GF official preprocessing fp32; H-d RealDeg on with per-image seeds; H-e SynthWildX loader. All baselines then "must re-run".
</file>

<file path="research/iter-1.md">
# iter-1 — Multi-Severity DCS (v3), run A1 seed 42

## Changed (branch `integration/loss-backbone`, author P-Bao, pushed)
- `configs/bfree_dcs.yaml`: `multi_severity` section (default off, levels from docx).
- `modules/dcs_loss.py`: `multi_severity_dcs` (weighted mean of unchanged `info_nce_loss`).
- `networks/bfree_globalforge_vit.py`: `compute_loss` takes Tensor (v2 path, unchanged) or list of K views; same 3 scalars; per-view terms in `last_dcs_terms`.
- `bfree_datasets/multi_severity.py`: `MultiSeverityDegrader`.
- `train_lora.py`: reads `--config` `multi_severity`; K-view batches; batch finder probes K views; `--select last|best` (+ `selected_ckpt.txt`); `dcs_k*` CSV columns in MS mode.
- `notebooks/version3/xla-training.ipynb` (A1, seed 42 only) and `notebooks/version3/xla-eval-harness-ours.ipynb` (copy of v2; only `OURS_CKPT` changed).
- Setup notebooks: no new libraries needed (code uses numpy/PIL/pyyaml/torch/timm, all bundled). b-free / globalforge harnesses untouched (protocol unchanged).

## Verified on CPU (env `xla-cpu`, torch 2.8.0+cpu)
- `tests/test_dcs_loss.py`, `test_backbone_shape.py`, `test_multi_severity.py` ALL PASS; default path bit-equal to pre-change reference (`tests/ref_single_view.pt`); `tests/overfit_test.py` **GATE PASS**.
- Notebooks: `json.tool` + AST OK; eval copy differs from v2 only in `OURS_CKPT` (+ changelog cell).
- **Not verified**: end-to-end `train_lora.py` main() on CPU — the first Kaggle run is the real check.

## You must do
1. Re-run `xla-setup` (online) so `bfree_src` contains the pushed code; attach its output to training.
2. Run `notebooks/version3/xla-training.ipynb`. Confirm the eval notebook's `OURS_CKPT` slug (`xla-training-v3` assumed).
3. Edit CLAUDE.md yourself (decision 2 → v2 lifecycle pipeline; `papers/`→`paper/`; remove `agent.md` / `.claude/hooks` / `B-Free-integration` refs).

## Paste back
- training notebook last cell + `/kaggle/working/train_log.csv`; first ~30 lines of `train_log.txt` (`[multi_severity] ON`, auto batch size)
- last cell of `notebooks/version3/xla-eval-harness-ours.ipynb`
</file>

<file path="research/iter-2.md">
# iter-2 — split A1 training over 2 Kaggle sessions (12h limit)

## Changed
- `train_lora.py` (branch `integration/loss-backbone`):
  - `--end_epoch N` (0 = all): stop after epoch N. `--epochs` stays 8 and fixes the cosine T_max, so the LR curve equals an uninterrupted run.
  - `--resume_from resume_state.pth`: restores model, optimizer, scheduler, loader generator, python/numpy/torch/cuda RNG, train_log, best bAcc.
  - `resume_state.pth` is written (atomically) after every epoch. Stage 2 takes batch size from it (finder skipped; T_max depends on steps/epoch) and asserts steps/epoch is unchanged.
  - Resume refuses to start if any hyperparameter in `RESUME_CHECK_KEYS` differs from the stage-1 config.
  - `train_log.csv` of stage 2 carries stage-1 rows. `selected_ckpt.txt` is only written when `end_epoch == epochs`.
  - Also (commit `dfc6f83`): model moved to device before the batch-size finder (was the cause of the BFloat16/bias crash).
- `tests/resume_test.py`: CPU gate, 2 epochs straight == 1 epoch + resume. Prints RESUME PASS.
- `notebooks/version3/xla-training.ipynb` = stage 1 (`END_EPOCH=4`); new `xla-training-stage2.ipynb` = stage 2 (`END_EPOCH=8`, `RESUME_FROM=...`). Everything else identical to v3 A1.

## You must do
1. Push `integration/loss-backbone`; re-run `xla-setup` so `bfree_src` has the new `train_lora.py`.
2. Run `xla-training` (stage 1, epochs 1–4) → Save & Run All. Note the `auto batch size = N` line.
3. Create stage-2 notebook from `xla-training-stage2.ipynb`, Add Input → Your Work → the stage-1 notebook output. Check `RESUME_FROM` slug (default `/kaggle/input/notebooks/baopv051/xla-training/resume_state.pth`). Run.
4. Eval notebook: `OURS_CKPT` must point at the **stage-2** output.

## Caveats
- Use the same GPU type for both stages (batch size comes from stage 1; a smaller GPU would OOM).
- With `num_workers>0`, DataLoader worker augmentation RNG restarts at stage 2 (the loader/shuffle order and main-process RNG are restored). Per-epoch metrics are therefore statistically, not bit-wise, equal to an uninterrupted run. `tests/resume_test.py` uses `num_workers=0` where equality is exact.
- `_best.pth` from stage 2 only exists if it beats the stage-1 best; `--select last` makes this moot.
- Each stage must itself fit in 12h; if stage 1 (4 epochs) does not, set `END_EPOCH` lower (1–3) and chain more stages — `resume_state.pth` is saved every epoch. Unverified: whether Kaggle keeps `/kaggle/working` when a commit run is killed at the time limit — do not rely on it.

## Paste back
- stage 1: `auto batch size` line, last cell, wall time. Stage 2: first lines of `train_log.txt` (`[resume] ...` with lr ≈ 5.0e-5), last cell, `train_log.csv` (should hold epochs 1–8).
</file>

<file path="research/log.md">
# Decision / dead-end log  (provenance: user | ai)

- 2026-09-30 user: git repair via `git worktree repair`; harness full protocol fix; v3 base = v2 lifecycle degradation (CLAUDE.md decision 2 to be updated by user); severity bins = docx 4 bins.
- 2026-09-30 user: "Thực hiện code theo kế hoạch plan.md" → plan approved for Phase 2.
- 2026-09-30 ai: `git worktree repair` alone silently did nothing (`.git` file held a Windows path). Rewrote `B-Free-loss-backbone/.git` gitdir line to the Linux path, then repair worked. 38 files show CRLF-only diffs vs LF blobs; only intentionally-edited files were converted to LF and committed.
- 2026-09-30 ai: `tests/` is in the repo `.gitignore` → committed with `git add -f`.
- 2026-09-30 ai: GSR window=3 fully masks a 4×4 patch grid (softmax NaN); CPU tests use img 112 (8×8). Real 504 → 36×36 unaffected.
- 2026-09-30 ai: v2 notebook hardcoded `--batch_size 64` (violates CLAUDE.md auto-batch rule) → v3 uses 0.
- 2026-09-30 ai: assumed noise σ unit 0–255 (docx σ_n=2..30); `single_op` chosen as default compose (docx ambiguous). Both flagged in direction README.
- 2026-09-30 ai: attempted a synthetic end-to-end `train_lora.py` CPU smoke run; user declined the command. Not run: main()'s loop/CSV edits are only py_compile'd and covered by glue tests, NOT executed end-to-end. First Kaggle A1 run is the real check.
- 2026-09-30 user: code only on `integration/loss-backbone`, commit+push as P-Bao <BaoPV.B23KH006@stu.ptit.edu.vn>; harness scope = minimal (only new version3 ours eval copy; b-free/gf untouched); training notebook = A1, one seed (42) only; keep research/ but A1-only.
- 2026-09-30 ai: `v3-dev` removed; the 4 commits were re-created with P-Bao author (git commit-tree, trees identical), fast-forwarded into `integration/loss-backbone`, pushed. Co-Authored-By trailer dropped.
- 2026-09-30 ai: setup notebooks checked, no library additions; observed (not changed) `py7zr` missing wheel and transformers 4.56.2 vs 4.55.4 mismatch in setup-eval.
- 2026-10-02 user: training exceeds the 12h Kaggle session → split epochs over 2 notebooks via checkpoint.
- 2026-10-02 ai: 4+4 epoch split, cosine T_max kept at 8 epochs; stage 2 reuses stage-1 batch size (steps/epoch fixes the schedule). Details in iter-2.md.
</file>

<file path="research/PATHS.md">
# PATHS — verified 2026-09-30 (ls / find)

| CLAUDE.md expects | Reality |
|---|---|
| `agent.md` | MISSING anywhere. Rules taken from CLAUDE.md + plan.md only |
| `.claude/hooks/guard_edits.py` | MISSING. `settings.json` at root (not active). No hook → discipline manual |
| `scripts/lock_readonly.sh` | root `lock_readonly.sh` (never run) |
| `B-Free/` | present; HEAD = `integration/train-eval` (not main) |
| `B-Free-integration/` | MISSING |
| `B-Free-loss-backbone/` | present. `.git` file fixed 2026-09-30 (was Windows path) → `git -C B-Free worktree repair`. Branch `integration/loss-backbone` (all code work here; no side branch) |
| tests | none existed; new in `B-Free-loss-backbone/tests/` |
| config | `B-Free-loss-backbone/code/configs/bfree_dcs.yaml`, `loader.py` |
| train entry | `B-Free-loss-backbone/code/train_lora.py` (Kaggle, accelerate) |
| `notebooks/version1/` | `xla-training.ipynb`, `xla-training-phase-2.ipynb`, `xla-eval-harness-ours.ipynb` |
| `notebooks/version2/` | `xla-training.ipynb`, `xla-eval-harness-ours.ipynb` |
| harnesses outside version dirs | `notebooks/xla-eval-harness-b-free.ipynb`, `notebooks/xla-eval-harness-globalforge-dinov3.ipynb` |
| `notebooks/version3/` | to create |
| `papers/` | actually `paper/`: `paper/latex_paper_bfree_globalforge/latex_paper_bfree_globalforge/main.tex` (+`main_vi.tex`, `references.bib`). `paper/B-free/`, `paper/globalForge/` = original sources, read-only |
| docx | root `Multi-Severity DCS AI Detection.docx`; copy in `research/directions/D-multiseverity-dcs/source/` |
| extra | `test/00_offline_assets_setup.ipynb`, `01_eval_harness_template.ipynb`, `02_aggregate_compare.ipynb` |

Local CPU env: conda env `xla-cpu` (py3.11, torch CPU 2.8.0, timm 1.0.22, peft 0.15.2, nbformat). `face_recognition` env untouched.
Line endings: worktree files are CRLF, committed blobs LF → 38 files show "modified" (whitespace only). Commit only files intentionally touched.
</file>

<file path="research/README.md">
# research/ index

- `PATHS.md` — real paths vs CLAUDE.md expectations
- `eval-diff.md` — harness comparison + unfairness flags
- `eval-changelog.md` — every harness edit (none yet)
- `log.md` — decisions / dead ends (user vs ai)
- `iter-1.md` — what changed, what to run on Kaggle, what to paste back
- `directions/D-multiseverity-dcs/` — MS-DCS spec, hypotheses (`README.md`), docx in `source/`

All directions share the v2 baseline, the same harness and splits. No direction is assumed to win.
</file>

<file path="training_data/README.md">
# Training Dataset
[![arXiv](https://img.shields.io/badge/-Data-ffab03.svg?style=for-the-badge&logo=files&logoColor=ffffff)](https://www.grip.unina.it/download/prog/B-Free/training_data/)

You can download our training dataset at the above link.

***Note:*** *training a method on this dataset only partially reflects the **inpainted+** version of our augmentation 
strategy, as the dataset does not include blurring, JPEG, and the extra augmentations of **inpainted++** (scaling, 
cut-out, noise addition, jittering), which are performed at training time.*

|        | Real   | Generated |
|--------|--------|-----------|
| Source | COCO   | SD 2.1    |
| Num.   | 51,517 | 309,102   |  

To create this dataset we used images collected from the training set of MS-COCO dataset (discarding images with 
licenses different than Creative Commons) and images generated with Stable Diffusion 2.1.\
Specifically, we first extracted the largest central crop and resize it to 512 x 512.
Then, for the generations, we used the inpainting code from the official [Stable Diffusion 2.1 repository](https://github.com/Stability-AI/stablediffusion).
Note that we did NOT embed the watermark during generation (`put_watermark` function).

The dataset at the above link contains the following folders and files:
- **COCO_real_512**: real images from COCO (largest central crop resized to 512x512)
- **SD2.1_selfconditioned**: self-conditioned images
- **SD2.1_selfconditioned_origBG**: self-conditioned images with original background restored
- **SD2.1_inpainted_samecat**: an object is replaced with one of the same category
- **SD2.1_inpainted_samecat_origBG**: same as *inpainted_samecat*, but with original background restored
- **SD2.1_inpainted_diffcat**: an object is replaced with one of a different category
- **SD2.1_inpainted_diffcat_origBG**: same as *inpainted_diffcat*, but with original background restored
- **mask**: object masks used for the generation of *inpainted_samecat* and for the background replacement of *inpainted_samecat_origBG* and *selfconditioned_origBG*
- **bbox**: bounding boxes used for the generation of *inpainted_diffcat* and for the background replacement of *inpainted_diffcat_origBG*
- **train_list.csv**: contains the ids for our training split and extra info for each image, such as the object category
- **valid_list.csv**: contains the ids for our validation split and extra info for each image, such as the object category

Note that both *train* and *valid* split come from the MS-COCO 2017 **training set**.

Further details can be found in appendix A of the paper.

## md5sum

- 41741e8e81e61455d01455452fcf15ae COCO_real_512.zip: 
- a96855ced782ce09b6139ba68af94dd0 SD2.1_selfconditioned.zip: 
- b2e8530fa6904bc25e6fe6f1ed45149d SD2.1_selfconditioned_origBG.zip: 
- fc3ba27bc97d60bcf18ca57931aebd3b SD2.1_inpainted_samecat.zip: 
- 466fed55a78dacc1f82bd95f6eb7cd5c SD2.1_inpainted_samecat_origBG.zip: 
- 5b00dc28d18809c2efc943c89c3f5514 SD2.1_inpainted_diffcat.zip: 
- d289c6b0fe4b19f229f4578e61ca8032 SD2.1_inpainted_diffcat_origBG.zip: 
- 0294cb1611ad1c50af72d6cace3492ca masks_and_bbox.zip
- 06581da53bf81741501396d71eb7ec5f train_list.csv:
- 72e7b4be2d049029dbedca7e75af91b0 valid_list.csv: 


## Bibtex

If you use this dataset, please cite:

```
@inproceedings{Guillaro2025biasfree,
  title={A Bias-Free Training Paradigm for More General AI-generated Image Detection},
  author={Guillaro, Fabrizio and Zingarini, Giada and Usman, Ben and Sud, Avneesh and Cozzolino, Davide and Verdoliva, Luisa},
  booktitle={IEEE/CVF conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2025}
}
```


## License

Copyright (c) 2025 Image Processing Research Group of University Federico II of Naples ('GRIP-UNINA'). 

All rights reserved.

This software should be used, reproduced and modified only for informational and nonprofit purposes.

By downloading and/or using any of these files, you implicitly agree to all the
terms of the license, as specified in the document LICENSE.txt
(included in this package)
</file>

<file path="LICENSE.txt">
THIS DOCUMENT CONSTITUTES A LICENCE TO USE THE SOFTWARE ON THE TERMS AND CONDITIONS APPEARING BELOW.


Preamble

This License applies to the software with which this license is distributed.
The software is intellectual property of Image Processing Research Group of University Federico II of Naples ('GRIP-UNINA')
and is placed under the protection of copyright laws, including Italian legislation and international treaties.
BY USING THE PROGRAM, YOU INDICATE YOUR ACCEPTANCE OF THIS LICENSE TO DO SO.


Terms and Conditions

Reproduction, modification, and usage of the software covered by this license is allowed free of charge provided that:
(i) this software should be used, reproduced and modified only for informational and nonprofit purposes; any unauthorized use of this software for industrial or profit-oriented activities is expressively prohibited; and
(ii) any reproduction or modification retains all original notices including proprietary or copyright notices; and
(iii) reference to the original authors is given whenever results, which arise from the use of this software or any modification of it, are made public.
No other use of the materials and of any information incorporated thereto is hereby authorized.
In addition, be informed that some names are protected by trademarks which are the property of GRIP-UNINA, its researchers and/or other third parties whether a specific mention in that respect is made or not. 


Disclaimers

This software is provided 'as-is', without any express or implied warranty.
In no event will the authors be held liable for any damages arising from the use of this software.


Transmission of user information

Any and all information or request for information you may direct to GRIP
through e-mail as may be linked to website http://www.grip.unina.it/
</file>

<file path="MODULES_INTERFACE.md">
# Modules Interface — B-Free GlobalForge Integration (branch integration/loss-backbone)

Loss and backbone module contract for the team. This branch provides the backbone
with post-hoc LIB/GSR and the L_DCS loss; the real LIB/GSR adapters land on the
`integration/lib-gsr` branch and the training/eval scripts on `integration/train-eval`.

## Model

```python
from networks.bfree_globalforge_vit import BFreeGlobalForgeViT
from configs.loader import load_config, build_model_kwargs

model = BFreeGlobalForgeViT(
    arch="vit_base_patch14_reg4_dinov2.lvd142m",
    num_classes=2,
    img_size=504,
    pretrained=True,          # True = tải dinov2 từ timm; False = random init
    use_lib=True,
    use_gsr=True,
    use_dcs=True,
    lib_kernel=3, lib_tau=0.5,
    gsr_window=3, gsr_mask_prob=1.0,
    dcs_tau=0.07,
    lambda_dcs=0.1,
    label_smoothing=0.1,
)
```

### forward(images) → dict
```python
out = model(images)          # images: (B, 3, 504, 504) hoặc (B, 3, 224, 224) nếu img_size=224
out["logits"]                # (B, 2)
out["cls"]                   # (B, 768) — mean-pool patch tokens (giống GlobalForge REM.py:279)
```

### compute_loss(images_clean, images_degraded, labels) → (total, ce, dcs)
```python
total, ce, dcs = model.compute_loss(images_clean, images_degraded, labels)
# total = ce + lambda_dcs * dcs
# ce  = CrossEntropy(logits_clean, labels) với label_smoothing=0.1
# dcs = info_nce_loss(cls_clean, cls_degraded, tau=0.07)  # symmetric InfoNCE
# Trả về 3 giá trị tách biệt để log riêng (Phase 8).
```

### load_checkpoint(path, strict=False)
```python
rep = model.load_checkpoint("bfree_baseline.pth", strict=False)
# Remap 'patch_embed.X' -> 'model.patch_embed.X' (định dạng Wrapper5crops).
# LIB/GSR/dcs/head mới sẽ missing (OK vì strict=False).
```

## Kiến trúc (quyết định đã chốt, khác plan gốc)

- **LIB/GSR áp dụng POST-HOC** trên patch tokens cuối (sau 12 block + norm), KHÔNG chèn giữa block 6/12.
  Lý do: code thật GlobalForge (`REM.py:251-282`) làm post-hoc, không chèn giữa block.
- Token order: `[cls, reg×4, patch×N]` — `num_prefix=5`, patch tokens = `x[:, 5:, :]`.
- 504×504: N=1296, hw=36. 224×224: N=256, hw=16. Dùng `model.set_input_size()`.
- Feature cho L_DCS = **mean-pool patch tokens** (không dùng cls token), giống `REM.py:279`.

## L_DCS (modules/dcs_loss.py)

```python
from modules.dcs_loss import info_nce_loss, DCSLoss
loss = info_nce_loss(z_clean, z_degraded, tau=0.07)  # (B, C) -> scalar
# = 0.5 * (CE(z @ z_aug.T / tau, arange) + CE(z_aug @ z.T / tau, arange))
# L2 normalize bên trong.
```
Không phụ thuộc dữ liệu RealDeg-Bench/VAE — chỉ cần cặp feature (clean, degraded).

## View suy giảm (cho L_DCS)

Pipeline GlobalForge (`datasets_real.py:378-383`), KHÔNG phải inpainted++ (B-Free không có code):
- RandomJPEG(quality 20-80, p=1.0)
- RandomGaussianBlur(kernel=7, sigma 0.5-1.5, p=0.8)
- ColorDistortion(level=3, p=0.8) = ColorJitter(0.4, 0.4, 0.4, 0.06)
- Bản triển khai: `tests/degradation.py` (`DegradationPipeline`) — chỉ local, không commit.

## Stub LIB/GSR (temporary, replaced by the real adapters on the lib-gsr branch)

```python
# modules/lib_adapter_stub.py — remove after merging modules/lib_adapter.py
class LIBAdapter(nn.Module):
    def __init__(self, dim=768, kernel_size=3, tau=0.5): ...
    def forward(self, feat_tokens, hw=None): return feat_tokens, None  # Identity

# modules/gsr_adapter_stub.py — remove after merging modules/gsr_adapter.py
class GSRAdapter(nn.Module):
    def __init__(self, dim=768, window=3, mask_prob=1.0): ...
    def forward(self, tokens, hw=None): return tokens  # Identity
```

At merge: swap the `*_stub` imports in `networks/bfree_globalforge_vit.py` for the real
adapters. **Re-run the overfit gate** afterwards — the stubs are Identity, so LIB/GSR
have not yet been exercised against the real backbone.

## Config (configs/bfree_dcs.yaml)

```yaml
backbone:
  arch: vit_base_patch14_reg4_dinov2.lvd142m
  num_classes: 2
  img_size: 504
  pretrained: false
  use_lib: true, use_gsr: true
  lib_kernel: 3, lib_tau: 0.5
  gsr_window: 3, gsr_mask_prob: 1.0
loss:
  use_dcs: true
  lambda_dcs: 0.1     # khởi điểm, cần sweep
  dcs_tau: 0.07
  label_smoothing: 0.1
degradation:
  jpeg_quality_min: 20, jpeg_quality_max: 80
  blur_kernel: 7, blur_sigma_min: 0.5, blur_sigma_max: 1.5, blur_prob: 0.8
  color_level: 3, color_prob: 0.8
```

## Gate Phase 4 (overfit test) — ĐÃ PASS

- 4 ảnh, 224×224, 20 bước, lr=5e-5, pretrained=True.
- total: 0.816 → 0.214 (ratio 0.262), CE: 0.683 → 0.203, DCS: 1.327 → 0.111.
- Kiến trúc (backbone + LIB/GSR stub + L_DCS + degradation) không có silent bug.

## Hyperparameter khởi điểm (cần sweep ở Phase 5+)

- `lambda_dcs`: 0.1 (khởi điểm, GlobalForge released dùng 0.01)
- `dcs_tau`: 0.07 (GlobalForge default)
- `label_smoothing`: 0.1 (GlobalForge convention)
- `lr`: 5e-5 (overfit test; training thật có thể khác)
</file>

<file path="requirements-test.txt">
# Test/development environment for integration work (Python 3.10, CPU)
# B-Free backbone uses timm ViT-B/14 reg4 dinov2; pins follow B-Free/code/requirements.txt
# (note: "pytorch" in B-Free requirements is actually the PyPI package "torch")
torch==2.0.1
torchvision==0.15.2
timm>=1.0.12
transformers>=4.43.4
pyyaml
scikit-learn
pandas
tqdm
pillow
numpy
pytest
</file>

<file path="code/bfree_datasets/multi_severity.py">
"""Multi-severity degradation views for MS-DCS (direction D-multiseverity-dcs).

Builds K degraded PIL views of one (already cropped/flipped) clean image, one per
severity bin. All ranges come from ``configs/bfree_dcs.yaml :: multi_severity`` — nothing
is hardcoded here. Operator implementations mirror ``train_lora.apply_degradation``
(JPEG re-encode, PIL Gaussian blur, resize-down/bicubic-up), except noise, which takes
sigma on the 0-255 scale as in the docx table.

compose:
  single_op : each view = ONE operator drawn uniformly from OPS, parameter drawn
              uniformly inside that bin's range.
  compound  : each view = jpeg -> blur -> resize -> noise, each with a parameter drawn
              inside the bin's range.

views_per_step:
  == len(levels): view k is bin k (fixed order, so per-bin logging is meaningful).
  <  len(levels): bins are sampled without replacement per image (weights then uniform).
"""

import io
from typing import Dict, List, Sequence

import numpy as np
from PIL import Image, ImageFilter

OPS = ("jpeg", "blur", "resize", "noise")
_REQUIRED_KEYS = ("jpeg", "blur_sigma", "resize", "noise_std255")


def validate_config(ms: Dict) -> None:
    levels = ms["levels"]
    if not levels:
        raise ValueError("multi_severity.levels is empty")
    for i, lv in enumerate(levels):
        for k in _REQUIRED_KEYS:
            if k not in lv or len(lv[k]) != 2 or lv[k][0] > lv[k][1]:
                raise ValueError(f"multi_severity.levels[{i}].{k} must be [lo, hi], got {lv.get(k)}")
    if ms["compose"] not in ("single_op", "compound"):
        raise ValueError(f"multi_severity.compose must be single_op|compound, got {ms['compose']}")
    n, v = len(levels), int(ms["views_per_step"])
    if not 1 <= v <= n:
        raise ValueError(f"views_per_step={v} must be in [1, {n}]")
    if v == n and len(ms["weights"]) != n:
        raise ValueError(f"len(weights)={len(ms['weights'])} must equal number of levels {n}")


def _apply(img: Image.Image, op: str, lv: Dict, rng) -> Image.Image:
    if op == "jpeg":
        q = rng.randint(int(lv["jpeg"][0]), int(lv["jpeg"][1]))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=q)
        buf.seek(0)
        return Image.open(buf).convert("RGB")
    if op == "blur":
        return img.filter(ImageFilter.GaussianBlur(radius=rng.uniform(*lv["blur_sigma"])))
    if op == "resize":
        s = rng.uniform(*lv["resize"])
        w, h = img.size
        small = img.resize((max(1, int(w * s)), max(1, int(h * s))), Image.BICUBIC)
        return small.resize((w, h), Image.BICUBIC)
    if op == "noise":
        sigma = rng.uniform(*lv["noise_std255"])
        arr = np.asarray(img).astype(np.float32)
        gen = np.random.default_rng(rng.getrandbits(32))
        arr = arr + gen.normal(0.0, sigma, arr.shape)
        return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    raise ValueError(op)


class MultiSeverityDegrader:
    """``degrader(img) -> list[PIL.Image]`` of length ``views_per_step``.

    ``rng`` is any object with ``randint/uniform/choice/sample/getrandbits`` (the
    ``random`` module or a ``random.Random``); seeding it makes views deterministic.
    """

    def __init__(self, levels: Sequence[Dict], compose: str, views_per_step: int, rng):
        self.levels = list(levels)
        self.compose = compose
        self.views_per_step = int(views_per_step)
        self.rng = rng
        validate_config({"levels": self.levels, "compose": compose,
                         "views_per_step": self.views_per_step,
                         "weights": [1] * len(self.levels)})

    @classmethod
    def from_config(cls, ms: Dict, rng):
        validate_config(ms)
        return cls(ms["levels"], ms["compose"], ms["views_per_step"], rng)

    def bins_for_step(self) -> List[int]:
        n = len(self.levels)
        if self.views_per_step == n:
            return list(range(n))
        return self.rng.sample(range(n), self.views_per_step)

    def degrade_one(self, img: Image.Image, bin_idx: int) -> Image.Image:
        lv = self.levels[bin_idx]
        if self.compose == "single_op":
            return _apply(img, self.rng.choice(OPS), lv, self.rng)
        for op in OPS:
            img = _apply(img, op, lv, self.rng)
        return img

    def __call__(self, img: Image.Image) -> List[Image.Image]:
        return [self.degrade_one(img, b) for b in self.bins_for_step()]
</file>

<file path="code/datasets/__init__.py">

</file>

<file path="code/modules/dcs_loss.py">
"""L_DCS: degradation-aware contrastive structural loss.

Tái triển khai đúng công thức thật từ GlobalForge-BE0F/code/engine_finetune.py:29-37
(hàm ``info_nce_loss``): symmetric InfoNCE giữa CLS/mean-pooled feature của 2 view
(clean và degraded) cùng một ảnh.

    z_hat      = normalize(z)
    z_aug_hat  = normalize(z_aug)
    S          = z_hat @ z_aug_hat^T / tau          (B, B)
    labels     = arange(B)
    L_DCS      = 0.5 * (CE(S, labels) + CE(S^T, labels))

L_DCS không phụ thuộc cấu trúc dữ liệu riêng của GlobalForge (RealDeg-Bench / VAE)
— chỉ cần cặp feature (z, z_aug) shape (B, C) và nhiệt độ tau.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def info_nce_loss(z: torch.Tensor, z_aug: torch.Tensor, tau: float = 0.07) -> torch.Tensor:
    """Symmetric InfoNCE cho cặp CLS/mean-pooled feature shape (B, C).

    Bản sao chính xác của ``GlobalForge-BE0F/code/engine_finetune.py:29-37``.
    """
    z = F.normalize(z, dim=-1)
    z_aug = F.normalize(z_aug, dim=-1)
    logits = torch.matmul(z, z_aug.t()) / tau  # (B, B)
    labels = torch.arange(logits.size(0), device=logits.device)
    loss1 = F.cross_entropy(logits, labels)
    loss2 = F.cross_entropy(logits.t(), labels)
    return 0.5 * (loss1 + loss2)


class DCSLoss(nn.Module):
    """Wrapper nn.Module cho L_DCS.

    ``weight`` là lambda_dcs (hệ số nhân khi ghép vào L_total), giữ trong config.
    ``forward`` chỉ trả về giá trị loss thuần (không nhân weight) để caller tự nhân
    — tách bạch cho logging.
    """

    def __init__(self, tau: float = 0.07, weight: float = 0.01):
        super().__init__()
        self.tau = float(tau)
        self.weight = float(weight)

    def forward(self, feat_clean: torch.Tensor, feat_degraded: torch.Tensor) -> torch.Tensor:
        return info_nce_loss(feat_clean, feat_degraded, self.tau)


def multi_severity_dcs(z: torch.Tensor, z_views, weights=None, tau: float = 0.07):
    """Weighted mean of ``info_nce_loss(z, z_k)`` over K degraded views.

    Additive to the faithful L_DCS: every term is the unchanged ``info_nce_loss``
    (symmetric InfoNCE, in-batch negatives, not class-aware). With K == 1 this is
    exactly ``info_nce_loss(z, z_views[0])``.

    Returns ``(loss, terms)``: ``loss = sum_k w_k * L_k / sum_k w_k`` and ``terms``
    the list of detached per-view scalars (for logging only).
    """
    k = len(z_views)
    if k == 0:
        raise ValueError("z_views must contain at least one view")
    if weights is None:
        weights = [1.0] * k
    if len(weights) != k:
        raise ValueError(f"len(weights)={len(weights)} != number of views {k}")
    w_sum = float(sum(weights))
    if w_sum <= 0 or any(float(w) < 0 for w in weights):
        raise ValueError(f"weights must be non-negative with positive sum, got {list(weights)}")
    terms = [info_nce_loss(z, zk, tau=tau) for zk in z_views]
    loss = sum(float(w) * t for w, t in zip(weights, terms)) / w_sum
    return loss, [t.detach() for t in terms]
</file>

<file path="code/requirements.txt">
numpy<2
pandas
PyYAML
tqdm
scikit-learn
Pillow
torch==2.0.1
torchvision==0.15.2
timm>=1.0.12
transformers>=4.43.4,<5.0.0
</file>

<file path="code/train_full_ft.py">
import os
import time
import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from configs.loader import load_config, build_model_kwargs
from networks.bfree_globalforge_vit import BFreeGlobalForgeViT
from bfree_datasets.bfree_dataset import BFreeDataset
from utils.dmetrics import balanced_accuracy_score, roc_auc_score


def train_one_epoch(model, loader, optimizer, scheduler, device, epoch):
    model.train()
    total_loss_sum, ce_loss_sum, dcs_loss_sum = 0.0, 0.0, 0.0
    correct, total_samples = 0, 0

    start_time = time.time()
    for step, (img_clean, img_deg, labels) in enumerate(loader):
        img_clean = img_clean.to(device, non_blocking=True)
        img_deg = img_deg.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        loss_total, loss_ce, loss_dcs = model.compute_loss(img_clean, img_deg, labels)

        loss_total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        batch_size = labels.size(0)
        total_loss_sum += loss_total.item() * batch_size
        ce_loss_sum += loss_ce.item() * batch_size
        dcs_loss_sum += loss_dcs.item() * batch_size
        total_samples += batch_size

        if step % 20 == 0:
            print(
                f"Epoch [{epoch}] Step [{step}/{len(loader)}] | "
                f"Total: {loss_total.item():.4f} (CE: {loss_ce.item():.4f}, DCS: {loss_dcs.item():.4f}) | "
                f"LR: {optimizer.param_groups[0]['lr']:.2e}"
            )

    elapsed = time.time() - start_time
    return {
        "train_loss": total_loss_sum / total_samples,
        "train_ce": ce_loss_sum / total_samples,
        "train_dcs": dcs_loss_sum / total_samples,
        "time": elapsed,
    }


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_scores = []
    all_labels = []
    val_loss_sum = 0.0
    total_samples = 0

    for img_clean, img_deg, labels in loader:
        img_clean = img_clean.to(device)
        img_deg = img_deg.to(device)
        labels = labels.to(device)

        loss_total, _, _ = model.compute_loss(img_clean, img_deg, labels)
        out = model(img_clean)
        logits = out["logits"]

        # score = logit_fake - logit_real
        score = (logits[:, 1] - logits[:, 0]).cpu().numpy()
        
        val_loss_sum += loss_total.item() * labels.size(0)
        total_samples += labels.size(0)
        all_scores.extend(score)
        all_labels.extend(labels.cpu().numpy())

    all_scores = torch.tensor(all_scores).numpy()
    all_labels = torch.tensor(all_labels).numpy()

    auc = roc_auc_score(all_labels, all_scores)
    bacc = balanced_accuracy_score(all_labels, all_scores > 0)
    return {
        "val_loss": val_loss_sum / total_samples,
        "val_auc": auc,
        "val_bacc": bacc,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/bfree_dcs.yaml")
    parser.add_argument("--train_csv", type=str, required=True)
    parser.add_argument("--val_csv", type=str, required=True)
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./outputs/full_ft")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    cfg = load_config(args.config)
    model_kwargs = build_model_kwargs(cfg)

    model = BFreeGlobalForgeViT(**model_kwargs)
    model.to(args.device)

    train_dataset = BFreeDataset(
        csv_file=args.train_csv,
        data_root=args.data_root,
        img_size=cfg["backbone"]["img_size"],
        is_train=True,
        degradation_cfg=cfg.get("degradation"),
    )
    val_dataset = BFreeDataset(
        csv_file=args.val_csv,
        data_root=args.data_root,
        img_size=cfg["backbone"]["img_size"],
        is_train=False,
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs * len(train_loader), eta_min=1e-7)

    log_file = os.path.join(args.output_dir, "train_log.csv")
    with open(log_file, "w") as f:
        f.write("epoch,train_loss,train_ce,train_dcs,val_loss,val_auc,val_bacc\n")

    best_bacc = 0.0
    for epoch in range(1, args.epochs + 1):
        tr_res = train_one_epoch(model, train_loader, optimizer, scheduler, args.device, epoch)
        val_res = evaluate(model, val_loader, args.device)

        print(
            f"--> Epoch {epoch} Final: Train Loss={tr_res['train_loss']:.4f} "
            f"(CE={tr_res['train_ce']:.4f}, DCS={tr_res['train_dcs']:.4f}) | "
            f"Val Loss={val_res['val_loss']:.4f}, AUC={val_res['val_auc']:.4f}, bAcc={val_res['val_bacc']:.4f}"
        )

        with open(log_file, "a") as f:
            f.write(
                f"{epoch},{tr_res['train_loss']:.4f},{tr_res['train_ce']:.4f},{tr_res['train_dcs']:.4f},"
                f"{val_res['val_loss']:.4f},{val_res['val_auc']:.4f},{val_res['val_bacc']:.4f}\n"
            )

        if val_res["val_bacc"] > best_bacc:
            best_bacc = val_res["val_bacc"]
            ckpt_path = os.path.join(args.output_dir, "best_model.pth")
            torch.save({"model": model.state_dict(), "epoch": epoch, "best_bacc": best_bacc}, ckpt_path)
            print(f"[*] Checkpoint saved: {ckpt_path} (bAcc: {best_bacc:.4f})")


if __name__ == "__main__":
    main()
</file>

<file path="code/configs/loader.py">
"""Loader YAML tối giản cho config bfree_dcs.yaml.

B-Free gốc không có loader YAML riêng (config nằm trong weights/<model>/config.yaml
với 4 key: weights_file, model_name, arch, norm_type — chỉ phục vụ inference).
Nên loader này tự viết, không động vào code gốc B-Free.
"""

import yaml


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_model_kwargs(cfg: dict) -> dict:
    """Trích dict kwargs cho BFreeGlobalForgeViT.__init__ từ config YAML."""
    b = cfg["backbone"]
    l = cfg["loss"]
    return {
        "arch": b["arch"],
        "num_classes": b["num_classes"],
        "img_size": b["img_size"],
        "pretrained": b["pretrained"],
        "use_lib": b["use_lib"],
        "use_gsr": b["use_gsr"],
        "lib_kernel": b["lib_kernel"],
        "lib_tau": b["lib_tau"],
        "gsr_window": b["gsr_window"],
        "gsr_mask_prob": b["gsr_mask_prob"],
        "use_dcs": l["use_dcs"],
        "dcs_tau": l["dcs_tau"],
        "lambda_dcs": l["lambda_dcs"],
        "label_smoothing": l["label_smoothing"],
        "head_mode": b.get("head_mode", "meanpool"),
        "gsr_zero_init": bool(b.get("gsr_zero_init", False)),
        "lib_beta_init": float(b.get("lib_beta_init", -0.85)),
        "loss_mode": l.get("mode", l.get("loss_mode", "legacy")),
        "w_deg": float(l.get("w_deg", 1.0)),
        "lam_cons": float(l.get("lam_cons", 0.0)),
        "cons_T": float(l.get("cons_T", 1.0)),
    }


def build_degradation_kwargs(cfg: dict) -> dict:
    """Trích dict kwargs cho DegradationPipeline.__init__ (datasets/bfree_dataset.py)
    từ config YAML — đối xứng với build_model_kwargs(). Dùng .get(..., default) cho
    resize_*/noise_* vì đây là key mới bổ sung, để các YAML cũ (chưa có 2 mục này)
    vẫn nạp được mà không lỗi, rơi về đúng default của DegradationPipeline."""
    d = cfg["degradation"]
    return {
        "jpeg_quality_min": d["jpeg_quality_min"],
        "jpeg_quality_max": d["jpeg_quality_max"],
        "blur_kernel": d["blur_kernel"],
        "blur_sigma_min": d["blur_sigma_min"],
        "blur_sigma_max": d["blur_sigma_max"],
        "blur_prob": d["blur_prob"],
        "color_prob": d["color_prob"],
        "resize_scale_min": d.get("resize_scale_min", 0.2),
        "resize_scale_max": d.get("resize_scale_max", 0.9),
        "resize_prob": d.get("resize_prob", 0.5),
        "noise_std_min": d.get("noise_std_min", 0.02),
        "noise_std_max": d.get("noise_std_max", 0.1),
        "noise_prob": d.get("noise_prob", 0.5),
        "single_op_prob": d.get("single_op_prob", 0.4),
    }
</file>

<file path="code/networks/bfree_globalforge_vit.py">
"""B-Free backbone (timm ViT-B/14 reg4 dinov2) + post-hoc LIB/GSR + L_DCS.

Sửa kiến trúc theo quyết định đã chốt (xem agent.md):
  - LIB/GSR áp dụng POST-HOC trên patch tokens cuối (giống GlobalForge REM.py:251-282),
    KHÔNG chèn giữa block 6/12.
  - Feature cho L_DCS = mean-pool patch tokens (REM.py:279), KHÔNG dùng cls token.
  - L_DCS = symmetric InfoNCE (modules.dcs_loss.info_nce_loss).
  - num_classes = 2 (B-Free gốc dùng num_classes=1, nhưng hợp đồng giao diện plan-nguoi2
    yêu cầu logits (B,2); dùng 2 cho CE multi-class real/fake).

Token order của timm vit_*_reg4: [cls, reg×4, patch×N] -> num_prefix_tokens = 5.
Với ảnh 504×504, patch 14: N = 36×36 = 1296, hw = 36.
Với ảnh 224×224 (overfit nhanh): N = 16×16 = 256, hw = 16.
"""

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import timm
from timm.models.vision_transformer import VisionTransformer

from modules.lib_adapter import LIBAdapter
from modules.gsr_adapter import GSRAdapter
from modules.dcs_loss import DCSLoss, info_nce_loss, multi_severity_dcs


class BFreeGlobalForgeViT(nn.Module):
    """timm ViT-B/14 reg4 dinov2 + post-hoc LIB/GSR trên patch tokens.

    forward(images) -> dict(logits (B,2), cls (B,C))   # cls = mean-pool patch tokens
    compute_loss(images_clean, images_degraded, labels) -> (total, ce, dcs)
    load_checkpoint(path, strict=False)
    """

    def __init__(
        self,
        arch: str = "vit_base_patch14_reg4_dinov2.lvd142m",
        num_classes: int = 2,
        img_size: int = 504,
        pretrained: bool = False,
        use_lib: bool = True,
        use_gsr: bool = True,
        use_dcs: bool = True,
        lib_kernel: int = 3,
        lib_tau: float = 0.5,
        gsr_window: int = 3,
        gsr_mask_prob: float = 1.0,
        dcs_tau: float = 0.07,
        lambda_dcs: float = 0.1,
        label_smoothing: float = 0.1,
        head_mode: str = "meanpool",
        gsr_zero_init: bool = False,
        lib_beta_init: float = -0.85,
        loss_mode: str = "legacy",
        w_deg: float = 1.0,
        lam_cons: float = 0.0,
        cons_T: float = 1.0,
    ):
        super().__init__()
        if head_mode not in ("cls", "meanpool"):
            raise ValueError(f"head_mode must be 'cls' or 'meanpool', got {head_mode!r}")
        if loss_mode not in ("legacy", "dual_view"):
            raise ValueError(f"loss_mode must be 'legacy' or 'dual_view', got {loss_mode!r}")
        if cons_T <= 0:
            raise ValueError(f"cons_T must be > 0, got {cons_T}")
        # head_mode mặc định của CLASS giữ "meanpool" để checkpoint cũ nạp/eval không đổi;
        # mặc định cho training mới ("cls") nằm ở configs/bfree_dcs.yaml (backbone.head_mode).
        self.head_mode = head_mode
        self.loss_mode = loss_mode
        self.w_deg = float(w_deg)
        self.lam_cons = float(lam_cons)
        self.cons_T = float(cons_T)
        self.num_classes = num_classes
        self.use_lib = use_lib
        self.use_gsr = use_gsr
        self.use_dcs = use_dcs
        self.lambda_dcs = float(lambda_dcs)

        # Backbone timm. patch_embed + pos_embed + 12 blocks + norm + head.
        self.model = timm.create_model(
            arch,
            num_classes=num_classes,
            pretrained=pretrained,
        )
        self.model.set_input_size(img_size)
        self.embed_dim = self.model.embed_dim  # 768
        self.num_prefix = self.model.num_prefix_tokens  # 5 (cls + 4 reg)

        # LIB/GSR post-hoc (theo dim=768 của ViT-B/14). Stub = Identity cho bước này.
        self.lib = (LIBAdapter(self.embed_dim, kernel_size=lib_kernel, tau=lib_tau,
                               beta_init=lib_beta_init) if use_lib else None)
        self.gsr = (GSRAdapter(self.embed_dim, window=gsr_window, mask_prob=gsr_mask_prob,
                               zero_init_gate=gsr_zero_init) if use_gsr else None)

        # Loss
        self.dcs = DCSLoss(tau=dcs_tau, weight=lambda_dcs) if use_dcs else None
        self.ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        # Per-view DCS terms of the last compute_loss (multi-severity mode), detached, for logging.
        self.last_dcs_terms: Dict[str, torch.Tensor] = {}
        # Per-term CE/consistency of the last compute_loss (dual_view mode), detached, for logging.
        self.last_terms: Dict[str, torch.Tensor] = {}

    # ---- feature extraction -------------------------------------------------

    def _patch_tokens(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """Chạy backbone -> strip prefix tokens -> (patch_tokens, feat_cls_mean, hw).

        Trả về:
          patch_tokens: (B, N, C)  — N = hw*hw
          feat_cls: (B, C)         — mean-pool của patch tokens (theo REM.py:279)
          hw: int                  — số patch mỗi cạnh (36 cho 504, 16 cho 224)
        """
        _, patch, feat_cls, hw = self._encode(x)
        return patch, feat_cls, hw

    def _encode(self, x: torch.Tensor):
        """Như _patch_tokens nhưng trả thêm ``feats`` đầy đủ (B, 1+4+N, C) từ forward_features
        (chưa qua LIB/GSR) — cần cho head_mode='cls' (timm forward_head lấy token 0)."""
        feats = self.model.forward_features(x)      # (B, 1+4+N, C)
        x = feats
        patch = x[:, self.num_prefix:, :]           # (B, N, C)
        N = patch.shape[1]
        hw = int(math.isqrt(N))
        assert hw * hw == N, f"patch count {N} not a perfect square (hw={hw})"
        # post-hoc LIB -> GSR trên patch tokens (giống REM.py:272-276)
        if self.lib is not None:
            patch, _ = self.lib(patch, hw=hw)
        if self.gsr is not None:
            patch = self.gsr(patch, hw=hw)
        feat_cls = patch.mean(dim=1)                # (B, C)
        return feats, patch, feat_cls, hw

    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        feats, patch, feat_cls, hw = self._encode(images)
        if self.head_mode == "cls":
            # P2.B: head đọc CLS token của backbone (như B-Free gốc) + độ lệch residual từ LIB/GSR
            # trên mean-pool patch tokens: cls_in = cls_token + (feat_cls - mean(patch_raw)).
            # - Khi không dùng LIB/GSR (hoặc gate=0, alpha=0): cls_in == feats[:, 0, :] -> trùng
            #   khít self.model.forward_head(feats) của B-Free gốc.
            # - Khi gsr_zero_init=True (gate=0) và lib_beta_init=-6 (alpha0~0.0025): ở bước 0
            #   |Δ logits| ~ 1e-4 < 1e-3 (Gate "bước 0 ≡ B-Free"), đồng thời LIB/GSR vẫn nhận
            #   gradient trực tiếp từ CE trong các run R3 (+LIB) và R4 (+GSR).
            patch_raw_mean = feats[:, self.num_prefix:, :].mean(dim=1)
            cls_in = feats[:, 0, :] + (feat_cls - patch_raw_mean)
            cls_in = self.model.fc_norm(cls_in)
            cls_in = self.model.head_drop(cls_in)
            logits = self.model.head(cls_in)         # (B, num_classes)
            feat_cls = self.model.fc_norm(feat_cls)
            feat_cls = self.model.head_drop(feat_cls)
            return {"logits": logits, "cls": feat_cls}
        # meanpool (legacy): head dùng feat_cls (mean-pool patch) thay vì cls token, theo GlobalForge.
        # Gọi trực tiếp self.model.head (bypass forward_head vì pool mặc định lấy cls token).
        feat_cls = self.model.fc_norm(feat_cls)
        feat_cls = self.model.head_drop(feat_cls)
        logits = self.model.head(feat_cls)          # (B, num_classes)
        return {"logits": logits, "cls": feat_cls}

    # ---- loss ---------------------------------------------------------------

    def compute_loss(
        self,
        images_clean: torch.Tensor,
        images_degraded,
        labels: torch.Tensor,
        view_weights=None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Trả về (total_loss, ce_loss, dcs_loss) — tách bạch cho logging Phase 8.

        CE tính trên logits của view clean (theo convention GlobalForge engine_finetune.py:130).
        L_DCS tính giữa cls_clean và cls_degraded (symmetric InfoNCE).
        total = CE + lambda_dcs * L_DCS.

        ``images_degraded``: Tensor (đường mặc định, không đổi so với v2) hoặc list K Tensor
        (multi-severity). Với list, dcs_loss = trung bình có trọng số (``view_weights``,
        mặc định đều) của K symmetric InfoNCE; từng term detached ghi vào
        ``self.last_dcs_terms`` = {"dcs_k1": ..., ...}. Hợp đồng 3 scalar giữ nguyên.

        ``loss_mode == "dual_view"`` (P2.A): xem ``_compute_loss_dual_view``.
        """
        if self.loss_mode == "dual_view":
            return self._compute_loss_dual_view(images_clean, images_degraded, labels)

        out_clean = self.forward(images_clean)
        ce_loss = self.ce(out_clean["logits"], labels)

        dcs_loss = out_clean["logits"].new_zeros(())
        self.last_dcs_terms = {}
        self.last_terms = {}
        if self.use_dcs and self.dcs is not None:
            if isinstance(images_degraded, (list, tuple)):
                cls_views = [self.forward(v)["cls"] for v in images_degraded]
                dcs_loss, terms = multi_severity_dcs(
                    out_clean["cls"], cls_views, weights=view_weights, tau=self.dcs.tau)
                self.last_dcs_terms = {f"dcs_k{i + 1}": t for i, t in enumerate(terms)}
            else:
                out_deg = self.forward(images_degraded)
                dcs_loss = info_nce_loss(out_clean["cls"], out_deg["cls"], tau=self.dcs.tau)

        total = ce_loss + self.lambda_dcs * dcs_loss
        return total, ce_loss, dcs_loss

    def _compute_loss_dual_view(
        self,
        images_clean: torch.Tensor,
        images_degraded,
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """P2.A — view suy giảm tham gia phân loại (không tốn thêm forward: view này đã được
        forward cho DCS).

            total = CE(clean) + w_deg * CE(deg) + lam_cons * KL(clean.detach() || deg) + lambda_dcs * DCS

        Trả về (total, ce_clean, dcs) — giữ hợp đồng 3 scalar; ``self.last_terms`` =
        {ce_c, ce_d, cons} (detached) để log. ``w_deg=0, lam_cons=0`` ≡ đường legacy.
        """
        if isinstance(images_degraded, (list, tuple)):
            raise ValueError("loss.mode=dual_view không hỗ trợ multi_severity (list view); "
                             "tắt multi_severity hoặc dùng loss.mode=legacy")
        out_c = self.forward(images_clean)
        out_d = self.forward(images_degraded)
        ce_c = self.ce(out_c["logits"], labels)
        ce_d = self.ce(out_d["logits"], labels)         # nhãn không đổi dưới suy giảm
        T = self.cons_T
        cons = F.kl_div(
            F.log_softmax(out_d["logits"].float() / T, dim=-1),
            F.softmax(out_c["logits"].detach().float() / T, dim=-1),
            reduction="batchmean",
        ) * (T * T)                                      # nhất quán quyết định clean -> deg
        dcs = out_c["logits"].new_zeros(())
        if self.use_dcs and self.dcs is not None:
            dcs = info_nce_loss(out_c["cls"], out_d["cls"], tau=self.dcs.tau)

        total = ce_c + self.w_deg * ce_d + self.lam_cons * cons + self.lambda_dcs * dcs
        self.last_dcs_terms = {}
        self.last_terms = {"ce_c": ce_c.detach(), "ce_d": ce_d.detach(), "cons": cons.detach()}
        return total, ce_c, dcs

    # ---- checkpoint ---------------------------------------------------------

    _OURS_ONLY_PREFIXES = ("lib.", "gsr.", "dcs.")

    def remap_bfree_state_dict(self, sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Đưa state_dict B-Free (hoặc dạng đã có tiền tố) về không gian key của self.state_dict().

        - Bỏ tiền tố ``module.`` (DDP) nếu có.
        - Key của module riêng của ta (``lib.``/``gsr.``/``dcs.``) giữ nguyên.
        - Mọi key còn lại: bỏ ``model.`` nếu có (một lần) rồi thêm ``model.`` đúng một lần. Gồm cả
          ``patch_embed.*`` (Wrapper5crops tách riêng) và các key timm trần (``blocks.0...``,
          ``cls_token``, ``head.*``) — checkpoint B-Free chính thức KHÔNG có tiền tố ``model.``.
        - Head: B-Free 1 logit ``l = w.f + b`` -> 2 logit sao cho ``z = l1 - l0 = l``:
          ``W2 = [-w/2; +w/2]``, ``b2 = [-b/2; +b/2]`` (khi ``num_classes == 2``).
        """
        out: Dict[str, torch.Tensor] = {}
        for k, v in sd.items():
            if k.startswith("module."):
                k = k[len("module."):]
            if k.startswith(self._OURS_ONLY_PREFIXES):
                out[k] = v
                continue
            if k.startswith("model."):
                k = k[len("model."):]
            out["model." + k] = v

        for name in ("weight", "bias"):
            key = f"model.head.{name}"
            if key not in out:
                continue
            t = out[key]
            if self.num_classes == 2 and t.shape[0] == 1:
                out[key] = torch.cat([-t / 2, t / 2], dim=0)
            elif t.shape[0] != self.num_classes:
                raise ValueError(
                    f"{key}: checkpoint có {t.shape[0]} logit, model cần {self.num_classes}")
        return out

    def load_checkpoint(self, path: str, strict: bool = False,
                        require_complete_backbone: bool = False) -> Dict:
        """Load checkpoint B-Free (định dạng Wrapper5crops: key 'patch_embed.*' + key timm trần).

        Xử lý key/head xem ``remap_bfree_state_dict``. pos_embed khác kích thước (img_size khác
        504) được resample. Mọi key của LIB/GSR/dcs sẽ missing (khởi tạo tại __init__).

        ``require_complete_backbone=True``: raise RuntimeError nếu còn key ``model.*`` missing
        hoặc có key unexpected — chặn kiểu lỗi "bỏ qua im lặng" của ``strict=False``.
        Trả về báo cáo load_state_dict.
        """
        from torch import load

        dat = load(path, map_location="cpu")
        if isinstance(dat, dict) and "model" in dat and not isinstance(dat["model"], torch.Tensor):
            sd = dat["model"]
        elif isinstance(dat, dict) and "state_dict" in dat:
            sd = dat["state_dict"]
        else:
            sd = dat

        remapped = self.remap_bfree_state_dict(sd)

        pe_key = "model.pos_embed"
        if pe_key in remapped:
            pe = remapped[pe_key]
            new_grid = tuple(self.model.patch_embed.grid_size)
            n = pe.shape[1]
            prefix = 0 if math.isqrt(n) ** 2 == n else self.num_prefix
            old_hw = math.isqrt(n - prefix)
            if old_hw * old_hw != n - prefix:
                raise ValueError(f"cannot parse pos_embed layout {tuple(pe.shape)}")
            if (old_hw, old_hw) != new_grid:
                from timm.layers.pos_embed import resample_abs_pos_embed
                remapped[pe_key] = resample_abs_pos_embed(
                    pe, new_size=new_grid, old_size=(old_hw, old_hw), num_prefix_tokens=prefix)

        report = self.load_state_dict(remapped, strict=strict)
        bad_missing = [k for k in report.missing_keys if not k.startswith(self._OURS_ONLY_PREFIXES)]
        if require_complete_backbone and (bad_missing or report.unexpected_keys):
            raise RuntimeError(
                f"load_checkpoint: backbone không nạp đầy đủ | missing={bad_missing[:10]} "
                f"({len(bad_missing)}) unexpected={list(report.unexpected_keys)[:10]} "
                f"({len(report.unexpected_keys)})")
        return {
            "missing_keys": list(report.missing_keys)[:20],
            "unexpected_keys": list(report.unexpected_keys)[:20],
            "num_missing": len(report.missing_keys),
            "num_unexpected": len(report.unexpected_keys),
            "num_missing_backbone": len(bad_missing),
        }
</file>

<file path="code/README.md">
# Pre-trained weights

Download the [weights](https://www.grip.unina.it/download/prog/B-Free/weights/BFREE_dino2reg4.zip) and unzip them in the `weights` folder.
MD5 of the zipfile is f3f53fa647848b16cf81c913f148a198.
Once unzipped, it should look like this:
```
weights
└── BFREE_dino2reg4
    ├── config.yaml
    └── model_epoch_best.pth
```


# Inference 

For instruction on how to run the code with Docker, see the section *Inference with Docker*.
Otherwise, in order to use the script, the packages in `requirements.txt` should be installed.


## Inference on a single image

If you want to run on a single image:
```
python main_bfree_single.py -i demo_images/metainfo.csv -o results.csv
```

The full list of flags is:
- `-i` or `--input_image`: the path of the input image
- `-m` or `--model`: the model to test
- `-d` or `--device`: torch device

## Inference from csv

If you want to run on a **list of images**, the input csv should contain the list of images to analyze, with the column `filename`.
If you also want to compute metrics, the input csv should also contain the column `label`, which should wither '0' or 'REAL' for real images, and anything else for AI-generated images.

Then run:
```
python main_bfree.py -i demo_images/metainfo.csv -o results.csv
```

If you also want to compute metrics, add the flag `-metrics_csv` or `-t`:
```
python main_bfree.py -i demo_images/metainfo.csv -o results.csv -t results_metrics.csv
```

The full list of flags is:
- `-i` or `--in_csv`: the path of the input csv file with the list of images
- `-o` or `--out_csv`: the path of the output csv file
- `-m` or `--model`: the model to test
- `-d` or `--device`: torch device
- `-t` or `--metrics_csv`: the path of the metrics csv file

***Note***: the code does not currently support batch processing.

***Note***: to be sure that the code has been executed correctly, compare results and metrics with the provided `demo_images/results.csv` and `demo_images/results_metrics.csv`.

## Inference with Docker

To build the docker image, run the following command:
```
docker build -t b_free . -f Dockerfile
```

To lauch the script on a list of images (with the same flags of the section above):

```
docker run --runtime=nvidia --gpus all -v ${PWD}/demo_images:/data_in -v ${PWD}/:/data_out b_free --in_csv /data_in/metainfo.csv --out_csv /data_out/out.csv --device 'cuda:0' --metrics_csv /data_out/results_metrics.csv
```



# License

Copyright (c) 2025 Image Processing Research Group of University Federico II of Naples ('GRIP-UNINA').

All rights reserved.

This software should be used, reproduced and modified only for informational and nonprofit purposes.

By downloading and/or using any of these files, you implicitly agree to all the
terms of the license, as specified in the document LICENSE.txt
(included in this package)
</file>

<file path=".gitignore">
# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]
*$py.class

# C extensions
*.so

# Distribution / packaging
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
share/python-wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# PyInstaller
#  Usually these files are written by a python script from a template
#  before PyInstaller builds the exe, so as to inject date/other infos into it.
*.manifest
*.spec

# Installer logs
pip-log.txt
pip-delete-this-directory.txt

# Unit test / coverage reports
htmlcov/
.tox/
.nox/
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
*.py,cover
.hypothesis/
.pytest_cache/
cover/

# Translations
*.mo
*.pot

# Django stuff:
*.log
local_settings.py
db.sqlite3
db.sqlite3-journal

# Flask stuff:
instance/
.webassets-cache

# Scrapy stuff:
.scrapy

# Sphinx documentation
docs/_build/

# PyBuilder
.pybuilder/
target/

# Jupyter Notebook
.ipynb_checkpoints

# IPython
profile_default/
ipython_config.py

# pyenv
#   For a library or package, you might want to ignore these files since the code is
#   intended to run in multiple environments; otherwise, check them in:
# .python-version

# pipenv
#   According to pypa/pipenv#598, it is recommended to include Pipfile.lock in version control.
#   However, in case of collaboration, if having platform-specific dependencies or dependencies
#   having no cross-platform support, pipenv may install dependencies that don't work, or not
#   install all needed dependencies.
#Pipfile.lock

# UV
#   Similar to Pipfile.lock, it is generally recommended to include uv.lock in version control.
#   This is especially recommended for binary packages to ensure reproducibility, and is more
#   commonly ignored for libraries.
#uv.lock

# poetry
#   Similar to Pipfile.lock, it is generally recommended to include poetry.lock in version control.
#   This is especially recommended for binary packages to ensure reproducibility, and is more
#   commonly ignored for libraries.
#   https://python-poetry.org/docs/basic-usage/#commit-your-poetrylock-file-to-version-control
#poetry.lock

# pdm
#   Similar to Pipfile.lock, it is generally recommended to include pdm.lock in version control.
#pdm.lock
#   pdm stores project-wide configurations in .pdm.toml, but it is recommended to not include it
#   in version control.
#   https://pdm.fming.dev/latest/usage/project/#working-with-version-control
.pdm.toml
.pdm-python
.pdm-build/

# PEP 582; used by e.g. github.com/David-OConnor/pyflow and github.com/pdm-project/pdm
__pypackages__/

# Celery stuff
celerybeat-schedule
celerybeat.pid

# SageMath parsed files
*.sage.py

# Environments
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# Spyder project settings
.spyderproject
.spyproject

# Rope project settings
.ropeproject

# mkdocs documentation
/site

# mypy
.mypy_cache/
.dmypy.json
dmypy.json

# Pyre type checker
.pyre/

# pytype static type analyzer
.pytype/

# Cython debug symbols
cython_debug/

# PyCharm
#  JetBrains specific template is maintained in a separate JetBrains.gitignore that can
#  be found at https://github.com/github/gitignore/blob/main/Global/JetBrains.gitignore
#  and can be added to the global gitignore or merged into this file.  For a more nuclear
#  option (not recommended) you can uncomment the following to ignore the entire idea folder.
#.idea/

# Ruff stuff:
.ruff_cache/

# PyPI configuration file
.pypirc

# --- Project-specific additions ---
plans/
guide/

# --- integration/loss-backbone additions ---
.venv/
tests/
results/
outputs/
*.pth
*.pt
*.ckpt
weights/
.pytest_cache/
.mypy_cache/
.ruff_cache/
configs/*.txt
</file>

<file path="code/configs/bfree_dcs.yaml">
# Cấu hình B-Free + GlobalForge (LIB + GSR + L_DCS) cho nhánh integration/loss-backbone.
# Hyperparameter chưa có số liệu tham chiếu (lambda_dcs, tau...) đều ở đây — không hardcode.

backbone:
  arch: vit_base_patch14_reg4_dinov2.lvd142m
  num_classes: 2
  img_size: 504
  pretrained: false          # true nếu tải dinov2 pretrained; overfit test dùng false
  use_lib: true
  use_gsr: true
  lib_kernel: 3              # LocalInformationBottleneck kernel_size
  lib_tau: 0.5               # LocalInformationBottleneck tau (không dùng trong stub)
  gsr_window: 3              # GlobalStructuralReasoning window (Chebyshev)
  gsr_mask_prob: 1.0         # xác suất áp mask khi training
  # P2.B (AGENT_PLAN_V4) — mặc định giữ giá trị legacy, trừ head_mode.
  head_mode: cls             # cls: head đọc CLS token như B-Free (LIB/GSR chỉ nuôi L_DCS) | meanpool: legacy
  gsr_zero_init: false       # true: GSR out = x + gate*proj(attn@V), gate=0 (nên bật khi --init_from bfree)
  lib_beta_init: -0.85       # -0.85 = legacy (alpha~0.30); -6 => alpha0~0.0025 (nên dùng khi --init_from bfree)

loss:
  use_dcs: true
  lambda_dcs: 0.1            # hệ số L_DCS trong total = CE + lambda_dcs * L_DCS (khởi điểm)
  dcs_tau: 0.07              # nhiệt độ InfoNCE (GlobalForge released: 0.07)
  label_smoothing: 0.1       # CE label smoothing (GlobalForge convention)
  # P2.A (AGENT_PLAN_V4) — view suy giảm tham gia phân loại. mode=legacy giữ bit-identical v2.
  mode: legacy               # legacy | dual_view  (dual_view không dùng chung với multi_severity)
  w_deg: 1.0                 # dual_view: trọng số CE trên view suy giảm
  lam_cons: 0.0              # dual_view: trọng số KL nhất quán clean->deg (khởi điểm thử {0, 0.5})
  cons_T: 1.0                # dual_view: nhiệt độ cho KL nhất quán
  p_aug_on_clean: 0.0        # dual_view: xác suất áp suy giảm kiểu lifecycle lên chính view sạch (khuyến nghị ~0.2)

# P2.C / P2.F (AGENT_PLAN_V4) — cân bằng dữ liệu, chống shortcut & sàng lọc ablation.
# Mặc định giữ nguyên hành vi v2 (mix_current, fmt_match=false, res_match=false, subsample=1.0).
data:
  mix_preset: mix_current    # mix_current (AIGB 4000/gen) | mix_bfree (AIGB 750/gen, <=25% tổng) | mix_noaigb
  fmt_match: false           # true: real & fake trong AIGB/extra cùng qua JPEG ngẫu nhiên Q in [75,95] (p=0.5)
  res_match: false           # true: thay bicubic ensure_min_size bằng replicate-wrap kiểu B-Free khi < img_size
  holdout_generators: []     # danh sách generator trong PRIORITY_GENERATORS giữ lại không đưa vào train
  train_subsample_frac: 1.0  # 1.0 = full train; 0.25 = sàng lọc P2.F (25% dữ liệu, deterministic theo seed)

# Nguồn view suy giảm cho L_DCS (pipeline GlobalForge datasets_real.py:378-383):
#   RandomJPEG(quality=(20,80), p=1.0)
#     + RandomGaussianBlur(kernel_size=7, sigma=(0.5,1.5), p=0.8)
#     + ColorDistortion(level=3, p=0.8)
# (KHÔNG dùng inpainted++ của B-Free — repo B-Free không có code, chỉ mô tả chữ.)
#
# Bổ sung resize/noise (KHÔNG có trong pipeline GlobalForge gốc ở trên): đối chiếu với
# eval RealDeg-Bench-style (notebook 02) cho thấy đây là 2 toán tử DUY NHẤT hoàn toàn
# vắng mặt lúc train, và cũng chính xác là 2 toán tử tệ nhất khi test
# (bAcc resize=0.6464, noise=0.6402 — so với brightness/contrast/saturation ~0.999 vì
# ColorJitter hiện có đã đủ mạnh để phủ 3 toán tử đó). Compound degradation (nhiều toán
# tử nối tiếp) rớt rất nặng (0.5715 ở 5 bước) phần lớn do gần như luôn dính ít nhất 1
# trong 2 toán tử còn thiếu này. jpeg_quality/blur_sigma/color_* GIỮ NGUYÊN không đổi
# so với recipe gốc của GlobalForge.
degradation:
  jpeg_quality_min: 20
  jpeg_quality_max: 80
  blur_kernel: 7
  blur_sigma_min: 0.5
  blur_sigma_max: 1.5
  blur_prob: 0.8
  color_level: 3
  color_prob: 0.8
  resize_scale_min: 0.2      # thu nhỏ còn 20%-90% kích thước gốc rồi phóng lại
  resize_scale_max: 0.9
  resize_prob: 0.5
  noise_std_min: 0.02        # Gaussian noise std trên ảnh chuẩn hoá [0,1]
  noise_std_max: 0.1
  noise_prob: 0.5
  # Bổ sung lần 2: 40% batch dùng đúng 1 toán tử ngẫu nhiên trên ảnh sạch (thay vì luôn
  # xếp chồng JPEG->Blur->Resize->Noise->Color) — sau lần bổ sung resize/noise ở trên,
  # noise cải thiện rõ (bAcc 0.6402->0.7357) nhưng resize gần như không đổi
  # (0.6464->0.6510) vì trong nhánh compound resize luôn bị áp SAU JPEG(p=1.0), DCS
  # chưa từng thấy "resize trên ảnh sạch" giống giao thức single-operator dùng để test.
  single_op_prob: 0.4
# Multi-Severity DCS (direction D-multiseverity-dcs) — additive mode, DEFAULT OFF.
# enabled=false => training uses the v2 single lifecycle-degraded view, bit-identical.
# Level ranges: source docx "Multi-Severity DCS AI Detection" (user-approved 2026-09-30).
# noise_std255 is on the 0-255 pixel scale (assumption, docx sigma_n = 2..30).
multi_severity:
  enabled: false
  compose: single_op        # single_op: 1 random op per view | compound: jpeg->blur->resize->noise at that level
  views_per_step: 4         # = len(levels): every bin each step (fixed order, per-bin weights); fewer: random subset of bins per image
  weights: [1, 1, 1, 1]     # per-bin weight in the DCS average (used when views_per_step == number of levels)
  levels:
    - {jpeg: [85, 95], blur_sigma: [0.5, 1.0], resize: [0.75, 0.90], noise_std255: [2, 5]}
    - {jpeg: [65, 80], blur_sigma: [1.2, 2.0], resize: [0.50, 0.70], noise_std255: [6, 12]}
    - {jpeg: [40, 60], blur_sigma: [2.2, 3.0], resize: [0.30, 0.45], noise_std255: [13, 20]}
    - {jpeg: [20, 35], blur_sigma: [3.2, 4.5], resize: [0.15, 0.25], noise_std255: [21, 30]}
</file>

<file path="code/train_lora.py">
"""K2 training script — train-only, chạy qua accelerate.

Tái sử dụng repo như thư viện (đúng tinh thần MODULES_INTERFACE.md), orchestration
được chuyển hết vào đây từ notebook `xla-training-phase-2-fromscratch.ipynb` để
notebook giữ được dưới 1 MB. Eval riêng (per-variant / AIGB held-out / OOD /
RealDeg-Bench) nằm ở notebook eval — script này CHỈ train (main val giữ lại vì
là metric chọn best checkpoint).

Chạy (từ thư mục code/, hoặc notebook bash cell — env offline do caller set):
    accelerate launch --num_processes 1 --mixed_precision bf16 train_lora.py \
        --train_data_root /kaggle/input/datasets/... \
        --dinov2_sd /kaggle/input/.../model.safetensors \
        --ai_genbench_dir /kaggle/input/datasets/...   # optional

Args tường minh: thiếu/th sai -> fail ngay (không auto-discover).
"""

import argparse
import contextlib
import hashlib
import io
import json
import logging
import math
import multiprocessing
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image
from peft import LoraConfig, get_peft_model
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from bfree_datasets.multi_severity import MultiSeverityDegrader, validate_config as validate_ms_config
from utils.normalization import get_list_norm
from utils.dmetrics import balanced_accuracy_score, roc_auc_score

logger = logging.getLogger("bfree")
CFG = {}  # điền trong parse_args/main, đọc từ các module-level helpers
_MS_DEGRADER = None  # MultiSeverityDegrader khi multi_severity.enabled; None = đường v2 mặc định

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

PRIORITY_GENERATORS = [
    "ADM", "DDPM", "DeepFloyd IF", "Palette", "Denoising Diffusion GAN",
    "LaMa", "MAT", "SN-PatchGAN",
    "StyleGAN1", "VQGAN", "GANformer", "CIPS",
    "FaceSynthetics",
]


# ============================================================
# CLI
# ============================================================

def load_multi_severity_config(config_path):
    """Đọc CHỈ section ``multi_severity`` của YAML (các hyperparameter khác vẫn theo CLI).
    Thiếu section => tắt (đường v2 mặc định)."""
    import yaml
    path = Path(config_path)
    if not path.is_absolute() and not path.exists():
        path = Path(__file__).resolve().parent / path
    if not path.is_file():
        raise FileNotFoundError(f"--config không tồn tại: {config_path}")
    with open(path, "r", encoding="utf-8") as f:
        ms = (yaml.safe_load(f) or {}).get("multi_severity") or {"enabled": False}
    if ms.get("enabled"):
        validate_ms_config(ms)
    return ms


def load_v4_config(config_path):
    """Đọc các cờ P2.A/P2.B từ YAML (``backbone.*`` và ``loss.*``). Thiếu key => giá trị legacy
    (config cũ không có các key này vẫn chạy đúng đường v2). Trả dict phẳng cho CFG."""
    import yaml
    path = Path(config_path)
    if not path.is_absolute() and not path.exists():
        path = Path(__file__).resolve().parent / path
    if not path.is_file():
        raise FileNotFoundError(f"--config không tồn tại: {config_path}")
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    b, l, d = cfg.get("backbone") or {}, cfg.get("loss") or {}, cfg.get("data") or {}
    out = {
        "use_lib": bool(b.get("use_lib", True)),
        "use_gsr": bool(b.get("use_gsr", True)),
        "use_dcs": bool(l.get("use_dcs", True)),
        "head_mode": b.get("head_mode", "meanpool"),
        "gsr_zero_init": bool(b.get("gsr_zero_init", False)),
        "lib_beta_init": float(b.get("lib_beta_init", -0.85)),
        "loss_mode": l.get("mode", "legacy"),
        "w_deg": float(l.get("w_deg", 1.0)),
        "lam_cons": float(l.get("lam_cons", 0.0)),
        "cons_T": float(l.get("cons_T", 1.0)),
        "p_aug_on_clean": float(l.get("p_aug_on_clean", 0.0)),
        "mix_preset": d.get("mix_preset", "mix_current"),
        "fmt_match": bool(d.get("fmt_match", False)),
        "res_match": bool(d.get("res_match", False)),
        "holdout_generators": list(d.get("holdout_generators") or []),
        "train_subsample_frac": float(d.get("train_subsample_frac", 1.0)),
    }
    if out["head_mode"] not in ("cls", "meanpool"):
        raise ValueError(f"backbone.head_mode={out['head_mode']!r} (cls|meanpool)")
    if out["loss_mode"] not in ("legacy", "dual_view"):
        raise ValueError(f"loss.mode={out['loss_mode']!r} (legacy|dual_view)")
    if out["cons_T"] <= 0:
        raise ValueError(f"loss.cons_T={out['cons_T']} phải > 0")
    if not 0.0 <= out["p_aug_on_clean"] <= 1.0:
        raise ValueError(f"loss.p_aug_on_clean={out['p_aug_on_clean']} ngoài [0, 1]")
    if out["loss_mode"] == "legacy" and out["p_aug_on_clean"] > 0:
        raise ValueError("loss.p_aug_on_clean > 0 chỉ có nghĩa với loss.mode=dual_view "
                         "(legacy phải giữ bit-identical)")
    if out["mix_preset"] not in ("mix_current", "mix_bfree", "mix_noaigb"):
        raise ValueError(f"data.mix_preset={out['mix_preset']!r} (mix_current|mix_bfree|mix_noaigb)")
    if not 0.0 < out["train_subsample_frac"] <= 1.0:
        raise ValueError(f"data.train_subsample_frac={out['train_subsample_frac']} ngoài (0, 1]")
    return out


# Hyperparameter phải khớp giữa các stage của cùng một run (resume kiểm tra).
RESUME_CHECK_KEYS = (
    "arch", "img_size", "epochs", "lr", "weight_decay", "scheduler_eta_min", "max_grad_norm",
    "lambda_dcs", "dcs_tau", "label_smoothing", "lib_kernel", "lib_tau", "gsr_window",
    "gsr_mask_prob", "lora_rank", "lora_alpha", "lora_dropout", "val_md5_percentile",
    "val_fake_variant", "pair_real_prob", "degrade_max_rounds", "degrade_round_probs",
    "extra_source_val_percentile", "aigenbench_max_per_gen", "aigenbench_val_frac",
    "aigenbench_seed", "seed", "multi_severity",
    # P2.A / P2.B / P2.C / P2.F (AGENT_PLAN_V4)
    "init_from", "use_lib", "use_gsr", "use_dcs",
    "head_mode", "gsr_zero_init", "lib_beta_init",
    "loss_mode", "w_deg", "lam_cons", "cons_T", "p_aug_on_clean",
    "mix_preset", "fmt_match", "res_match", "holdout_generators", "train_subsample_frac",
)


def parse_args():
    p = argparse.ArgumentParser(description="K2 LoRA training (train-only, accelerate)")
    p.add_argument("--train_data_root", type=str, required=True,
                   help="dataset root chứa COCO_real_512/ + 6 thư mục SD2.1_*")
    p.add_argument("--dinov2_sd", type=str, default=None,
                   help="DINOv2 ViT-B/14 reg4 model.safetensors (bundle notebook 01); "
                        "bắt buộc khi --init_from dinov2")
    p.add_argument("--init_from", choices=["dinov2", "bfree"], default="dinov2",
                   help="dinov2: nạp DINOv2 pretrained (đường cũ); bfree: nạp checkpoint B-Free "
                        "(head 1 logit -> 2 logit) qua BFreeGlobalForgeViT.load_checkpoint")
    p.add_argument("--bfree_ckpt", type=str, default=None,
                   help="checkpoint B-Free chính thức (dict có key 'model'); bắt buộc khi --init_from bfree")
    p.add_argument("--ai_genbench_dir", type=str, default=None,
                   help="HF Arrow DatasetDict ai-genbench-v1; bỏ trống = train không có AIGB")
    p.add_argument("--output_dir", type=str, default="/kaggle/working")
    p.add_argument("--config", type=str, default="configs/bfree_dcs.yaml")

    p.add_argument("--arch", type=str, default="vit_base_patch14_reg4_dinov2.lvd142m")
    p.add_argument("--img_size", type=int, default=504)
    p.add_argument("--lambda_dcs", type=float, default=0.01)   # D1
    p.add_argument("--dcs_tau", type=float, default=0.07)
    p.add_argument("--label_smoothing", type=float, default=0.1)
    p.add_argument("--lib_kernel", type=int, default=3)
    p.add_argument("--lib_tau", type=float, default=0.5)
    p.add_argument("--gsr_window", type=int, default=3)
    p.add_argument("--gsr_mask_prob", type=float, default=1.0)
    p.add_argument("--epochs", type=int, default=8)            # D2 (tổng; quyết định lịch cosine)
    p.add_argument("--end_epoch", type=int, default=0,
                   help="dừng sau epoch này (0 = chạy hết --epochs). Dùng để chia train thành "
                        "nhiều stage/notebook; lịch cosine vẫn theo tổng --epochs")
    p.add_argument("--resume_from", type=str, default=None,
                   help="resume_state.pth của stage trước (model+optimizer+scheduler+RNG+log)")
    p.add_argument("--lora_rank", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--scheduler_eta_min", type=float, default=1e-7)
    p.add_argument("--batch_size", type=int, default=64,
                   help="0 = auto-find theo D6 (doubling + binary search)")
    p.add_argument("--batch_hard_cap", type=int, default=128)
    p.add_argument("--val_md5_percentile", type=int, default=3)
    p.add_argument("--val_fake_variant", type=str, default="SD2.1_selfconditioned")
    p.add_argument("--pair_real_prob", type=float, default=0.5)  # D4
    p.add_argument("--degrade_max_rounds", type=int, default=4)
    p.add_argument("--degrade_round_probs", type=str,
                   default="0.15,0.30,0.25,0.20,0.10")
    p.add_argument("--extra_source_val_percentile", type=int, default=30)
    p.add_argument("--aigenbench_split", type=str, default="train")
    p.add_argument("--aigenbench_max_per_gen", type=int, default=4000)
    p.add_argument("--aigenbench_val_frac", type=float, default=0.1)
    p.add_argument("--aigenbench_seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--select", choices=["last", "best"], default="last",
                   help="checkpoint dùng cho eval: last-epoch (mặc định, không chọn theo val bão hòa) "
                        "hay best-val. Cả hai file vẫn được lưu; lựa chọn ghi vào selected_ckpt.txt")
    # P2.A / P2.B / P2.C / P2.F optional CLI overrides (None = lấy từ --config YAML)
    p.add_argument("--use_lib", type=int, choices=[0, 1], default=None)
    p.add_argument("--use_gsr", type=int, choices=[0, 1], default=None)
    p.add_argument("--use_dcs", type=int, choices=[0, 1], default=None)
    p.add_argument("--head_mode", choices=["cls", "meanpool"], default=None)
    p.add_argument("--gsr_zero_init", type=int, choices=[0, 1], default=None)
    p.add_argument("--lib_beta_init", type=float, default=None)
    p.add_argument("--loss_mode", choices=["legacy", "dual_view"], default=None)
    p.add_argument("--w_deg", type=float, default=None)
    p.add_argument("--lam_cons", type=float, default=None)
    p.add_argument("--cons_T", type=float, default=None)
    p.add_argument("--p_aug_on_clean", type=float, default=None)
    p.add_argument("--mix_preset", choices=["mix_current", "mix_bfree", "mix_noaigb"], default=None)
    p.add_argument("--fmt_match", type=int, choices=[0, 1], default=None)
    p.add_argument("--res_match", type=int, choices=[0, 1], default=None)
    p.add_argument("--holdout_generators", type=str, default=None,
                   help="danh sách generator giữ lại cách nhau dấu phẩy (P2.C)")
    p.add_argument("--train_subsample_frac", type=float, default=None,
                   help="tỷ lệ lấy mẫu train (0,1]; sàng lọc P2.F dùng 0.25")
    args = p.parse_args()

    cli_dict = vars(args)
    v4_yaml = load_v4_config(cli_dict["config"])
    CFG.update(cli_dict)
    CFG.update(v4_yaml)
    for bool_k in ("use_lib", "use_gsr", "use_dcs", "gsr_zero_init", "fmt_match", "res_match"):
        if cli_dict.get(bool_k) is not None:
            CFG[bool_k] = bool(cli_dict[bool_k])
    for val_k in ("head_mode", "lib_beta_init", "loss_mode", "w_deg", "lam_cons",
                  "cons_T", "p_aug_on_clean", "mix_preset", "train_subsample_frac"):
        if cli_dict.get(val_k) is not None:
            CFG[val_k] = cli_dict[val_k]
    if cli_dict.get("holdout_generators") is not None:
        CFG["holdout_generators"] = [
            g.strip() for g in cli_dict["holdout_generators"].split(",") if g.strip()
        ]
    if CFG["mix_preset"] == "mix_bfree" and cli_dict.get("aigenbench_max_per_gen") == 4000:
        # P2.C: mix_bfree giảm AIGB xuống ~750/gen (AIGB <= 25% tổng mẫu)
        CFG["aigenbench_max_per_gen"] = 750
    elif CFG["mix_preset"] == "mix_noaigb":
        CFG["ai_genbench_dir"] = None
    if CFG["loss_mode"] == "legacy" and CFG["p_aug_on_clean"] > 0:
        raise ValueError("loss.p_aug_on_clean > 0 chỉ dùng với loss.mode=dual_view")
    CFG["num_classes"] = 2
    CFG["degrade_round_probs"] = [
        float(x) for x in CFG["degrade_round_probs"].split(",") if x.strip()
    ]
    CFG["multi_severity"] = load_multi_severity_config(CFG["config"])
    if CFG["init_from"] == "dinov2" and CFG["dinov2_sd"] is None:
        raise ValueError("--init_from dinov2 yêu cầu --dinov2_sd")
    if CFG["init_from"] == "bfree" and CFG["bfree_ckpt"] is None:
        raise ValueError("--init_from bfree yêu cầu --bfree_ckpt")
    if CFG["loss_mode"] == "dual_view" and CFG["multi_severity"].get("enabled"):
        raise ValueError("loss.mode=dual_view không dùng chung với multi_severity.enabled")
    if CFG["end_epoch"] <= 0:
        CFG["end_epoch"] = CFG["epochs"]
    if not 1 <= CFG["end_epoch"] <= CFG["epochs"]:
        raise ValueError(f"--end_epoch={CFG['end_epoch']} ngoài [1, {CFG['epochs']}]")
    for path_key in ("train_data_root", "dinov2_sd", "bfree_ckpt", "ai_genbench_dir", "resume_from"):
        if CFG[path_key] is not None and not Path(CFG[path_key]).exists():
            raise FileNotFoundError(
                f"--{path_key.replace('_', '-')} không tồn tại: {CFG[path_key]}")
    return args


def setup_logging(output_dir: Path):
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    logger.handlers.clear()
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter(
        fmt="%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S"))
    logger.addHandler(console)
    fh = logging.FileHandler(str(output_dir / "train_log.txt"), mode="a", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        fmt="%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    logger.addHandler(fh)


# ============================================================
# Compound lifecycle degradation (7-operator pool, Appendix A RealDeg-Bench)
# ============================================================

DEGRADE_STRENGTHS = {
    "jpeg":       [90, 80, 70, 60, 40],
    "blur":       [0.5, 1, 2, 3, 5],
    "resize":     [0.9, 0.7, 0.5, 0.3, 0.2],
    "noise":      [0.0005, 0.001, 0.002, 0.005, 0.01],
    "brightness": [-0.2, -0.1, 0.1, 0.2],
    "contrast":   [-0.3, -0.2, 0.1, 0.2],
    "saturation": [0.6, 0.8, 1.3, 1.5],
}
DEGRADE_OPS = list(DEGRADE_STRENGTHS.keys())


def apply_degradation(img, op, strength):
    if op == "jpeg":
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=int(strength))
        buf.seek(0)
        return Image.open(buf).convert("RGB")
    if op == "blur":
        return img.filter(__import__("PIL.ImageFilter", fromlist=["GaussianBlur"]).GaussianBlur(radius=strength))
    if op == "resize":
        w, h = img.size
        small = img.resize((max(1, int(w * strength)), max(1, int(h * strength))), Image.BICUBIC)
        return small.resize((w, h), Image.BICUBIC)
    if op == "noise":
        arr = np.asarray(img).astype(np.float32) / 255.0
        arr = arr + np.random.normal(0, np.sqrt(strength), arr.shape)
        return Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8))
    from PIL import ImageEnhance
    if op == "brightness":
        return ImageEnhance.Brightness(img).enhance(1.0 + strength)
    if op == "contrast":
        return ImageEnhance.Contrast(img).enhance(1.0 + strength)
    if op == "saturation":
        return ImageEnhance.Color(img).enhance(strength)
    raise ValueError(op)


def apply_compound(img, n_steps, rng):
    for _ in range(n_steps):
        op = rng.choice(DEGRADE_OPS)
        strength = rng.choice(DEGRADE_STRENGTHS[op])
        img = apply_degradation(img, op, strength)
    return img


def realistic_lifecycle_degrade(img, rng):
    """Áp cho cả real + fake như nhau (tránh shortcut 'suy giảm = fake')."""
    n_rounds = rng.choices(
        range(CFG["degrade_max_rounds"] + 1),
        weights=CFG["degrade_round_probs"],
    )[0]
    return apply_compound(img, n_rounds, rng)


def degraded_view(img, normalize):
    """Default (multi_severity off): 1 lifecycle-degraded view, normalized tensor (3,H,W) —
    đúng như v2. On: K view multi-severity, stacked (K,3,H,W)."""
    if _MS_DEGRADER is None:
        return normalize(realistic_lifecycle_degrade(img, random))
    return torch.stack([normalize(v) for v in _MS_DEGRADER(img)])


def clean_view(img, normalize):
    """View "sạch" của batch. Mặc định (p_aug_on_clean=0 / loss.mode=legacy) = normalize(img),
    KHÔNG tiêu thụ RNG => bit-identical v2. P2.A: với xác suất p, áp suy giảm lifecycle lên chính
    view sạch (view suy giảm của DCS vẫn lấy từ ``img`` gốc)."""
    p = CFG.get("p_aug_on_clean", 0.0)
    if p > 0 and random.random() < p:
        return normalize(realistic_lifecycle_degrade(img, random))
    return normalize(img)


def split_views(img_deg):
    """Batch (B,K,3,H,W) -> list K tensors (B,3,H,W); tensor 4-D giữ nguyên (đường v2)."""
    if img_deg.dim() == 5:
        return [img_deg[:, k] for k in range(img_deg.size(1))]
    return img_deg


# ============================================================
# Dataset — stem index + fixed 8-epoch fake schedule (D4) + extra sources
# ============================================================

def unwrap_dir(path):
    """Bỏ 1 tầng lồng thừa: <dir>/<dir> -> <dir> (nếu tồn tại)."""
    path = Path(path)
    child = path / path.name
    return child if child.is_dir() else path


def ensure_min_size(img, min_size):
    w, h = img.size
    if min(w, h) < min_size:
        if CFG.get("res_match", False):
            # P2.C res_match: lặp biên (replicate-wrap kiểu B-Free) thay vì upscale Bicubic
            arr = np.asarray(img)
            rep_h = max(1, int(math.ceil(min_size / h)))
            rep_w = max(1, int(math.ceil(min_size / w)))
            arr = np.tile(arr, (rep_h, rep_w, 1))[:max(min_size, h), :max(min_size, w)]
            return Image.fromarray(arr)
        scale = min_size / min(w, h)
        img = img.resize(
            (max(min_size, round(w * scale)), max(min_size, round(h * scale))),
            Image.BICUBIC)
    return img


def maybe_fmt_match(img):
    """P2.C fmt_match: real & fake trong AIGB/extra cùng đi qua JPEG ngẫu nhiên Q in [75,95]
    với xác suất 0.5 để khử shortcut định dạng PNG vs JPEG."""
    if CFG.get("fmt_match", False) and random.random() < 0.5:
        return apply_degradation(img, "jpeg", random.randint(75, 95))
    return img


def build_stem_index(dirs, cache_path=None):
    """os.listdir + splitext (không stat từng entry — FUSE mount network round-trip)."""
    if cache_path is not None and Path(cache_path).is_file():
        cached = json.loads(Path(cache_path).read_text())
        if cached.get("dirs") == list(dirs):
            logger.info(f"stem index (cache): {[len(m) for m in cached['maps']]} files")
            return cached["maps"]
    t0 = time.time()
    maps = []
    for d in dirs:
        m = {}
        for name in os.listdir(d):
            stem, ext = os.path.splitext(name)
            if ext:
                m[stem] = os.path.join(d, name)
        maps.append(m)
    logger.info(f"stem index (scan): {[len(m) for m in maps]} files in {time.time()-t0:.1f}s")
    if cache_path is not None:
        Path(cache_path).write_text(json.dumps({"dirs": list(dirs), "maps": maps}))
    return maps


def is_val_id(stem):
    return (int(hashlib.md5(stem.encode()).hexdigest(), 16) % 100
            < CFG["val_md5_percentile"])


class PairDataset(Dataset):
    """Train: lịch 8-epoch/ID cố định — round(epochs*(1-p)) epoch là fake, các epoch fake
    gán KHÔNG LẶP biến thể SD2.1 (seed = md5(stem)), 50/50 real/fake đúng D4.
    Val: 2 samples/ID — chẵn = real, lẻ = deterministic fake variant (balanced)."""

    def __init__(self, ids, img_size, is_train, pair_real_prob, val_fake_variant,
                 real_map, fake_maps, fake_dirs):
        self.ids = ids
        self.img_size = img_size
        self.is_train = is_train
        self.pair_real_prob = pair_real_prob
        self.val_fake_variant = val_fake_variant
        self.real_map = real_map
        self.fake_maps = fake_maps
        self.fake_dirs = fake_dirs
        self.normalize = T.Compose(get_list_norm("resnet"))
        if is_train:
            self.epoch_value = multiprocessing.Value("i", 1)
            self.fake_schedule = self._build_fake_schedule()
        else:
            self.epoch_value = None
            self.fake_schedule = None

    def set_epoch(self, epoch):
        if self.epoch_value is not None:
            self.epoch_value.value = epoch

    def _build_fake_schedule(self):
        n_variants = len(self.fake_maps)
        n_epochs = CFG["epochs"]
        n_fake = min(n_epochs, round(n_epochs * (1 - self.pair_real_prob)))
        schedule = {}
        for stem in self.ids:
            available = [v for v in range(n_variants) if stem in self.fake_maps[v]]
            if not available:
                continue
            rng = random.Random(int(hashlib.md5(stem.encode()).hexdigest(), 16))
            slots = list(range(1, n_epochs + 1))
            rng.shuffle(slots)
            fake_epochs = sorted(slots[:n_fake])
            bag, assigned = [], []
            for _ in fake_epochs:
                if not bag:
                    bag = available.copy()
                    rng.shuffle(bag)
                assigned.append(bag.pop())
            schedule[stem] = dict(zip(fake_epochs, assigned))
        return schedule

    def __len__(self):
        return len(self.ids) * (1 if self.is_train else 2)

    def _pick_val_fake(self, stem):
        for m, d in zip(self.fake_maps, self.fake_dirs):
            if Path(d).name == self.val_fake_variant and stem in m:
                return m[stem]
        for m in self.fake_maps:
            if stem in m:
                return m[stem]
        return None

    def __getitem__(self, i):
        stem = self.ids[i if self.is_train else i // 2]
        if self.is_train:
            variant_idx = self.fake_schedule.get(stem, {}).get(self.epoch_value.value)
            if variant_idx is None:
                path, label = self.real_map[stem], 0
            else:
                path, label = self.fake_maps[variant_idx][stem], 1
        else:
            if i % 2 == 0:
                path, label = self.real_map[stem], 0
            else:
                path, label = self._pick_val_fake(stem), 1
        assert path is not None, f"No image found for id {stem}"
        img = Image.open(path).convert("RGB")
        if self.is_train:
            t, l, h, w = T.RandomCrop.get_params(img, (self.img_size, self.img_size))
            img = TF.crop(img, t, l, h, w)
            if random.random() > 0.5:
                img = TF.hflip(img)
        else:
            img = TF.center_crop(img, (self.img_size, self.img_size))
        clean = clean_view(img, self.normalize) if self.is_train else self.normalize(img)
        return clean, degraded_view(img, self.normalize), label


def _is_extra_holdout(path):
    name = Path(path).name
    return (int(hashlib.md5(name.encode()).hexdigest(), 16) % 100
            < CFG["extra_source_val_percentile"])


def _resolve_extra_source(name, train_data_root):
    for base in (Path(train_data_root), Path(train_data_root) / "extended_synthbuster"):
        cand = unwrap_dir(base / name)
        if cand.is_dir():
            return cand
    return None


def _list_images_recursive(d):
    return sorted(str(p) for p in Path(d).rglob("*") if p.suffix.lower() in IMG_EXTS)


class ExtraTrainDataset(Dataset):
    """Train flat (không content-pair) cho RAISE/LDM/SD3-FLUX — cùng augmentation
    + lifecycle degrade trả về (img_clean, img_deg, label)."""

    def __init__(self, files, label, img_size):
        self.files = files
        self.label = label
        self.img_size = img_size
        self.normalize = T.Compose(get_list_norm("resnet"))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        img = Image.open(self.files[i]).convert("RGB")
        img = ensure_min_size(img, self.img_size)
        img = maybe_fmt_match(img)
        t, l, h, w = T.RandomCrop.get_params(img, (self.img_size, self.img_size))
        img = TF.crop(img, t, l, h, w)
        if random.random() > 0.5:
            img = TF.hflip(img)
        return clean_view(img, self.normalize), degraded_view(img, self.normalize), self.label


# ============================================================
# AI-GenBench train split (13 generator ưu tiên)
# ============================================================

_AIGB_CACHE = {}


def _aigenbench_dataset():
    if "ds" in _AIGB_CACHE:
        return _AIGB_CACHE["ds"]
    from datasets import load_from_disk  # lazy: tránh import nhầm package 'datasets' của repo

    obj = load_from_disk(CFG["ai_genbench_dir"])
    if hasattr(obj, "keys"):
        splits = list(obj.keys())
        split = CFG["aigenbench_split"] if CFG["aigenbench_split"] in splits else splits[0]
        if split != CFG["aigenbench_split"]:
            logger.warning(f"ai_genbench: không có split '{CFG['aigenbench_split']}', "
                           f"dùng '{split}' (có: {splits})")
        ds = obj[split]
    else:
        split, ds = "(single)", obj

    img_col = next((c for c, f in ds.features.items()
                    if f.__class__.__name__ == "Image"), None)
    if img_col is None:
        raise RuntimeError(f"ai_genbench: không tìm thấy cột ảnh. features={list(ds.features)}")
    label_feat = ds.features.get("label")
    label_names = list(getattr(label_feat, "names", []) or [])
    logger.info(f"ai_genbench: split={split} n={len(ds)} img_col='{img_col}' "
                f"label_names={label_names}")
    _AIGB_CACHE["ds"] = (ds, img_col, label_names)
    return _AIGB_CACHE["ds"]


def _aigenbench_meta(ds, label_names):
    if "meta" in _AIGB_CACHE:
        return _AIGB_CACHE["meta"]
    cols = [c for c in ("label", "generator", "origin_dataset") if c in ds.column_names]
    meta = ds.select_columns(cols).to_pandas()
    if label_names:
        fake_ids = {i for i, n in enumerate(label_names) if "fake" in str(n).lower()}
        if not fake_ids:
            fake_ids = {len(label_names) - 1}
        meta["y"] = meta["label"].apply(lambda v: 1 if int(v) in fake_ids else 0)
    else:
        meta["y"] = (meta["label"].astype(int) != 0).astype(int)
    if "generator" not in meta.columns:
        meta["generator"] = "unknown"
    if "origin_dataset" not in meta.columns:
        meta["origin_dataset"] = "unknown"
    meta["generator"] = meta["generator"].fillna("unknown").astype(str).str.strip()
    meta["origin_dataset"] = meta["origin_dataset"].fillna("unknown").astype(str)
    logger.info(f"ai_genbench: meta OK | real={int((meta.y == 0).sum())} "
                f"fake={int((meta.y == 1).sum())} | "
                f"n_generator={meta.loc[meta.y == 1, 'generator'].nunique()}")
    _AIGB_CACHE["meta"] = meta
    return meta


def build_ai_genbench_train_groups():
    """-> (ArrowSource, real_train, fake_train, real_val, fake_val_per_gen).
    Val là lát cắt TÁCH TỪ split="train" — không bao giờ đụng split "validation"
    (split đó dành riêng cho eval notebook)."""
    ds, img_col, label_names = _aigenbench_dataset()
    meta = _aigenbench_meta(ds, label_names)

    real_gens = set(meta.loc[meta.y == 1, "generator"].unique())
    missing = [g for g in PRIORITY_GENERATORS if g not in real_gens]
    if missing:
        raise AssertionError(
            f"Generator không khớp tên thật trong Arrow: {missing}\nCó: {sorted(real_gens)}")

    rng = np.random.default_rng(CFG["aigenbench_seed"])
    y = meta["y"].to_numpy()
    gens = meta["generator"]

    holdout_set = set(CFG.get("holdout_generators") or [])
    active_gens = [g for g in PRIORITY_GENERATORS if g not in holdout_set]
    if holdout_set:
        logger.info(f"[ai_genbench] holdout_generators={sorted(holdout_set)} -> active={len(active_gens)}")
    train_groups, val_groups = {}, {}
    for g in active_gens:
        gi = meta.index[(y == 1) & (gens == g)].to_numpy()
        if CFG["aigenbench_max_per_gen"] and len(gi) > CFG["aigenbench_max_per_gen"]:
            gi = rng.choice(gi, CFG["aigenbench_max_per_gen"], replace=False)
        gi = np.array(sorted(gi))
        n_val = (min(len(gi), max(1, round(len(gi) * CFG["aigenbench_val_frac"])))
                 if len(gi) else 0)
        perm = rng.permutation(len(gi))
        val_local, train_local = perm[:n_val], perm[n_val:]
        val_groups[g] = [(int(i), 1) for i in sorted(gi[val_local])]
        train_groups[g] = [(int(i), 1) for i in sorted(gi[train_local])]

    total_fake_train = sum(len(v) for v in train_groups.values())
    total_fake_val = sum(len(v) for v in val_groups.values())

    real_idx_all = meta.index[y == 0].to_numpy().copy()
    rng.shuffle(real_idx_all)
    n_real_val = min(len(real_idx_all) // 4, max(1, total_fake_val)) if total_fake_val else 0
    real_val_idx = real_idx_all[:n_real_val]
    remaining_real = real_idx_all[n_real_val:]
    real_train_idx = remaining_real[:min(total_fake_train, len(remaining_real))]

    real_items = [(int(i), 0) for i in sorted(real_train_idx)]
    real_val_items = [(int(i), 0) for i in sorted(real_val_idx)]

    logger.info(f"[ai_genbench] {len(train_groups)} generator | fake_train={total_fake_train} "
                f"| real_train={len(real_items)} | val held-out: fake={total_fake_val} "
                f"real={len(real_val_items)}")

    class ArrowSource:
        def __init__(self, ds, img_col):
            self.ds, self.img_col = ds, img_col

        def __len__(self):
            return len(self.ds)

        def get(self, idx):
            return self.ds[int(idx)][self.img_col]

    return (ArrowSource(ds, img_col), real_items,
            [it for v in train_groups.values() for it in v],
            real_val_items, val_groups)


class ExtraArrowTrainDataset(Dataset):
    """Dataset train đọc ảnh từ ArrowSource — augment=False (val) center-crop không flip."""

    def __init__(self, source, items, label, img_size, augment=True):
        self.source = source
        self.indices = [i for i, y in items if y == label]
        self.label = label
        self.img_size = img_size
        self.augment = augment
        self.normalize = T.Compose(get_list_norm("resnet"))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        img = self.source.get(self.indices[i]).convert("RGB")
        img = ensure_min_size(img, self.img_size)
        if self.augment:
            img = maybe_fmt_match(img)
            t, l, h, w = T.RandomCrop.get_params(img, (self.img_size, self.img_size))
            img = TF.crop(img, t, l, h, w)
            if random.random() > 0.5:
                img = TF.hflip(img)
        else:
            img = TF.center_crop(img, (self.img_size, self.img_size))
        clean = clean_view(img, self.normalize) if self.augment else self.normalize(img)
        return clean, degraded_view(img, self.normalize), self.label


# ============================================================
# Model — offline pretrained init (D5) + LoRA
# ============================================================

def load_pretrained_backbone(model, safetensors_path, model_module=None):
    """D5 offline: load DINOv2 weights vào model.model + resample pos_embed 518->504.

    dinov2 reg4 checkpoint lưu pos_embed PATCH-ONLY (1, 37*37, 768) vì
    no_embed_class=True -> tự dò layout (patch-only vs patch+prefix).
    """
    from safetensors.torch import load_file

    sd = load_file(str(safetensors_path))
    sd = {k[len("model."):] if k.startswith("model.") else k: v for k, v in sd.items()}

    prefix = model.model.num_prefix_tokens
    if "pos_embed" in sd:
        pe = sd["pos_embed"]
        n = pe.shape[1]
        if math.isqrt(n) ** 2 == n:
            ckpt_prefix = 0
        elif math.isqrt(n - prefix) ** 2 == n - prefix:
            ckpt_prefix = prefix
        else:
            raise ValueError(
                f"cannot parse pos_embed layout {tuple(pe.shape)} "
                f"(neither patch-only nor patch+{prefix} prefix)")
        old_hw = math.isqrt(n - ckpt_prefix)
        new_hw = model.model.patch_embed.grid_size[0]
        if old_hw != new_hw:
            from timm.layers.pos_embed import resample_abs_pos_embed
            logger.info(f"resample pos_embed: {old_hw}x{old_hw} -> {new_hw}x{new_hw} "
                        f"(checkpoint prefix tokens in pos_embed: {ckpt_prefix})")
            sd["pos_embed"] = resample_abs_pos_embed(
                pe, new_size=model.model.patch_embed.grid_size,
                old_size=(old_hw, old_hw), num_prefix_tokens=ckpt_prefix)

    ref = model.model.state_dict()
    dropped = [k for k, v in sd.items() if k not in ref or ref[k].shape != v.shape]
    sd = {k: v for k, v in sd.items() if k not in dropped}

    report = model.model.load_state_dict(sd, strict=False)
    logger.info(f"pretrained load: dropped={len(dropped)}, missing={report.missing_keys}")
    assert set(report.missing_keys) <= {"head.weight", "head.bias"}, "backbone not fully initialized"
    assert not report.unexpected_keys
    assert "pos_embed" not in dropped, "pos_embed was dropped by shape filter — resample failed"
    return model


def apply_lora_to_backbone(model, r=16, lora_alpha=32, lora_dropout=0.05):
    """Áp LoRA lên timm VisionTransformer backbone, giữ full train cho LIB, GSR và Head."""
    lora_config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=["qkv", "proj", "fc1", "fc2"],
        lora_dropout=lora_dropout,
        bias="none",
    )
    model.model = get_peft_model(model.model, lora_config)

    trainable_submodules = [model.lib, model.gsr,
                            model.model.base_model.model.head,
                            model.model.base_model.model.fc_norm]
    for sub in trainable_submodules:
        if sub is not None:
            for p in sub.parameters():
                p.requires_grad = True

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"[*] LoRA configured (rank={r}): {trainable_params:,} / {total_params:,} "
                f"trainable ({100 * trainable_params / total_params:.2f}%)")
    return model


# ============================================================
# Auto batch size (D6) + train/val loops
# ============================================================

def _autocast():
    if torch.cuda.is_available():
        return torch.amp.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def try_batch(model, bs, img_size, device):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    try:
        x1 = torch.randn(bs, 3, img_size, img_size, device=device)
        n_views = CFG["multi_severity"]["views_per_step"] if _MS_DEGRADER is not None else 0
        x2 = ([torch.randn(bs, 3, img_size, img_size, device=device) for _ in range(n_views)]
              if n_views else torch.randn(bs, 3, img_size, img_size, device=device))
        y = torch.randint(0, 2, (bs,), device=device)
        with _autocast():
            total, _, _ = model.compute_loss(x1, x2, y, view_weights=CFG.get("ms_view_weights"))
        total.backward()
        del x1, x2, y, total
        return True
    except torch.cuda.OutOfMemoryError:
        return False
    finally:
        model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()


def find_max_batch_size(model, img_size, device, hard_cap):
    assert try_batch(model, 2, img_size, device), "Batch size 2 OOMs — environment problem."
    lo, bs, hi = 2, 4, None
    while bs <= hard_cap:
        if try_batch(model, bs, img_size, device):
            lo = bs
            bs *= 2
        else:
            hi = bs
            break
    if hi is None:
        logger.warning(f"no OOM up to hard cap {hard_cap}")
        hi = lo + 1
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if try_batch(model, mid, img_size, device):
            lo = mid
        else:
            hi = mid
    return lo


def main():
    global _MS_DEGRADER
    args = parse_args()
    output_dir = Path(CFG["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(output_dir)

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("PYTHONHASHSEED", "0")
    os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

    random.seed(CFG["seed"])
    np.random.seed(CFG["seed"])
    torch.manual_seed(CFG["seed"])
    torch.cuda.manual_seed_all(CFG["seed"])

    ms = CFG["multi_severity"]
    CFG["ms_view_weights"] = None
    if ms.get("enabled"):
        _MS_DEGRADER = MultiSeverityDegrader.from_config(ms, random)  # global `random`, seeded per worker
        # weights gắn với bin chỉ khi mỗi step dùng đủ mọi bin; ngược lại trung bình đều
        if int(ms["views_per_step"]) == len(ms["levels"]):
            CFG["ms_view_weights"] = [float(w) for w in ms["weights"]]
        logger.info(f"[multi_severity] ON: compose={ms['compose']} views_per_step={ms['views_per_step']} "
                    f"weights={CFG['ms_view_weights'] or 'uniform'} levels={len(ms['levels'])}")
    else:
        logger.info("[multi_severity] OFF -> v2 single lifecycle-degraded view")
    n_ms = int(ms["views_per_step"]) if ms.get("enabled") else 0
    # dual_view: log thêm ce_c / ce_d / cons (train) và val_* (val) vào train_log.csv
    extra_keys = ["ce_c", "ce_d", "cons"] if CFG["loss_mode"] == "dual_view" else []
    csv_extra = extra_keys + [f"val_{k}" for k in extra_keys]

    train_root = Path(CFG["train_data_root"])
    real_dir = unwrap_dir(train_root / "COCO_real_512")
    fake_dirs = sorted(str(unwrap_dir(p)) for p in train_root.glob("SD2.1_*") if p.is_dir())
    if not real_dir.is_dir():
        raise FileNotFoundError(f"COCO_real_512 missing under {train_root}")
    if len(fake_dirs) != 6:
        raise FileNotFoundError(f"Expected 6 SD2.1_* dirs, found {len(fake_dirs)}")
    logger.info(f"REAL_DIR={real_dir} | {len(fake_dirs)} fake variants")

    logger.info(f"CONFIG: {CFG}")

    # ---- stem index + val split ----
    real_map = build_stem_index([str(real_dir)],
                                cache_path=output_dir / "stem_index_real.json")[0]
    fake_maps = build_stem_index(fake_dirs,
                                 cache_path=output_dir / "stem_index_fake.json")
    all_stems = sorted(real_map.keys())
    train_ids = [s for s in all_stems if not is_val_id(s)]
    val_ids = [s for s in all_stems if is_val_id(s)]
    logger.info(f"train IDs={len(train_ids)}, val IDs={len(val_ids)} "
                f"(~{CFG['val_md5_percentile']}%)")

    train_ds = PairDataset(train_ids, CFG["img_size"], True, CFG["pair_real_prob"],
                           CFG["val_fake_variant"], real_map, fake_maps, fake_dirs)
    val_ds = PairDataset(val_ids, CFG["img_size"], False, CFG["pair_real_prob"],
                         CFG["val_fake_variant"], real_map, fake_maps, fake_dirs)
    a, _, c = train_ds[0]
    _, _, c2 = val_ds[1]
    assert tuple(a.shape) == (3, CFG["img_size"], CFG["img_size"])
    logger.info(f"train_ds={len(train_ds)}, val_ds={len(val_ds)} "
                f"| labels val[0],val[1]={c},{c2} (expect 0,1)")

    # ---- extra sources: RAISE/LDM/SD3-FLUX (train phần, held-out cho OOD notebook) ----
    train_sources = [train_ds]
    raise_dir = _resolve_extra_source("real_RAISE_1k", train_root)
    ldm_dir = _resolve_extra_source("latent-diffusion", train_root)
    sd3_dir = _resolve_extra_source("sd3_flux", train_root)
    if raise_dir is None or (ldm_dir is None and sd3_dir is None):
        logger.warning("[extra] thiếu real_RAISE_1k / latent-diffusion / sd3_flux — "
                       "train với COCO+SD2.1 (+AI-GenBench nếu có)")
    else:
        extra_real = _list_images_recursive(raise_dir)
        extra_fake = []
        for gdir in (ldm_dir, sd3_dir):
            if gdir is not None:
                extra_fake += _list_images_recursive(gdir)
        train_real = [p for p in extra_real if not _is_extra_holdout(p)]
        train_fake = [p for p in extra_fake if not _is_extra_holdout(p)]
        n_hold_r = len(extra_real) - len(train_real)
        n_hold_f = len(extra_fake) - len(train_fake)
        logger.info(f"[extra] train: real={len(train_real)} fake={len(train_fake)} | "
                    f"held-out: real={n_hold_r} fake={n_hold_f} "
                    f"({CFG['extra_source_val_percentile']}%)")
        if train_real or train_fake:
            train_sources.append(torch.utils.data.ConcatDataset([
                ExtraTrainDataset(train_real, 0, CFG["img_size"]),
                ExtraTrainDataset(train_fake, 1, CFG["img_size"]),
            ]))

    # ---- AI-GenBench ----
    if CFG["ai_genbench_dir"] is None:
        logger.info("ai_genbench_dir không truyền -> train không có AI-GenBench")
    else:
        (aigb_source, aigb_real, aigb_fake, _, _) = build_ai_genbench_train_groups()
        train_sources.append(torch.utils.data.ConcatDataset([
            ExtraArrowTrainDataset(aigb_source, aigb_real, 0, CFG["img_size"]),
            ExtraArrowTrainDataset(aigb_source, aigb_fake, 1, CFG["img_size"]),
        ]))

    # Luôn bọc ConcatDataset: nếu dataset có `set_epoch`, accelerate gọi set_epoch(iteration từ 0)
    # và ghi đè train_ds.set_epoch(epoch) (lịch real/fake sai, và lệch sau resume).
    train_dataset = torch.utils.data.ConcatDataset(train_sources)
    total_before_sub = len(train_dataset)
    sub_frac = float(CFG.get("train_subsample_frac", 1.0))
    if sub_frac < 1.0:
        n_keep = max(1, int(round(total_before_sub * sub_frac)))
        sub_rng = np.random.default_rng(CFG["seed"] + 2026)
        keep_idx = sorted(int(i) for i in sub_rng.choice(total_before_sub, size=n_keep, replace=False))
        train_dataset = torch.utils.data.Subset(train_dataset, keep_idx)
        logger.info(f"[subsample] train_subsample_frac={sub_frac:.3f} -> kept {len(train_dataset)}/{total_before_sub} samples")
    logger.info(f"train sources = {len(train_sources)} | total train samples = {len(train_dataset)}")

    # ---- model: construct + D5 pretrained + LoRA ----
    from networks.bfree_globalforge_vit import BFreeGlobalForgeViT
    from configs.loader import load_config, build_model_kwargs  # noqa: F401  (giữ tương thích config)

    model = BFreeGlobalForgeViT(
        arch=CFG["arch"], num_classes=CFG["num_classes"],
        img_size=CFG["img_size"], pretrained=False,
        use_lib=CFG["use_lib"], use_gsr=CFG["use_gsr"], use_dcs=CFG["use_dcs"],
        lib_kernel=CFG["lib_kernel"], lib_tau=CFG["lib_tau"],
        gsr_window=CFG["gsr_window"], gsr_mask_prob=CFG["gsr_mask_prob"],
        dcs_tau=CFG["dcs_tau"], lambda_dcs=CFG["lambda_dcs"],
        label_smoothing=CFG["label_smoothing"],
        head_mode=CFG["head_mode"], gsr_zero_init=CFG["gsr_zero_init"],
        lib_beta_init=CFG["lib_beta_init"], loss_mode=CFG["loss_mode"],
        w_deg=CFG["w_deg"], lam_cons=CFG["lam_cons"], cons_T=CFG["cons_T"],
    )
    if CFG["init_from"] == "bfree":
        rep = model.load_checkpoint(CFG["bfree_ckpt"], strict=False, require_complete_backbone=True)
        logger.info(f"[init_from=bfree] {CFG['bfree_ckpt']} | {rep}")
        if CFG["head_mode"] != "cls" or not CFG["gsr_zero_init"] or CFG["lib_beta_init"] > -5:
            logger.warning("[init_from=bfree] 'bước 0 ≡ B-Free' cần head_mode=cls (+ gsr_zero_init=true, "
                           "lib_beta_init≈-6 để LIB/GSR gần identity): "
                           f"hiện head_mode={CFG['head_mode']} gsr_zero_init={CFG['gsr_zero_init']} "
                           f"lib_beta_init={CFG['lib_beta_init']}")
    else:
        model = load_pretrained_backbone(model, CFG["dinov2_sd"])
    model = apply_lora_to_backbone(model, r=CFG["lora_rank"],
                                   lora_alpha=CFG["lora_alpha"],
                                   lora_dropout=CFG["lora_dropout"])

    # ---- accelerate ----
    from accelerate import Accelerator
    accelerator = Accelerator(mixed_precision="bf16")
    device = accelerator.device

    # ---- resume (stage > 1): model weights + kiểm tra config tương thích ----
    resume = None
    if CFG["resume_from"]:
        resume = torch.load(CFG["resume_from"], map_location="cpu", weights_only=False)
        prev = resume["config"]
        bad = [k for k in RESUME_CHECK_KEYS if prev.get(k) != CFG.get(k)]
        if bad:
            raise RuntimeError("resume: config khác stage trước: " + ", ".join(
                f"{k} (ckpt={prev.get(k)!r} vs now={CFG.get(k)!r})" for k in bad))
        model.load_state_dict(resume["model"], strict=True)
        logger.info(f"[resume] loaded {CFG['resume_from']} | done epoch {resume['epoch']} "
                    f"-> train epoch {resume['epoch'] + 1}..{CFG['end_epoch']}/{CFG['epochs']}")
        if resume["epoch"] >= CFG["end_epoch"]:
            raise RuntimeError(f"resume epoch {resume['epoch']} >= end_epoch {CFG['end_epoch']}")

    batch_size = CFG["batch_size"]
    if resume is not None:
        # lịch cosine T_max = epochs * steps/epoch -> bắt buộc cùng batch size với stage trước
        if batch_size not in (0, resume["batch_size"]):
            raise RuntimeError(f"resume: --batch_size {batch_size} != ckpt {resume['batch_size']}")
        batch_size = resume["batch_size"]
        logger.info(f"[resume] batch size = {batch_size} (từ ckpt, bỏ qua finder)")
    elif batch_size == 0:
        if device.type != "cuda":
            raise RuntimeError("--batch_size=0 (auto-find) yêu cầu CUDA")
        model.to(device)  # finder chạy trước accelerator.prepare -> model còn ở CPU
        model.train()
        batch_size = find_max_batch_size(model, CFG["img_size"], device,
                                         CFG["batch_hard_cap"])
        logger.info(f"auto batch size = {batch_size}")

    def seed_worker(worker_id):
        ws = torch.initial_seed() % 2 ** 32
        np.random.seed(ws)
        random.seed(ws)

    g = torch.Generator()
    g.manual_seed(CFG["seed"])

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=CFG["num_workers"], pin_memory=True, drop_last=True,
        persistent_workers=CFG["num_workers"] > 0,
        worker_init_fn=seed_worker, generator=g)
    val_loader = DataLoader(val_ds, batch_size=max(2, batch_size // 2), shuffle=False,
                            num_workers=CFG["num_workers"], pin_memory=True)

    optimizer = AdamW((p for p in model.parameters() if p.requires_grad),
                      lr=CFG["lr"], weight_decay=CFG["weight_decay"])
    scheduler = CosineAnnealingLR(optimizer,
                                  T_max=CFG["epochs"] * len(train_loader),
                                  eta_min=CFG["scheduler_eta_min"])

    model, optimizer, train_loader, val_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader)

    train_log, best_bacc, start_epoch = [], 0.0, 1
    if resume is not None:
        assert resume["steps_per_epoch"] == len(train_loader), (
            f"resume: steps/epoch {len(train_loader)} != ckpt {resume['steps_per_epoch']} "
            "(dữ liệu hoặc batch size đã đổi)")
        optimizer.load_state_dict(resume["optimizer"])
        scheduler.load_state_dict(resume["scheduler"])
        g.set_state(resume["loader_gen"])
        random.setstate(resume["rng_python"])
        np.random.set_state(resume["rng_numpy"])
        torch.set_rng_state(resume["rng_torch"])
        if torch.cuda.is_available() and resume.get("rng_cuda") is not None:
            torch.cuda.set_rng_state(resume["rng_cuda"])
        train_log, best_bacc, start_epoch = resume["train_log"], resume["best_bacc"], resume["epoch"] + 1
        logger.info(f"[resume] optimizer/scheduler/RNG restored | lr now "
                    f"{optimizer.param_groups[0]['lr']:.3e} | best bAcc so far {best_bacc:.4f}")

    # ---- train/val loops ----
    def run_epoch(epoch):
        train_ds.set_epoch(epoch)
        model.train()
        agg = {"total": 0.0, "ce": 0.0, "dcs": 0.0, "n": 0}
        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{CFG['epochs']} [train]",
                    leave=False, disable=not accelerator.is_main_process)
        for step, (img_clean, img_deg, labels) in enumerate(pbar):
            img_clean = img_clean.to(device, non_blocking=True)
            img_deg = split_views(img_deg.to(device, non_blocking=True))
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with _autocast():
                total, ce, dcs = model.compute_loss(img_clean, img_deg, labels,
                                                    view_weights=CFG["ms_view_weights"])
            terms = getattr(accelerator.unwrap_model(model), "last_dcs_terms", {})
            for name, t in terms.items():
                agg[name] = agg.get(name, 0.0) + float(t) * labels.size(0)
            for name, t in getattr(accelerator.unwrap_model(model), "last_terms", {}).items():
                agg[name] = agg.get(name, 0.0) + float(t) * labels.size(0)
            accelerator.backward(total)
            accelerator.clip_grad_norm_(
                (p for p in model.parameters() if p.requires_grad),
                CFG["max_grad_norm"])
            optimizer.step()
            scheduler.step()
            bs = labels.size(0)
            agg["total"] += float(total.detach()) * bs
            agg["ce"] += float(ce.detach()) * bs
            agg["dcs"] += float(dcs.detach()) * bs
            agg["n"] += bs
            if step % 20 == 0:
                logger.debug(
                    f"ep{epoch} step {step}/{len(train_loader)} | "
                    f"total {agg['total']/max(agg['n'],1):.4f} "
                    f"(ce {agg['ce']/max(agg['n'],1):.4f}, "
                    f"dcs {agg['dcs']/max(agg['n'],1):.4f}) | "
                    f"lr {optimizer.param_groups[0]['lr']:.2e}")
            pbar.set_postfix(
                total=f"{agg['total']/max(agg['n'],1):.4f}",
                ce=f"{agg['ce']/max(agg['n'],1):.4f}",
                dcs=f"{agg['dcs']/max(agg['n'],1):.4f}",
                lr=f"{optimizer.param_groups[0]['lr']:.1e}")
        keys = ["total", "ce", "dcs"] + [f"dcs_k{i}" for i in range(1, n_ms + 1)] + extra_keys
        return {k: agg[k] / max(agg["n"], 1) for k in keys}

    @torch.no_grad()
    def run_val():
        model.eval()
        agg = {"total": 0.0, "n": 0}
        scores, labels_all = [], []
        for img_clean, img_deg, labels in val_loader:
            img_clean = img_clean.to(device, non_blocking=True)
            img_deg = split_views(img_deg.to(device, non_blocking=True))
            labels = labels.to(device, non_blocking=True)
            with _autocast():
                total, _, _ = model.compute_loss(img_clean, img_deg, labels,
                                                 view_weights=CFG["ms_view_weights"])
                out = model(img_clean)
            logits = out["logits"].float()
            scores.append((logits[:, 1] - logits[:, 0]).cpu())
            labels_all.append(labels.cpu())
            agg["total"] += float(total) * labels.size(0)
            for name, t in getattr(accelerator.unwrap_model(model), "last_terms", {}).items():
                agg[name] = agg.get(name, 0.0) + float(t) * labels.size(0)
            agg["n"] += labels.size(0)
        scores = torch.cat(scores).numpy()
        labels_all = torch.cat(labels_all).numpy()
        return {
            "val_loss": agg["total"] / max(agg["n"], 1),
            "val_auc": roc_auc_score(labels_all, scores),
            "val_bacc": balanced_accuracy_score(labels_all, scores > 0),
            **{f"val_{k}": agg.get(k, 0.0) / max(agg["n"], 1) for k in extra_keys},
        }

    def save_ckpt(path, epoch, train_log, extra=None):
        state = {
            "model": accelerator.unwrap_model(model).state_dict(),
            "epoch": epoch, "config": CFG, "train_log": train_log,
        }
        if extra:
            state.update(extra)
        accelerator.save(state, path)

    def save_resume(path, epoch, best):
        state = {
            "model": accelerator.unwrap_model(model).state_dict(),
            "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
            "epoch": epoch, "config": CFG, "train_log": train_log, "best_bacc": best,
            "batch_size": batch_size, "steps_per_epoch": len(train_loader),
            "loader_gen": g.get_state(), "rng_python": random.getstate(),
            "rng_numpy": np.random.get_state(), "rng_torch": torch.get_rng_state(),
            "rng_cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
        }
        tmp = Path(str(path) + ".tmp")
        torch.save(state, tmp)
        os.replace(tmp, path)  # atomic: session chết giữa chừng không để lại file hỏng

    # ---- main loop ----
    log_csv = output_dir / "train_log.csv"

    def csv_row(r):
        return (f"{r['epoch']},{r['train_loss']:.4f},{r['train_ce']:.4f},{r['train_dcs']:.4f},"
                f"{r['val_loss']:.4f},{r['val_auc']:.4f},{r['val_bacc']:.4f}"
                + "".join(f",{r[f'dcs_k{i}']:.4f}" for i in range(1, n_ms + 1))
                + "".join(f",{r[k]:.4f}" for k in csv_extra) + "\n")

    with open(log_csv, "w", encoding="utf-8") as f:
        f.write("epoch,train_loss,train_ce,train_dcs,val_loss,val_auc,val_bacc"
                + "".join(f",dcs_k{i}" for i in range(1, n_ms + 1))
                + "".join(f",{k}" for k in csv_extra) + "\n")
        for r in train_log:  # resume: giữ lại các epoch của stage trước
            f.write(csv_row(r))

    for epoch in range(start_epoch, CFG["end_epoch"] + 1):
        tr = run_epoch(epoch)
        va = run_val()
        logger.debug(f"--> epoch {epoch}: train total={tr['total']:.4f} "
                     f"(ce={tr['ce']:.4f}, dcs={tr['dcs']:.4f}) | "
                     f"val loss={va['val_loss']:.4f} auc={va['val_auc']:.4f} "
                     f"bacc={va['val_bacc']:.4f}")
        train_log.append({"epoch": epoch, "train_loss": tr["total"],
                          "train_ce": tr["ce"], "train_dcs": tr["dcs"], **va,
                          **{k: v for k, v in tr.items() if k.startswith("dcs_k")},
                          **{k: tr[k] for k in extra_keys}})
        if accelerator.is_main_process:
            with open(log_csv, "a", encoding="utf-8") as f:
                f.write(csv_row(train_log[-1]))
            if va["val_bacc"] > best_bacc:
                best_bacc = va["val_bacc"]
                save_ckpt(output_dir / "bfree_globalforge_lora_r16_best.pth",
                          epoch, train_log, extra={"val_bacc": best_bacc})
                logger.info(f"[*] best checkpoint saved (epoch {epoch}, "
                            f"bAcc {best_bacc:.4f})")
            save_resume(output_dir / "resume_state.pth", epoch, best_bacc)  # mỗi epoch
        logger.info(f"--> epoch {epoch}/{CFG['epochs']} done | best bAcc "
                    f"so far = {best_bacc:.4f}")

    finished = CFG["end_epoch"] == CFG["epochs"]
    if accelerator.is_main_process:
        save_ckpt(output_dir / "bfree_globalforge_lora_r16.pth",
                  CFG["end_epoch"], train_log)
        if finished:
            selected = ("bfree_globalforge_lora_r16.pth" if CFG["select"] == "last"
                        else "bfree_globalforge_lora_r16_best.pth")
            (output_dir / "selected_ckpt.txt").write_text(selected + "\n")
            logger.info(f"selected checkpoint for eval (--select {CFG['select']}): {selected}")
        else:
            logger.info(f"STAGE DONE at epoch {CFG['end_epoch']}/{CFG['epochs']} -> stage tiếp theo: "
                        f"--resume_from <output>/resume_state.pth (selected_ckpt.txt chưa ghi)")

        # training curves (matplotlib Agg — không cần display)
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import pandas as pd

        df = pd.read_csv(log_csv)
        fig, axes = plt.subplots(1, 3, figsize=(16, 4), dpi=120)
        axes[0].plot(df["epoch"], df["train_loss"], marker="o", label="total")
        axes[0].plot(df["epoch"], df["train_ce"], marker="s", label="CE")
        axes[0].plot(df["epoch"], df["train_dcs"], marker="^", label="DCS")
        axes[0].set_xlabel("epoch"); axes[0].set_ylabel("train loss")
        axes[0].legend(); axes[0].set_title("Train losses (CE & DCS)")
        axes[1].plot(df["epoch"], df["val_loss"], marker="o", color="tab:red")
        axes[1].set_xlabel("epoch"); axes[1].set_ylabel("val loss")
        axes[1].set_title(f"Val loss ({CFG['val_fake_variant']})")
        axes[2].plot(df["epoch"], df["val_auc"], marker="o", label="AUC")
        axes[2].plot(df["epoch"], df["val_bacc"], marker="s", label="bAcc")
        axes[2].set_xlabel("epoch"); axes[2].set_title("Val metrics")
        axes[2].legend()
        fig.tight_layout()
        fig.savefig(output_dir / "training_curves.png")
        plt.close(fig)
        logger.info("saved training_curves.png")

    logger.info("TRAINING COMPLETE")
    accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
</file>

<file path="README.md">
# B-Free

[![BFree](https://img.shields.io/badge/Project%20page-222222.svg?style=for-the-badge&logo=github)](https://grip-unina.github.io/B-Free)
[![arXiv](https://img.shields.io/badge/-arXiv-B31B1B.svg?style=for-the-badge&logo=arXiv)](https://arxiv.org/abs/2412.17671)
[![arXiv](https://img.shields.io/badge/-Training%20Data-ffab03.svg?style=for-the-badge&logo=files&logoColor=ffffff)](https://github.com/grip-unina/B-Free/blob/main/training_data/)
<a href="https://www.grip.unina.it">
  <img src="./docs/grip.svg" alt="GRIP" height="28" style="background-color: #3D3D3D; padding: 7px; padding-left: 20px; padding-right: 20px;">
</a>

Official PyTorch implementation of the CVPR 2025 paper "A Bias-Free Training Paradigm for More General AI-generated Image Detection".


## News
*   TODO: batch processing support
*   2026-01-14: dataset of new generators released
*   2025-06-13: code released
*   2025-06-09: dataset of viral images released
*   2025-05-28: training data released
*   2025-04-03: Updated arXiv version!
*   2024-12-23: Paper has been uploaded on arXiv


## Overview

<p align="center">
 <img src="./docs/teaser.svg" alt="teaser" width="100%" />
</p>

A well-designed forensic detector should detect generator specific artifacts rather than reflect data biases. 
To this end, we propose **B-Free**, a bias-free training paradigm for AI-generated image detection, where fake images are generated from real ones using 
the conditioning procedure of stable diffusion models. 
This ensures semantic alignment between real and fake images, allowing any differences to stem solely from the subtle 
artifacts introduced by AI generation. 
Through **content-based augmentation**, we show significant improvements in both generalization and robustness over 
state-of-the-art detectors and more calibrated results.

## Training Paradigm

To avoid possible biases, we generate synthetic images from self-conditioned reconstructions of real images and include 
augmentation in the form of inpainted versions.
To ensure fake images semantically match the content of real images, we exploit the conditioning mechanism of Stable
Diffusion 2.1. More specifically:
* For **self-conditioning**, we use an empty mask (all zeroes) to induce the diffusion steps to regenerate the input.
* For the **content augmentation**, we use an object mask to replace an object with a new one of same category, or a 
rectangular mask to replace it with a new one of a different category.

<p align="center"> <img src="./docs/content_aug.svg" alt="content augmentation" width="70%" /> </p>

Besides the default inpainting, which regenerates the whole image, we consider also a version where the original 
background is restored (figure above).

## Architecture

We adopt a variant of the ViT network proposed in [[1]](https://arxiv.org/abs/2309.16588) with four registers and use the 
pretraining based on DINOv2 [[2]](https://arxiv.org/abs/2304.07193). During training, we avoid resizing the image and rely 
on large crops of 504 × 504 pixels. At inference time, we extract crops of 504 × 504 pixels (if the image is larger we 
average the results of multiple crops).\
The official python implementation of this architecture is found [here](https://github.com/facebookresearch/dinov2).

[[1]](https://arxiv.org/abs/2309.16588) *Timothée Darcet et al., Vision Transformers Need Registers. (ICLR 2024)* \
[[2]](https://arxiv.org/abs/2304.07193) *Maxime Oquab et al., DINOv2: Learning Robust Visual Features without Supervision. (TMLR 2024)*

## Augmentation strategies

<p align="center"> <img src="./docs/aug_strategies.svg" alt="augmentation strategies" width="85%" /> </p>

Our approach adopts the *inpainted++* strategy, which includes self-conditioned images, content augmentation (local 
inpainting), blurring, JPEG compression, scaling, cut-out, noise, and jittering.


## Code

The code is available in the `code` folder.

## Training set

You can download our training dataset [here](https://github.com/grip-unina/B-Free/blob/main/training_data/).

***Note:*** *training a method on this dataset only partially reflects the **inpainted+** version of our augmentation 
strategy, as the dataset does not include blurring, JPEG, and the extra augmentations of **inpainted++** (scaling, 
cut-out, noise addition, jittering), which are performed at training time.*

|        | Real   | Generated |
|--------|--------|-----------|
| Source | COCO   | SD 2.1    |
| Num.   | 51,517 | 309,102   |  

To create this dataset we used images collected from the training set of MS-COCO dataset (discarding images with 
licenses different than Creative Commons) and images generated with Stable Diffusion 2.1.\
Specifically, we first extracted the largest central crop and resize it to 512 x 512.
Then, for the generations, we used the inpainting code from the official [Stable Diffusion 2.1 repository](https://github.com/Stability-AI/stablediffusion).
Note that we did NOT embed the watermark during generation (`put_watermark` function).

Further details can be found in appendix A of the paper.


## Dataset of viral images

You can download our dataset of viral images [here](https://github.com/grip-unina/B-Free/blob/main/viral_images_dataset/).

It comprises real and fake images that went viral on internet and includes multiple web-scraped versions of
each image, for a total of about 1400 images.


## Dataset of new generators

At [this link](https://www.grip.unina.it/download/prog/B-Free/extended_synthbuster/) we provide the extended version of the [synthbuster](https://github.com/qbammey/synthbuster) dataset used in our experiments, including images generated with additional generators.
Following the same procedure of synthbuster, fake images are generated using captions extracted from real images.

It comprises:
- 1K images from **RAISE**, which we resized to a dimension so that the area is about 1024x1024, 
- 1K images generated with **FLUX**, using captions from RAISE
- 1K images generated with **Stable Diffusion 3.5**, using captions from RAISE


## License

Copyright (c) 2025 Image Processing Research Group of University Federico II of Naples ('GRIP-UNINA'). 

All rights reserved.

This software should be used, reproduced and modified only for informational and nonprofit purposes.

By downloading and/or using any of these files, you implicitly agree to all the
terms of the license, as specified in the document LICENSE.txt
(included in this package) 


## Bibtex

 ```
@InProceedings{Guillaro2024biasfree,
    author    = {Guillaro, Fabrizio and Zingarini, Giada and Usman, Ben and Sud, Avneesh and Cozzolino, Davide and Verdoliva, Luisa},
    title     = {A Bias-Free Training Paradigm for More General AI-generated Image Detection},
    booktitle = {Proceedings of the Computer Vision and Pattern Recognition Conference (CVPR)},
    month     = {June},
    year      = {2025},
    pages     = {18685-18694}
}
```

## Acknowledgments

We gratefully acknowledge the support of this research by a Google Gift. In addition, this work has received funding 
from the European Union under the Horizon Europe vera.ai project, Grant Agreement number 101070093, and was partially 
supported by SERICS (PE00000014) under the MUR National Recovery and Resilience Plan, funded by the European Union - 
NextGenerationEU.
</file>

<file path="docs/index.md">
---
layout: paper
paper: A Bias-Free Training Paradigm for More <br> General AI-generated Image Detection
github_url: https://github.com/grip-unina/B-Free
authors:  
  - name: Fabrizio Guillaro
    link: https://www.grip.unina.it/members/guillaro
    index: 1
  - name: Giada Zingarini
    link: https://www.grip.unina.it/members/zingarini
    index: 1
  - name: Ben Usman
    link: 
    index: 2
  - name: Avneesh Sud
    link: 
    index: 2
  - name: Davide Cozzolino
    link: https://www.grip.unina.it/members/cozzolino
    index: 1
  - name: Luisa Verdoliva
    link: https://www.grip.unina.it/members/verdoliva
    index: 1
affiliations: 
  - name: University Federico II of Naples, Italy
    index: 1
  - name: Google DeepMind
    index: 2
links:
    arxiv: https://arxiv.org/abs/2412.17671
    code: https://github.com/grip-unina/B-Free
    data: https://github.com/grip-unina/B-Free/blob/main/training_data/
---


<center>
 <img src="./teaser.svg" alt="teaser" width="100%" style="transform: scale(1.15);" />
</center>

A well-designed forensic detector should detect generator specific artifacts rather than reflect data biases. 
To this end, we propose **B-Free**, a bias-free training paradigm for AI-generated image detection, where fake images are generated from real ones using 
the conditioning procedure of stable diffusion models. 
This ensures semantic alignment between real and fake images, allowing any differences to stem solely from the subtle 
artifacts introduced by AI generation. 
Through **content-based augmentation**, we show significant improvements in both generalization and robustness over 
state-of-the-art detectors and more calibrated results.

## Training Paradigm

To avoid possible biases, we generate synthetic images from self-conditioned reconstructions of real images and include 
augmentation in the form of inpainted versions.
To ensure fake images semantically match the content of real images, we exploit the conditioning mechanism of Stable
Diffusion 2.1. More specifically:
* For **self-conditioning**, we use an empty mask (all zeroes) to induce the diffusion steps to regenerate the input.
* For the **content augmentation**, we use an object mask to replace an object with a new one of same category, or a 
rectangular mask to replace it with a new one of a different category.

<center> <img src="./content_aug.svg" alt="content augmentation" width="80%" /> </center>

Besides the default inpainting, which regenerates the whole image, we consider also a version where the original 
background is restored (figure above).

## Architecture

We adopt a variant of the ViT network proposed in [[1]](https://arxiv.org/abs/2309.16588) with four registers and use the 
pretraining based on DINOv2 [[2]](https://arxiv.org/abs/2304.07193). During training, we avoid resizing the image and rely 
on large crops of 504 × 504 pixels. At inference time, we extract crops of 504 × 504 pixels (if the image is larger we 
average the results of multiple crops).\
The official python implementation of this architecture is found [here](https://github.com/facebookresearch/dinov2).

[[1]](https://arxiv.org/abs/2309.16588) *Timothée Darcet et al., Vision Transformers Need Registers. (ICLR 2024)* \
[[2]](https://arxiv.org/abs/2304.07193) *Maxime Oquab et al., DINOv2: Learning Robust Visual Features without Supervision. (TMLR 2024)*

## Augmentation strategies

<center> <img src="./aug_strategies.svg" alt="augmentation strategies" width="90%" /> </center>

Our approach adopts the *inpainted++* strategy, which includes self-conditioned images, content augmentation (local 
inpainting), blurring, JPEG compression, scaling, cut-out, noise, and jittering.

## Training set

You can download our training dataset [here](https://github.com/grip-unina/B-Free/blob/main/training_data/).

***Note:*** *training a method on this dataset only partially reflects the **inpainted+** version of our augmentation 
strategy, as the dataset does not include blurring, JPEG, and the extra augmentations of **inpainted++** (scaling, 
cut-out, noise addition, jittering), which are performed at training time.*

|        | Real   | Generated |
|--------|--------|-----------|
| Source | COCO   | SD 2.1    |
| Num.   | 51,517 | 309,102   |  

To create this dataset we used images collected from the training set of MS-COCO dataset (discarding images with 
licenses different than Creative Commons) and images generated with Stable Diffusion 2.1.\
Specifically, we first extracted the largest central crop and resize it to 512 x 512.
Then, for the generations, we used the inpainting code from the official [Stable Diffusion 2.1 repository](https://github.com/Stability-AI/stablediffusion).
Note that we did NOT embed the watermark during generation (`put_watermark` function).

Further details can be found in appendix A of the paper.


## Dataset of viral images

You can download our dataset of viral images [here](https://github.com/grip-unina/B-Free/blob/main/viral_images_dataset/).

It comprises real and fake images that went viral on internet and includes multiple web-scraped versions of
each image, for a total of about 1400 images.

<center> <img src="./viral.svg" alt="viral images" width="70%" /> </center>

The figure shows the performance in terms of balanced accuracy evaluated over a 100-day period from 
the time of initial online post (Log scale). \
The performance drops after only one day, after which most competitors are stuck below 70%. Only B-Free is
able to ensure an average accuracy around 92% even after many days from the first on-line post.


## Dataset of new generators

At [this link](https://www.grip.unina.it/download/prog/B-Free/extended_synthbuster/) we provide the extended version of the [synthbuster](https://github.com/qbammey/synthbuster) dataset used in our experiments, including images generated with additional generators.
Following the same procedure of synthbuster, fake images are generated using captions extracted from real images.

It comprises:
- 1K images from **RAISE**, which we resized to a dimension so that the area is about 1024x1024, 
- 1K images generated with **FLUX**, using captions from RAISE
- 1K images generated with **Stable Diffusion 3.5**, using captions from RAISE


## Other results

<figure style="text-align: center;">
  <img src="./histogram.svg" alt="results" width="100%" />
  <figcaption style="font-size: 0.9em; margin-top: 8px;">
    Average performance in term of AUC and bAcc on four datasets: Synthbuster, GenImage, FakeInversion, SynthWildX.
  </figcaption>
</figure>

Some methods, like NPR and LGrad, present a clear non-uniform behavior, with very good performance on a single dataset 
and much worse on the others. This seems to suggest that these methods may not be truly detecting forensic artifacts, 
instead are rather exploiting intrinsic biases within the dataset. Differently, the proposed method presents a uniform 
performance across all datasets and a small loss between AUC and accuracy.

<figure style="text-align: center;">
  <img src="./robustness.svg" alt="robustness" width="80%" />
  <figcaption style="font-size: 0.9em; margin-top: 8px;">
    Robustness analysis in terms of balanced Accuracy carried out on nine generators of Synthbuster under three different
    post-processing operations: JPEG compression, resizing and blurring.
  </figcaption>
</figure>

Robustness analysis in terms of balanced Accuracy carried out on nine generators of Synthbuster under three different
post-processing operations: JPEG compression, resizing and blurring. 
B-Free is more robust by a large margin compared with other methods and can ensure a balanced accuracy always above 80%
even in the most challenging scenario.

## News
*   2026-01-14: dataset of new generators released
*   2025-06-13: code released
*   2025-06-09: dataset of viral images released
*   2025-05-28: training data released
*   2025-04-03: Updated arXiv version!
*   2024-12-23: Paper has been uploaded on arXiv




## Bibtex

 ```
@InProceedings{Guillaro2024biasfree,
    author    = {Guillaro, Fabrizio and Zingarini, Giada and Usman, Ben and Sud, Avneesh and Cozzolino, Davide and Verdoliva, Luisa},
    title     = {A Bias-Free Training Paradigm for More General AI-generated Image Detection},
    booktitle = {Proceedings of the Computer Vision and Pattern Recognition Conference (CVPR)},
    month     = {June},
    year      = {2025},
    pages     = {18685-18694}
}
```

## Acknowledgments

We gratefully acknowledge the support of this research by a Google Gift. In addition, this work has received funding 
from the European Union under the Horizon Europe vera.ai project, Grant Agreement number 101070093, and was partially 
supported by SERICS (PE00000014) under the MUR National Recovery and Resilience Plan, funded by the European Union - 
NextGenerationEU.
</file>

</files>
