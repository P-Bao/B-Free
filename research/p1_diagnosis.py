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
