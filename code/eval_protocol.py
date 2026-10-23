"""Shared evaluation protocol v2 (AGENT_PLAN_V5 Phase 0).

One implementation used by all three harness copies (ours / b-free /
globalforge), so preprocessing and metrics cannot drift between them:

  * ``bfree_five_crops``: the 5 patch-grid crops of B-Free's ``Wrapper5crops``
    computed in PIL space (center, TL, BL, BR, TR), replicate-tiling small images
    instead of bicubic upscaling.
  * ``metrics_from_logits``: metrics on the raw fake-vs-real logit, using the
    official ``utils.dmetrics`` definitions (balanced ECE/NLL, Pd@FAR, EER).
  * ``bootstrap_auc_ci`` / ``paired_bootstrap_diff``: image-level CIs, no
    assumption of independent generators.
  * ``write_scores_parquet``: per-image scores + provenance.

PENDING USER APPROVAL: changes nothing in the old harness files; it backs the
new version4 harnesses only (see research/eval-v2-proposal.md).
"""

import hashlib
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.special import expit
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, roc_curve

HARNESS_VERSION = 2


# ---- copies of B-Free utils/dmetrics.py (kept inline so this module imports no
# ---- repo package and cannot shadow the official bundle's `utils`/`networks`) ----

def calculate_eer2(y_true, y_score):
    from scipy.interpolate import interp1d
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    err = 1. - tpr - fpr
    val = (fpr + 1 - tpr) / 2
    return np.float64(interp1d(err, val)(0))


def pd_at_far(y_true, y_score, fpr_th):
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    return np.interp(fpr_th, fpr, tpr)


def balanced_cllr(y_true, y_score):
    _, y_true = np.unique(y_true, return_inverse=True)
    y_true = 2 * y_true - 1
    y_score = np.logaddexp(-y_true * y_score, 0)
    y_score0 = np.mean(y_score[y_true < 0])
    y_score1 = np.mean(y_score[y_true > 0])
    return (y_score0 + y_score1) / (2 * np.log(2))


def balanced_nll_binary(y_true, y_score):
    return balanced_cllr(y_true, y_score) * np.log(2)


def balanced_ece_binary(y_true, y_score, bins=15):
    _, yp, yc = np.unique(y_true, return_inverse=True, return_counts=True)
    sample_weight = (1 / yc)[yp]
    assert np.max(yp) == 1
    total_weight = sample_weight.sum()
    prob = expit(y_score)
    correctness = (yp == 1)
    interval = np.floor(bins * prob)
    ece = 0.0
    for _ in range(bins):
        in_bin = (interval == _)
        if np.any(in_bin):
            weight = sample_weight[in_bin]
            accuracy_in_bin = (weight * correctness[in_bin]).mean() / weight.mean()
            avg_confidence_in_bin = (weight * prob[in_bin]).mean() / weight.mean()
            prop_in_bin = weight.sum() / total_weight
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    return ece


def bfree_five_crops(img: Image.Image, size: int = 504, patch: int = 14):
    """5 crops of `size` px on the patch grid, matching B-Free `Wrapper5crops`.

    Mirrors ``Wrapper5crops.forward`` exactly: reduce to whole patches (conv
    stride=patch drops the remainder), replicate-tile (then truncate) each axis
    that is shorter than ``size//patch``, then cut five windows at the offsets
    of the wrapper: center, top-left, bottom-left, bottom-right, top-right.
    No resize, no bicubic upscale.
    """
    if size % patch != 0:
        raise ValueError(f"size={size} is not a multiple of patch={patch}")
    grid = size // patch
    img = img.convert("RGB")
    gw, gh = img.size[0] // patch, img.size[1] // patch
    if gw < 1 or gh < 1:
        raise ValueError(f"image {img.size} smaller than one patch ({patch}px)")
    base = img.crop((0, 0, gw * patch, gh * patch))          # drop partial patches
    # replicate_wrap: repeat by ceil(grid/n) then truncate to exactly `grid`
    nw, nh = max(gw, grid), max(gh, grid)
    tiled = Image.new("RGB", (nw * patch, nh * patch))
    for ty in range(0, nh, gh):
        for tx in range(0, nw, gw):
            tiled.paste(base, (tx * patch, ty * patch))
    W, H = tiled.size
    ws_c, hs_c = (W - size) // 2 // patch * patch, (H - size) // 2 // patch * patch
    boxes = [
        (ws_c, hs_c),                     # center
        (0, 0),                           # top-left
        (0, H - size),                    # bottom-left
        (W - size, H - size),             # bottom-right
        (W - size, 0),                    # top-right
    ]
    return [tiled.crop((x, y, x + size, y + size)) for x, y in boxes]


def mean_logit(per_crop_logits) -> float:
    """Mean of the fake-vs-real logit over crops (never average probabilities)."""
    vals = [float(v) for v in per_crop_logits]
    if not vals:
        raise ValueError("no crop logits")
    return float(np.mean(vals))


def _fpr_at_tpr(y_true, y_score, target=0.95):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    ok = np.where(tpr >= target)[0]
    return float(fpr[ok[0]]) if len(ok) else float("nan")


def metrics_from_logits(y_true, y_logit) -> dict:
    """All metrics from the raw logit. bAcc@0.5 = (logit > 0)."""
    y_true = np.asarray(y_true).astype(int)
    z = np.asarray(y_logit, dtype=float)
    p = 1.0 / (1.0 + np.exp(-z))
    y05 = (z > 0).astype(int)
    out = {"n": int(len(y_true)), "n_real": int((y_true == 0).sum()),
           "n_fake": int((y_true == 1).sum())}
    out["auc"] = float(roc_auc_score(y_true, z)) if len(np.unique(y_true)) > 1 else float("nan")
    out["bacc"] = float(balanced_accuracy_score(y_true, y05)) * 100
    out["tpr05"] = float((y05[y_true == 1] == 1).mean()) if (y_true == 1).any() else float("nan")
    out["fpr05"] = float((y05[y_true == 0] == 1).mean()) if (y_true == 0).any() else float("nan")
    if len(np.unique(y_true)) > 1:
        out["ece_bal"] = float(balanced_ece_binary(y_true, z))
        out["nll_bal"] = float(balanced_nll_binary(y_true, z))
        out["pd10"] = float(pd_at_far(y_true, z, 0.10))
        out["eer"] = float(calculate_eer2(y_true, z))
        out["fpr95"] = _fpr_at_tpr(y_true, z, 0.95)
    else:
        for k in ("ece_bal", "nll_bal", "pd10", "eer", "fpr95"):
            out[k] = float("nan")
    out["mean_p_fake"] = float(p.mean())
    return out


def bootstrap_auc_ci(y_true, y_logit, n_boot=1000, seed=42, alpha=0.05):
    """Image-level bootstrap CI of AUC. Returns (lo, hi)."""
    y_true = np.asarray(y_true)
    z = np.asarray(y_logit, dtype=float)
    if len(np.unique(y_true)) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y_true), len(y_true))
        if len(np.unique(y_true[idx])) < 2:
            continue
        vals.append(roc_auc_score(y_true[idx], z[idx]))
    lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def paired_bootstrap_diff(y_true, z_a, z_b, n_boot=2000, seed=42):
    """Paired bootstrap of AUC(a) - AUC(b) on the same images. Returns (diff, lo, hi)."""
    y_true = np.asarray(y_true)
    z_a = np.asarray(z_a, dtype=float)
    z_b = np.asarray(z_b, dtype=float)
    if not (len(y_true) == len(z_a) == len(z_b)):
        raise ValueError("paired bootstrap needs equal-length arrays")
    if len(np.unique(y_true)) < 2:
        return float("nan"), float("nan"), float("nan")
    diff = roc_auc_score(y_true, z_a) - roc_auc_score(y_true, z_b)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y_true), len(y_true))
        if len(np.unique(y_true[idx])) < 2:
            continue
        vals.append(roc_auc_score(y_true[idx], z_a[idx]) - roc_auc_score(y_true[idx], z_b[idx]))
    lo, hi = np.quantile(vals, [0.025, 0.975])
    return float(diff), float(lo), float(hi)


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def config_hash(cfg) -> str:
    import json
    return hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:16]


def write_scores_parquet(rows, path, provenance):
    """rows: list of dicts with at least id,label,logit,score,benchmark,generator,condition.

    Provenance columns are overwritten from `provenance` (git_sha, ckpt_sha256,
    config_hash, seed, harness_version) so every file carries it (rule #8).
    """
    import pandas as pd
    df = pd.DataFrame(rows)
    for k, v in provenance.items():
        df[k] = v
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path


def resolve_arrow_dir(root):
    """Find the directory holding a HF dataset without unzipping/extracting.

    `load_from_disk` needs the directory that CONTAINS `dataset_dict.json` /
    `dataset_info.json` / `state.json` (or a top-level `*.arrow`), not the
    dataset root with an extra wrapper level.
    """
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"dataset root not found: {root}")
    markers = ("dataset_dict.json", "dataset_info.json", "state.json")
    if any((root / m).is_file() for m in markers) or list(root.glob("*.arrow")):
        return root
    for p in sorted(root.rglob("*")):
        if p.is_dir() and (any((p / m).is_file() for m in markers) or list(p.glob("*.arrow"))):
            return p
    raise FileNotFoundError(
        f"no HF dataset directory (dataset_dict.json/dataset_info.json/*.arrow) under {root}")


def load_hf_dataset_any_split(root):
    """load_from_disk at the resolved dir; concat splits if DatasetDict."""
    from datasets import concatenate_datasets, load_from_disk
    obj = load_from_disk(str(resolve_arrow_dir(root)))
    if hasattr(obj, "keys"):
        return concatenate_datasets([obj[k] for k in obj.keys()])
    return obj
