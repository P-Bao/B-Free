"""eval_protocol: 5-crop math == Wrapper5crops patch selection; metric sanity."""
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import _common
from eval_protocol import (bfree_five_crops, bootstrap_auc_ci, mean_logit,
                           metrics_from_logits, paired_bootstrap_diff, resolve_arrow_dir)

SIZE, PATCH = 504, 14


def _img(w, h, seed=0):
    g = np.random.default_rng(seed)
    return Image.fromarray(g.integers(0, 256, (h, w, 3), dtype=np.uint8))


def _reference_patch_crops(img, conv):
    """Wrapper5crops.forward logic: conv on whole image, replicate only if smaller, 5 slices."""
    gw, gh = img.size[0] // PATCH, img.size[1] // PATCH
    x = torch.from_numpy(np.asarray(img.crop((0, 0, gw * PATCH, gh * PATCH))).copy())
    x = x.permute(2, 0, 1).float().div(255.0).unsqueeze(0)
    emb = conv(x)                                   # (1, C, gh, gw)
    grid = SIZE // PATCH
    if emb.shape[2] < grid or emb.shape[3] < grid:
        reps = [1, 1, max(1, -(-grid // emb.shape[2])), max(1, -(-grid // emb.shape[3]))]
        emb = emb.repeat(*reps)[..., :grid, :grid]
    hs = max((emb.shape[2] - grid) // 2, 0)
    ws = max((emb.shape[3] - grid) // 2, 0)
    starts = [(ws, hs), (0, 0), (0, emb.shape[2] - grid),
              (emb.shape[3] - grid, emb.shape[2] - grid), (emb.shape[3] - grid, 0)]
    return [emb[..., y:y + grid, x:x + grid] for x, y in starts]


def test_five_crops_match_wrapper5crops():
    torch.manual_seed(0)
    conv = torch.nn.Conv2d(3, 8, PATCH, stride=PATCH, bias=False)
    for (w, h) in [(700, 520), (512, 512), (300, 260), (504, 504)]:
        img = _img(w, h, seed=w + h)
        ref = _reference_patch_crops(img, conv)
        crops = bfree_five_crops(img, SIZE, PATCH)
        assert len(crops) == 5 and all(c.size == (SIZE, SIZE) for c in crops)
        for i, c in enumerate(crops):
            x = torch.from_numpy(np.asarray(c).copy()).permute(2, 0, 1).float().div(255.0).unsqueeze(0)
            got = conv(x)
            assert torch.allclose(got, ref[i], atol=1e-5), (w, h, i, float((got - ref[i]).abs().max()))


def test_mean_logit_averages_logits_not_probs():
    assert abs(mean_logit([0.0, 2.0]) - 1.0) < 1e-9


def _toy(n_real=60, n_fake=60, seed=0):
    rng = np.random.default_rng(seed)
    z = np.concatenate([rng.normal(-1.5, 1.0, n_real), rng.normal(1.5, 1.0, n_fake)])
    y = np.array([0] * n_real + [1] * n_fake)
    return y, z


def test_metrics_and_bootstrap():
    y, z = _toy()
    m = metrics_from_logits(y, z)
    assert m["auc"] > 0.9 and abs(m["bacc"] - 50.0) < 50.0
    for k in ("ece_bal", "nll_bal", "pd10", "eer", "fpr95", "tpr05", "fpr05"):
        assert np.isfinite(m[k]), k
    lo, hi = bootstrap_auc_ci(y, z, n_boot=200)
    assert lo <= m["auc"] <= hi
    rng = np.random.default_rng(1)
    z_weak = 0.3 * z + rng.normal(0, 2.0, len(z))   # strictly worse ranker
    d, dlo, dhi = paired_bootstrap_diff(y, z, z_weak, n_boot=200)
    assert d > 0 and dlo <= d <= dhi
    # degenerate labels must not raise
    m0 = metrics_from_logits(np.zeros(5), np.zeros(5))
    assert np.isnan(m0["auc"]) and m0["bacc"] == m0["bacc"]


def test_resolve_arrow_dir_finds_nested_dataset():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "synthwildx"
        (root / "v1" / "train").mkdir(parents=True)
        (root / "v1" / "train" / "dataset_info.json").write_text("{}")
        assert resolve_arrow_dir(root) == root / "v1" / "train"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            print("--", name)
            fn()
    print("OK eval_protocol")
