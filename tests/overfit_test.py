"""Phase-4 overfit gate (CPU): ViT-B/14 random init, 8 synthetic images, both DCS modes.

Passes when, in each mode, the loss stays finite, one backward step gives finite grads,
and CE + total loss drop clearly on the memorised batch. Prints GATE PASS.
"""
import random
import time

import numpy as np
import torch
from PIL import Image

import _common
from networks.bfree_globalforge_vit import BFreeGlobalForgeViT
from bfree_datasets.multi_severity import MultiSeverityDegrader
import yaml

STEPS, LR, N = 40, 3e-4, 8
CFG = yaml.safe_load(open(f"{_common.CODE}/configs/bfree_dcs.yaml"))["multi_severity"]


def make_data():
    g = np.random.default_rng(0)
    imgs = [Image.fromarray(g.integers(0, 256, (_common.IMG, _common.IMG, 3), dtype=np.uint8)) for _ in range(N)]
    to_t = lambda im: torch.from_numpy(np.asarray(im).copy()).permute(2, 0, 1).float().div(127.5).sub(1)
    clean = torch.stack([to_t(i) for i in imgs])
    labels = torch.tensor([0, 1] * (N // 2))
    return imgs, clean, labels, to_t


def run(mode):
    torch.manual_seed(0)
    imgs, clean, labels, to_t = make_data()
    if mode == "single":
        # stand-in single degraded view: JPEG-ish/blur via the q=Mild..Extreme bin 1 view (fixed once)
        d = MultiSeverityDegrader(CFG["levels"], "single_op", 1, random.Random(0))
        deg = torch.stack([to_t(d(i)[0]) for i in imgs])
        view_w = None
    else:
        d = MultiSeverityDegrader(CFG["levels"], "single_op", 4, random.Random(0))
        per_img = [d(i) for i in imgs]
        deg = [torch.stack([to_t(per_img[n][k]) for n in range(N)]) for k in range(4)]
        view_w = [float(w) for w in CFG["weights"]]
    m = BFreeGlobalForgeViT(img_size=_common.IMG, pretrained=False, lambda_dcs=0.01).train()
    opt = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=1e-4)
    hist = []
    t0 = time.time()
    for step in range(STEPS):
        opt.zero_grad()
        total, ce, dcs = m.compute_loss(clean, deg, labels, view_weights=view_w)
        assert all(torch.isfinite(x) for x in (total, ce, dcs)), f"{mode} step {step} non-finite"
        total.backward()
        gn = torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        assert torch.isfinite(gn), f"{mode} step {step} grad non-finite"
        opt.step()
        hist.append((float(total.detach()), float(ce.detach()), float(dcs.detach())))
    (t_a, c_a, _), (t_b, c_b, d_b) = hist[0], hist[-1]
    print(f"[{mode}] {time.time()-t0:.0f}s total {t_a:.3f}->{t_b:.3f} ce {c_a:.3f}->{c_b:.3f} dcs_last {d_b:.3f}")
    ok = c_b < 0.5 * c_a + 0.1 and t_b < 0.6 * t_a
    return ok


if __name__ == "__main__":
    results = {mode: run(mode) for mode in ("single", "multi_severity")}
    print(results)
    print("GATE PASS" if all(results.values()) else "GATE FAIL")
    raise SystemExit(0 if all(results.values()) else 1)
