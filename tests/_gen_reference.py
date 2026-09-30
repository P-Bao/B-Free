"""Generate pre-change reference (total, ce, dcs) for the default (single-view) path.

Run ONCE at the commit before multi-severity changes; the result is committed as
tests/ref_single_view.pt and compared bit-for-bit by test_dcs_loss.py.
"""
import os, sys
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))
from networks.bfree_globalforge_vit import BFreeGlobalForgeViT


def build():
    torch.manual_seed(0)
    return BFreeGlobalForgeViT(img_size=112, pretrained=False, lambda_dcs=0.01, label_smoothing=0.1)


def run():
    m = build().train()
    g = torch.Generator().manual_seed(1)
    xc = torch.randn(4, 3, 112, 112, generator=g)
    xd = torch.randn(4, 3, 112, 112, generator=g)
    y = torch.tensor([0, 1, 0, 1])
    torch.manual_seed(123)  # GSR masking randomness
    total, ce, dcs = m.compute_loss(xc, xd, y)
    return {"xc": xc, "xd": xd, "y": y, "total": total.detach(), "ce": ce.detach(), "dcs": dcs.detach()}


if __name__ == "__main__":
    out = run()
    torch.save(out, os.path.join(os.path.dirname(__file__), "ref_single_view.pt"))
    print({k: v for k, v in out.items() if k in ("total", "ce", "dcs")})
