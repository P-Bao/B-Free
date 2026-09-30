"""Backbone shape contract: logits (B,2), cls (B,768), ViT-B/14 reg4 (5 prefix tokens)."""
import torch

import _common
from networks.bfree_globalforge_vit import BFreeGlobalForgeViT


def test_shapes():
    torch.manual_seed(0)
    m = BFreeGlobalForgeViT(img_size=_common.IMG, pretrained=False).eval()
    assert m.embed_dim == 768 and m.num_prefix == 5
    with torch.no_grad():
        out = m(torch.randn(3, 3, _common.IMG, _common.IMG))
        patch, feat, hw = m._patch_tokens(torch.randn(3, 3, _common.IMG, _common.IMG))
    assert out["logits"].shape == (3, 2) and out["cls"].shape == (3, 768)
    assert hw == _common.IMG // 14 and patch.shape == (3, hw * hw, 768) and feat.shape == (3, 768)
    assert torch.isfinite(out["logits"]).all()


if __name__ == "__main__":
    test_shapes()
    print("ALL PASS")
