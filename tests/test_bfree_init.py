"""init_from=bfree: key remap + 1-logit -> 2-logit head conversion (synthetic ckpt).

Real B-Free weights are not in this repo; this test reproduces the *format*
(Wrapper5crops state dict: 'patch_embed.*' + timm keys without 'model.').
"""
import tempfile
from pathlib import Path

import torch

import _common
from networks.bfree_globalforge_vit import BFreeGlobalForgeViT

KW = dict(img_size=_common.IMG, pretrained=False, use_lib=False, use_gsr=False, use_dcs=False)


def _wrapper_format(sd):
    out = {}
    for k, v in sd.items():
        if k.startswith("model."):
            k = k[len("model."):]
        out[k] = v
    return out


def _x(seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(2, 3, _common.IMG, _common.IMG, generator=g)


def test_remap_and_head_conversion():
    torch.manual_seed(0)
    ref = BFreeGlobalForgeViT(num_classes=1, head_mode="cls", **KW).eval()
    sd = _wrapper_format(ref.state_dict())
    assert "patch_embed.proj.weight" in sd and not any(k.startswith("model.") for k in sd)

    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "bfree.pth"
        torch.save({"model": sd}, p)
        new = BFreeGlobalForgeViT(num_classes=2, head_mode="cls", **KW).eval()
        report = new.load_checkpoint(str(p), strict=False)

    assert report["num_unexpected"] == 0 and report["num_skipped_shape"] == 0, report
    assert report["missing_keys"] == [], report["missing_keys"]
    with torch.no_grad():
        z_ref = ref(_x())["logits"][:, 0]
        logits = new(_x())["logits"]
        z_new = logits[:, 1] - logits[:, 0]
    assert torch.allclose(z_ref, z_new, atol=1e-5), float((z_ref - z_new).abs().max())
    # sanity: head weights are +/- half of the reference row
    w_ref = sd["head.weight"]
    assert torch.allclose(new.model.head.weight[1] - new.model.head.weight[0], w_ref[0], atol=1e-6)


def test_shape_mismatch_is_reported_not_silent():
    torch.manual_seed(0)
    m = BFreeGlobalForgeViT(**KW)
    sd = _wrapper_format(m.state_dict())
    sd["blocks.0.attn.qkv.weight"] = torch.zeros(3, 3)
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "bad.pth"
        torch.save({"model": sd}, p)
        report = m.load_checkpoint(str(p), strict=False)
    assert report["num_skipped_shape"] == 1 and report["skipped_shape"] == ["model.blocks.0.attn.qkv.weight"]


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            print("--", name)
            fn()
    print("OK bfree init")
