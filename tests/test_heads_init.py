"""Step-0 equivalence gates (CPU, random init, no data):
  * cls_plus_patch with gamma=0 == cls head (bit-wise on logits)
  * gsr_zero_init == no GSR (bit-wise on logits)
  * lib_beta_init=-10 ~= no LIB
  * head modes return 2 logits / finite loss
"""
import torch

import _common
from networks.bfree_globalforge_vit import BFreeGlobalForgeViT

KW = dict(img_size=_common.IMG, pretrained=False, use_dcs=False)


def _model(**kw):
    torch.manual_seed(0)
    return BFreeGlobalForgeViT(**{**KW, **kw}).eval()


def _x(seed=0, n=2):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, 3, _common.IMG, _common.IMG, generator=g)


def _logits(m, x):
    with torch.no_grad():
        return m(x)["logits"]


def test_cls_plus_patch_gamma_zero_equals_cls():
    a = _model(use_lib=False, use_gsr=False, head_mode="cls")
    b = _model(use_lib=False, use_gsr=False, head_mode="cls_plus_patch")
    b.load_state_dict(a.state_dict(), strict=False)
    assert float(b.extra_head.gamma.detach()) == 0.0
    la, lb = _logits(a, _x()), _logits(b, _x())
    assert la.shape == lb.shape == (2, 2)
    assert torch.equal(la, lb), float((la - lb).abs().max())


def test_gsr_zero_init_equals_no_gsr():
    a = _model(use_lib=False, use_gsr=False)
    b = _model(use_lib=False, use_gsr=True, gsr_zero_init=True)
    b.load_state_dict(a.state_dict(), strict=False)
    assert b.gsr is not None and b.gsr.gate is not None
    assert torch.equal(_logits(a, _x()), _logits(b, _x()))


def test_lib_beta_minus10_near_identity():
    a = _model(use_lib=False, use_gsr=False)
    b = _model(use_lib=True, use_gsr=False, lib_beta_init=-10.0)
    b.load_state_dict(a.state_dict(), strict=False)
    la, lb = _logits(a, _x()), _logits(b, _x())
    assert torch.allclose(la, lb, atol=1e-3), float((la - lb).abs().max())
    # beta=-0.85 (v2) must NOT be identity-like
    c = _model(use_lib=True, use_gsr=False)
    c.load_state_dict(a.state_dict(), strict=False)
    assert not torch.allclose(la, _logits(c, _x()), atol=1e-3)


def test_class_aware_loss_runs_on_forward_features():
    torch.manual_seed(0)
    m = BFreeGlobalForgeViT(img_size=_common.IMG, pretrained=False, use_lib=False,
                            use_gsr=False, use_dcs=True, dcs_mode="class_aware")
    x = _x()
    y = torch.tensor([0, 1])
    total, ce, dcs = m.compute_loss(x, x * 0.9, y)
    for t in (total, ce, dcs):
        assert torch.isfinite(t)
    total.backward()
    assert any(p.grad is not None for p in m.model.head.parameters())


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            print("--", name)
            fn()
    print("OK heads/init")
