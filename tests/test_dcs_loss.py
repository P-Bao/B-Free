"""Unit tests: faithful L_DCS untouched, multi-severity additive, default path bit-identical."""
import os
import torch
import torch.nn.functional as F

import _common  # noqa: F401
from modules.dcs_loss import info_nce_loss, multi_severity_dcs
from networks.bfree_globalforge_vit import BFreeGlobalForgeViT
import _gen_reference as ref


def gf_info_nce_loss(z, z_aug, tau=0.07):
    """Verbatim copy of GlobalForge-BE0F/code/engine_finetune.py::info_nce_loss."""
    z = F.normalize(z, dim=-1)
    z_aug = F.normalize(z_aug, dim=-1)
    logits = torch.matmul(z, z_aug.t()) / tau
    labels = torch.arange(logits.size(0), device=logits.device)
    loss1 = F.cross_entropy(logits, labels)
    loss2 = F.cross_entropy(logits.t(), labels)
    return 0.5 * (loss1 + loss2)


def test_matches_globalforge():
    g = torch.Generator().manual_seed(0)
    z, za = torch.randn(6, 16, generator=g), torch.randn(6, 16, generator=g)
    assert torch.equal(info_nce_loss(z, za), gf_info_nce_loss(z, za))


def test_k1_equals_info_nce():
    g = torch.Generator().manual_seed(1)
    z, za = torch.randn(5, 8, generator=g), torch.randn(5, 8, generator=g)
    loss, terms = multi_severity_dcs(z, [za])
    assert torch.equal(loss, info_nce_loss(z, za)) and len(terms) == 1


def test_weight_normalisation():
    g = torch.Generator().manual_seed(2)
    z = torch.randn(5, 8, generator=g)
    zs = [torch.randn(5, 8, generator=g) for _ in range(3)]
    l1, t = multi_severity_dcs(z, zs, weights=[1, 2, 3])
    expect = (1 * t[0] + 2 * t[1] + 3 * t[2]) / 6
    assert torch.allclose(l1, expect, atol=1e-6)
    l2, _ = multi_severity_dcs(z, zs, weights=[10, 20, 30])   # scale-invariant
    assert torch.allclose(l1, l2, atol=1e-6)
    lu, tu = multi_severity_dcs(z, zs)                        # default = uniform mean
    assert torch.allclose(lu, torch.stack(tu).mean(), atol=1e-6)
    for bad in ([1, 2], [0, 0, 0], [-1, 1, 1]):
        try:
            multi_severity_dcs(z, zs, weights=bad)
        except ValueError:
            continue
        raise AssertionError(f"weights {bad} should raise")


def test_default_path_bit_identical_to_pre_change_reference():
    saved = torch.load(os.path.join(os.path.dirname(__file__), "ref_single_view.pt"))
    out = ref.run()
    for k in ("total", "ce", "dcs"):
        assert torch.equal(out[k], saved[k]), f"{k}: {out[k]} != {saved[k]}"


def test_list_path_contract_and_equivalence():
    m = ref.build().train()
    g = torch.Generator().manual_seed(1)
    xc = torch.randn(4, 3, _common.IMG, _common.IMG, generator=g)
    xd = [torch.randn(4, 3, _common.IMG, _common.IMG, generator=g) for _ in range(3)]
    y = torch.tensor([0, 1, 0, 1])
    m.eval()  # remove GSR stochasticity for exact comparisons
    with torch.no_grad():
        total, ce, dcs = m.compute_loss(xc, xd, y)
        assert total.shape == ce.shape == dcs.shape == ()
        assert list(m.last_dcs_terms) == ["dcs_k1", "dcs_k2", "dcs_k3"]
        assert torch.allclose(dcs, torch.stack(list(m.last_dcs_terms.values())).mean(), atol=1e-6)
        assert torch.allclose(total, ce + m.lambda_dcs * dcs)
        # K=1 list == plain tensor path
        t1, c1, d1 = m.compute_loss(xc, [xd[0]], y)
        t0, c0, d0 = m.compute_loss(xc, xd[0], y)
        assert torch.equal(d1, d0) and torch.equal(t1, t0)
        assert m.last_dcs_terms == {}  # tensor path clears per-view terms


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
    print("ALL PASS")
