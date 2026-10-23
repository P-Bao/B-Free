"""class_aware_info_nce: permutation-invariant, no NaN, one class -> 0."""
import torch

import _common
from modules.dcs_loss import class_aware_info_nce


def _batch(seed=0, n=8, dim=16):
    g = torch.Generator().manual_seed(seed)
    z = torch.randn(n, dim, generator=g)
    z_aug = z + 0.05 * torch.randn(n, dim, generator=g)
    labels = torch.tensor([0, 1] * (n // 2))
    return z, z_aug, labels


def test_finite_and_positive():
    z, z_aug, y = _batch()
    loss = class_aware_info_nce(z, z_aug, y, tau=0.07)
    assert torch.isfinite(loss) and loss.ndim == 0
    assert float(loss) > 0


def test_permutation_invariant():
    z, z_aug, y = _batch()
    perm = torch.tensor([3, 0, 7, 2, 5, 1, 6, 4])
    a = class_aware_info_nce(z, z_aug, y, tau=0.07)
    b = class_aware_info_nce(z[perm], z_aug[perm], y[perm], tau=0.07)
    assert torch.allclose(a, b, atol=1e-6), (float(a), float(b))


def test_single_class_returns_zero_not_nan():
    z, z_aug, _ = _batch()
    loss = class_aware_info_nce(z, z_aug, torch.zeros(len(z), dtype=torch.long), tau=0.07)
    assert float(loss) == 0.0


def test_identical_views_are_finite():
    z, _, y = _batch()
    loss = class_aware_info_nce(z, z.clone(), y, tau=0.07)
    assert torch.isfinite(loss)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("OK class_aware_info_nce")
