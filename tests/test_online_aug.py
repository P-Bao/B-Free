"""Online augmentation: size/type invariant, deterministic under a given rng."""
import random

import numpy as np
from PIL import Image

import _common
from modules.online_aug import OnlineAug


def _img(seed=0):
    g = np.random.default_rng(seed)
    return Image.fromarray(g.integers(0, 256, (80, 96, 3), dtype=np.uint8))


def test_off_returns_same_object():
    aug = OnlineAug({"enabled": False})
    im = _img()
    assert aug(im, random.Random(0)) is im


def test_size_and_mode_preserved():
    aug = OnlineAug({"enabled": True, "p_any": 1.0, "cutout_p": 1.0, "jpeg_p": 1.0,
                     "blur_p": 1.0, "resize_p": 1.0, "noise_p": 1.0, "jitter_p": 1.0})
    for seed in range(20):
        out = aug(_img(seed), random.Random(seed))
        assert out.size == (96, 80) and out.mode == "RGB"


def test_deterministic_for_same_seed():
    aug = OnlineAug({"enabled": True, "p_any": 1.0, "preset": "strong"})
    a = aug(_img(3), random.Random(7))
    b = aug(_img(3), random.Random(7))
    assert np.array_equal(np.asarray(a), np.asarray(b))


def test_preset_changes_probabilities_only():
    lo = OnlineAug({"enabled": True}, preset="light")
    hi = OnlineAug({"enabled": True}, preset="strong")
    assert lo.cfg["p_any"] < hi.cfg["p_any"]
    assert lo.cfg["blur_sigma_max"] < hi.cfg["blur_sigma_max"]


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("OK online_aug")
