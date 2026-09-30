"""MultiSeverityDegrader: view count, bin ranges, determinism, config validation, train_lora glue."""
import random
import numpy as np
import yaml
from PIL import Image

import _common
from bfree_datasets.multi_severity import MultiSeverityDegrader, validate_config, _apply

CFG = yaml.safe_load(open(f"{_common.CODE}/configs/bfree_dcs.yaml"))["multi_severity"]


def _img(seed=0, size=96):
    a = np.random.default_rng(seed).integers(0, 256, (size, size, 3), dtype=np.uint8)
    return Image.fromarray(a)


def test_yaml_default_off_and_valid():
    assert CFG["enabled"] is False
    validate_config(CFG)
    assert len(CFG["levels"]) == 4 and len(CFG["weights"]) == 4


def test_view_count_and_size():
    for compose in ("single_op", "compound"):
        d = MultiSeverityDegrader(CFG["levels"], compose, 4, random.Random(0))
        views = d(_img())
        assert len(views) == 4 and all(v.size == (96, 96) and v.mode == "RGB" for v in views)
    d1 = MultiSeverityDegrader(CFG["levels"], "single_op", 1, random.Random(0))
    assert len(d1(_img())) == 1


def test_params_inside_bin_ranges():
    rng = random.Random(3)
    seen = {"q": [], "sigma": [], "scale": [], "noise": []}
    orig = _apply

    class Spy(random.Random):
        def randint(self, a, b):
            v = super().randint(a, b); seen["q"].append((a, b, v)); return v

        def uniform(self, a, b):
            v = super().uniform(a, b); seen["sigma"].append((a, b, v)); return v

    spy = Spy(5)
    for b, lv in enumerate(CFG["levels"]):
        d = MultiSeverityDegrader(CFG["levels"], "compound", 4, spy)
        d.degrade_one(_img(), b)
    for lo, hi, v in seen["q"] + seen["sigma"]:
        assert lo <= v <= hi
    # each drawn interval must equal one of the configured intervals
    allowed = {tuple(lv[k]) for lv in CFG["levels"] for k in ("jpeg", "blur_sigma", "resize", "noise_std255")}
    assert {(lo, hi) for lo, hi, _ in seen["q"] + seen["sigma"]} <= allowed


def test_determinism_with_seed():
    def run():
        d = MultiSeverityDegrader(CFG["levels"], "single_op", 4, random.Random(42))
        return [np.asarray(v) for v in d(_img())]
    a, b = run(), run()
    assert all(np.array_equal(x, y) for x, y in zip(a, b))


def test_severity_monotone_damage():
    """Harsher bin => larger mean abs pixel change (compound, averaged over draws)."""
    base = np.asarray(_img(1)).astype(np.float32)
    dev = []
    for b in range(4):
        rng = random.Random(7)
        d = MultiSeverityDegrader(CFG["levels"], "compound", 4, rng)
        dev.append(np.mean([np.abs(np.asarray(d.degrade_one(_img(1), b)).astype(np.float32) - base).mean()
                            for _ in range(5)]))
    assert dev == sorted(dev), dev


def test_validation_errors():
    bad = dict(CFG, views_per_step=5)
    for cfg in (bad, dict(CFG, compose="x"), dict(CFG, weights=[1, 1])):
        try:
            validate_config(cfg)
        except ValueError:
            continue
        raise AssertionError("should raise")


def test_train_lora_glue():
    import torch
    import train_lora as tl
    from torchvision import transforms as T
    norm = T.ToTensor()
    tl.CFG.update(degrade_max_rounds=4, degrade_round_probs=[0.15, 0.30, 0.25, 0.20, 0.10])
    img = _img()
    random.seed(0)
    v = tl.degraded_view(img, norm)                 # default path
    assert v.shape == (3, 96, 96)
    tl._MS_DEGRADER = MultiSeverityDegrader(CFG["levels"], "single_op", 4, random)
    try:
        v = tl.degraded_view(img, norm)
        assert v.shape == (4, 3, 96, 96)
        views = tl.split_views(torch.stack([v, v]))  # batch (2,4,3,H,W)
        assert len(views) == 4 and views[0].shape == (2, 3, 96, 96)
        assert tl.split_views(torch.zeros(2, 3, 8, 8)).shape == (2, 3, 8, 8)
    finally:
        tl._MS_DEGRADER = None


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
    print("ALL PASS")
