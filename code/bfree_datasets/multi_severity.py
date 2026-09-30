"""Multi-severity degradation views for MS-DCS (direction D-multiseverity-dcs).

Builds K degraded PIL views of one (already cropped/flipped) clean image, one per
severity bin. All ranges come from ``configs/bfree_dcs.yaml :: multi_severity`` — nothing
is hardcoded here. Operator implementations mirror ``train_lora.apply_degradation``
(JPEG re-encode, PIL Gaussian blur, resize-down/bicubic-up), except noise, which takes
sigma on the 0-255 scale as in the docx table.

compose:
  single_op : each view = ONE operator drawn uniformly from OPS, parameter drawn
              uniformly inside that bin's range.
  compound  : each view = jpeg -> blur -> resize -> noise, each with a parameter drawn
              inside the bin's range.

views_per_step:
  == len(levels): view k is bin k (fixed order, so per-bin logging is meaningful).
  <  len(levels): bins are sampled without replacement per image (A2 ablation: 1 bin).
"""

import io
from typing import Dict, List, Sequence

import numpy as np
from PIL import Image, ImageFilter

OPS = ("jpeg", "blur", "resize", "noise")
_REQUIRED_KEYS = ("jpeg", "blur_sigma", "resize", "noise_std255")


def validate_config(ms: Dict) -> None:
    levels = ms["levels"]
    if not levels:
        raise ValueError("multi_severity.levels is empty")
    for i, lv in enumerate(levels):
        for k in _REQUIRED_KEYS:
            if k not in lv or len(lv[k]) != 2 or lv[k][0] > lv[k][1]:
                raise ValueError(f"multi_severity.levels[{i}].{k} must be [lo, hi], got {lv.get(k)}")
    if ms["compose"] not in ("single_op", "compound"):
        raise ValueError(f"multi_severity.compose must be single_op|compound, got {ms['compose']}")
    n, v = len(levels), int(ms["views_per_step"])
    if not 1 <= v <= n:
        raise ValueError(f"views_per_step={v} must be in [1, {n}]")
    if v == n and len(ms["weights"]) != n:
        raise ValueError(f"len(weights)={len(ms['weights'])} must equal number of levels {n}")


def _apply(img: Image.Image, op: str, lv: Dict, rng) -> Image.Image:
    if op == "jpeg":
        q = rng.randint(int(lv["jpeg"][0]), int(lv["jpeg"][1]))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=q)
        buf.seek(0)
        return Image.open(buf).convert("RGB")
    if op == "blur":
        return img.filter(ImageFilter.GaussianBlur(radius=rng.uniform(*lv["blur_sigma"])))
    if op == "resize":
        s = rng.uniform(*lv["resize"])
        w, h = img.size
        small = img.resize((max(1, int(w * s)), max(1, int(h * s))), Image.BICUBIC)
        return small.resize((w, h), Image.BICUBIC)
    if op == "noise":
        sigma = rng.uniform(*lv["noise_std255"])
        arr = np.asarray(img).astype(np.float32)
        gen = np.random.default_rng(rng.getrandbits(32))
        arr = arr + gen.normal(0.0, sigma, arr.shape)
        return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    raise ValueError(op)


class MultiSeverityDegrader:
    """``degrader(img) -> list[PIL.Image]`` of length ``views_per_step``.

    ``rng`` is any object with ``randint/uniform/choice/sample/getrandbits`` (the
    ``random`` module or a ``random.Random``); seeding it makes views deterministic.
    """

    def __init__(self, levels: Sequence[Dict], compose: str, views_per_step: int, rng):
        self.levels = list(levels)
        self.compose = compose
        self.views_per_step = int(views_per_step)
        self.rng = rng
        validate_config({"levels": self.levels, "compose": compose,
                         "views_per_step": self.views_per_step,
                         "weights": [1] * len(self.levels)})

    @classmethod
    def from_config(cls, ms: Dict, rng):
        validate_config(ms)
        return cls(ms["levels"], ms["compose"], ms["views_per_step"], rng)

    def bins_for_step(self) -> List[int]:
        n = len(self.levels)
        if self.views_per_step == n:
            return list(range(n))
        return self.rng.sample(range(n), self.views_per_step)

    def degrade_one(self, img: Image.Image, bin_idx: int) -> Image.Image:
        lv = self.levels[bin_idx]
        if self.compose == "single_op":
            return _apply(img, self.rng.choice(OPS), lv, self.rng)
        for op in OPS:
            img = _apply(img, op, lv, self.rng)
        return img

    def __call__(self, img: Image.Image) -> List[Image.Image]:
        return [self.degrade_one(img, b) for b in self.bins_for_step()]
