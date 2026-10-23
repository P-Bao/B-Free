"""Online augmentation for the CE path (B-Free `inpainted++` style).

Ported by description, not from B-Free source: the repo ships no augmentation
code (README only names the operators). All numeric parameters below are
PLACEHOLDERS from AGENT_PLAN_V5 Phase 1.1 and MUST be cross-checked against the
B-Free paper appendix before any claim about reproducing its recipe.

Applied with the same distribution to real and fake (no "degraded == fake" cue).
Image size is preserved; the caller crops first, then augments, then normalizes.

    aug = OnlineAug(cfg)                   # cfg = YAML section `online_aug`
    img = aug(img, rng=random.Random(...)) # deterministic when rng is given
"""

import io
import math
import random

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

# Placeholders (AGENT_PLAN_V5 Phase 1.1). `preset` scales probabilities and
# strength ranges: light=0.5, bfree=1.0, strong=1.5.
DEFAULT_CFG = {
    "enabled": False,
    "preset": "bfree",
    "p_any": 0.9,
    "jpeg_p": 0.5,
    "jpeg_q_min": 30,
    "jpeg_q_max": 100,
    "blur_p": 0.4,
    "blur_sigma_min": 0.1,
    "blur_sigma_max": 3.0,
    "resize_p": 0.4,
    "resize_scale_min": 0.25,
    "resize_scale_max": 1.5,
    "noise_p": 0.3,
    "noise_std_min": 0.002,
    "noise_std_max": 0.03,
    "jitter_p": 0.4,
    "jitter_min": 0.7,
    "jitter_max": 1.3,
    "cutout_p": 0.3,
    "cutout_frac_min": 0.02,
    "cutout_frac_max": 0.20,
}

_PRESET_SCALE = {"light": 0.5, "bfree": 1.0, "strong": 1.5}


def resolve_cfg(cfg=None):
    """YAML section -> full cfg dict. Unknown keys are kept (forward compatible)."""
    out = dict(DEFAULT_CFG)
    out.update({k: v for k, v in (cfg or {}).items() if v is not None})
    return out


def _scale_probs(cfg, factor):
    if factor == 1.0:
        return cfg
    out = dict(cfg)
    for key in ("p_any", "jpeg_p", "blur_p", "resize_p", "noise_p", "jitter_p", "cutout_p"):
        out[key] = min(1.0, float(cfg[key]) * factor)
    # strength ranges widen/narrow around their midpoint
    for lo_key, hi_key in (("jpeg_q_min", "jpeg_q_max"), ("blur_sigma_min", "blur_sigma_max"),
                           ("noise_std_min", "noise_std_max"), ("cutout_frac_min", "cutout_frac_max")):
        lo, hi = float(cfg[lo_key]), float(cfg[hi_key])
        mid, half = (lo + hi) / 2.0, (hi - lo) / 2.0 * factor
        out[lo_key], out[hi_key] = mid - half, mid + half
    out["resize_scale_min"] = max(0.1, 1.0 - (1.0 - float(cfg["resize_scale_min"])) * factor)
    out["resize_scale_max"] = max(1.0, 1.0 + (float(cfg["resize_scale_max"]) - 1.0) * factor)
    out["jitter_min"] = max(0.2, 1.0 - (1.0 - float(cfg["jitter_min"])) * factor)
    out["jitter_max"] = max(1.0, 1.0 + (float(cfg["jitter_max"]) - 1.0) * factor)
    return out


class OnlineAug:
    """Random subset of {jpeg, blur, resize, noise, jitter, cutout} per image."""

    OPS = ("jpeg", "blur", "resize", "noise", "jitter", "cutout")

    def __init__(self, cfg=None, preset=None):
        cfg = resolve_cfg(cfg)
        self.preset = str(preset or cfg.get("preset") or "bfree")
        if self.preset not in _PRESET_SCALE:
            raise ValueError(f"online_aug.preset must be one of {sorted(_PRESET_SCALE)}, got {self.preset!r}")
        self.cfg = _scale_probs(cfg, _PRESET_SCALE[self.preset])
        self.enabled = bool(self.cfg.get("enabled", True))

    def __call__(self, img, rng=None):
        if not self.enabled:
            return img
        rng = rng or random
        if rng.random() >= float(self.cfg["p_any"]):
            return img
        ops = [op for op in self.OPS if rng.random() < float(self.cfg[f"{op}_p"])]
        rng.shuffle(ops)
        for op in ops:
            img = getattr(self, f"_{op}")(img, rng)
        return img

    # ---- operators -------------------------------------------------------
    def _jpeg(self, img, rng):
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=int(rng.uniform(self.cfg["jpeg_q_min"], self.cfg["jpeg_q_max"])))
        buf.seek(0)
        return Image.open(buf).convert("RGB")

    def _blur(self, img, rng):
        return img.filter(ImageFilter.GaussianBlur(radius=rng.uniform(
            self.cfg["blur_sigma_min"], self.cfg["blur_sigma_max"])))

    def _resize(self, img, rng):
        w, h = img.size
        scale = rng.uniform(self.cfg["resize_scale_min"], self.cfg["resize_scale_max"])
        small = img.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.BICUBIC)
        return small.resize((w, h), Image.BICUBIC)

    def _noise(self, img, rng):
        arr = np.asarray(img).astype(np.float32) / 255.0
        std = rng.uniform(self.cfg["noise_std_min"], self.cfg["noise_std_max"])
        arr = np.clip(arr + np.random.default_rng(rng.randrange(2 ** 31)).normal(0, std, arr.shape), 0, 1)
        return Image.fromarray((arr * 255).astype(np.uint8))

    def _jitter(self, img, rng):
        img = ImageEnhance.Brightness(img).enhance(rng.uniform(self.cfg["jitter_min"], self.cfg["jitter_max"]))
        img = ImageEnhance.Contrast(img).enhance(rng.uniform(self.cfg["jitter_min"], self.cfg["jitter_max"]))
        return ImageEnhance.Color(img).enhance(rng.uniform(self.cfg["jitter_min"], self.cfg["jitter_max"]))

    def _cutout(self, img, rng):
        w, h = img.size
        frac = rng.uniform(self.cfg["cutout_frac_min"], self.cfg["cutout_frac_max"])
        cw, ch = max(1, round(w * math.sqrt(frac))), max(1, round(h * math.sqrt(frac)))
        x, y = rng.randrange(max(1, w - cw + 1)), rng.randrange(max(1, h - ch + 1))
        color = tuple(int(v) for v in np.random.default_rng(rng.randrange(2 ** 31)).integers(0, 256, 3))
        out = img.copy()
        ImageDraw.Draw(out).rectangle([x, y, x + cw, y + ch], fill=color)
        return out
