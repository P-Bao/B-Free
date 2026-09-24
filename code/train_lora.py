"""K2 training script — train-only, chạy qua accelerate.

Tái sử dụng repo như thư viện (đúng tinh thần MODULES_INTERFACE.md), orchestration
được chuyển hết vào đây từ notebook `xla-training-phase-2-fromscratch.ipynb` để
notebook giữ được dưới 1 MB. Eval riêng (per-variant / AIGB held-out / OOD /
RealDeg-Bench) nằm ở notebook eval — script này CHỈ train (main val giữ lại vì
là metric chọn best checkpoint).

Chạy (từ thư mục code/, hoặc notebook bash cell — env offline do caller set):
    accelerate launch --num_processes 1 --mixed_precision bf16 train_lora.py \
        --train_data_root /kaggle/input/datasets/... \
        --dinov2_sd /kaggle/input/.../model.safetensors \
        --ai_genbench_dir /kaggle/input/datasets/...   # optional

Args tường minh: thiếu/th sai -> fail ngay (không auto-discover).
"""

import argparse
import contextlib
import hashlib
import io
import json
import logging
import math
import multiprocessing
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image
from peft import LoraConfig, get_peft_model
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from utils.normalization import get_list_norm
from utils.dmetrics import balanced_accuracy_score, roc_auc_score

logger = logging.getLogger("bfree")
CFG = {}  # điền trong parse_args/main, đọc từ các module-level helpers

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

PRIORITY_GENERATORS = [
    "ADM", "DDPM", "DeepFloyd IF", "Palette", "Denoising Diffusion GAN",
    "LaMa", "MAT", "SN-PatchGAN",
    "StyleGAN1", "VQGAN", "GANformer", "CIPS",
    "FaceSynthetics",
]


# ============================================================
# CLI
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(description="K2 LoRA training (train-only, accelerate)")
    p.add_argument("--train_data_root", type=str, required=True,
                   help="dataset root chứa COCO_real_512/ + 6 thư mục SD2.1_*")
    p.add_argument("--dinov2_sd", type=str, required=True,
                   help="DINOv2 ViT-B/14 reg4 model.safetensors (bundle notebook 01)")
    p.add_argument("--ai_genbench_dir", type=str, default=None,
                   help="HF Arrow DatasetDict ai-genbench-v1; bỏ trống = train không có AIGB")
    p.add_argument("--output_dir", type=str, default="/kaggle/working")
    p.add_argument("--config", type=str, default="configs/bfree_dcs.yaml")

    p.add_argument("--arch", type=str, default="vit_base_patch14_reg4_dinov2.lvd142m")
    p.add_argument("--img_size", type=int, default=504)
    p.add_argument("--lambda_dcs", type=float, default=0.01)   # D1
    p.add_argument("--dcs_tau", type=float, default=0.07)
    p.add_argument("--label_smoothing", type=float, default=0.1)
    p.add_argument("--lib_kernel", type=int, default=3)
    p.add_argument("--lib_tau", type=float, default=0.5)
    p.add_argument("--gsr_window", type=int, default=3)
    p.add_argument("--gsr_mask_prob", type=float, default=1.0)
    p.add_argument("--epochs", type=int, default=8)            # D2
    p.add_argument("--lora_rank", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--scheduler_eta_min", type=float, default=1e-7)
    p.add_argument("--batch_size", type=int, default=64,
                   help="0 = auto-find theo D6 (doubling + binary search)")
    p.add_argument("--batch_hard_cap", type=int, default=128)
    p.add_argument("--val_md5_percentile", type=int, default=3)
    p.add_argument("--val_fake_variant", type=str, default="SD2.1_selfconditioned")
    p.add_argument("--pair_real_prob", type=float, default=0.5)  # D4
    p.add_argument("--degrade_max_rounds", type=int, default=4)
    p.add_argument("--degrade_round_probs", type=str,
                   default="0.15,0.30,0.25,0.20,0.10")
    p.add_argument("--extra_source_val_percentile", type=int, default=30)
    p.add_argument("--aigenbench_split", type=str, default="train")
    p.add_argument("--aigenbench_max_per_gen", type=int, default=4000)
    p.add_argument("--aigenbench_val_frac", type=float, default=0.1)
    p.add_argument("--aigenbench_seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    CFG.update(vars(args))
    CFG["num_classes"] = 2
    CFG["degrade_round_probs"] = [
        float(x) for x in CFG["degrade_round_probs"].split(",") if x.strip()
    ]
    for path_key in ("train_data_root", "dinov2_sd", "ai_genbench_dir"):
        if CFG[path_key] is not None and not Path(CFG[path_key]).exists():
            raise FileNotFoundError(
                f"--{path_key.replace('_', '-')} không tồn tại: {CFG[path_key]}")
    return args


def setup_logging(output_dir: Path):
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    logger.handlers.clear()
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter(
        fmt="%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S"))
    logger.addHandler(console)
    fh = logging.FileHandler(str(output_dir / "train_log.txt"), mode="a", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        fmt="%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    logger.addHandler(fh)


# ============================================================
# Compound lifecycle degradation (7-operator pool, Appendix A RealDeg-Bench)
# ============================================================

DEGRADE_STRENGTHS = {
    "jpeg":       [90, 80, 70, 60, 40],
    "blur":       [0.5, 1, 2, 3, 5],
    "resize":     [0.9, 0.7, 0.5, 0.3, 0.2],
    "noise":      [0.0005, 0.001, 0.002, 0.005, 0.01],
    "brightness": [-0.2, -0.1, 0.1, 0.2],
    "contrast":   [-0.3, -0.2, 0.1, 0.2],
    "saturation": [0.6, 0.8, 1.3, 1.5],
}
DEGRADE_OPS = list(DEGRADE_STRENGTHS.keys())


def apply_degradation(img, op, strength):
    if op == "jpeg":
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=int(strength))
        buf.seek(0)
        return Image.open(buf).convert("RGB")
    if op == "blur":
        return img.filter(__import__("PIL.ImageFilter", fromlist=["GaussianBlur"]).GaussianBlur(radius=strength))
    if op == "resize":
        w, h = img.size
        small = img.resize((max(1, int(w * strength)), max(1, int(h * strength))), Image.BICUBIC)
        return small.resize((w, h), Image.BICUBIC)
    if op == "noise":
        arr = np.asarray(img).astype(np.float32) / 255.0
        arr = arr + np.random.normal(0, np.sqrt(strength), arr.shape)
        return Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8))
    from PIL import ImageEnhance
    if op == "brightness":
        return ImageEnhance.Brightness(img).enhance(1.0 + strength)
    if op == "contrast":
        return ImageEnhance.Contrast(img).enhance(1.0 + strength)
    if op == "saturation":
        return ImageEnhance.Color(img).enhance(strength)
    raise ValueError(op)


def apply_compound(img, n_steps, rng):
    for _ in range(n_steps):
        op = rng.choice(DEGRADE_OPS)
        strength = rng.choice(DEGRADE_STRENGTHS[op])
        img = apply_degradation(img, op, strength)
    return img


def realistic_lifecycle_degrade(img, rng):
    """Áp cho cả real + fake như nhau (tránh shortcut 'suy giảm = fake')."""
    n_rounds = rng.choices(
        range(CFG["degrade_max_rounds"] + 1),
        weights=CFG["degrade_round_probs"],
    )[0]
    return apply_compound(img, n_rounds, rng)


# ============================================================
# Dataset — stem index + fixed 8-epoch fake schedule (D4) + extra sources
# ============================================================

def unwrap_dir(path):
    """Bỏ 1 tầng lồng thừa: <dir>/<dir> -> <dir> (nếu tồn tại)."""
    path = Path(path)
    child = path / path.name
    return child if child.is_dir() else path


def ensure_min_size(img, min_size):
    w, h = img.size
    if min(w, h) < min_size:
        scale = min_size / min(w, h)
        img = img.resize(
            (max(min_size, round(w * scale)), max(min_size, round(h * scale))),
            Image.BICUBIC)
    return img


def build_stem_index(dirs, cache_path=None):
    """os.listdir + splitext (không stat từng entry — FUSE mount network round-trip)."""
    if cache_path is not None and Path(cache_path).is_file():
        cached = json.loads(Path(cache_path).read_text())
        if cached.get("dirs") == list(dirs):
            logger.info(f"stem index (cache): {[len(m) for m in cached['maps']]} files")
            return cached["maps"]
    t0 = time.time()
    maps = []
    for d in dirs:
        m = {}
        for name in os.listdir(d):
            stem, ext = os.path.splitext(name)
            if ext:
                m[stem] = os.path.join(d, name)
        maps.append(m)
    logger.info(f"stem index (scan): {[len(m) for m in maps]} files in {time.time()-t0:.1f}s")
    if cache_path is not None:
        Path(cache_path).write_text(json.dumps({"dirs": list(dirs), "maps": maps}))
    return maps


def is_val_id(stem):
    return (int(hashlib.md5(stem.encode()).hexdigest(), 16) % 100
            < CFG["val_md5_percentile"])


class PairDataset(Dataset):
    """Train: lịch 8-epoch/ID cố định — round(epochs*(1-p)) epoch là fake, các epoch fake
    gán KHÔNG LẶP biến thể SD2.1 (seed = md5(stem)), 50/50 real/fake đúng D4.
    Val: 2 samples/ID — chẵn = real, lẻ = deterministic fake variant (balanced)."""

    def __init__(self, ids, img_size, is_train, pair_real_prob, val_fake_variant,
                 real_map, fake_maps, fake_dirs):
        self.ids = ids
        self.img_size = img_size
        self.is_train = is_train
        self.pair_real_prob = pair_real_prob
        self.val_fake_variant = val_fake_variant
        self.real_map = real_map
        self.fake_maps = fake_maps
        self.fake_dirs = fake_dirs
        self.normalize = T.Compose(get_list_norm("resnet"))
        if is_train:
            self.epoch_value = multiprocessing.Value("i", 1)
            self.fake_schedule = self._build_fake_schedule()
        else:
            self.epoch_value = None
            self.fake_schedule = None

    def set_epoch(self, epoch):
        if self.epoch_value is not None:
            self.epoch_value.value = epoch

    def _build_fake_schedule(self):
        n_variants = len(self.fake_maps)
        n_epochs = CFG["epochs"]
        n_fake = min(n_epochs, round(n_epochs * (1 - self.pair_real_prob)))
        schedule = {}
        for stem in self.ids:
            available = [v for v in range(n_variants) if stem in self.fake_maps[v]]
            if not available:
                continue
            rng = random.Random(int(hashlib.md5(stem.encode()).hexdigest(), 16))
            slots = list(range(1, n_epochs + 1))
            rng.shuffle(slots)
            fake_epochs = sorted(slots[:n_fake])
            bag, assigned = [], []
            for _ in fake_epochs:
                if not bag:
                    bag = available.copy()
                    rng.shuffle(bag)
                assigned.append(bag.pop())
            schedule[stem] = dict(zip(fake_epochs, assigned))
        return schedule

    def __len__(self):
        return len(self.ids) * (1 if self.is_train else 2)

    def _pick_val_fake(self, stem):
        for m, d in zip(self.fake_maps, self.fake_dirs):
            if Path(d).name == self.val_fake_variant and stem in m:
                return m[stem]
        for m in self.fake_maps:
            if stem in m:
                return m[stem]
        return None

    def __getitem__(self, i):
        stem = self.ids[i if self.is_train else i // 2]
        if self.is_train:
            variant_idx = self.fake_schedule.get(stem, {}).get(self.epoch_value.value)
            if variant_idx is None:
                path, label = self.real_map[stem], 0
            else:
                path, label = self.fake_maps[variant_idx][stem], 1
        else:
            if i % 2 == 0:
                path, label = self.real_map[stem], 0
            else:
                path, label = self._pick_val_fake(stem), 1
        assert path is not None, f"No image found for id {stem}"
        img = Image.open(path).convert("RGB")
        if self.is_train:
            t, l, h, w = T.RandomCrop.get_params(img, (self.img_size, self.img_size))
            img = TF.crop(img, t, l, h, w)
            if random.random() > 0.5:
                img = TF.hflip(img)
        else:
            img = TF.center_crop(img, (self.img_size, self.img_size))
        img_deg = realistic_lifecycle_degrade(img, random)
        return self.normalize(img), self.normalize(img_deg), label


def _is_extra_holdout(path):
    name = Path(path).name
    return (int(hashlib.md5(name.encode()).hexdigest(), 16) % 100
            < CFG["extra_source_val_percentile"])


def _resolve_extra_source(name, train_data_root):
    for base in (Path(train_data_root), Path(train_data_root) / "extended_synthbuster"):
        cand = unwrap_dir(base / name)
        if cand.is_dir():
            return cand
    return None


def _list_images_recursive(d):
    return sorted(str(p) for p in Path(d).rglob("*") if p.suffix.lower() in IMG_EXTS)


class ExtraTrainDataset(Dataset):
    """Train flat (không content-pair) cho RAISE/LDM/SD3-FLUX — cùng augmentation
    + lifecycle degrade trả về (img_clean, img_deg, label)."""

    def __init__(self, files, label, img_size):
        self.files = files
        self.label = label
        self.img_size = img_size
        self.normalize = T.Compose(get_list_norm("resnet"))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        img = Image.open(self.files[i]).convert("RGB")
        img = ensure_min_size(img, self.img_size)
        t, l, h, w = T.RandomCrop.get_params(img, (self.img_size, self.img_size))
        img = TF.crop(img, t, l, h, w)
        if random.random() > 0.5:
            img = TF.hflip(img)
        img_deg = realistic_lifecycle_degrade(img, random)
        return self.normalize(img), self.normalize(img_deg), self.label


# ============================================================
# AI-GenBench train split (13 generator ưu tiên)
# ============================================================

_AIGB_CACHE = {}


def _aigenbench_dataset():
    if "ds" in _AIGB_CACHE:
        return _AIGB_CACHE["ds"]
    from datasets import load_from_disk  # lazy: tránh import nhầm package 'datasets' của repo

    obj = load_from_disk(CFG["ai_genbench_dir"])
    if hasattr(obj, "keys"):
        splits = list(obj.keys())
        split = CFG["aigenbench_split"] if CFG["aigenbench_split"] in splits else splits[0]
        if split != CFG["aigenbench_split"]:
            logger.warning(f"ai_genbench: không có split '{CFG['aigenbench_split']}', "
                           f"dùng '{split}' (có: {splits})")
        ds = obj[split]
    else:
        split, ds = "(single)", obj

    img_col = next((c for c, f in ds.features.items()
                    if f.__class__.__name__ == "Image"), None)
    if img_col is None:
        raise RuntimeError(f"ai_genbench: không tìm thấy cột ảnh. features={list(ds.features)}")
    label_feat = ds.features.get("label")
    label_names = list(getattr(label_feat, "names", []) or [])
    logger.info(f"ai_genbench: split={split} n={len(ds)} img_col='{img_col}' "
                f"label_names={label_names}")
    _AIGB_CACHE["ds"] = (ds, img_col, label_names)
    return _AIGB_CACHE["ds"]


def _aigenbench_meta(ds, label_names):
    if "meta" in _AIGB_CACHE:
        return _AIGB_CACHE["meta"]
    cols = [c for c in ("label", "generator", "origin_dataset") if c in ds.column_names]
    meta = ds.select_columns(cols).to_pandas()
    if label_names:
        fake_ids = {i for i, n in enumerate(label_names) if "fake" in str(n).lower()}
        if not fake_ids:
            fake_ids = {len(label_names) - 1}
        meta["y"] = meta["label"].apply(lambda v: 1 if int(v) in fake_ids else 0)
    else:
        meta["y"] = (meta["label"].astype(int) != 0).astype(int)
    if "generator" not in meta.columns:
        meta["generator"] = "unknown"
    if "origin_dataset" not in meta.columns:
        meta["origin_dataset"] = "unknown"
    meta["generator"] = meta["generator"].fillna("unknown").astype(str).str.strip()
    meta["origin_dataset"] = meta["origin_dataset"].fillna("unknown").astype(str)
    logger.info(f"ai_genbench: meta OK | real={int((meta.y == 0).sum())} "
                f"fake={int((meta.y == 1).sum())} | "
                f"n_generator={meta.loc[meta.y == 1, 'generator'].nunique()}")
    _AIGB_CACHE["meta"] = meta
    return meta


def build_ai_genbench_train_groups():
    """-> (ArrowSource, real_train, fake_train, real_val, fake_val_per_gen).
    Val là lát cắt TÁCH TỪ split="train" — không bao giờ đụng split "validation"
    (split đó dành riêng cho eval notebook)."""
    ds, img_col, label_names = _aigenbench_dataset()
    meta = _aigenbench_meta(ds, label_names)

    real_gens = set(meta.loc[meta.y == 1, "generator"].unique())
    missing = [g for g in PRIORITY_GENERATORS if g not in real_gens]
    if missing:
        raise AssertionError(
            f"Generator không khớp tên thật trong Arrow: {missing}\nCó: {sorted(real_gens)}")

    rng = np.random.default_rng(CFG["aigenbench_seed"])
    y = meta["y"].to_numpy()
    gens = meta["generator"]

    train_groups, val_groups = {}, {}
    for g in PRIORITY_GENERATORS:
        gi = meta.index[(y == 1) & (gens == g)].to_numpy()
        if CFG["aigenbench_max_per_gen"] and len(gi) > CFG["aigenbench_max_per_gen"]:
            gi = rng.choice(gi, CFG["aigenbench_max_per_gen"], replace=False)
        gi = np.array(sorted(gi))
        n_val = (min(len(gi), max(1, round(len(gi) * CFG["aigenbench_val_frac"])))
                 if len(gi) else 0)
        perm = rng.permutation(len(gi))
        val_local, train_local = perm[:n_val], perm[n_val:]
        val_groups[g] = [(int(i), 1) for i in sorted(gi[val_local])]
        train_groups[g] = [(int(i), 1) for i in sorted(gi[train_local])]

    total_fake_train = sum(len(v) for v in train_groups.values())
    total_fake_val = sum(len(v) for v in val_groups.values())

    real_idx_all = meta.index[y == 0].to_numpy().copy()
    rng.shuffle(real_idx_all)
    n_real_val = min(len(real_idx_all) // 4, max(1, total_fake_val)) if total_fake_val else 0
    real_val_idx = real_idx_all[:n_real_val]
    remaining_real = real_idx_all[n_real_val:]
    real_train_idx = remaining_real[:min(total_fake_train, len(remaining_real))]

    real_items = [(int(i), 0) for i in sorted(real_train_idx)]
    real_val_items = [(int(i), 0) for i in sorted(real_val_idx)]

    logger.info(f"[ai_genbench] {len(train_groups)} generator | fake_train={total_fake_train} "
                f"| real_train={len(real_items)} | val held-out: fake={total_fake_val} "
                f"real={len(real_val_items)}")

    class ArrowSource:
        def __init__(self, ds, img_col):
            self.ds, self.img_col = ds, img_col

        def __len__(self):
            return len(self.ds)

        def get(self, idx):
            return self.ds[int(idx)][self.img_col]

    return (ArrowSource(ds, img_col), real_items,
            [it for v in train_groups.values() for it in v],
            real_val_items, val_groups)


class ExtraArrowTrainDataset(Dataset):
    """Dataset train đọc ảnh từ ArrowSource — augment=False (val) center-crop không flip."""

    def __init__(self, source, items, label, img_size, augment=True):
        self.source = source
        self.indices = [i for i, y in items if y == label]
        self.label = label
        self.img_size = img_size
        self.augment = augment
        self.normalize = T.Compose(get_list_norm("resnet"))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        img = self.source.get(self.indices[i]).convert("RGB")
        img = ensure_min_size(img, self.img_size)
        if self.augment:
            t, l, h, w = T.RandomCrop.get_params(img, (self.img_size, self.img_size))
            img = TF.crop(img, t, l, h, w)
            if random.random() > 0.5:
                img = TF.hflip(img)
        else:
            img = TF.center_crop(img, (self.img_size, self.img_size))
        img_deg = realistic_lifecycle_degrade(img, random)
        return self.normalize(img), self.normalize(img_deg), self.label


# ============================================================
# Model — offline pretrained init (D5) + LoRA
# ============================================================

def load_pretrained_backbone(model, safetensors_path, model_module=None):
    """D5 offline: load DINOv2 weights vào model.model + resample pos_embed 518->504.

    dinov2 reg4 checkpoint lưu pos_embed PATCH-ONLY (1, 37*37, 768) vì
    no_embed_class=True -> tự dò layout (patch-only vs patch+prefix).
    """
    from safetensors.torch import load_file

    sd = load_file(str(safetensors_path))
    sd = {k[len("model."):] if k.startswith("model.") else k: v for k, v in sd.items()}

    prefix = model.model.num_prefix_tokens
    if "pos_embed" in sd:
        pe = sd["pos_embed"]
        n = pe.shape[1]
        if math.isqrt(n) ** 2 == n:
            ckpt_prefix = 0
        elif math.isqrt(n - prefix) ** 2 == n - prefix:
            ckpt_prefix = prefix
        else:
            raise ValueError(
                f"cannot parse pos_embed layout {tuple(pe.shape)} "
                f"(neither patch-only nor patch+{prefix} prefix)")
        old_hw = math.isqrt(n - ckpt_prefix)
        new_hw = model.model.patch_embed.grid_size[0]
        if old_hw != new_hw:
            from timm.layers.pos_embed import resample_abs_pos_embed
            logger.info(f"resample pos_embed: {old_hw}x{old_hw} -> {new_hw}x{new_hw} "
                        f"(checkpoint prefix tokens in pos_embed: {ckpt_prefix})")
            sd["pos_embed"] = resample_abs_pos_embed(
                pe, new_size=model.model.patch_embed.grid_size,
                old_size=(old_hw, old_hw), num_prefix_tokens=ckpt_prefix)

    ref = model.model.state_dict()
    dropped = [k for k, v in sd.items() if k not in ref or ref[k].shape != v.shape]
    sd = {k: v for k, v in sd.items() if k not in dropped}

    report = model.model.load_state_dict(sd, strict=False)
    logger.info(f"pretrained load: dropped={len(dropped)}, missing={report.missing_keys}")
    assert set(report.missing_keys) <= {"head.weight", "head.bias"}, "backbone not fully initialized"
    assert not report.unexpected_keys
    assert "pos_embed" not in dropped, "pos_embed was dropped by shape filter — resample failed"
    return model


def apply_lora_to_backbone(model, r=16, lora_alpha=32, lora_dropout=0.05):
    """Áp LoRA lên timm VisionTransformer backbone, giữ full train cho LIB, GSR và Head."""
    lora_config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=["qkv", "proj", "fc1", "fc2"],
        lora_dropout=lora_dropout,
        bias="none",
    )
    model.model = get_peft_model(model.model, lora_config)

    trainable_submodules = [model.lib, model.gsr,
                            model.model.base_model.model.head,
                            model.model.base_model.model.fc_norm]
    for sub in trainable_submodules:
        if sub is not None:
            for p in sub.parameters():
                p.requires_grad = True

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"[*] LoRA configured (rank={r}): {trainable_params:,} / {total_params:,} "
                f"trainable ({100 * trainable_params / total_params:.2f}%)")
    return model


# ============================================================
# Auto batch size (D6) + train/val loops
# ============================================================

def _autocast():
    if torch.cuda.is_available():
        return torch.amp.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def try_batch(model, bs, img_size, device):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    try:
        x1 = torch.randn(bs, 3, img_size, img_size, device=device)
        x2 = torch.randn(bs, 3, img_size, img_size, device=device)
        y = torch.randint(0, 2, (bs,), device=device)
        with _autocast():
            total, _, _ = model.compute_loss(x1, x2, y)
        total.backward()
        del x1, x2, y, total
        return True
    except torch.cuda.OutOfMemoryError:
        return False
    finally:
        torch.cuda.empty_cache()


def find_max_batch_size(model, img_size, device, hard_cap):
    assert try_batch(model, 2, img_size, device), "Batch size 2 OOMs — environment problem."
    lo, bs, hi = 2, 4, None
    while bs <= hard_cap:
        if try_batch(model, bs, img_size, device):
            lo = bs
            bs *= 2
        else:
            hi = bs
            break
    if hi is None:
        logger.warning(f"no OOM up to hard cap {hard_cap}")
        hi = lo + 1
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if try_batch(model, mid, img_size, device):
            lo = mid
        else:
            hi = mid
    return lo


def main():
    args = parse_args()
    output_dir = Path(CFG["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(output_dir)

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("PYTHONHASHSEED", "0")
    os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

    random.seed(CFG["seed"])
    np.random.seed(CFG["seed"])
    torch.manual_seed(CFG["seed"])
    torch.cuda.manual_seed_all(CFG["seed"])

    train_root = Path(CFG["train_data_root"])
    real_dir = unwrap_dir(train_root / "COCO_real_512")
    fake_dirs = sorted(str(unwrap_dir(p)) for p in train_root.glob("SD2.1_*") if p.is_dir())
    if not real_dir.is_dir():
        raise FileNotFoundError(f"COCO_real_512 missing under {train_root}")
    if len(fake_dirs) != 6:
        raise FileNotFoundError(f"Expected 6 SD2.1_* dirs, found {len(fake_dirs)}")
    logger.info(f"REAL_DIR={real_dir} | {len(fake_dirs)} fake variants")

    logger.info(f"CONFIG: {CFG}")

    # ---- stem index + val split ----
    real_map = build_stem_index([str(real_dir)],
                                cache_path=output_dir / "stem_index_real.json")[0]
    fake_maps = build_stem_index(fake_dirs,
                                 cache_path=output_dir / "stem_index_fake.json")
    all_stems = sorted(real_map.keys())
    train_ids = [s for s in all_stems if not is_val_id(s)]
    val_ids = [s for s in all_stems if is_val_id(s)]
    logger.info(f"train IDs={len(train_ids)}, val IDs={len(val_ids)} "
                f"(~{CFG['val_md5_percentile']}%)")

    train_ds = PairDataset(train_ids, CFG["img_size"], True, CFG["pair_real_prob"],
                           CFG["val_fake_variant"], real_map, fake_maps, fake_dirs)
    val_ds = PairDataset(val_ids, CFG["img_size"], False, CFG["pair_real_prob"],
                         CFG["val_fake_variant"], real_map, fake_maps, fake_dirs)
    a, _, c = train_ds[0]
    _, _, c2 = val_ds[1]
    assert tuple(a.shape) == (3, CFG["img_size"], CFG["img_size"])
    logger.info(f"train_ds={len(train_ds)}, val_ds={len(val_ds)} "
                f"| labels val[0],val[1]={c},{c2} (expect 0,1)")

    # ---- extra sources: RAISE/LDM/SD3-FLUX (train phần, held-out cho OOD notebook) ----
    train_sources = [train_ds]
    raise_dir = _resolve_extra_source("real_RAISE_1k", train_root)
    ldm_dir = _resolve_extra_source("latent-diffusion", train_root)
    sd3_dir = _resolve_extra_source("sd3_flux", train_root)
    if raise_dir is None or (ldm_dir is None and sd3_dir is None):
        logger.warning("[extra] thiếu real_RAISE_1k / latent-diffusion / sd3_flux — "
                       "train với COCO+SD2.1 (+AI-GenBench nếu có)")
    else:
        extra_real = _list_images_recursive(raise_dir)
        extra_fake = []
        for gdir in (ldm_dir, sd3_dir):
            if gdir is not None:
                extra_fake += _list_images_recursive(gdir)
        train_real = [p for p in extra_real if not _is_extra_holdout(p)]
        train_fake = [p for p in extra_fake if not _is_extra_holdout(p)]
        n_hold_r = len(extra_real) - len(train_real)
        n_hold_f = len(extra_fake) - len(train_fake)
        logger.info(f"[extra] train: real={len(train_real)} fake={len(train_fake)} | "
                    f"held-out: real={n_hold_r} fake={n_hold_f} "
                    f"({CFG['extra_source_val_percentile']}%)")
        if train_real or train_fake:
            train_sources.append(torch.utils.data.ConcatDataset([
                ExtraTrainDataset(train_real, 0, CFG["img_size"]),
                ExtraTrainDataset(train_fake, 1, CFG["img_size"]),
            ]))

    # ---- AI-GenBench ----
    if CFG["ai_genbench_dir"] is None:
        logger.info("ai_genbench_dir không truyền -> train không có AI-GenBench")
    else:
        (aigb_source, aigb_real, aigb_fake, _, _) = build_ai_genbench_train_groups()
        train_sources.append(torch.utils.data.ConcatDataset([
            ExtraArrowTrainDataset(aigb_source, aigb_real, 0, CFG["img_size"]),
            ExtraArrowTrainDataset(aigb_source, aigb_fake, 1, CFG["img_size"]),
        ]))

    train_dataset = (train_sources[0] if len(train_sources) == 1
                     else torch.utils.data.ConcatDataset(train_sources))
    logger.info(f"train sources = {len(train_sources)}")

    # ---- model: construct + D5 pretrained + LoRA ----
    from networks.bfree_globalforge_vit import BFreeGlobalForgeViT
    from configs.loader import load_config, build_model_kwargs  # noqa: F401  (giữ tương thích config)

    model = BFreeGlobalForgeViT(
        arch=CFG["arch"], num_classes=CFG["num_classes"],
        img_size=CFG["img_size"], pretrained=False,
        use_lib=True, use_gsr=True, use_dcs=True,
        lib_kernel=CFG["lib_kernel"], lib_tau=CFG["lib_tau"],
        gsr_window=CFG["gsr_window"], gsr_mask_prob=CFG["gsr_mask_prob"],
        dcs_tau=CFG["dcs_tau"], lambda_dcs=CFG["lambda_dcs"],
        label_smoothing=CFG["label_smoothing"],
    )
    model = load_pretrained_backbone(model, CFG["dinov2_sd"])
    model = apply_lora_to_backbone(model, r=CFG["lora_rank"],
                                   lora_alpha=CFG["lora_alpha"],
                                   lora_dropout=CFG["lora_dropout"])

    # ---- accelerate ----
    from accelerate import Accelerator
    accelerator = Accelerator(mixed_precision="bf16")
    device = accelerator.device

    batch_size = CFG["batch_size"]
    if batch_size == 0:
        if device.type != "cuda":
            raise RuntimeError("--batch_size=0 (auto-find) yêu cầu CUDA")
        batch_size = find_max_batch_size(model, CFG["img_size"], device,
                                         CFG["batch_hard_cap"])
        logger.info(f"auto batch size = {batch_size}")

    def seed_worker(worker_id):
        ws = torch.initial_seed() % 2 ** 32
        np.random.seed(ws)
        random.seed(ws)

    g = torch.Generator()
    g.manual_seed(CFG["seed"])

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=CFG["num_workers"], pin_memory=True, drop_last=True,
        persistent_workers=CFG["num_workers"] > 0,
        worker_init_fn=seed_worker, generator=g)
    val_loader = DataLoader(val_ds, batch_size=max(2, batch_size // 2), shuffle=False,
                            num_workers=CFG["num_workers"], pin_memory=True)

    optimizer = AdamW((p for p in model.parameters() if p.requires_grad),
                      lr=CFG["lr"], weight_decay=CFG["weight_decay"])
    scheduler = CosineAnnealingLR(optimizer,
                                  T_max=CFG["epochs"] * len(train_loader),
                                  eta_min=CFG["scheduler_eta_min"])

    model, optimizer, train_loader, val_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader)

    # ---- train/val loops ----
    def run_epoch(epoch):
        train_ds.set_epoch(epoch)
        model.train()
        agg = {"total": 0.0, "ce": 0.0, "dcs": 0.0, "n": 0}
        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{CFG['epochs']} [train]",
                    leave=False, disable=not accelerator.is_main_process)
        for step, (img_clean, img_deg, labels) in enumerate(pbar):
            img_clean = img_clean.to(device, non_blocking=True)
            img_deg = img_deg.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with _autocast():
                total, ce, dcs = model.compute_loss(img_clean, img_deg, labels)
            accelerator.backward(total)
            accelerator.clip_grad_norm_(
                (p for p in model.parameters() if p.requires_grad),
                CFG["max_grad_norm"])
            optimizer.step()
            scheduler.step()
            bs = labels.size(0)
            agg["total"] += float(total.detach()) * bs
            agg["ce"] += float(ce.detach()) * bs
            agg["dcs"] += float(dcs.detach()) * bs
            agg["n"] += bs
            if step % 20 == 0:
                logger.debug(
                    f"ep{epoch} step {step}/{len(train_loader)} | "
                    f"total {agg['total']/max(agg['n'],1):.4f} "
                    f"(ce {agg['ce']/max(agg['n'],1):.4f}, "
                    f"dcs {agg['dcs']/max(agg['n'],1):.4f}) | "
                    f"lr {optimizer.param_groups[0]['lr']:.2e}")
            pbar.set_postfix(
                total=f"{agg['total']/max(agg['n'],1):.4f}",
                ce=f"{agg['ce']/max(agg['n'],1):.4f}",
                dcs=f"{agg['dcs']/max(agg['n'],1):.4f}",
                lr=f"{optimizer.param_groups[0]['lr']:.1e}")
        return {k: agg[k] / max(agg["n"], 1) for k in ("total", "ce", "dcs")}

    @torch.no_grad()
    def run_val():
        model.eval()
        agg = {"total": 0.0, "n": 0}
        scores, labels_all = [], []
        for img_clean, img_deg, labels in val_loader:
            img_clean = img_clean.to(device, non_blocking=True)
            img_deg = img_deg.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with _autocast():
                total, _, _ = model.compute_loss(img_clean, img_deg, labels)
                out = model(img_clean)
            logits = out["logits"].float()
            scores.append((logits[:, 1] - logits[:, 0]).cpu())
            labels_all.append(labels.cpu())
            agg["total"] += float(total) * labels.size(0)
            agg["n"] += labels.size(0)
        scores = torch.cat(scores).numpy()
        labels_all = torch.cat(labels_all).numpy()
        return {
            "val_loss": agg["total"] / max(agg["n"], 1),
            "val_auc": roc_auc_score(labels_all, scores),
            "val_bacc": balanced_accuracy_score(labels_all, scores > 0),
        }

    def save_ckpt(path, epoch, train_log, extra=None):
        state = {
            "model": accelerator.unwrap_model(model).state_dict(),
            "epoch": epoch, "config": CFG, "train_log": train_log,
        }
        if extra:
            state.update(extra)
        accelerator.save(state, path)

    # ---- main loop ----
    log_csv = output_dir / "train_log.csv"
    with open(log_csv, "w", encoding="utf-8") as f:
        f.write("epoch,train_loss,train_ce,train_dcs,val_loss,val_auc,val_bacc\n")

    train_log = []
    best_bacc = 0.0
    for epoch in range(1, CFG["epochs"] + 1):
        tr = run_epoch(epoch)
        va = run_val()
        logger.debug(f"--> epoch {epoch}: train total={tr['total']:.4f} "
                     f"(ce={tr['ce']:.4f}, dcs={tr['dcs']:.4f}) | "
                     f"val loss={va['val_loss']:.4f} auc={va['val_auc']:.4f} "
                     f"bacc={va['val_bacc']:.4f}")
        train_log.append({"epoch": epoch, "train_loss": tr["total"],
                          "train_ce": tr["ce"], "train_dcs": tr["dcs"], **va})
        if accelerator.is_main_process:
            with open(log_csv, "a", encoding="utf-8") as f:
                f.write(f"{epoch},{tr['total']:.4f},{tr['ce']:.4f},{tr['dcs']:.4f},"
                        f"{va['val_loss']:.4f},{va['val_auc']:.4f},"
                        f"{va['val_bacc']:.4f}\n")
            if va["val_bacc"] > best_bacc:
                best_bacc = va["val_bacc"]
                save_ckpt(output_dir / "bfree_globalforge_lora_r16_best.pth",
                          epoch, train_log, extra={"val_bacc": best_bacc})
                logger.info(f"[*] best checkpoint saved (epoch {epoch}, "
                            f"bAcc {best_bacc:.4f})")
        logger.info(f"--> epoch {epoch}/{CFG['epochs']} done | best bAcc "
                    f"so far = {best_bacc:.4f}")

    if accelerator.is_main_process:
        save_ckpt(output_dir / "bfree_globalforge_lora_r16.pth",
                  CFG["epochs"], train_log)

        # training curves (matplotlib Agg — không cần display)
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import pandas as pd

        df = pd.read_csv(log_csv)
        fig, axes = plt.subplots(1, 3, figsize=(16, 4), dpi=120)
        axes[0].plot(df["epoch"], df["train_loss"], marker="o", label="total")
        axes[0].plot(df["epoch"], df["train_ce"], marker="s", label="CE")
        axes[0].plot(df["epoch"], df["train_dcs"], marker="^", label="DCS")
        axes[0].set_xlabel("epoch"); axes[0].set_ylabel("train loss")
        axes[0].legend(); axes[0].set_title("Train losses (CE & DCS)")
        axes[1].plot(df["epoch"], df["val_loss"], marker="o", color="tab:red")
        axes[1].set_xlabel("epoch"); axes[1].set_ylabel("val loss")
        axes[1].set_title(f"Val loss ({CFG['val_fake_variant']})")
        axes[2].plot(df["epoch"], df["val_auc"], marker="o", label="AUC")
        axes[2].plot(df["epoch"], df["val_bacc"], marker="s", label="bAcc")
        axes[2].set_xlabel("epoch"); axes[2].set_title("Val metrics")
        axes[2].legend()
        fig.tight_layout()
        fig.savefig(output_dir / "training_curves.png")
        plt.close(fig)
        logger.info("saved training_curves.png")

    logger.info("TRAINING COMPLETE")
    accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
