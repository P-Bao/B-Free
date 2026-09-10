"""Regenerate the three Kaggle notebooks (01/02/03) as .ipynb JSON.

Usage: python notebooks/_build_notebooks.py
After writing, validates each notebook with json.tool + AST compile
(rule from agent.md: never skip notebook validation).

Edit cell sources here and re-run to regenerate the .ipynb files;
do not hand-edit the .ipynb outputs.
"""

import ast
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))

KAGGLE_NOTEBOOK_META = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
}


def nb(cells):
    out_cells = []
    for kind, src in cells:
        if kind == "md":
            cell = {"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)}
        else:
            cell = {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": src.splitlines(keepends=True),
            }
        cell["id"] = f"cell-{len(out_cells):03d}"
        out_cells.append(cell)
    return {"cells": out_cells, "metadata": KAGGLE_NOTEBOOK_META, "nbformat": 4, "nbformat_minor": 5}


def strip_magics(src):
    """Replace IPython magics/shell-escapes (!pip, %magic, multi-line continuations)
    with a `pass` line at the same indentation so the remaining source stays
    valid Python for AST compile."""
    out, in_magic = [], False
    for line in src.splitlines(keepends=True):
        if in_magic:
            if line.rstrip().endswith("\\"):
                continue
            in_magic = False
            continue
        stripped = line.lstrip()
        if stripped.startswith(("!", "%")):
            indent = line[: len(line) - len(stripped)]
            out.append(f"{indent}pass  # <ipython-magic>\n")
            if line.rstrip().endswith("\\"):
                in_magic = True
            continue
        out.append(line)
    return "".join(out)


def write_nb(name, cells):
    path = os.path.join(HERE, name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb(cells), f, indent=1, ensure_ascii=False)
        f.write("\n")
    with open(path, encoding="utf-8") as f:
        doc = json.load(f)
    assert doc["nbformat"] == 4
    n_code = 0
    for cell in doc["cells"]:
        if cell["cell_type"] == "code":
            n_code += 1
            ast.parse(strip_magics("".join(cell["source"])), filename=f"{name}#cell")
    print(f"OK {name}: {len(doc['cells'])} cells ({n_code} code, all AST-compiled)")
    return path


OFFLINE_ENV = """import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1\""""

PIP_INSTALL_CELLS = [
    ("code", f"""{OFFLINE_ENV}
import os, sys, glob

WHEELS_DIR = "/kaggle/input/bfree-wheels"
assert os.path.isdir(WHEELS_DIR), (
    f"Wheel bundle not found at {{WHEELS_DIR}}. "
    "Attach the 'bfree-wheels' Kaggle dataset before running this notebook."
)
wheels = sorted(glob.glob(os.path.join(WHEELS_DIR, "*.whl")))
print(f"Found {{len(wheels)}} wheels in {{WHEELS_DIR}}")
assert wheels, "No .whl files found in the wheel bundle."

!pip install --no-index --find-links={{WHEELS_DIR}} \\
    torch==2.8.0+cu128 torchvision==0.23.0+cu128
!pip install --no-index --find-links={{WHEELS_DIR}} \\
    timm==1.0.22 peft==0.15.2 transformers==4.55.4 \\
    pandas==2.3.3 numpy==1.26.4 matplotlib==3.11.1 seaborn==0.13.2 \\
    scikit-learn scipy pyyaml pillow tqdm safetensors"""),
]

CLONE_CELL = ("code", """REPO_URL = "https://github.com/P-Bao/B-Free.git"
REPO_DIR = "/kaggle/working/B-Free"
BRANCH = "integration/loss-backbone"

import os, sys, glob
if not os.path.isdir(REPO_DIR):
    !git clone --branch {BRANCH} {REPO_URL} {REPO_DIR}
else:
    print(f"{REPO_DIR} already cloned.")

assert os.path.isfile(os.path.join(REPO_DIR, "code", "networks", "bfree_globalforge_vit.py")), "Clone failed: backbone file missing."
stubs = glob.glob(os.path.join(REPO_DIR, "code", "modules", "*_stub.py"))
assert not stubs, f"Stub files still present (K0 not merged?): {stubs}"
print("OK: repo cloned on integration/loss-backbone, no stub files (K0 verified).")""")

# =====================================================================
# Notebook 01 — B-Free Setup Bundle (K1)
# =====================================================================

NB01_CELLS = [
    ("md", """# B-Free x GlobalForge — Notebook 01: Setup Bundle Builder (K1)

Chạy trên session **CPU hoặc T4×2, Internet ON**. Notebook này **tạo** các assets offline cho notebooks 02/03 (những notebook chạy trên RTX PRO 6000 Blackwell 96GB, hoàn toàn offline — không pip internet, không HF hub):

1. `/kaggle/working/bfree-wheels/` — wheel bundle cho **Linux x86_64, Python 3.11** (Kaggle dùng chung image Linux x86_64 Python 3.11 cho mọi accelerator, nên wheels tải ở session CPU/T4 khớp session RTX PRO 6000):
   `torch==2.8.0+cu128`, `torchvision==0.23.0+cu128` (Blackwell sm_120 cần >=2.8 — risk table), `timm==1.0.22`, `peft==0.15.2`, `transformers==4.55.4`, `pandas==2.3.3` (MUST <3), `numpy==1.26.4`, `matplotlib==3.11.1`, `seaborn==0.13.2`, `scikit-learn`, `scipy`, `pyyaml`, `pillow`, `tqdm`, `safetensors`, `huggingface_hub` (+ dependencies).
2. `/kaggle/working/dinov2-vitb14-reg4-pretrain/` — weights DINOv2 ViT-B/14 reg4 (`timm/vit_base_patch14_reg4_dinov2.lvd142m`, `model.safetensors`) cho offline pretrained init (D5) của notebook 02.

Notebook 02/03 sẽ `pip install --no-index` từ bundle này (rule agent.md: offline, không apt-get, không HF hub).

Sau khi chạy xong: **Save Version → Save & Run All (Commit)**, rồi từ tab **Output** tạo 2 Kaggle Datasets (`bfree-wheels`, `dinov2-vitb14-reg4-pretrain`) và attach vào notebooks 02/03."""),

    ("code", """import platform
import sys

print(f"Python  : {sys.version.split()[0]} (bundle target: Kaggle Linux x86_64, Python 3.11)")
print(f"Platform: {platform.platform()}")
try:
    import torch
    print(f"torch (session, preinstalled): {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)} — not required here (CPU session is fine)")
except ImportError:
    print("torch not preinstalled in this session — fine, this notebook only downloads wheels.")"""),

    ("code", """WHEELS_DIR = "/kaggle/working/bfree-wheels"

!pip download --quiet --dest {WHEELS_DIR} \\
    --index-url https://download.pytorch.org/whl/cu128 \\
    --extra-index-url https://pypi.org/simple \\
    torch==2.8.0+cu128 torchvision==0.23.0+cu128

!pip download --quiet --dest {WHEELS_DIR} \\
    timm==1.0.22 peft==0.15.2 transformers==4.55.4 \\
    pandas==2.3.3 numpy==1.26.4 matplotlib==3.11.1 seaborn==0.13.2 \\
    scikit-learn scipy pyyaml pillow tqdm safetensors huggingface_hub

import glob
import os

wheels = sorted(glob.glob(os.path.join(WHEELS_DIR, "*.whl")))
assert wheels, "pip download produced no wheels."
print(f"{len(wheels)} wheels, {sum(os.path.getsize(w) for w in wheels) / 1024**3:.2f} GB -> {WHEELS_DIR}")"""),

    ("code", """!pip install --quiet timm==1.0.22 peft==0.15.2 transformers==4.55.4 pyyaml safetensors huggingface_hub"""),
    CLONE_CELL,

    ("code", """import sys

sys.path.insert(0, os.path.join(REPO_DIR, "code"))

import torch
import timm
import peft
import transformers
import yaml

print(f"torch (session) = {torch.__version__}")
print(f"timm            = {timm.__version__}")
print(f"peft            = {peft.__version__}")
print(f"transformers    = {transformers.__version__}")

from networks.bfree_globalforge_vit import BFreeGlobalForgeViT
from configs.loader import load_config, build_model_kwargs
from datasets.bfree_dataset import BFreeDataset, DegradationPipeline
from modules.lib_adapter import LIBAdapter
from modules.gsr_adapter import GSRAdapter
from modules.dcs_loss import DCSLoss, info_nce_loss

cfg = load_config(os.path.join(REPO_DIR, "code", "configs", "bfree_dcs.yaml"))
kwargs = build_model_kwargs(cfg)
print("\\nModel kwargs from bfree_dcs.yaml:")
for k, v in kwargs.items():
    print(f"  {k} = {v}")
print("\\nIMPORT VALIDATION OK")"""),

    ("code", """import os
import shutil

from huggingface_hub import hf_hub_download

PRETRAIN_DIR = "/kaggle/working/dinov2-vitb14-reg4-pretrain"
os.makedirs(PRETRAIN_DIR, exist_ok=True)

src = hf_hub_download(repo_id="timm/vit_base_patch14_reg4_dinov2.lvd142m",
                      filename="model.safetensors")
dst = os.path.join(PRETRAIN_DIR, "model.safetensors")
shutil.copyfile(src, dst)
print(f"saved: {dst} ({os.path.getsize(dst) / 1024**2:.1f} MB)")

from safetensors.torch import load_file

sd = load_file(dst)
embed_dim = sd["patch_embed.proj.weight"].shape[0]
pe = sd["pos_embed"]
print(f"state dict: {len(sd)} tensors | embed_dim={embed_dim} | pos_embed={tuple(pe.shape)}")
assert embed_dim == 768, "not a ViT-B checkpoint"
assert tuple(pe.shape) == (1, 37 * 37 + 5, 768), (
    f"unexpected pos_embed shape {tuple(pe.shape)} — expected 518px grid (37x37) + 5 prefix tokens")
print("DINOv2 ViT-B/14 reg4 weights OK (518px grid 37x37 + 5 prefix tokens)")"""),

    ("code", """import torch

model_smoke = BFreeGlobalForgeViT(img_size=224, pretrained=False)
x = torch.randn(2, 3, 224, 224)
with torch.no_grad():
    out = model_smoke(x)
print(f"CPU smoke test: logits {tuple(out['logits'].shape)}, cls {tuple(out['cls'].shape)}")
assert out["logits"].shape == (2, 2) and out["cls"].shape == (2, 768)
del model_smoke, x, out
print("MODEL SMOKE OK")"""),

    ("code", """import glob
import os

PIN_STACK = {
    "torch": "2.8.0+cu128",
    "torchvision": "0.23.0+cu128",
    "timm": "1.0.22",
    "peft": "0.15.2",
    "transformers": "4.55.4",
    "pandas": "2.3.3",
    "numpy": "1.26.4",
    "matplotlib": "3.11.1",
    "seaborn": "0.13.2",
}

wheels = sorted(glob.glob(os.path.join(WHEELS_DIR, "*.whl")))
names = {os.path.basename(w).split("-")[0].replace("_", "-").lower() for w in wheels}
missing = [f"{pkg}=={ver}" for pkg, ver in PIN_STACK.items() if pkg not in names]
assert not missing, f"Missing pinned wheels in bundle: {missing}"
assert os.path.isfile(os.path.join(PRETRAIN_DIR, "model.safetensors")), "model.safetensors missing"

total_gb = sum(os.path.getsize(w) for w in wheels) / 1024**3
print(f"bundle check OK: {len(wheels)} wheels ({total_gb:.2f} GB) + DINOv2 ViT-B/14 reg4 weights")
for w in wheels[:40]:
    print(f"  {os.path.basename(w)}")"""),

    ("md", """## Bundle Ready (K1 pass)

Output trong `/kaggle/working/`:
- `bfree-wheels/` — wheel bundle pin stack đầy đủ (Linux x86_64, Python 3.11) cho `pip install --no-index`
- `dinov2-vitb14-reg4-pretrain/model.safetensors` — DINOv2 ViT-B/14 reg4 pretrained (D5); notebook 02 sẽ resample `pos_embed` 518px→504px khi load

**Bước tiếp theo (bắt buộc):**
1. **Save Version → Save & Run All (Commit)** để chốt output.
2. Tab **Output** của notebook này → **New Dataset** cho từng thư mục:
   - `bfree-wheels/` → dataset `bfree-wheels`
   - `dinov2-vitb14-reg4-pretrain/` → dataset `dinov2-vitb14-reg4-pretrain`
3. Attach 2 dataset đó vào `02_bfree_kaggle_train.ipynb` và `03_bfree_kaggle_eval.ipynb` (chạy offline trên RTX PRO 6000)."""),
]

# =====================================================================
# Notebook 02 — Kaggle Training (K2)
# =====================================================================

NB02_CELLS = [
    ("md", """# B-Free x GlobalForge — Notebook 02: Kaggle Training (K2, LoRA r=16)

Huấn luyện **DINOv2 ViT-B/14 reg4 + LIB + GSR + L_DCS** (LoRA r=16, α=32) trên B-Free training data (51.517 real + 309.102 fake = 6 fake variants/ID), 504×504, 8 epochs, bf16.

**Locked decisions (plan.md D1–D10):**
- D1 `lambda_dcs=0.01`, `dcs_tau=0.07`, `label_smoothing=0.1`
- D2 8 epochs @ 504px — D3 full data (no subsample)
- D4 50/50 per-ID pairing: p=0.5 real else random 1/6 fake variant (resample mỗi epoch)
- D5 DINOv2 pretrained init **offline từ local weights** (không HF hub)
- D6 auto batch size (0.9× VRAM) — D7 bf16 autocast
- AdamW lr=1e-4 wd=1e-4 · CosineAnnealingLR eta_min=1e-7 per-step · grad clip 1.0
- Val split `md5(id)%100 < 3`; mỗi val ID = 1 real + 1 deterministic fake (balanced)

**Yêu cầu Kaggle inputs:**
- `bfree-wheels` — wheel bundle **do notebook 01 tạo** (CPU/T4 online session, Save Version → New Dataset)
- `bfree-training-data` — B-Free training dataset (tải từ grip.unina.it): `COCO_real_512/` + 6 thư mục `SD2.1_*/`, fake variant dùng **cùng tên file** với ảnh real tương ứng
- `dinov2-vitb14-reg4-pretrain` — weights `timm/vit_base_patch14_reg4_dinov2.lvd142m` (`model.safetensors`, ~330MB) **do notebook 01 tạo**, cho offline pretrained init (D5)

> Notebook này chạy **offline hoàn toàn** trên RTX PRO 6000: KHÔNG apt-get, KHÔNG internet.pip (chỉ `--no-index` từ bundle), KHÔNG HF hub download trong runtime. Log riêng CE và DCS mỗi epoch (rule agent.md) → `train_log.csv` cho K4 / Phase 8."""),

    PIP_INSTALL_CELLS[0],

    CLONE_CELL,

    ("code", """import sys

sys.path.insert(0, os.path.join(REPO_DIR, "code"))

import glob
import hashlib
import random

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as TF

from networks.bfree_globalforge_vit import BFreeGlobalForgeViT
from datasets.bfree_dataset import DegradationPipeline
from train_lora import apply_lora_to_backbone
from utils.normalization import get_list_norm
from utils.dmetrics import balanced_accuracy_score, roc_auc_score

DEVICE = "cuda:0"
print(f"torch={torch.__version__}")
print(f"GPU: {torch.cuda.get_device_name(0)}, "
      f"{(getattr(torch.cuda.get_device_properties(0), 'total_mem', None) or getattr(torch.cuda.get_device_properties(0), 'total_memory')) / 1024**3:.1f} GB")"""),

    ("code", """# ============ Hyperparameters (locked D1-D10 — mọi giá trị ở đây, không hardcode chỗ khác) ============
CONFIG = {
    "arch": "vit_base_patch14_reg4_dinov2.lvd142m",
    "num_classes": 2,
    "img_size": 504,
    "pretrained": True,            # D5: DINOv2 pretrained init (loaded offline ở cell model)
    "lambda_dcs": 0.01,            # D1
    "dcs_tau": 0.07,
    "label_smoothing": 0.1,
    "lib_kernel": 3,
    "lib_tau": 0.5,
    "gsr_window": 3,
    "gsr_mask_prob": 1.0,
    "epochs": 8,                   # D2
    "lora_rank": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "lora_targets": ["qkv", "proj", "fc1", "fc2"],   # GlobalForge convention (apply_lora_to_backbone)
    "lr": 1e-4,
    "weight_decay": 1e-4,
    "max_grad_norm": 1.0,
    "scheduler_eta_min": 1e-7,
    "max_vram_frac": 0.9,           # D6
    "batch_hard_cap": 128,
    "val_md5_percentile": 3,        # md5(id)%100 < 3
    "val_fake_variant": "SD2.1_selfconditioned",
    "pair_real_prob": 0.5,          # D4
    "num_workers": 4,
    "seed": 42,
    "data_root": "/kaggle/input/bfree-training-data",
    "pretrain_weights_glob": "/kaggle/input/dinov2-vitb14-reg4-pretrain/*",
    "output_dir": "/kaggle/working",
}

DATA_ROOT = CONFIG["data_root"]
assert os.path.isdir(DATA_ROOT), (
    f"Training data not found at {DATA_ROOT}. Attach 'bfree-training-data' "
    "(download from https://www.grip.unina.it/download/prog/B-Free/training_data/)."
)

random.seed(CONFIG["seed"])
np.random.seed(CONFIG["seed"])
torch.manual_seed(CONFIG["seed"])
torch.cuda.manual_seed_all(CONFIG["seed"])

REAL_DIR = os.path.join(DATA_ROOT, "COCO_real_512")
FAKE_DIRS = sorted(d for d in glob.glob(os.path.join(DATA_ROOT, "SD2.1_*")) if os.path.isdir(d))
assert os.path.isdir(REAL_DIR), f"COCO_real_512 missing under {DATA_ROOT}"
assert len(FAKE_DIRS) == 6, f"Expected 6 SD2.1_* variant dirs, found {len(FAKE_DIRS)}: {FAKE_DIRS}"
for d in FAKE_DIRS:
    print(f"  {os.path.basename(d)}: {len(glob.glob(os.path.join(d, '*')))} files")"""),

    ("md", """### Dataset — 50/50 per-ID pairing (D4)

`BFreeDataset` của repo nhận CSV tĩnh (filename,label) nên không pairing động theo ID được; cell dưới reuses `DegradationPipeline` của repo và cài đúng convention:

- **Train**: 1 sample = 1 ID; `p=0.5` → real, ngược lại random 1/6 fake variant (resample mỗi `__getitem__` → 8 epochs phủ hết các variant).
- **Val** (`md5(stem)%100 < 3`): mỗi ID cho 2 samples — index chẵn = real, lẻ = deterministic fake `SD2.1_selfconditioned` (fallback: variant khác có sẵn) → balanced.
- Crop: train RandomCrop 504 (ảnh 512×512) + hflip 0.5; val center crop 504 — giống `BFreeDataset`.
- Degraded view (cho L_DCS): áp cho cả train lẫn val (matching `BFreeDataset` — degradation unconditional), pipeline GlobalForge JPEG 20-80 → blur k7 σ0.5-1.5 p0.8 → jitter p0.8."""),

    ("code", """def build_stem_index(dirs):
    maps = []
    for d in dirs:
        m = {}
        for p in glob.glob(os.path.join(d, "*")):
            if os.path.isfile(p):
                m[os.path.splitext(os.path.basename(p))[0]] = p
        maps.append(m)
    return maps

REAL_MAP = build_stem_index([REAL_DIR])[0]
FAKE_MAPS = build_stem_index(FAKE_DIRS)
ALL_STEMS = sorted(REAL_MAP.keys())
print(f"real={len(REAL_MAP)}, fake variants={[len(m) for m in FAKE_MAPS]}")

def is_val_id(stem):
    return int(hashlib.md5(stem.encode()).hexdigest(), 16) % 100 < CONFIG["val_md5_percentile"]

TRAIN_IDS = [s for s in ALL_STEMS if not is_val_id(s)]
VAL_IDS = [s for s in ALL_STEMS if is_val_id(s)]
print(f"train IDs={len(TRAIN_IDS)}, val IDs={len(VAL_IDS)} (~{CONFIG['val_md5_percentile']}%)")

class PairDataset(Dataset):
    \"\"\"Train: p=0.5 real else random 1/6 fake variant (resample mỗi epoch).
    Val: 2 samples/ID — chẵn=real, lẻ=deterministic fake variant (balanced).\"\"\"

    def __init__(self, ids, img_size, is_train, pair_real_prob, val_fake_variant):
        self.ids = ids
        self.img_size = img_size
        self.is_train = is_train
        self.pair_real_prob = pair_real_prob
        self.val_fake_variant = val_fake_variant
        self.normalize = T.Compose(get_list_norm("resnet"))
        self.degradation = DegradationPipeline()

    def __len__(self):
        return len(self.ids) * (1 if self.is_train else 2)

    def _pick_val_fake(self, stem):
        for m, d in zip(FAKE_MAPS, FAKE_DIRS):
            if os.path.basename(d) == self.val_fake_variant and stem in m:
                return m[stem]
        for m in FAKE_MAPS:
            if stem in m:
                return m[stem]
        return None

    def __getitem__(self, i):
        stem = self.ids[i if self.is_train else i // 2]
        if self.is_train:
            if random.random() < self.pair_real_prob or not any(stem in m for m in FAKE_MAPS):
                path, label = REAL_MAP[stem], 0
            else:
                maps = [m for m in FAKE_MAPS if stem in m]
                path = maps[random.randrange(len(maps))][stem]
                label = 1
        else:
            if i % 2 == 0:
                path, label = REAL_MAP[stem], 0
            else:
                path = self._pick_val_fake(stem)
                label = 1
        assert path is not None, f"No image found for id {stem}"
        img = Image.open(path).convert("RGB")

        if self.is_train:
            t, l, h, w = T.RandomCrop.get_params(img, (self.img_size, self.img_size))
            img = TF.crop(img, t, l, h, w)
            if random.random() > 0.5:
                img = TF.hflip(img)
        else:
            img = TF.center_crop(img, (self.img_size, self.img_size))

        img_deg = self.degradation(img)
        return self.normalize(img), self.normalize(img_deg), label

train_ds = PairDataset(TRAIN_IDS, CONFIG["img_size"], True, CONFIG["pair_real_prob"], CONFIG["val_fake_variant"])
val_ds = PairDataset(VAL_IDS, CONFIG["img_size"], False, CONFIG["pair_real_prob"], CONFIG["val_fake_variant"])

a, b, c = train_ds[0]
print(f"train_ds: {len(train_ds)} samples | clean {tuple(a.shape)}, deg {tuple(b.shape)}, label {c}")
a, b, c = val_ds[0]
_, _, c2 = val_ds[1]
print(f"val_ds  : {len(val_ds)} samples | labels {c},{c2} (expect 0,1)")
assert tuple(a.shape) == (3, CONFIG["img_size"], CONFIG["img_size"])

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(CONFIG["seed"])"""),

    ("md", """### Model — BFreeGlobalForgeViT + LoRA r=16 (offline pretrained init)

- Backbone `timm vit_base_patch14_reg4_dinov2.lvd142m`, `set_input_size(504)`, `num_classes=2`.
- **Offline pretrained init (D5):** checkpoint DINOv2 mặc định là 518px (grid 37×37) → load vào model 504 (grid 36×36) cần **resample `pos_embed`** bằng `timm.layers.resample_abs_pos_embed` (đúng cách timm làm nội bộ). Head 2-class + LIB/GSR missing khi `strict=False` là behavior đúng.
- LoRA r=16 α=32 dropout 0.05 target `qkv/proj/fc1/fc2` qua `apply_lora_to_backbone` của repo; LIB + GSR + `fc_norm` + `head` vẫn full-train."""),

    ("code", """import math

import torch


def load_pretrained_backbone(model, glob_pattern):
    \"\"\"D5 offline: load DINOv2 pretrained weights (.safetensors/.pth) vào model.model.
    Resample pos_embed nếu grid checkpoint (thường 518px/37×37) khác grid model (504px/36×36).
    Filter key+shape (bỏ head 1000-class của ckpt nếu có) trước khi load strict=False.\"\"\"
    matches = [m for m in sorted(glob.glob(glob_pattern)) if m.endswith((".safetensors", ".pth", ".pt"))]
    assert matches, f"No pretrained weights match {glob_pattern} — attach the dinov2 dataset."
    path = matches[0]
    print(f"Loading pretrained backbone from: {path}")

    if path.endswith(".safetensors"):
        from safetensors.torch import load_file
        sd = load_file(path)
    else:
        sd = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(sd, dict) and isinstance(sd.get("model"), dict):
        sd = sd["model"]
    sd = {k[len("model."):] if k.startswith("model.") else k: v for k, v in sd.items()}

    prefix = model.model.num_prefix_tokens
    if "pos_embed" in sd:
        pe = sd["pos_embed"]
        n_patch_old = pe.shape[1] - prefix
        old_hw = int(math.isqrt(n_patch_old))
        new_hw = model.model.patch_embed.grid_size[0]
        if old_hw != new_hw:
            from timm.layers.pos_embed import resample_abs_pos_embed
            print(f"resample pos_embed: {old_hw}x{old_hw} -> {new_hw}x{new_hw}")
            sd["pos_embed"] = resample_abs_pos_embed(
                pe, new_size=model.model.patch_embed.grid_size,
                old_size=(old_hw, old_hw), num_prefix_tokens=prefix,
            )

    ref = model.model.state_dict()
    dropped = [k for k, v in sd.items() if k not in ref or ref[k].shape != v.shape]
    if dropped:
        print(f"dropped {len(dropped)} incompatible keys (e.g. {dropped[:4]})")
    sd = {k: v for k, v in sd.items() if k not in dropped}

    report = model.model.load_state_dict(sd, strict=False)
    print(f"missing={len(report.missing_keys)} unexpected={len(report.unexpected_keys)}")
    if report.missing_keys:
        print("  missing:", report.missing_keys[:8])
    assert set(report.missing_keys) <= {"head.weight", "head.bias"}, (
        f"Backbone not fully initialized, missing: {report.missing_keys[:10]}")
    assert not report.unexpected_keys
    return model


model = BFreeGlobalForgeViT(
    arch=CONFIG["arch"], num_classes=CONFIG["num_classes"],
    img_size=CONFIG["img_size"], pretrained=False,
    use_lib=True, use_gsr=True, use_dcs=True,
    lib_kernel=CONFIG["lib_kernel"], lib_tau=CONFIG["lib_tau"],
    gsr_window=CONFIG["gsr_window"], gsr_mask_prob=CONFIG["gsr_mask_prob"],
    dcs_tau=CONFIG["dcs_tau"], lambda_dcs=CONFIG["lambda_dcs"],
    label_smoothing=CONFIG["label_smoothing"],
)
model = load_pretrained_backbone(model, CONFIG["pretrain_weights_glob"])
model = apply_lora_to_backbone(model, r=CONFIG["lora_rank"],
                               lora_alpha=CONFIG["lora_alpha"], lora_dropout=CONFIG["lora_dropout"])
model.to(DEVICE)

n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
n_total = sum(p.numel() for p in model.parameters())
print(f"trainable params: {n_train:,} / {n_total:,} ({100 * n_train / n_total:.2f}%)")"""),

    ("md", """### Auto batch size finder (D6)

Binary search batch size lớn nhất mà `compute_loss` (2 forwards: clean + degraded, bf16 autocast, backward) chạy được không OOM. Bắt đầu từ 2, doubling tới khi OOM hoặc hard cap, rồi binary search giữa khoảng đó."""),

    ("code", """def try_batch(model, bs, img_size, device):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    try:
        x1 = torch.randn(bs, 3, img_size, img_size, device=device)
        x2 = torch.randn(bs, 3, img_size, img_size, device=device)
        y = torch.randint(0, 2, (bs,), device=device)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
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
        print(f"[warn] no OOM up to hard cap {hard_cap}")
        hi = lo + 1
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if try_batch(model, mid, img_size, device):
            lo = mid
        else:
            hi = mid
    return lo

BATCH_SIZE = find_max_batch_size(model, CONFIG["img_size"], DEVICE, CONFIG["batch_hard_cap"])
CONFIG["batch_size"] = BATCH_SIZE
print(f"AUTO BATCH SIZE = {BATCH_SIZE}")"""),

    ("md", """### Training loop — 8 epochs, bf16 autocast, grad clip 1.0

- `total = CE + lambda_dcs * DCS` qua `model.compute_loss` — log **riêng** CE và DCS.
- AdamW (lr 1e-4, wd 1e-4) trên trainable params; CosineAnnealingLR `T_max = epochs * len(train_loader)`, `eta_min=1e-7`, step mỗi batch.
- Lưu checkpoint **best theo val_bAcc** + checkpoint cuối; `train_log.csv` (epoch, train_loss, train_ce, train_dcs, val_loss, val_auc, val_bacc)."""),

    ("code", """train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                         num_workers=CONFIG["num_workers"], pin_memory=True, drop_last=True,
                         persistent_workers=True, worker_init_fn=seed_worker, generator=g)
val_loader = DataLoader(val_ds, batch_size=max(2, BATCH_SIZE // 2), shuffle=False,
                        num_workers=CONFIG["num_workers"], pin_memory=True)

optimizer = AdamW((p for p in model.parameters() if p.requires_grad),
                  lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"])
scheduler = CosineAnnealingLR(optimizer, T_max=CONFIG["epochs"] * len(train_loader),
                              eta_min=CONFIG["scheduler_eta_min"])


def run_epoch(epoch):
    model.train()
    agg = {"total": 0.0, "ce": 0.0, "dcs": 0.0, "n": 0}
    for step, (img_clean, img_deg, labels) in enumerate(train_loader):
        img_clean = img_clean.to(DEVICE, non_blocking=True)
        img_deg = img_deg.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            total, ce, dcs = model.compute_loss(img_clean, img_deg, labels)
        total.backward()
        torch.nn.utils.clip_grad_norm_(
            (p for p in model.parameters() if p.requires_grad), CONFIG["max_grad_norm"])
        optimizer.step()
        scheduler.step()
        bs = labels.size(0)
        agg["total"] += float(total.detach()) * bs
        agg["ce"] += float(ce.detach()) * bs
        agg["dcs"] += float(dcs.detach()) * bs
        agg["n"] += bs
        if step % 20 == 0:
            print(f"ep{epoch} step {step}/{len(train_loader)} | "
                  f"total {agg['total']/max(agg['n'],1):.4f} (ce {agg['ce']/max(agg['n'],1):.4f}, "
                  f"dcs {agg['dcs']/max(agg['n'],1):.4f}) | lr {optimizer.param_groups[0]['lr']:.2e}",
                  flush=True)
    return {k: agg[k] / max(agg["n"], 1) for k in ("total", "ce", "dcs")}


@torch.no_grad()
def run_val():
    model.eval()
    agg = {"total": 0.0, "n": 0}
    scores, labels_all = [], []
    for img_clean, img_deg, labels in val_loader:
        img_clean = img_clean.to(DEVICE, non_blocking=True)
        img_deg = img_deg.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
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


LOG_CSV = os.path.join(CONFIG["output_dir"], "train_log.csv")
with open(LOG_CSV, "w", encoding="utf-8") as f:
    f.write("epoch,train_loss,train_ce,train_dcs,val_loss,val_auc,val_bacc\\n")

TRAIN_LOG = []
best_bacc = 0.0
for epoch in range(1, CONFIG["epochs"] + 1):
    tr = run_epoch(epoch)
    va = run_val()
    print(f"--> epoch {epoch}: train total={tr['total']:.4f} (ce={tr['ce']:.4f}, dcs={tr['dcs']:.4f}) | "
          f"val loss={va['val_loss']:.4f} auc={va['val_auc']:.4f} bacc={va['val_bacc']:.4f}", flush=True)
    row = {"epoch": epoch, "train_loss": tr["total"], "train_ce": tr["ce"],
           "train_dcs": tr["dcs"], **va}
    TRAIN_LOG.append(row)
    with open(LOG_CSV, "a", encoding="utf-8") as f:
        f.write(f"{epoch},{tr['total']:.4f},{tr['ce']:.4f},{tr['dcs']:.4f},"
                f"{va['val_loss']:.4f},{va['val_auc']:.4f},{va['val_bacc']:.4f}\\n")
    if va["val_bacc"] > best_bacc:
        best_bacc = va["val_bacc"]
        torch.save({"model": model.state_dict(), "epoch": epoch, "val_bacc": best_bacc,
                    "config": CONFIG},
                   os.path.join(CONFIG["output_dir"], "bfree_globalforge_lora_r16_best.pth"))
        print(f"[*] best checkpoint saved (epoch {epoch}, bAcc {best_bacc:.4f})", flush=True)

torch.save({"model": model.state_dict(), "epoch": CONFIG["epochs"], "config": CONFIG,
            "train_log": TRAIN_LOG},
           os.path.join(CONFIG["output_dir"], "bfree_globalforge_lora_r16.pth"))
print("TRAINING COMPLETE")"""),

    ("code", """import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv(LOG_CSV)
display(df)

fig, axes = plt.subplots(1, 3, figsize=(16, 4), dpi=120)
axes[0].plot(df["epoch"], df["train_loss"], marker="o", label="total")
axes[0].plot(df["epoch"], df["train_ce"], marker="s", label="CE")
axes[0].plot(df["epoch"], df["train_dcs"], marker="^", label="DCS")
axes[0].set_xlabel("epoch"); axes[0].set_ylabel("train loss"); axes[0].legend()
axes[0].set_title("Train losses (CE & DCS logged separately)")
axes[1].plot(df["epoch"], df["val_loss"], marker="o", color="tab:red")
axes[1].set_xlabel("epoch"); axes[1].set_ylabel("val loss"); axes[1].set_title("Val loss")
axes[2].plot(df["epoch"], df["val_auc"], marker="o", label="AUC")
axes[2].plot(df["epoch"], df["val_bacc"], marker="s", label="bAcc")
axes[2].set_xlabel("epoch"); axes[2].set_title("Val metrics"); axes[2].legend()
fig.tight_layout()
fig.savefig(os.path.join(CONFIG["output_dir"], "training_curves.png"))
print("saved training_curves.png")"""),

    ("md", """## Training Complete (K2)

Outputs trong `/kaggle/working/`:
- `bfree_globalforge_lora_r16.pth` — checkpoint cuối (model + config + train_log)
- `bfree_globalforge_lora_r16_best.pth` — checkpoint best theo val_bAcc
- `train_log.csv` — CE/DCS log riêng từng epoch (dùng cho K4 / Phase 8)
- `training_curves.png`

**K2 checklist:** chạy hết 8 epochs không lỗi · checkpoint saved · training log CSV + curves plot.

**Tiếp theo:** `03_bfree_kaggle_eval.ipynb` (3 models × 17 wild subsets + standard benchmarks)."""),
]

# =====================================================================
# Notebook 03 — Kaggle Evaluation (K3)
# =====================================================================

NB03_CELLS = [
    ("md", """# B-Free x GlobalForge — Notebook 03: Evaluation (K3)

Đánh giá **3 models** trên **17 in-the-wild subsets** + **standard benchmarks** (AIGCDetect, GenImage, UnivFD, DRCT, Synthbuster, EvalGEN):

| # | Model | Protocol |
|---|-------|----------|
| 1 | Integrated LoRA (output K2) | B-Free family: multi-crop 504 — 5 crops (center + 4 corners), replicate-pad ảnh < 504, **average logits** |
| 2 | B-Free baseline `BFREE_dino2reg4` | Wrapper5crops gốc của repo (replicate + 5-crop ở mức embedding), num_classes=1 → score = logit |
| 3 | GlobalForge released `REM vit_l_a` | **Protocol code thật** (`eval_in_the_wild.py` + `Get_Transforms`): short side ≤ 1296 → center-crop 224; > 1296 → resize 1296 rồi center-crop 224; PNG → JPEG round-trip q100; `Norm(imagenet)` trong model; softmax → log-odds |

Score = `logit_fake − logit_real` (models 1, 2); model 3 quy về logit bằng `log(p_fake/p_real)` để đồng bộ metrics. Metrics: **AUC, bAcc, NLL, ECE, Pd10, EER** (từ `code/utils/dmetrics.py` của repo).

**CO-SPY: `fake_only`** (paper default) — chỉ ảnh fake, bAcc = accuracy trên fake, các metric còn lại NaN.
**Grouped average** (SynthWildx / WildRF / AIGIBench / CO-SPY / BFree): `combine_parent_dataset_average` — trung bình của trung bình các parent group (giống `eval_in_the_wild.py`).

**Yêu cầu Kaggle inputs:**
- `bfree-wheels` — wheel bundle **do notebook 01 tạo** (CPU/T4 online session, Save Version → New Dataset)
- `k2-output` — output notebook 02 (`bfree_globalforge_lora_r16.pth` hoặc `..._best.pth`)
- `bfree-baseline-weights` — thư mục `BFREE_dino2reg4/` (config.yaml + weights .pth) từ https://grip-unina.github.io/B-Free/ (weights table), upload làm Dataset
- `globalforge-code` — thư mục `code/` của GlobalForge (chứa `models/REM.py`), upload làm Dataset
- `globalforge-backbone-vitla` — HF backbone ViT-L của GlobalForge: hoặc chính là thư mục model (`config.json` ở root) hoặc chứa thư mục con `vit_l_a/`, upload làm Dataset
- `globalforge-weights` — checkpoint 13 parts `checkpoint-best.pth.part_*` (~1.25GB total), upload làm Dataset
- `wild-benchmarks` — DATA_ROOT 17 subsets, layout như `eval_in_the_wild.py` (Chameleon, synthwildx/{dalle3,firefly,midjourney_v5}, WildRF/test/{facebook,reddit,twitter}, AIGIBench/{SocialRF,CommunityAI}, CO-SPY-In-the-Wild/{civitai,dalle3,instavibeai,lexica,midjourney}, RRDataset, B-Free, realchain_CD — mỗi cái có `0_real/` + `1_fake/`); standard benchmarks nằm cùng root (`AIGCDetect/`, `GenImage/`, ... cũng `0_real/1_fake`)

> Notebook này chạy **offline hoàn toàn** trên RTX PRO 6000: KHÔNG apt-get, KHÔNG internet.pip (chỉ `--no-index` từ bundle), KHÔNG HF hub download trong runtime."""),

    PIP_INSTALL_CELLS[0],

    CLONE_CELL,

    ("code", """import sys

sys.path.insert(0, os.path.join(REPO_DIR, "code"))

import glob
import io
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as TF

from networks.bfree_globalforge_vit import BFreeGlobalForgeViT
from train_lora import apply_lora_to_backbone
from utils import dmetrics
from utils.normalization import get_list_norm

GF_CODE_DIR = "/kaggle/input/globalforge-code/code"
GF_BACKBONE_INPUT = "/kaggle/input/globalforge-backbone-vitla"
GF_CKPT_PARTS = "/kaggle/input/globalforge-weights"
K2_OUTPUT_DIR = "/kaggle/input/k2-output"
BFREE_WEIGHTS_DIR = "/kaggle/input/bfree-baseline-weights"
DATA_ROOT = "/kaggle/input/wild-benchmarks"

sys.path.append(GF_CODE_DIR)

DEVICE = "cuda:0"
print(f"torch={torch.__version__}")
print(f"GPU: {torch.cuda.get_device_name(0)}")"""),

    ("code", """# ============ Model 1: Integrated LoRA (K2 output) ============

def load_integrated_lora(k2_output_dir, device=DEVICE):
    candidates = ["bfree_globalforge_lora_r16.pth", "bfree_globalforge_lora_r16_best.pth"]
    ckpt_path = next((os.path.join(k2_output_dir, c) for c in candidates
                      if os.path.isfile(os.path.join(k2_output_dir, c))), None)
    assert ckpt_path, f"No K2 checkpoint found in {k2_output_dir} (expected {candidates})."
    print(f"Loading Integrated LoRA from: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt.get("config", {})
    model = BFreeGlobalForgeViT(
        arch=cfg.get("arch", "vit_base_patch14_reg4_dinov2.lvd142m"),
        num_classes=cfg.get("num_classes", 2),
        img_size=cfg.get("img_size", 504),
        pretrained=False, use_lib=True, use_gsr=True, use_dcs=True,
        lib_kernel=cfg.get("lib_kernel", 3), lib_tau=cfg.get("lib_tau", 0.5),
        gsr_window=cfg.get("gsr_window", 3), gsr_mask_prob=cfg.get("gsr_mask_prob", 1.0),
        dcs_tau=cfg.get("dcs_tau", 0.07), lambda_dcs=cfg.get("lambda_dcs", 0.01),
        label_smoothing=cfg.get("label_smoothing", 0.1),
    )
    model = apply_lora_to_backbone(model, r=cfg.get("lora_rank", 16))
    report = model.load_state_dict(ckpt["model"], strict=False)
    print(f"missing={len(report.missing_keys)} unexpected={len(report.unexpected_keys)} "
          f"epoch={ckpt.get('epoch')} val_bAcc={ckpt.get('val_bacc', 'n/a')}")
    assert not report.unexpected_keys, f"Unexpected keys: {report.unexpected_keys[:10]}"
    return model.to(device).eval()

model_integrated = load_integrated_lora(K2_OUTPUT_DIR)
print("Model 1 (Integrated LoRA) loaded.")"""),

    ("code", """# ============ Model 2: B-Free baseline (BFREE_dino2reg4, protocol repo) ============
import yaml

from networks import get_network, load_weights

with open(os.path.join(BFREE_WEIGHTS_DIR, "config.yaml")) as f:
    bfree_cfg = yaml.safe_load(f)
model_path = os.path.join(BFREE_WEIGHTS_DIR, bfree_cfg["weights_file"])
print(f"B-Free baseline: arch={bfree_cfg['arch']} norm={bfree_cfg['norm_type']} weights={model_path}")
model_bfree = load_weights(get_network(bfree_cfg["arch"]), model_path)
model_bfree = model_bfree.to(DEVICE).eval()
BFREE_NORM = bfree_cfg["norm_type"]
print("Model 2 (B-Free baseline) loaded.")"""),

    ("code", """# ============ Model 3: GlobalForge released (REM vit_l_a, reassembled 13 parts) ============
GF_CKPT = "/kaggle/working/checkpoint-best.pth"

parts = sorted(glob.glob(os.path.join(GF_CKPT_PARTS, "checkpoint-best.pth.part_*")))
print(f"checkpoint parts: {len(parts)} found")
assert parts, f"No checkpoint parts under {GF_CKPT_PARTS}"
if not os.path.isfile(GF_CKPT) or os.path.getsize(GF_CKPT) < 1_000_000_000:
    with open(GF_CKPT, "wb") as out:
        for p in parts:
            with open(p, "rb") as f:
                while True:
                    chunk = f.read(1024 * 1024 * 64)
                    if not chunk:
                        break
                    out.write(chunk)
print(f"reassembled checkpoint: {os.path.getsize(GF_CKPT) / 1024**3:.2f} GB")

GF_WEIGHTS_ROOT = "/kaggle/working/gf_weights"
os.makedirs(GF_WEIGHTS_ROOT, exist_ok=True)
vit_la_link = os.path.join(GF_WEIGHTS_ROOT, "vit_l_a")
if not os.path.exists(vit_la_link):
    if os.path.isdir(os.path.join(GF_BACKBONE_INPUT, "vit_l_a")):
        src = os.path.join(GF_BACKBONE_INPUT, "vit_l_a")
    elif os.path.isfile(os.path.join(GF_BACKBONE_INPUT, "config.json")):
        src = GF_BACKBONE_INPUT
    else:
        raise AssertionError(
            f"{GF_BACKBONE_INPUT} must contain either a vit_l_a/ subdir or be the HF model dir itself (config.json).")
    os.symlink(src, vit_la_link)
os.environ["GLOBALFORGE_WEIGHTS_DIR"] = GF_WEIGHTS_ROOT

import models.REM as REM


def _infer_lora_rank_from_state_dict(state_dict):
    for key, value in state_dict.items():
        if "lora_A.default.weight" in key:
            return int(value.shape[0])
    return 16


def _infer_module_switches_from_state_dict(state_dict):
    use_lib = any(key.startswith("lib.") for key in state_dict)
    use_gsr = any(key.startswith("gsr.") for key in state_dict)
    return use_lib, use_gsr


def load_released_globalforge(ckpt_path, device=DEVICE):
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = checkpoint.get("model", checkpoint)
    lora_rank = _infer_lora_rank_from_state_dict(state)
    use_lib, use_gsr = _infer_module_switches_from_state_dict(state)
    print(f"GlobalForge released: lora_rank={lora_rank}, use_lib={use_lib}, use_gsr={use_gsr}")
    model = REM.__dict__["REM"](
        mode="vit_l_a", use_lib=use_lib, use_gsr=use_gsr,
        lib_layer=0, gsr_layer=0, lib_kernel=3, lib_tau=0.5,
        gsr_window=3, gsr_mask_prob=1.0,
    )
    model.load_state_dict(state)
    return model.to(device).eval()

model_gf = load_released_globalforge(GF_CKPT)
print("Model 3 (GlobalForge released) loaded.")"""),

    ("md", """### Scoring kernels (theo protocol code thật của từng model)

- **Model 1 (Integrated)**: replicate-pad ảnh < 504 (numpy `edge` pad — tương đương `replicate_wrap` của Wrapper5crops), 5 crops 504 (center + 4 corners), batch forward, **average logits** → score = `l1 − l0`.
- **Model 2 (B-Free baseline)**: nguyên bản repo — feed full ảnh (ToTensor + Normalize theo config), `Wrapper5crops` tự replicate + 5-crop ở mức patch-embedding, mean 5 views; num_classes=1 → score = logit.
- **Model 3 (GlobalForge)**: `short256_center` với `eval_resize_short=1296` (paper) — KHÔNG phải resize 224 trực tiếp; PNG round-trip JPEG q100 (giống `compress_image`); `Norm(imagenet)` đã nằm trong `REM_Model.forward`; softmax → log-odds."""),

    ("code", """IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}
GF_EVAL_RESIZE_SHORT = 1296


def list_images(root, limit=None):
    out = [str(p) for p in sorted(Path(root).iterdir())
           if p.is_file() and p.suffix.lower() in IMG_EXT]
    return out[:limit] if limit else out


def replicate_pad_504(img):
    \"\"\"Replicate-pad ảnh (PIL) để >= 504x504 (tương đương replicate_wrap của Wrapper5crops).\"\"\"
    w, h = img.size
    if w >= 504 and h >= 504:
        return img
    arr = np.asarray(img)
    pad_h, pad_w = max(0, 504 - h), max(0, 504 - w)
    if pad_h or pad_w:
        arr = np.pad(arr, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")
        img = Image.fromarray(arr)
    return img


def five_crop_boxes(img):
    w, h = img.size
    s = 504
    if w < s or h < s:
        raise ValueError(f"image smaller than 504 after padding: {w}x{h}")
    cx, cy = (w - s) // 2, (h - s) // 2
    return [(cx, cy, s, s), (0, 0, s, s), (w - s, 0, s, s), (0, h - s, s, s), (w - s, h - s, s, s)]


@torch.no_grad()
def score_image_integrated(model, img_path, device=DEVICE):
    img = Image.open(img_path).convert("RGB")
    img = replicate_pad_504(img)
    views = torch.stack([T.Compose(get_list_norm("resnet"))(img.crop(b)) for b in five_crop_boxes(img)])
    logits = model(views.to(device))["logits"].float().mean(dim=0)
    return float(logits[1] - logits[0])


@torch.no_grad()
def score_image_bfree(model, img_path, device=DEVICE, norm_type=BFREE_NORM):
    x = T.Compose(get_list_norm(norm_type))(Image.open(img_path).convert("RGB")).unsqueeze(0).to(device)
    out = model(x)
    if out.shape[1] == 1:
        return float(out[0, 0])
    return float(out[0, 1] - out[0, 0])


@torch.no_grad()
def score_image_globalforge(model, img_path, device=DEVICE):
    img = Image.open(img_path).convert("RGB")
    w, h = img.size
    short = min(w, h)
    if short > GF_EVAL_RESIZE_SHORT:
        scale = GF_EVAL_RESIZE_SHORT / short
        img = TF.resize(img, [round(h * scale), round(w * scale)],
                        interpolation=T.InterpolationMode.BICUBIC)
    img = TF.center_crop(img, [224, 224])
    if img_path.lower().endswith(".png"):
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=100, optimize=True)
        buf.seek(0)
        img = Image.open(buf).convert("RGB").copy()
    x = T.functional.to_tensor(img).unsqueeze(0).to(device)
    out = model(x)
    logits = (out["logits"] if isinstance(out, dict) else out).float().clamp(-30, 30)
    p = float(torch.softmax(logits, dim=1)[0, 1])
    return float(np.log(p / max(1.0 - p, 1e-12)))


SCORERS = {
    "Integrated-LoRA": lambda p: score_image_integrated(model_integrated, p),
    "B-Free-baseline": lambda p: score_image_bfree(model_bfree, p, norm_type=BFREE_NORM),
    "GlobalForge-REM": lambda p: score_image_globalforge(model_gf, p),
}
print("scorers ready:", list(SCORERS))"""),

    ("code", """# ============ Wild subset registry (17 subsets, layout 0_real/1_fake) ============

WILD_SUBSETS = {
    "Chameleon":           ("Chameleon/0_real", "Chameleon/1_fake", False),
    "SynthWildx-DALLE3":   ("synthwildx/dalle3/0_real", "synthwildx/dalle3/1_fake", False),
    "SynthWildx-Firefly":  ("synthwildx/firefly/0_real", "synthwildx/firefly/1_fake", False),
    "SynthWildx-Midj.":    ("synthwildx/midjourney_v5/0_real", "synthwildx/midjourney_v5/1_fake", False),
    "WildRF-FB":           ("WildRF/test/facebook/0_real", "WildRF/test/facebook/1_fake", False),
    "WildRF-Reddit":       ("WildRF/test/reddit/0_real", "WildRF/test/reddit/1_fake", False),
    "WildRF-Twitter":      ("WildRF/test/twitter/0_real", "WildRF/test/twitter/1_fake", False),
    "AIGIBench-SocRF":     ("AIGIBench/SocialRF/0_real", "AIGIBench/SocialRF/1_fake", False),
    "AIGIBench-ComAI":     ("AIGIBench/CommunityAI/0_real", "AIGIBench/CommunityAI/1_fake", False),
    "CO-SPY-Civitai":      ("CO-SPY-In-the-Wild/civitai/0_real", "CO-SPY-In-the-Wild/civitai/1_fake", True),
    "CO-SPY-DALLE3":       ("CO-SPY-In-the-Wild/dalle3/0_real", "CO-SPY-In-the-Wild/dalle3/1_fake", True),
    "CO-SPY-instavibe.ai": ("CO-SPY-In-the-Wild/instavibeai/0_real", "CO-SPY-In-the-Wild/instavibeai/1_fake", True),
    "CO-SPY-Lexica":       ("CO-SPY-In-the-Wild/lexica/0_real", "CO-SPY-In-the-Wild/lexica/1_fake", True),
    "CO-SPY-Midj.v6":      ("CO-SPY-In-the-Wild/midjourney/0_real", "CO-SPY-In-the-Wild/midjourney/1_fake", True),
    "RR-Dataset":          ("RRDataset/0_real", "RRDataset/1_fake", False),
    "BFree-Online":        ("B-Free/0_real", "B-Free/1_fake", False),
    "real-chain":          ("realchain_CD/0_real", "realchain_CD/1_fake", False),
}

STANDARD_BENCHMARKS = {
    "AIGCDetect":  ("AIGCDetect/0_real", "AIGCDetect/1_fake"),
    "GenImage":    ("GenImage/0_real", "GenImage/1_fake"),
    "UnivFD":      ("UnivFD/0_real", "UnivFD/1_fake"),
    "DRCT":        ("DRCT/0_real", "DRCT/1_fake"),
    "Synthbuster": ("Synthbuster/0_real", "Synthbuster/1_fake"),
    "EvalGEN":     ("EvalGEN/0_real", "EvalGEN/1_fake"),
}

assert os.path.isdir(DATA_ROOT), f"DATA_ROOT {DATA_ROOT} does not exist — attach 'wild-benchmarks' dataset."
available = [k for k, (r, f, _) in WILD_SUBSETS.items()
             if os.path.isdir(os.path.join(DATA_ROOT, r)) and os.path.isdir(os.path.join(DATA_ROOT, f))]
missing = [k for k in WILD_SUBSETS if k not in available]
print(f"wild subsets available: {len(available)}/17")
if missing:
    print("[WARN] missing subsets (skipped):", missing)
assert available, "No wild subset folders found under DATA_ROOT."

MAX_IMAGES = None"""),

    ("code", """import tqdm


def run_subset(model_name, real_dir, fake_dir, data_root, fake_only=False, max_images=None):
    real_paths = list_images(os.path.join(data_root, real_dir), limit=max_images)
    fake_paths = list_images(os.path.join(data_root, fake_dir), limit=max_images)
    if fake_only:
        paths, labels = fake_paths, [1] * len(fake_paths)
    else:
        paths = real_paths + fake_paths
        labels = [0] * len(real_paths) + [1] * len(fake_paths)
    if not paths:
        return None

    scorer = SCORERS[model_name]
    scores, y, n_skipped = [], [], 0
    for path, label in tqdm.tqdm(list(zip(paths, labels)),
                                 desc=f"{model_name}::{os.path.basename(os.path.dirname(fake_dir))}",
                                 leave=False):
        try:
            scores.append(scorer(path))
            y.append(label)
        except (OSError, UnidentifiedImageError, ValueError):
            n_skipped += 1
    if n_skipped:
        print(f"  [WARN] {n_skipped} unreadable images skipped")
    if not scores:
        return None
    scores = np.asarray(scores, dtype=float)
    y = np.asarray(y, dtype=int)

    if fake_only:
        return {"Model": model_name, "n_images": len(scores), "n_skipped": n_skipped,
                "AUC": float("nan"), "bAcc": float((scores > 0).mean() * 100.0),
                "NLL": float("nan"), "ECE": float("nan"),
                "Pd10": float("nan"), "EER": float("nan"), "fake_only": True}
    return {"Model": model_name, "n_images": len(scores), "n_skipped": n_skipped,
            "AUC": float(dmetrics.roc_auc_score(y, scores)),
            "bAcc": float(dmetrics.balanced_accuracy_score(y, scores > 0) * 100.0),
            "NLL": float(dmetrics.balanced_nll_binary(y, scores)),
            "ECE": float(dmetrics.balanced_ece_binary(y, scores)),
            "Pd10": float(dmetrics.pd_at_far(y, scores, 0.10) * 100.0),
            "EER": float(dmetrics.calculate_eer2(y, scores) * 100.0),
            "fake_only": False}


all_results = []
for subset, (real_dir, fake_dir, fake_only) in WILD_SUBSETS.items():
    if not (os.path.isdir(os.path.join(DATA_ROOT, real_dir))
            and os.path.isdir(os.path.join(DATA_ROOT, fake_dir))):
        print(f"[skip] {subset}: folders missing")
        continue
    for model_name in SCORERS:
        r = run_subset(model_name, real_dir, fake_dir, DATA_ROOT,
                       fake_only=fake_only, max_images=MAX_IMAGES)
        if r:
            all_results.append({"Benchmark": subset, **r})
            print(f"{subset:20s} | {model_name:16s} | bAcc={r['bAcc']:6.2f} AUC={r['AUC']:.4f} n={r['n_images']}",
                  flush=True)

wild_df = pd.DataFrame(all_results)
display(wild_df)"""),

    ("code", """# ============ Standard benchmarks (optional — skip nếu chưa attach) ============
std_results = []
for bench, (real_dir, fake_dir) in STANDARD_BENCHMARKS.items():
    if not (os.path.isdir(os.path.join(DATA_ROOT, real_dir))
            and os.path.isdir(os.path.join(DATA_ROOT, fake_dir))):
        print(f"[skip] {bench}: folders missing under {DATA_ROOT}")
        continue
    for model_name in SCORERS:
        r = run_subset(model_name, real_dir, fake_dir, DATA_ROOT,
                       fake_only=False, max_images=MAX_IMAGES)
        if r:
            std_results.append({"Benchmark": bench, **r})
            print(f"{bench:14s} | {model_name:16s} | bAcc={r['bAcc']:6.2f} AUC={r['AUC']:.4f} n={r['n_images']}",
                  flush=True)

std_df = pd.DataFrame(std_results)
if len(std_df):
    display(std_df)"""),

    ("code", """# ============ Combine + grouped parent average + save ============
def get_parent_dataset_key(result_key):
    if result_key.startswith("SynthWildx-"): return "SynthWildx"
    if result_key.startswith("WildRF-"): return "WildRF"
    if result_key.startswith("AIGIBench-"): return "AIGIBench"
    if result_key.startswith("CO-SPY-"): return "CO-SPY"
    if result_key.startswith("BFree-"): return "BFree"
    return result_key


def combine_parent_dataset_average(df, metric="bAcc"):
    \"\"\"Trung bình từng parent group, rồi trung bình các group (giống eval_in_the_wild.py).\"\"\"
    out = {}
    for model in df["Model"].unique():
        groups = {}
        for subset, value in df[df["Model"] == model].groupby("Benchmark")[metric].last().items():
            groups.setdefault(get_parent_dataset_key(subset), []).append(float(value))
        out[model] = sum(sum(v) / len(v) for v in groups.values()) / len(groups)
    return out


results_df = pd.concat([wild_df, std_df], ignore_index=True)
results_df.to_csv("/kaggle/working/eval_results.csv", index=False)

wild_only = results_df[results_df["Benchmark"].isin(WILD_SUBSETS)]
avg_rows = [{"Model": m, "Benchmark": "Avg B.Acc (wild, parent-avg)", "bAcc": v, "fake_only": False}
            for m, v in combine_parent_dataset_average(wild_only).items()]
results_df = pd.concat([results_df, pd.DataFrame(avg_rows)], ignore_index=True)
results_df.to_csv("/kaggle/working/eval_results.csv", index=False)
print("saved /kaggle/working/eval_results.csv")
display(results_df.pivot_table(index="Benchmark", columns="Model", values="bAcc", dropna=False))"""),

    ("code", """# ============ Visualization: bar chart + heatmap ============
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

plot_df = results_df[results_df["Benchmark"] != "Avg B.Acc (wild, parent-avg)"].copy()
piv_bacc = plot_df.pivot_table(index="Benchmark", columns="Model", values="bAcc")

fig, ax = plt.subplots(figsize=(13, max(6, 0.35 * len(piv_bacc))), dpi=130)
piv_bacc.plot(kind="barh", ax=ax)
ax.set_xlabel("bAcc (%)")
ax.set_title("Balanced Accuracy per subset × model")
ax.legend(loc="lower right")
fig.tight_layout()
fig.savefig("/kaggle/working/eval_bacc_bar.png")

fig, ax = plt.subplots(figsize=(9, max(6, 0.35 * len(piv_bacc))), dpi=130)
sns.heatmap(piv_bacc, annot=True, fmt=".1f", cmap="RdYlGn", vmin=50, vmax=100, ax=ax)
ax.set_title("bAcc heatmap (model × subset)")
fig.tight_layout()
fig.savefig("/kaggle/working/eval_bacc_heatmap.png")
print("saved eval_bacc_bar.png + eval_bacc_heatmap.png")"""),

    ("md", """## Evaluation Complete (K3)

Outputs trong `/kaggle/working/`:
- `eval_results.csv` — model × subset × {AUC, bAcc, NLL, ECE, Pd10, EER} + hàng `Avg B.Acc (wild, parent-avg)`
- `eval_bacc_bar.png` — bar chart bAcc per subset × model
- `eval_bacc_heatmap.png` — heatmap model × subset

**K3 checklist:** notebook chạy được · eval results CSV + heatmap + bar chart.

**Tiếp theo:** K4 (loss interaction analysis) — dùng `train_log.csv` từ K2 với `eval/analyze_loss.py` của repo (Pearson/Spearman correlation CE vs DCS + loss dynamics plot)."""),
]

# =====================================================================

if __name__ == "__main__":
    write_nb("01_bfree_setup_bundle.ipynb", NB01_CELLS)
    write_nb("02_bfree_kaggle_train.ipynb", NB02_CELLS)
    write_nb("03_bfree_kaggle_eval.ipynb", NB03_CELLS)
    print("\nAll notebooks generated and validated (json.tool + AST compile).")
