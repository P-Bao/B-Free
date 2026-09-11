"""Regenerate the three Kaggle notebooks (01/02/03) as .ipynb JSON.

Pipeline (mirrors StructLoRA notebook pattern):
  01 setup-online (Kaggle CPU/T4x2, Internet ON)  -> builds the offline bundle
  02 training    (RTX PRO 6000, offline)           -> consumes bundle + B-Free data
  03 eval        (RTX PRO 6000, offline)           -> consumes bundle + K2 output + benchmarks

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


# =====================================================================
# Shared cell sources (offline notebooks 02/03)
# =====================================================================

OFFLINE_LOGGER_SRC = """import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger("bfree")
logger.setLevel(logging.INFO)
logger.handlers.clear()
_console = logging.StreamHandler()
_console.setFormatter(logging.Formatter(fmt="%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S"))
logger.addHandler(_console)

logger.info("stdlib + logging ready")"""

BUNDLE_DISCOVERY_SRC = """import glob
from pathlib import Path

WORK = Path("/kaggle/working")


def find_one(patterns, what, optional=False):
    \"\"\"Return the first existing match across glob patterns (in priority order).\"""
    for pat in patterns:
        hits = sorted(glob.glob(pat))
        if hits:
            return hits[0]
    if optional:
        return None
    raise FileNotFoundError(
        f"NOT FOUND: {what}\\nsearched: {list(patterns)}\\n"
        "Attach the missing Kaggle dataset / notebook-output input and re-run.")


def find_dir_containing(marker_rel, patterns, what, optional=False):
    \"\"\"Return the parent dir that contains marker_rel, searched via glob patterns.\"""
    for pat in patterns:
        for hit in sorted(glob.glob(pat)):
            root = Path(hit)
            while root != Path("/kaggle/input") and root != root.parent:
                if (root / marker_rel).is_file():
                    return root
                root = root.parent
    if optional:
        return None
    raise FileNotFoundError(
        f"NOT FOUND: {what} (marker: {marker_rel})\\nsearched: {list(patterns)}\\n"
        "Attach the missing Kaggle dataset / notebook-output input and re-run.")


# ---- bundle produced by notebook 01 (attach its output as input here) ----
WHEELS_DIR = find_one(
    ["/kaggle/input/*/wheels_rtxpro6000",
     "/kaggle/input/wheels_rtxpro6000",
     "/kaggle/input/*/wheels*",
     "/kaggle/input/wheels*"],
    "wheel bundle 'wheels_rtxpro6000' (output of notebook 01)")

REPO_MARKER = "code/networks/bfree_globalforge_vit.py"
REPO_SRC = find_dir_containing(
    REPO_MARKER,
    ["/kaggle/input/*/bfree_src/code/networks/bfree_globalforge_vit.py",
     "/kaggle/input/bfree_src*/code/networks/bfree_globalforge_vit.py",
     "/kaggle/input/*/B-Free/code/networks/bfree_globalforge_vit.py",
     "/kaggle/input/B-Free*/code/networks/bfree_globalforge_vit.py",
     "/kaggle/input/*/code/networks/bfree_globalforge_vit.py"],
    "B-Free repo source 'bfree_src' (output of notebook 01)")

DINOV2_SD = find_one(
    ["/kaggle/input/*/models/vit_base_patch14_reg4_dinov2/model.safetensors",
     "/kaggle/input/models/vit_base_patch14_reg4_dinov2/model.safetensors",
     "/kaggle/input/dinov2*/*.safetensors",
     "/kaggle/input/*/dinov2*/*.safetensors"],
    "DINOv2 ViT-B/14 reg4 weights (output of notebook 01)")

wheels = sorted(glob.glob(str(Path(WHEELS_DIR) / "*.whl")))
assert wheels, f"No .whl files inside {WHEELS_DIR}"
logger.info(f"WHEELS_DIR  = {WHEELS_DIR} ({len(wheels)} wheels)")
logger.info(f"REPO_SRC    = {REPO_SRC}")
logger.info(f"DINOV2_SD   = {DINOV2_SD}")"""

OFFLINE_ENV_SRC = """os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTHONHASHSEED"] = "0"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

logger.info("offline env vars set (HF hub blocked, bytecode off — repo source stays read-only in input)")"""

PIP_INSTALL_SRC = """!pip install --no-index --find-links="{WHEELS_DIR}" \\
    torch torchvision timm peft transformers accelerate \\
    pandas numpy matplotlib seaborn scikit-learn scipy \\
    pyyaml pillow tqdm safetensors huggingface_hub

import torch
import timm
import peft
import transformers

assert torch.__version__.startswith("2.8.0"), f"torch {torch.__version__} != 2.8.0+cu128 (bundle wrong?)"
assert transformers.__version__ == "4.55.4", f"transformers {transformers.__version__} != 4.55.4"
assert peft.__version__ == "0.15.2", f"peft {peft.__version__} != 0.15.2"
import pandas
assert pandas.__version__.split(".")[0] == "2", "pandas>=3 breaks sklearn (risk table)"
logger.info(f"pip --no-index OK | torch={torch.__version__} timm={timm.__version__} "
            f"peft={peft.__version__} transformers={transformers.__version__}")"""

REPO_IMPORT_SRC = """import sys

sys.path.insert(0, str(Path(REPO_SRC) / "code"))

stubs = sorted(glob.glob(str(Path(REPO_SRC) / "code" / "modules" / "*_stub.py")))
assert not stubs, f"Stub files still present (K0 not merged?): {stubs}"

from networks.bfree_globalforge_vit import BFreeGlobalForgeViT
from configs.loader import load_config, build_model_kwargs
from datasets.bfree_dataset import BFreeDataset, DegradationPipeline
from modules.lib_adapter import LIBAdapter
from modules.gsr_adapter import GSRAdapter
from modules.dcs_loss import DCSLoss, info_nce_loss
import networks.bfree_globalforge_vit as _bgv

logger.info(f"repo import OK from {_bgv.__file__} (no stubs — K0 verified)")"""

# =====================================================================
# Notebook 01 — Setup Online (bundle builder)
# =====================================================================

NB01_CELLS = [
    ("md", """# B-Free x GlobalForge — Notebook 01: Setup Online (bundle builder)

**Target A: Kaggle CPU hoặc T4×2, Internet ON.** Notebook này tạo **toàn bộ assets offline** cho notebooks 02 (train) / 03 (eval), vốn chạy trên **Target B: RTX PRO 6000 Blackwell 96GB, hoàn toàn offline** (không pip internet, không git, không HF hub).

## Chuỗi pipeline (3 notebook, khép kín)

```
[01] setup-online (CPU/T4, online)
      |- wheels_rtxpro6000/   : pin stack wheels (linux x86_64, py3.11)
      |- models/vit_base_patch14_reg4_dinov2/  : DINOv2 ViT-B/14 reg4 (D5)
      |- bfree_src/            : repo P-Bao/B-Free @ integration/loss-backbone
      |- manifest.json        : versions + SHA256 + sizes
      v  (Save Version -> attach output này làm Input của 02)
[02] training (RTX PRO 6000, offline)  -> bfree_globalforge_lora_r16.pth + train_log.csv
      v  (Save Version -> attach output này làm Input của 03)
[03] eval (RTX PRO 6000, offline)      -> eval_results.csv + bar + heatmap
```

Giữa các notebook chỉ có thao tác Kaggle-native: **Save Version → attach output làm input** — không bước chuẩn bị thủ công nào khác.

**Input ngoài duy nhất** (attach thêm vào 02/03, upload thủ công một lần): B-Free training data (grip.unina.it) cho 02; baseline weights + GlobalForge assets + wild benchmarks cho 03 (liệt kê chi tiết trong từng notebook)."""),

    ("code", """# ============================================================
# Cell 1 - Pin versions (locked stack, plan.md)
# ============================================================
TORCH_VERSION = "2.8.0+cu128"     # RTX Pro 6000 Blackwell (sm_120) needs torch>=2.8 cu128
TORCHVISION_VERSION = "0.23.0+cu128"
TIMM_VERSION = "1.0.22"
PEFT_VERSION = "0.15.2"
TRANSFORMERS_VERSION = "4.55.4"
PANDAS_VERSION = "2.3.3"         # MUST <3 (breaks sklearn at >=3)
NUMPY_VERSION = "1.26.4"
CUDA_TAG = "cu128"
REPO_URL = "https://github.com/P-Bao/B-Free.git"
BRANCH = "integration/loss-backbone"
DINOV2_HF_REPO = "timm/vit_base_patch14_reg4_dinov2.lvd142m"

print(f"TORCH_VERSION         = {TORCH_VERSION}")
print(f"TORCHVISION_VERSION   = {TORCHVISION_VERSION}")
print(f"TIMM_VERSION          = {TIMM_VERSION}")
print(f"PEFT_VERSION          = {PEFT_VERSION}")
print(f"TRANSFORMERS_VERSION  = {TRANSFORMERS_VERSION}")
print(f"PANDAS_VERSION        = {PANDAS_VERSION}")
print(f"NUMPY_VERSION         = {NUMPY_VERSION}")
print(f"CUDA_TAG              = {CUDA_TAG}")
print(f"REPO                 = {REPO_URL} @ {BRANCH}")"""),

    ("md", """## 1. Chốt version môi trường

Probe môi trường **Target A** (session này). Wheels tải ở đây phải khớp **Target B** — Kaggle dùng chung image Linux x86_64 + Python 3.11 cho mọi accelerator, nên assert Python 3.11 để chắc chắn."""),

    ("code", """import platform
import subprocess
import sys

print("=" * 60)
print("Environment probe (Target A: online session)")
print("=" * 60)
print(f"python   = {platform.python_version()}")
print(f"platform = {platform.platform()}")

assert sys.version_info[:2] == (3, 11), (
    f"This session runs Python {sys.version_info[:2]} but Kaggle Target B uses 3.11 — "
    "wheels would not match. Switch the session to Python 3.11 and re-run.")

try:
    out = subprocess.run(["nvidia-smi"], capture_output=True, text=True, check=True)
    print(out.stdout.splitlines()[2] if len(out.stdout.splitlines()) > 2 else "nvidia-smi ok")
except Exception:
    print("nvidia-smi not available (CPU session) — fine, this notebook only downloads assets.")"""),

    ("md", """## 2. Tải wheel cho Target B (RTX PRO 6000)

- `torch` / `torchvision`: từ **PyTorch cu128 index**, `--no-deps` (wheel cu128 đã bundle CUDA runtime, không cần kéo `nvidia-*`/`pytorch-triton`).
- Còn lại: **`--no-deps` từng package + đầy đủ leaf dependencies** (filelock, fsspec, tokenizers, ...) để `pip install --no-index` trên Target B tự chủ hoàn toàn, không phụ thuộc version preinstall của image đích."""),

    ("code", """from pathlib import Path

WHEELS = Path("/kaggle/working/wheels_rtxpro6000")
WHEELS.mkdir(parents=True, exist_ok=True)
PYTORCH_INDEX = f"https://download.pytorch.org/whl/{CUDA_TAG}"
print(f"PyTorch index: {PYTORCH_INDEX}\\n")

!pip download torch=={TORCH_VERSION} \\
    --index-url {PYTORCH_INDEX} --no-deps -d {WHEELS} 2>&1 | tail -3
!pip download torchvision=={TORCHVISION_VERSION} \\
    --index-url {PYTORCH_INDEX} --no-deps -d {WHEELS} 2>&1 | tail -3

for p in sorted(WHEELS.glob("*.whl")):
    print(f"  {p.name:60s} {p.stat().st_size/1024/1024:8.1f} MB")"""),

    ("code", """# Pin stack còn lại + đầy đủ leaf deps (đủ cho pip --no-index trên Target B)
_pkgs_pinned = [
    f"timm=={TIMM_VERSION}",
    f"peft=={PEFT_VERSION}",
    f"transformers=={TRANSFORMERS_VERSION}",
    f"pandas=={PANDAS_VERSION}",
    f"numpy=={NUMPY_VERSION}",
    "matplotlib==3.11.1",
    "seaborn==0.13.2",
]
_pkgs_loose = [
    "scikit-learn", "scipy", "pyyaml", "pillow", "tqdm", "safetensors",
    "huggingface_hub", "accelerate",
    # leaf dependencies (thiếu cái nào pip --no-index sẽ fail nếu image đích không có)
    "filelock", "fsspec", "packaging", "typing-extensions", "regex",
    "requests", "certifi", "charset-normalizer", "idna", "urllib3",
    "tokenizers", "psutil", "joblib", "threadpoolctl",
    "python-dateutil", "six", "pytz", "tzdata",
    "contourpy", "cycler", "fonttools", "kiwisolver", "pyparsing",
]
for pkg in _pkgs_pinned + _pkgs_loose:
    print(f">>> pip download {pkg}")
    !pip download {pkg} --no-deps -d {WHEELS} 2>&1 | tail -2

import subprocess
n_whl = len(list(WHEELS.glob("*.whl")))
total_gb = sum(p.stat().st_size for p in WHEELS.glob("*.whl")) / 1024**3
print(f"\\n{WHEELS}: {n_whl} wheels, {total_gb:.2f} GB")
assert n_whl >= 35, "wheels look incomplete (expected >=35 incl. leaf deps)""" + '"'),

    ("md", """## 3. Tải DINOv2 ViT-B/14 reg4 (backbone pretrained — D5)

Tải `model.safetensors` từ HF hub `timm/vit_base_patch14_reg4_dinov2.lvd142m` (~330MB) — Target B offline sẽ init backbone từ file này (notebook 02 resample `pos_embed` 518px → 504px khi load)."""),

    ("code", """import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download

MODELS = Path("/kaggle/working/models")
MODELS.mkdir(parents=True, exist_ok=True)
DINO_OUT = MODELS / "vit_base_patch14_reg4_dinov2"
DINO_OUT.mkdir(exist_ok=True)

dst = DINO_OUT / "model.safetensors"
if dst.exists():
    print(f"[skip] already at {dst}")
else:
    src = hf_hub_download(repo_id=DINOV2_HF_REPO, filename="model.safetensors")
    shutil.copyfile(src, dst)
print(f"saved: {dst} ({dst.stat().st_size/1024**2:.1f} MB)")

from safetensors.torch import load_file

sd = load_file(str(dst))
pe = sd["pos_embed"]
assert sd["patch_embed.proj.weight"].shape[0] == 768, "not a ViT-B checkpoint"
assert tuple(pe.shape) == (1, 37 * 37 + 5, 768), (
    f"unexpected pos_embed {tuple(pe.shape)} — expected 518px grid (37x37) + 5 prefix tokens")
print(f"OK: {len(sd)} tensors | embed_dim=768 | pos_embed {tuple(pe.shape)} (518px grid + 5 prefix)")"""),

    ("md", """## 4. Copy B-Free repo source (branch `integration/loss-backbone`)

Target B offline không git clone được → đóng gói source vào bundle (bỏ `.git`, `__pycache__`). Sau đó **smoke test trên CPU** để xác nhận bundle model (backbone + real LIB/GSR + L_DCS) import + forward + `compute_loss` + backward chạy đúng — đây là verify cuối cùng trước khi vào offline."""),

    ("code", """import glob
import shutil
import subprocess
import sys
from pathlib import Path

BFREE_SRC = Path("/kaggle/working/bfree_src")
if BFREE_SRC.exists():
    shutil.rmtree(BFREE_SRC)
subprocess.run(["git", "clone", "--depth", "1", "--branch", BRANCH, REPO_URL, str(BFREE_SRC)], check=True)

shutil.rmtree(BFREE_SRC / ".git", ignore_errors=True)
for p in BFREE_SRC.rglob("__pycache__"):
    shutil.rmtree(p, ignore_errors=True)

assert (BFREE_SRC / "code/networks/bfree_globalforge_vit.py").is_file(), "clone failed: backbone file missing"
assert not glob.glob(str(BFREE_SRC / "code/modules/*_stub.py")), "stub files present — K0 not merged?"
n_py = len(list(BFREE_SRC.rglob("*.py")))
print(f"OK: {BFREE_SRC} ({n_py} .py files, .git removed, no stubs — K0 verified)")"""),

    ("code", """# Smoke test tren CPU: import + forward + compute_loss + backward (224px)
import sys

sys.path.insert(0, str(BFREE_SRC / "code"))

import torch

from networks.bfree_globalforge_vit import BFreeGlobalForgeViT

model = BFreeGlobalForgeViT(img_size=224, pretrained=False)
x1 = torch.randn(2, 3, 224, 224)
x2 = torch.randn(2, 3, 224, 224)
y = torch.tensor([0, 1])
with torch.no_grad():
    out = model(x1)
assert out["logits"].shape == (2, 2) and out["cls"].shape == (2, 768)
total, ce, dcs = model.compute_loss(x1, x2, y)
total.backward()
print(f"SMOKE OK | logits {tuple(out['logits'].shape)} | total={float(total):.4f} "
      f"ce={float(ce):.4f} dcs={float(dcs):.4f} | backward OK")"""),

    ("md", """## 5. Manifest + tổng kết

Ghi `manifest.json` (versions + SHA256 từng file + size từng component) — Target B dùng để kiểm tra tính toàn vẹn của bundle sau khi attach."""),

    ("code", """import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

WORK = Path("/kaggle/working")


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


manifest = {
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "python": platform.python_version(),
    "cuda_tag_for_rtxpro6000": CUDA_TAG,
    "repo": {"url": REPO_URL, "branch": BRANCH},
    "dinov2_hf_repo": DINOV2_HF_REPO,
    "pins": {
        "torch": TORCH_VERSION, "torchvision": TORCHVISION_VERSION,
        "timm": TIMM_VERSION, "peft": PEFT_VERSION,
        "transformers": TRANSFORMERS_VERSION, "pandas": PANDAS_VERSION,
        "numpy": NUMPY_VERSION, "matplotlib": "3.11.1", "seaborn": "0.13.2",
    },
    "components": {},
}

for sd in ["wheels_rtxpro6000", "models", "bfree_src"]:
    root = WORK / sd
    if not root.exists():
        continue
    files = sorted(p for p in root.rglob("*") if p.is_file())
    manifest["components"][sd] = {
        "file_count": len(files),
        "total_mb": sum(p.stat().st_size for p in files) / 1024**2,
        "sha256": {str(p.relative_to(WORK)): sha256_file(p) for p in files},
    }

out = WORK / "manifest.json"
out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
print(f"[OK] {out} ({out.stat().st_size/1024:.1f} KB)\\n")
print(f"{'component':<22} {'files':>7} {'size (MB)':>10}")
print("-" * 45)
for name, info in manifest["components"].items():
    print(f"{name:<22} {info['file_count']:>7} {info['total_mb']:>10.1f}")"""),

    ("md", """## Bundle Ready (K1 pass)

`/kaggle/working/` giờ chứa **đầy đủ assets offline**:
- `wheels_rtxpro6000/` — pin stack + leaf deps (linux x86_64, py3.11)
- `models/vit_base_patch14_reg4_dinov2/model.safetensors` — DINOv2 ViT-B/14 reg4 (D5)
- `bfree_src/` — repo `integration/loss-backbone` (đã smoke test)
- `manifest.json` — versions + SHA256

**Bước giao tiếp giữa notebook (Kaggle-native, bắt buộc):**
1. **Save Version → Save & Run All (Commit)** để chốt output.
2. Mở notebooks 02/03 → **Add Input → Your Work → notebook 01 này** (output của nó thành input `/kaggle/input/<slug>/...` chứa `wheels_rtxpro6000/`, `bfree_src/`, `models/`).
3. 02 cần thêm input ngoài: **B-Free training data** (upload dataset có `COCO_real_512/` + 6 thư mục `SD2.1_*/`).
4. 03 cần thêm: **K2 output** (notebook 02), **B-Free baseline weights**, **GlobalForge assets**, **wild benchmarks** (chi tiết trong notebook 03)."""),
]

# =====================================================================
# Notebook 02 — Training (offline)
# =====================================================================

NB02_CELLS = [
    ("md", """# B-Free x GlobalForge — Notebook 02: Kaggle Training (K2, LoRA r=16)

**Target B: RTX PRO 6000 Blackwell 96GB, hoàn toàn OFFLINE.** Huấn luyện **DINOv2 ViT-B/14 reg4 + LIB + GSR + L_DCS** (LoRA r=16, α=32) trên B-Free training data (51.517 real + 309.102 fake), 504×504, 8 epochs, bf16.

**Inputs (attach trước khi chạy):**
1. **Output notebook 01** (wheels + bfree_src + DINOv2 weights + manifest) — qua *Add Input → Your Work → notebook 01*
2. **B-Free training data** (upload một lần từ grip.unina.it): dataset chứa `COCO_real_512/` + 6 thư mục `SD2.1_*/` (fake variants dùng cùng tên file với real)

**Locked decisions (plan.md D1–D10):** D1 lambda_dcs=0.01, tau=0.07, ls=0.1 · D2 8 epochs @504px · D3 full data · D4 50/50 per-ID pairing (p=0.5 real else random 1/6 fake) · D5 DINOv2 pretrained init offline · D6 auto batch size (0.9×VRAM) · D7 bf16 autocast · AdamW lr=1e-4 wd=1e-4, cosine eta_min=1e-7 per-step, clip 1.0 · val split `md5(id)%100<3`, mỗi ID = 1 real + 1 deterministic fake (balanced).

> **Vì sao không gọi `train_lora.py` trực tiếp?** Script của repo (a) yêu cầu CSV tĩnh `filename,label` — không thực hiện được pairing động D4 (resample mỗi epoch); (b) loop của nó **fp32**, vi phạm D7 (bf16 autocast); (c) không có đường load pretrained offline D5 (resample pos_embed 518→504); (d) checkpoint của nó thiếu CONFIG + train_log mà K3 cần. Notebook này **tái sử dụng repo như thư viện** — `BFreeGlobalForgeViT`, `DegradationPipeline`, `apply_lora_to_backbone`, `dmetrics` — đúng tinh thần hợp đồng `MODULES_INTERFACE.md`, chỉ thay phần orchestration cho khớp D4/D5/D6/D7."""),

    ("code", OFFLINE_LOGGER_SRC),

    ("code", BUNDLE_DISCOVERY_SRC + """

# ---- external input: B-Free training data (COCO_real_512 + SD2.1_*) ----
TRAIN_DATA_ROOT = None
for cand in sorted(glob.glob("/kaggle/input/*/")) + ["/kaggle/input/"]:
    for rel in ("COCO_real_512", "bfree-training-data/COCO_real_512"):
        if Path(cand, rel).is_dir():
            TRAIN_DATA_ROOT = Path(cand, rel).parent
            break
    if TRAIN_DATA_ROOT is not None:
        break
if TRAIN_DATA_ROOT is None:
    raise FileNotFoundError(
        "B-Free training data not found: no /kaggle/input/*/COCO_real_512. "
        "Upload it (grip.unina.it training_data) as a Kaggle dataset and attach.")
logger.info(f"TRAIN_DATA_ROOT = {TRAIN_DATA_ROOT}")

OUTPUTS = WORK
OUTPUTS.mkdir(parents=True, exist_ok=True)
_fh = logging.FileHandler(str(OUTPUTS / "train_log.txt"), mode="a", encoding="utf-8")
_fh.setFormatter(logging.Formatter(fmt="%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
logger.addHandler(_fh)"""),

    ("code", OFFLINE_ENV_SRC),

    ("code", PIP_INSTALL_SRC),

    ("code", REPO_IMPORT_SRC),

    ("code", """# ============================================================
# Hyperparameters (locked D1-D10) - everything lives here
# ============================================================
import random

import numpy as np

CONFIG = {
    "arch": "vit_base_patch14_reg4_dinov2.lvd142m",
    "num_classes": 2,
    "img_size": 504,
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
    "lora_targets": ["qkv", "proj", "fc1", "fc2"],   # GlobalForge convention
    "lr": 1e-4,
    "weight_decay": 1e-4,
    "max_grad_norm": 1.0,
    "scheduler_eta_min": 1e-7,
    "max_vram_frac": 0.9,           # D6
    "batch_hard_cap": 128,
    "val_md5_percentile": 3,        # md5(id)%100 < 3
    "val_fake_variant": "SD2.1_selfconditioned",
    "pair_real_prob": 0.5,         # D4
    "num_workers": 4,
    "seed": 42,
}
DEVICE = "cuda:0"

random.seed(CONFIG["seed"])
np.random.seed(CONFIG["seed"])
torch.manual_seed(CONFIG["seed"])
torch.cuda.manual_seed_all(CONFIG["seed"])

REAL_DIR = str(TRAIN_DATA_ROOT / "COCO_real_512")
FAKE_DIRS = sorted(str(p) for p in TRAIN_DATA_ROOT.glob("SD2.1_*") if p.is_dir())
assert Path(REAL_DIR).is_dir(), f"COCO_real_512 missing under {TRAIN_DATA_ROOT}"
assert len(FAKE_DIRS) == 6, f"Expected 6 SD2.1_* dirs, found {len(FAKE_DIRS)}"
for d in FAKE_DIRS:
    logger.info(f"  {Path(d).name}: {len(list(Path(d).glob('*')))} files")
logger.info(f"CONFIG: {CONFIG}")"""),

    ("md", """### Dataset — 50/50 per-ID pairing (D4)

`BFreeDataset` của repo nhận CSV tĩnh nên không pairing động theo ID được; cell dưới tái dùng `DegradationPipeline` của repo và cài đúng convention D4:

- **Train**: 1 sample = 1 ID; `p=0.5` → real, ngược lại random 1/6 fake variant (resample mỗi `__getitem__` → 8 epochs phủ hết các variant).
- **Val** (`md5(stem)%100 < 3`): mỗi ID 2 samples — chẵn = real, lẻ = deterministic fake `SD2.1_selfconditioned` → balanced.
- Crop: train RandomCrop 504 (ảnh 512×512) + hflip 0.5; val center crop 504.
- Degraded view cho L_DCS: JPEG 20-80 → GaussianBlur k7 σ0.5-1.5 p0.8 → ColorJitter p0.8 (pipeline GlobalForge)."""),

    ("code", """import hashlib

from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as TF

from utils.normalization import get_list_norm


def build_stem_index(dirs):
    maps = []
    for d in dirs:
        m = {}
        for p in Path(d).glob("*"):
            if p.is_file():
                m[p.stem] = str(p)
        maps.append(m)
    return maps

REAL_MAP = build_stem_index([REAL_DIR])[0]
FAKE_MAPS = build_stem_index(FAKE_DIRS)
ALL_STEMS = sorted(REAL_MAP.keys())
logger.info(f"real={len(REAL_MAP)}, fake variants={[len(m) for m in FAKE_MAPS]}")


def is_val_id(stem):
    return int(hashlib.md5(stem.encode()).hexdigest(), 16) % 100 < CONFIG["val_md5_percentile"]

TRAIN_IDS = [s for s in ALL_STEMS if not is_val_id(s)]
VAL_IDS = [s for s in ALL_STEMS if is_val_id(s)]
logger.info(f"train IDs={len(TRAIN_IDS)}, val IDs={len(VAL_IDS)} (~{CONFIG['val_md5_percentile']}%)")


class PairDataset(Dataset):
    '''Train: p=0.5 real else random 1/6 fake variant (resample moi epoch).
    Val: 2 samples/ID - chan=real, le=deterministic fake variant (balanced).'''

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
            if Path(d).name == self.val_fake_variant and stem in m:
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
_, _, c2 = val_ds[1]
assert tuple(a.shape) == (3, CONFIG["img_size"], CONFIG["img_size"])
logger.info(f"train_ds={len(train_ds)}, val_ds={len(val_ds)} | labels val[0],val[1]={c},{c2} (expect 0,1)")

def seed_worker(worker_id):
    ws = torch.initial_seed() % 2**32
    np.random.seed(ws)
    random.seed(ws)

g = torch.Generator()
g.manual_seed(CONFIG["seed"])"""),

    ("md", """### Model — BFreeGlobalForgeViT + LoRA r=16 (offline pretrained init, D5)

Load `model.safetensors` từ bundle notebook 01 vào backbone; **resample `pos_embed` 518px (37×37) → 504px (36×36)** vì checkpoint DINOv2 mặc định 518px; filter key+shape mismatch (head 2-class mới sẽ missing — đúng). Sau đó áp LoRA r=16 α=32 qua `apply_lora_to_backbone` của repo (target `qkv/proj/fc1/fc2`; LIB/GSR/fc_norm/head vẫn full-train)."""),

    ("code", """import math

import torch


def load_pretrained_backbone(model, safetensors_path):
    '''D5 offline: load DINOv2 weights vao model.model + resample pos_embed 518->504.'''
    from safetensors.torch import load_file

    sd = load_file(str(safetensors_path))
    sd = {k[len("model."):] if k.startswith("model.") else k: v for k, v in sd.items()}

    prefix = model.model.num_prefix_tokens
    if "pos_embed" in sd:
        pe = sd["pos_embed"]
        old_hw = int(math.isqrt(pe.shape[1] - prefix))
        new_hw = model.model.patch_embed.grid_size[0]
        if old_hw != new_hw:
            from timm.layers.pos_embed import resample_abs_pos_embed
            logger.info(f"resample pos_embed: {old_hw}x{old_hw} -> {new_hw}x{new_hw}")
            sd["pos_embed"] = resample_abs_pos_embed(
                pe, new_size=model.model.patch_embed.grid_size,
                old_size=(old_hw, old_hw), num_prefix_tokens=prefix)

    ref = model.model.state_dict()
    dropped = [k for k, v in sd.items() if k not in ref or ref[k].shape != v.shape]
    sd = {k: v for k, v in sd.items() if k not in dropped}

    report = model.model.load_state_dict(sd, strict=False)
    logger.info(f"pretrained load: dropped={len(dropped)}, missing={report.missing_keys}")
    assert set(report.missing_keys) <= {"head.weight", "head.bias"}, "backbone not fully initialized"
    assert not report.unexpected_keys
    return model

from train_lora import apply_lora_to_backbone

model = BFreeGlobalForgeViT(
    arch=CONFIG["arch"], num_classes=CONFIG["num_classes"],
    img_size=CONFIG["img_size"], pretrained=False,
    use_lib=True, use_gsr=True, use_dcs=True,
    lib_kernel=CONFIG["lib_kernel"], lib_tau=CONFIG["lib_tau"],
    gsr_window=CONFIG["gsr_window"], gsr_mask_prob=CONFIG["gsr_mask_prob"],
    dcs_tau=CONFIG["dcs_tau"], lambda_dcs=CONFIG["lambda_dcs"],
    label_smoothing=CONFIG["label_smoothing"],
)
model = load_pretrained_backbone(model, DINOV2_SD)
model = apply_lora_to_backbone(model, r=CONFIG["lora_rank"],
                               lora_alpha=CONFIG["lora_alpha"], lora_dropout=CONFIG["lora_dropout"])
model.to(DEVICE)

n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
n_total = sum(p.numel() for p in model.parameters())
logger.info(f"trainable params: {n_train:,} / {n_total:,} ({100 * n_train / n_total:.2f}%)")"""),

    ("md", """### Auto batch size finder (D6)

Doubling từ 2 → OOM/hard-cap, rồi binary search — đo bằng chính `compute_loss` (2 forwards clean+degraded, bf16, backward)."""),

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
        logger.warning(f"no OOM up to hard cap {hard_cap}")
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
logger.info(f"AUTO BATCH SIZE = {BATCH_SIZE}")"""),

    ("md", """### Training loop — 8 epochs, bf16 autocast (D7), clip 1.0

`total = CE + lambda_dcs * DCS` qua `model.compute_loss`; **log riêng CE và DCS** (rule agent.md) → `train_log.csv`; AdamW lr 1e-4 wd 1e-4; CosineAnnealingLR `T_max=epochs*len(train_loader)`, eta_min=1e-7, step mỗi batch; lưu checkpoint best theo val_bAcc + checkpoint cuối (kèm CONFIG — K3 cần)."""),

    ("code", """from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from utils.dmetrics import balanced_accuracy_score, roc_auc_score

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
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
            logger.info(f"ep{epoch} step {step}/{len(train_loader)} | "
                        f"total {agg['total']/max(agg['n'],1):.4f} (ce {agg['ce']/max(agg['n'],1):.4f}, "
                        f"dcs {agg['dcs']/max(agg['n'],1):.4f}) | lr {optimizer.param_groups[0]['lr']:.2e}")
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

LOG_CSV = OUTPUTS / "train_log.csv"
with open(LOG_CSV, "w", encoding="utf-8") as f:
    f.write("epoch,train_loss,train_ce,train_dcs,val_loss,val_auc,val_bacc\\n")

TRAIN_LOG = []
best_bacc = 0.0
for epoch in range(1, CONFIG["epochs"] + 1):
    tr = run_epoch(epoch)
    va = run_val()
    logger.info(f"--> epoch {epoch}: train total={tr['total']:.4f} (ce={tr['ce']:.4f}, dcs={tr['dcs']:.4f}) | "
                f"val loss={va['val_loss']:.4f} auc={va['val_auc']:.4f} bacc={va['val_bacc']:.4f}")
    TRAIN_LOG.append({"epoch": epoch, "train_loss": tr["total"], "train_ce": tr["ce"],
                      "train_dcs": tr["dcs"], **va})
    with open(LOG_CSV, "a", encoding="utf-8") as f:
        f.write(f"{epoch},{tr['total']:.4f},{tr['ce']:.4f},{tr['dcs']:.4f},"
                f"{va['val_loss']:.4f},{va['val_auc']:.4f},{va['val_bacc']:.4f}\\n")
    if va["val_bacc"] > best_bacc:
        best_bacc = va["val_bacc"]
        torch.save({"model": model.state_dict(), "epoch": epoch, "val_bacc": best_bacc,
                    "config": CONFIG},
                   OUTPUTS / "bfree_globalforge_lora_r16_best.pth")
        logger.info(f"[*] best checkpoint saved (epoch {epoch}, bAcc {best_bacc:.4f})")

torch.save({"model": model.state_dict(), "epoch": CONFIG["epochs"], "config": CONFIG,
            "train_log": TRAIN_LOG},
           OUTPUTS / "bfree_globalforge_lora_r16.pth")
logger.info("TRAINING COMPLETE")"""),

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
fig.savefig(OUTPUTS / "training_curves.png")
logger.info("saved training_curves.png")"""),

    ("md", """## Training Complete (K2)

Outputs trong `/kaggle/working/`:
- `bfree_globalforge_lora_r16.pth` / `..._best.pth` — checkpoint cuối / best-val_bAcc (kèm CONFIG — K3 cần để tái tạo kiến trúc)
- `train_log.csv` — CE/DCS riêng từng epoch (cho K4 / Phase 8)
- `training_curves.png` · `train_log.txt` — log đầy đủ

**K2 checklist:** chạy hết 8 epochs không lỗi · checkpoint saved · training log CSV + curves.

**Bước giao tiếp:** **Save Version → Save & Run All (Commit)**, rồi mở notebook 03 → *Add Input → Your Work → notebook 02 này*.""" ),
]

# =====================================================================
# Notebook 03 — Evaluation (offline)
# =====================================================================

NB03_CELLS = [
    ("md", """# B-Free x GlobalForge — Notebook 03: Evaluation (K3)

**Target B: RTX PRO 6000, hoàn toàn OFFLINE.** Đánh giá **3 models** trên **17 in-the-wild subsets** + **standard benchmarks** (AIGCDetect, GenImage, UnivFD, DRCT, Synthbuster, EvalGEN):

| # | Model | Protocol |
|---|-------|----------|
| 1 | Integrated LoRA (output K2) — **bắt buộc** | multi-crop 504: replicate-pad ảnh < 504, 5 crops (center + 4 corners), average logits |
| 2 | B-Free baseline `BFREE_dino2reg4` | Wrapper5crops gốc của repo (replicate + 5-crop ở mức embedding), num_classes=1 |
| 3 | GlobalForge released `REM vit_l_a` | **protocol code thật** `eval_in_the_wild.py`: short-side ≤ 1296 → center-crop 224, ngược lại resize 1296 rồi crop; PNG → JPEG q100; softmax → log-odds |

Score = `logit_fake − logit_real`; model 3 quy về logit `log(p/(1-p))`. Metrics: AUC, bAcc, NLL, ECE, Pd10, EER (`utils/dmetrics.py`). CO-SPY: `fake_only` (paper default). Grouped average theo parent dataset (giống `combine_parent_dataset_average`).

**Inputs (attach trước khi chạy):**
1. **Output notebook 01** (wheels + repo source)
2. **Output notebook 02** (checkpoint LoRA) — qua *Add Input → Your Work*
3. `bfree-baseline-weights` — thư mục `BFREE_dino2reg4/` (config.yaml + weights .pth)
4. `globalforge-code` (thư mục `code/` có `models/REM.py`) + `globalforge-backbone-vitla` (HF ViT-L `vit_l_a/`) + `globalforge-weights` (`checkpoint-best.pth.part_*` × 13, ~1.25GB)
5. `wild-benchmarks` — DATA_ROOT 17 subsets layout `eval_in_the_wild.py` (Chameleon, synthwildx/…, WildRF/test/…, AIGIBench/…, CO-SPY-In-the-Wild/…, RRDataset, B-Free, realchain_CD; mỗi cái `0_real/` + `1_fake/`)

> Models 2/3: nếu thiếu input thì **skip có cảnh báo** (đặt `SKIP_MISSING_MODELS = True`) — kết quả vẫn ra cho các model có sẵn; đặt `False` để fail sớm."""),

    ("code", OFFLINE_LOGGER_SRC),

    ("code", """import glob
from pathlib import Path

WORK = Path("/kaggle/working")


def find_one(patterns, what, optional=False):
    for pat in patterns:
        hits = sorted(glob.glob(pat))
        if hits:
            return hits[0]
    if optional:
        return None
    raise FileNotFoundError(f"NOT FOUND: {what}\\nsearched: {list(patterns)}")


def find_dir_containing(marker_rel, patterns, what, optional=False):
    for pat in patterns:
        for hit in sorted(glob.glob(pat)):
            root = Path(hit)
            while root != Path("/kaggle/input") and root != root.parent:
                if (root / marker_rel).is_file():
                    return root
                root = root.parent
    if optional:
        return None
    raise FileNotFoundError(f"NOT FOUND: {what} (marker: {marker_rel})")


SKIP_MISSING_MODELS = True   # False = fail sớm nếu model 2/3 thiếu input

# ---- bundle cua notebook 01 ----
WHEELS_DIR = find_one(
    ["/kaggle/input/*/wheels_rtxpro6000", "/kaggle/input/wheels_rtxpro6000",
     "/kaggle/input/*/wheels*", "/kaggle/input/wheels*"],
    "wheel bundle (output of notebook 01)")
REPO_SRC = find_dir_containing(
    "code/networks/bfree_globalforge_vit.py",
    ["/kaggle/input/*/bfree_src/code/networks/bfree_globalforge_vit.py",
     "/kaggle/input/bfree_src*/code/networks/bfree_globalforge_vit.py",
     "/kaggle/input/*/B-Free/code/networks/bfree_globalforge_vit.py",
     "/kaggle/input/B-Free*/code/networks/bfree_globalforge_vit.py",
     "/kaggle/input/*/code/networks/bfree_globalforge_vit.py"],
    "B-Free repo source (output of notebook 01)")

# ---- K2 checkpoint (output notebook 02) ----
K2_CKPT = find_one(
    ["/kaggle/input/*/bfree_globalforge_lora_r16.pth",
     "/kaggle/input/bfree_globalforge_lora_r16.pth",
     "/kaggle/input/*/bfree_globalforge_lora_r16_best.pth",
     "/kaggle/input/bfree_globalforge_lora_r16_best.pth"],
    "K2 LoRA checkpoint (output of notebook 02)")

# ---- optional: B-Free baseline weights ----
BFREE_WEIGHTS_DIR = find_dir_containing(
    "config.yaml",
    ["/kaggle/input/*/BFREE_dino2reg4/config.yaml",
     "/kaggle/input/BFREE_dino2reg4/config.yaml",
     "/kaggle/input/*/bfree-baseline-weights/BFREE_dino2reg4/config.yaml"],
    "B-Free baseline weights (BFREE_dino2reg4/)", optional=SKIP_MISSING_MODELS)

# ---- optional: GlobalForge code + backbone + ckpt parts ----
GF_CODE_DIR = find_dir_containing(
    "models/REM.py",
    ["/kaggle/input/*/code/models/REM.py",
     "/kaggle/input/code/models/REM.py",
     "/kaggle/input/*/globalforge*/code/models/REM.py"],
    "GlobalForge code (models/REM.py)", optional=SKIP_MISSING_MODELS)
GF_BACKBONE_VITLA = find_one(
    ["/kaggle/input/*/vit_l_a/config.json", "/kaggle/input/vit_l_a/config.json",
     "/kaggle/input/*/globalforge-backbone-vitla/vit_l_a/config.json"],
    "GlobalForge HF backbone vit_l_a/", optional=SKIP_MISSING_MODELS)
GF_CKPT_PARTS_DIR = find_one(
    ["/kaggle/input/*/checkpoint-best.pth.part_aa", "/kaggle/input/checkpoint-best.pth.part_aa",
     "/kaggle/input/*/globalforge-weights/checkpoint-best.pth.part_aa"],
    "GlobalForge checkpoint parts (checkpoint-best.pth.part_*)", optional=SKIP_MISSING_MODELS)
GF_CKPT_ASSEMBLED = find_one(
    ["/kaggle/input/*/checkpoint-best.pth", "/kaggle/input/checkpoint-best.pth"],
    "assembled GlobalForge checkpoint", optional=SKIP_MISSING_MODELS)

# ---- wild + standard benchmarks root ----
WILD_MARKERS = ["Chameleon/0_real", "synthwildx", "WildRF/test", "AIGIBench",
                "CO-SPY-In-the-Wild", "RRDataset", "realchain_CD"]
DATA_ROOT = None
for cand in sorted(glob.glob("/kaggle/input/*/")) + ["/kaggle/input/"]:
    if any(Path(cand, m).exists() for m in WILD_MARKERS):
        DATA_ROOT = Path(cand)
        break
if DATA_ROOT is None:
    raise FileNotFoundError("wild benchmarks root not found under /kaggle/input — attach 'wild-benchmarks'.")

wheels = sorted(glob.glob(str(Path(WHEELS_DIR) / "*.whl")))
assert wheels, f"No .whl files inside {WHEELS_DIR}"
logger.info(f"WHEELS_DIR   = {WHEELS_DIR} ({len(wheels)} wheels)")
logger.info(f"REPO_SRC     = {REPO_SRC}")
logger.info(f"K2_CKPT      = {K2_CKPT}")
logger.info(f"BFREE_WEIGHT = {BFREE_WEIGHTS_DIR}")
logger.info(f"GF_CODE      = {GF_CODE_DIR}")
logger.info(f"GF_BACKBONE  = {GF_BACKBONE_VITLA}")
logger.info(f"GF_PARTS     = {GF_CKPT_PARTS_DIR} | assembled={GF_CKPT_ASSEMBLED}")
logger.info(f"DATA_ROOT    = {DATA_ROOT}")"""),

    ("code", OFFLINE_ENV_SRC),

    ("code", PIP_INSTALL_SRC),

    ("code", REPO_IMPORT_SRC),

    ("md", """### Model 1 — Integrated LoRA (K2 output, bắt buộc)

Tái tạo đúng kiến trúc từ CONFIG trong checkpoint (kể cả LoRA wrap cùng rank) rồi load state dict."""),

    ("code", """import torch


def load_integrated_lora(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt.get("config", {})
    m = BFreeGlobalForgeViT(
        arch=cfg.get("arch", "vit_base_patch14_reg4_dinov2.lvd142m"),
        num_classes=cfg.get("num_classes", 2),
        img_size=cfg.get("img_size", 504),
        pretrained=False, use_lib=True, use_gsr=True, use_dcs=True,
        lib_kernel=cfg.get("lib_kernel", 3), lib_tau=cfg.get("lib_tau", 0.5),
        gsr_window=cfg.get("gsr_window", 3), gsr_mask_prob=cfg.get("gsr_mask_prob", 1.0),
        dcs_tau=cfg.get("dcs_tau", 0.07), lambda_dcs=cfg.get("lambda_dcs", 0.01),
        label_smoothing=cfg.get("label_smoothing", 0.1),
    )
    from train_lora import apply_lora_to_backbone
    m = apply_lora_to_backbone(m, r=cfg.get("lora_rank", 16))
    report = m.load_state_dict(ckpt["model"], strict=False)
    logger.info(f"Integrated LoRA: missing={len(report.missing_keys)} "
                f"unexpected={len(report.unexpected_keys)} epoch={ckpt.get('epoch')} "
                f"val_bAcc={ckpt.get('val_bacc', 'n/a')}")
    assert not report.unexpected_keys, f"unexpected keys: {report.unexpected_keys[:10]}"
    return m.to(device).eval()

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
model_integrated = load_integrated_lora(K2_CKPT, DEVICE)
logger.info("Model 1 (Integrated LoRA) loaded.")"""),

    ("md", """### Model 2 — B-Free baseline (`BFREE_dino2reg4`) — nguyên protocol repo

`get_network` + `load_weights` + `Wrapper5crops` (tự replicate + 5-crop ở mức patch-embedding); feed full ảnh qua ToTensor+Normalize theo `norm_type` trong config.yaml (giống `main_bfree.py`)."""),

    ("code", """model_bfree = None
BFREE_NORM = "resnet"
if BFREE_WEIGHTS_DIR is None:
    logger.warning("[SKIP] Model 2 (B-Free baseline): weights input not attached")
elif not SKIP_MISSING_MODELS:
    raise FileNotFoundError("B-Free baseline weights required but not attached")
else:
    import yaml

    from networks import get_network, load_weights

    wdir = Path(BFREE_WEIGHTS_DIR)
    with open(wdir / "config.yaml") as f:
        bcfg = yaml.safe_load(f)
    model_bfree = load_weights(get_network(bcfg["arch"]), str(wdir / bcfg["weights_file"]))
    model_bfree = model_bfree.to(DEVICE).eval()
    BFREE_NORM = bcfg["norm_type"]
    logger.info(f"Model 2 (B-Free baseline) loaded: arch={bcfg['arch']} norm={BFREE_NORM}")"""),

    ("md", """### Model 3 — GlobalForge released (`REM vit_l_a`, reassemble 13 parts)

Ghép `checkpoint-best.pth.part_*` → `checkpoint-best.pth` (~1.25GB), suy ra lora_rank/use_lib/use_gsr từ state dict, dựng `REM(vit_l_a)` với `GLOBALFORGE_WEIGHTS_DIR` trỏ backbone HF local, load weights."""),

    ("code", """model_gf = None
if GF_CODE_DIR is None or (GF_CKPT_PARTS_DIR is None and GF_CKPT_ASSEMBLED is None):
    logger.warning("[SKIP] Model 3 (GlobalForge REM): code/weights/backbone input not attached")
elif not SKIP_MISSING_MODELS:
    raise FileNotFoundError("GlobalForge inputs required but not attached")
else:
    import sys as _sys

    _sys.path.insert(0, str(GF_CODE_DIR))
    if GF_CKPT_ASSEMBLED is None:
        GF_CKPT = WORK / "checkpoint-best.pth"
        if not GF_CKPT.exists() or GF_CKPT.stat().st_size < 1_000_000_000:
            parts = sorted(glob.glob(str(Path(GF_CKPT_PARTS_DIR) / "checkpoint-best.pth.part_*")))
            assert parts, f"no checkpoint-best.pth.part_* under {GF_CKPT_PARTS_DIR}"
            logger.info(f"reassembling {len(parts)} parts ...")
            with open(GF_CKPT, "wb") as out:
                for p in parts:
                    with open(p, "rb") as f:
                        while True:
                            chunk = f.read(1024 * 1024 * 64)
                            if not chunk:
                                break
                            out.write(chunk)
        GF_CKPT_ASSEMBLED = GF_CKPT
    logger.info(f"checkpoint: {GF_CKPT_ASSEMBLED} "
                f"({Path(GF_CKPT_ASSEMBLED).stat().st_size/1024**3:.2f} GB)")

    GF_WEIGHTS_ROOT = WORK / "gf_weights"
    GF_WEIGHTS_ROOT.mkdir(exist_ok=True)
    vit_link = GF_WEIGHTS_ROOT / "vit_l_a"
    if not vit_link.exists():
        if Path(GF_BACKBONE_VITLA, "config.json").is_file() and Path(GF_BACKBONE_VITLA).name == "vit_l_a":
            src_backbone = Path(GF_BACKBONE_VITLA)
        elif Path(GF_BACKBONE_VITLA, "vit_l_a", "config.json").is_file():
            src_backbone = Path(GF_BACKBONE_VITLA) / "vit_l_a"
        else:
            src_backbone = None
        if src_backbone is not None:
            import shutil as _shutil
            _shutil.copytree(src_backbone, vit_link)
            logger.info(f"backbone vit_l_a staged at {vit_link}")
    os.environ["GLOBALFORGE_WEIGHTS_DIR"] = str(GF_WEIGHTS_ROOT)

    import models.REM as REM

    ckpt = torch.load(str(GF_CKPT_ASSEMBLED), map_location="cpu", weights_only=False)
    state = ckpt.get("model", ckpt)
    lora_rank = next((int(v.shape[0]) for k, v in state.items()
                      if "lora_A.default.weight" in k), 16)
    use_lib = any(k.startswith("lib.") for k in state)
    use_gsr = any(k.startswith("gsr.") for k in state)
    logger.info(f"GlobalForge released: lora_rank={lora_rank} use_lib={use_lib} use_gsr={use_gsr}")
    model_gf = REM.__dict__["REM"](
        mode="vit_l_a", use_lib=use_lib, use_gsr=use_gsr,
        lib_layer=0, gsr_layer=0, lib_kernel=3, lib_tau=0.5,
        gsr_window=3, gsr_mask_prob=1.0,
    )
    model_gf.load_state_dict(state)
    model_gf = model_gf.to(DEVICE).eval()
    logger.info("Model 3 (GlobalForge released) loaded.")"""),

    ("md", """### Scoring kernels (protocol code thật của từng model)

- **Model 1**: replicate-pad ảnh < 504 (numpy `edge` — tương đương `replicate_wrap`), 5 crops 504 (center + 4 corners), batch forward, average logits → `l1 − l0`.
- **Model 2**: feed full ảnh, Wrapper5crops tự xử lý; num_classes=1 → score = logit.
- **Model 3**: `short256_center` với `eval_resize_short=1296`; PNG round-trip JPEG q100; `Norm(imagenet)` nằm trong model; softmax → log-odds."""),

    ("code", """import io

import numpy as np
from PIL import Image, UnidentifiedImageError
import torchvision.transforms as T
import torchvision.transforms.functional as TF

from utils import dmetrics
from utils.normalization import get_list_norm

IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}
GF_EVAL_RESIZE_SHORT = 1296


def list_images(root, limit=None):
    out = [str(p) for p in sorted(Path(root).iterdir())
           if p.is_file() and p.suffix.lower() in IMG_EXT]
    return out[:limit] if limit else out


def replicate_pad_504(img):
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
    img = replicate_pad_504(Image.open(img_path).convert("RGB"))
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


SCORERS = {"Integrated-LoRA": (lambda p: score_image_integrated(model_integrated, p), True)}
if model_bfree is not None:
    SCORERS["B-Free-baseline"] = (lambda p: score_image_bfree(model_bfree, p), True)
if model_gf is not None:
    SCORERS["GlobalForge-REM"] = (lambda p: score_image_globalforge(model_gf, p), True)
logger.info(f"scorers ready: {list(SCORERS)}")"""),

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

available = [k for k, (r, f, _) in WILD_SUBSETS.items()
             if (DATA_ROOT / r).is_dir() and (DATA_ROOT / f).is_dir()]
missing = [k for k in WILD_SUBSETS if k not in available]
logger.info(f"wild subsets available: {len(available)}/17" + (f" (skipped: {missing})" if missing else ""))
assert available, "no wild subset folders found under DATA_ROOT"
MAX_IMAGES = None"""),

    ("code", """import tqdm


def run_subset(model_name, scorer, real_dir, fake_dir, fake_only=False, max_images=None):
    real_paths = list_images(DATA_ROOT / real_dir, limit=max_images)
    fake_paths = list_images(DATA_ROOT / fake_dir, limit=max_images)
    if fake_only:
        paths, labels = fake_paths, [1] * len(fake_paths)
    else:
        paths = real_paths + fake_paths
        labels = [0] * len(real_paths) + [1] * len(fake_paths)
    if not paths:
        return None

    scores, y, n_skipped = [], [], 0
    for path, label in tqdm.tqdm(list(zip(paths, labels)),
                                 desc=f"{model_name}::{Path(fake_dir).parent.name}",
                                 leave=False):
        try:
            scores.append(scorer(path))
            y.append(label)
        except (OSError, UnidentifiedImageError, ValueError):
            n_skipped += 1
    if n_skipped:
        logger.warning(f"  {n_skipped} unreadable images skipped")
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
    if not ((DATA_ROOT / real_dir).is_dir() and (DATA_ROOT / fake_dir).is_dir()):
        logger.info(f"[skip] {subset}: folders missing")
        continue
    for model_name, (scorer, _) in SCORERS.items():
        r = run_subset(model_name, scorer, real_dir, fake_dir,
                       fake_only=fake_only, max_images=MAX_IMAGES)
        if r:
            all_results.append({"Benchmark": subset, **r})
            logger.info(f"{subset:20s} | {model_name:16s} | bAcc={r['bAcc']:6.2f} "
                        f"AUC={r['AUC']:.4f} n={r['n_images']}")

import pandas as pd

wild_df = pd.DataFrame(all_results)
display(wild_df)"""),

    ("code", """# ============ Standard benchmarks (optional - skip neu chua attach) ============
std_results = []
for bench, (real_dir, fake_dir) in STANDARD_BENCHMARKS.items():
    if not ((DATA_ROOT / real_dir).is_dir() and (DATA_ROOT / fake_dir).is_dir()):
        logger.info(f"[skip] {bench}: folders missing under {DATA_ROOT}")
        continue
    for model_name, (scorer, _) in SCORERS.items():
        r = run_subset(model_name, scorer, real_dir, fake_dir,
                       fake_only=False, max_images=MAX_IMAGES)
        if r:
            std_results.append({"Benchmark": bench, **r})
            logger.info(f"{bench:14s} | {model_name:16s} | bAcc={r['bAcc']:6.2f} "
                        f"AUC={r['AUC']:.4f} n={r['n_images']}")

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
    '''Trung binh tung parent group, roi trung binh cac group (giong eval_in_the_wild.py).'''
    out = {}
    for model in df["Model"].unique():
        groups = {}
        for subset, value in df[df["Model"] == model].groupby("Benchmark")[metric].last().items():
            groups.setdefault(get_parent_dataset_key(subset), []).append(float(value))
        out[model] = sum(sum(v) / len(v) for v in groups.values()) / len(groups)
    return out


results_df = pd.concat([wild_df, std_df], ignore_index=True)
results_df.to_csv(WORK / "eval_results.csv", index=False)

wild_only = results_df[results_df["Benchmark"].isin(WILD_SUBSETS)]
avg_rows = [{"Model": m, "Benchmark": "Avg B.Acc (wild, parent-avg)", "bAcc": v, "fake_only": False}
            for m, v in combine_parent_dataset_average(wild_only).items()]
results_df = pd.concat([results_df, pd.DataFrame(avg_rows)], ignore_index=True)
results_df.to_csv(WORK / "eval_results.csv", index=False)
logger.info("saved /kaggle/working/eval_results.csv")
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
ax.set_title("Balanced Accuracy per subset x model")
ax.legend(loc="lower right")
fig.tight_layout()
fig.savefig(WORK / "eval_bacc_bar.png")

fig, ax = plt.subplots(figsize=(9, max(6, 0.35 * len(piv_bacc))), dpi=130)
sns.heatmap(piv_bacc, annot=True, fmt=".1f", cmap="RdYlGn", vmin=50, vmax=100, ax=ax)
ax.set_title("bAcc heatmap (model x subset)")
fig.tight_layout()
fig.savefig(WORK / "eval_bacc_heatmap.png")
logger.info("saved eval_bacc_bar.png + eval_bacc_heatmap.png")"""),

    ("md", """## Evaluation Complete (K3)

Outputs trong `/kaggle/working/`:
- `eval_results.csv` — model × subset × {AUC, bAcc, NLL, ECE, Pd10, EER} + hàng `Avg B.Acc (wild, parent-avg)`
- `eval_bacc_bar.png` / `eval_bacc_heatmap.png`

**K3 checklist:** notebook chạy được · eval results CSV + heatmap + bar chart.

**K4 (post-training analysis):** dùng `train_log.csv` từ K2 — Pearson/Spearman CE vs DCS + loss dynamics (tương đương `eval/analyze_loss.py` của repo), cập nhật vào báo cáo Phase 8."""),
]

# =====================================================================

if __name__ == "__main__":
    write_nb("01_bfree_setup_bundle.ipynb", NB01_CELLS)
    write_nb("02_bfree_kaggle_train.ipynb", NB02_CELLS)
    write_nb("03_bfree_kaggle_eval.ipynb", NB03_CELLS)
    print("\nAll notebooks generated and validated (json.tool + AST compile).")
