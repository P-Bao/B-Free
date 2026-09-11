"""Smoke-test notebook cell logic locally (CPU) with mocked data.

Executes pure-Python parts of the notebook cells (bundle discovery helpers,
PairDataset, scoring kernels, run_subset metrics, combine_parent_dataset_average)
against tiny synthetic image folders. Run: python notebooks/_smoke_test.py
"""

import ast
import hashlib
import json
import math
import os
import random
import sys
import tempfile
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_CODE = os.path.join(os.path.dirname(HERE), "code")
sys.path.insert(0, REPO_CODE)

import numpy as np
import torch
import torch.utils.data
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image


def get_code_cells(nb_name):
    with open(os.path.join(HERE, nb_name), encoding="utf-8") as f:
        doc = json.load(f)
    return ["".join(c["source"]) for c in doc["cells"] if c["cell_type"] == "code"]


def strip_magics(src):
    out, in_magic = [], False
    for line in src.splitlines(keepends=True):
        if in_magic:
            if line.rstrip().endswith("\\"):
                continue
            in_magic = False
            continue
        s = line.lstrip()
        if s.startswith(("!", "%")):
            indent = line[: len(line) - len(s)]
            out.append(f"{indent}pass  # <ipython-magic>\n")
            if line.rstrip().endswith("\\"):
                in_magic = True
            continue
        out.append(line)
    return "".join(out)


def run_bundle_helpers(tmp):
    """Test find_one / find_dir_containing from notebook 02's discovery cell."""
    cells = get_code_cells("02_bfree_kaggle_train.ipynb")
    src = strip_magics(cells[1])
    # extract just the two helper functions (between their defs and the bundle section)
    start = src.index("def find_one")
    end = src.index("# ---- bundle produced by notebook 01")
    src = src[start:end]

    # fake /kaggle/input structure
    fake_input = Path(tmp) / "input"
    (fake_input / "notebook01-output" / "wheels_rtxpro6000").mkdir(parents=True)
    (fake_input / "notebook01-output" / "wheels_rtxpro6000" / "torch-2.8.0.whl").write_bytes(b"x" * 16)
    repo = fake_input / "notebook01-output" / "bfree_src"
    (repo / "code" / "networks").mkdir(parents=True)
    (repo / "code" / "networks" / "bfree_globalforge_vit.py").write_text("# ok")

    env = {"glob": __import__("glob"), "Path": Path, "FileNotFoundError": FileNotFoundError}
    exec(compile(ast.parse(src), "<helpers>", "exec"), env)

    find_one = env["find_one"]
    find_dir_containing = env["find_dir_containing"]

    hit = find_one(["NONEXIST", str(fake_input / "*" / "wheels_rtxpro6000")], "wheels")
    assert Path(hit).name == "wheels_rtxpro6000", hit

    root = find_dir_containing(
        "code/networks/bfree_globalforge_vit.py",
        [str(fake_input / "*" / "bfree_src" / "code" / "networks" / "bfree_globalforge_vit.py")],
        "repo")
    assert Path(root).name == "bfree_src", root

    try:
        find_one(["NOPE-*"], "nothing")
        raise AssertionError("should have raised")
    except FileNotFoundError:
        pass
    assert find_one(["NOPE-*"], "nothing", optional=True) is None
    print("NB02/03 bundle discovery helpers OK")


def run_nb02_pair_dataset(tmp):
    data_root = os.path.join(tmp, "train_data")
    real_dir = os.path.join(data_root, "COCO_real_512")
    fake_dirs = [os.path.join(data_root, f"SD2.1_{v}")
                 for v in ["selfconditioned", "selfconditioned_origBG",
                           "inpainted_samecat", "inpainted_samecat_origBG",
                           "inpainted_diffcat", "inpainted_diffcat_origBG"]]
    for d in [real_dir] + fake_dirs:
        os.makedirs(d, exist_ok=True)
    ids = [f"{n:08d}" for n in range(200)]
    for d in [real_dir] + fake_dirs:
        for i, stem in enumerate(ids):
            Image.new("RGB", (512, 512), color=(i * 30 % 255, 50, 150)).save(
                os.path.join(d, stem + ".jpg"), quality=90)

    cells = get_code_cells("02_bfree_kaggle_train.ipynb")
    env = {"__name__": "nb", "os": os, "glob": __import__("glob"), "hashlib": hashlib,
           "random": random, "np": np, "torch": torch, "Image": Image, "Path": Path,
           "T": T, "TF": TF,
           "DegradationPipeline": __import__("datasets.bfree_dataset", fromlist=["x"]).DegradationPipeline,
           "get_list_norm": __import__("utils.normalization", fromlist=["x"]).get_list_norm,
           "Dataset": torch.utils.data.Dataset,
           "CONFIG": {"val_md5_percentile": 3, "img_size": 504, "pair_real_prob": 0.5,
                      "val_fake_variant": "SD2.1_selfconditioned", "seed": 42},
           "REAL_DIR": real_dir, "FAKE_DIRS": fake_dirs}
    src = cells[6]  # dataset cell (build_stem_index + PairDataset + generator)
    tree = ast.parse(strip_magics(src))
    tree.body = [n for n in tree.body
                if not (isinstance(n, ast.Expr) and ast.unparse(n).startswith(("logger.", "display")))]
    exec(compile(tree, "<pair-cell>", "exec"), env)

    train_ds, val_ds = env["train_ds"], env["val_ds"]
    a, b, c = train_ds[0]
    assert tuple(a.shape) == (3, 504, 504) and tuple(b.shape) == (3, 504, 504)
    v0, v1 = val_ds[0], val_ds[1]
    assert v0[2] == 0 and v1[2] == 1, f"val not balanced: {v0[2]}, {v1[2]}"

    labels_seen = {0: 0, 1: 0}
    for i in range(len(train_ds) * 3):
        _, _, lab = train_ds[i % len(train_ds)]
        labels_seen[lab] += 1
    assert labels_seen[1] > 0 and labels_seen[0] > 0
    print(f"NB02 PairDataset OK: train={len(train_ds)}, val={len(val_ds)}, draws={labels_seen}")


def run_nb03_eval(tmp):
    cells = get_code_cells("03_bfree_kaggle_eval.ipynb")
    data_root = os.path.join(tmp, "wild")
    for sub in ["Chameleon/0_real", "Chameleon/1_fake"]:
        os.makedirs(os.path.join(data_root, sub), exist_ok=True)
    for i in range(3):
        Image.new("RGB", (300, 300), (i * 40, 20, 20)).save(os.path.join(data_root, "Chameleon/0_real", f"r{i}.jpg"))
        Image.new("RGB", (700, 500), (10, i * 40, 20)).save(os.path.join(data_root, "Chameleon/1_fake", f"f{i}.png"))

    dmetrics = __import__("utils.dmetrics", fromlist=["x"])
    env = {"__name__": "nb", "os": os, "glob": __import__("glob"), "np": np,
           "torch": torch, "tqdm": __import__("tqdm"), "pd": __import__("pandas"),
           "Image": Image, "Path": Path, "io": __import__("io"),
           "T": T, "TF": TF, "DEVICE": "cpu", "BFREE_NORM": "resnet",
           "UnidentifiedImageError": __import__("PIL").UnidentifiedImageError,
           "dmetrics": dmetrics, "get_list_norm": __import__("utils.normalization", fromlist=["x"]).get_list_norm,
           "logger": __import__("logging").getLogger("test"),
           "model_bfree": None, "model_gf": None, "model_integrated": None}

    # code cell order in NB03: 0 logger, 1 discovery, 2 env, 3 pip, 4 repo-import,
    # 5 model1, 6 model2, 7 model3, 8 kernels, 9 registry, 10 run_subset loop...
    kernels_src = strip_magics(cells[8])
    exec(compile(ast.parse(kernels_src), "<kernels>", "exec"), env)

    def fake_scorer_integrated(p):
        img = env["replicate_pad_504"](Image.open(p).convert("RGB"))
        boxes = env["five_crop_boxes"](img)
        return float(len(boxes))

    def fake_scorer_gf(p):
        img = Image.open(p).convert("RGB")
        w, h = img.size
        if min(w, h) > 1296:
            scale = 1296 / min(w, h)
            img = TF.resize(img, [round(h * scale), round(w * scale)])
        img = TF.center_crop(img, [224, 224])
        if p.lower().endswith(".png"):
            import io as _io
            buf = _io.BytesIO()
            img.save(buf, format="JPEG", quality=100)
            buf.seek(0)
            img = Image.open(buf).convert("RGB").copy()
        return 1.0

    env["SCORERS"] = {"Integrated-LoRA": (lambda p: fake_scorer_integrated(p), True),
                      "GlobalForge-REM": (fake_scorer_gf, True)}
    env["DATA_ROOT"] = Path(data_root)

    reg_src = strip_magics(cells[9])
    reg_src = reg_src.replace("logger.info(", "print(")
    exec(compile(ast.parse(reg_src), "<registry>", "exec"), env)

    kernel_src = strip_magics(cells[10]).split("all_results = []")[0]
    exec(compile(ast.parse(kernel_src), "<run_subset>", "exec"), env)

    r = env["run_subset"]("Integrated-LoRA", env["SCORERS"]["Integrated-LoRA"][0],
                          "Chameleon/0_real", "Chameleon/1_fake",
                          fake_only=False, max_images=None)
    assert r["n_images"] == 6 and 0 <= r["bAcc"] <= 100 and 0 <= r["AUC"] <= 1
    assert math.isfinite(r["NLL"]) and math.isfinite(r["EER"])
    rf = env["run_subset"]("GlobalForge-REM", env["SCORERS"]["GlobalForge-REM"][0],
                           "Chameleon/0_real", "Chameleon/1_fake",
                           fake_only=True, max_images=None)
    assert rf["fake_only"] and rf["n_images"] == 3 and math.isnan(rf["AUC"])
    print(f"NB03 run_subset OK: pair bAcc={r['bAcc']:.1f}, fake_only n={rf['n_images']}")

    combine_src = strip_magics(cells[12]).split("results_df = pd.concat")[0]
    env2 = {"pd": __import__("pandas")}
    exec(compile(ast.parse(combine_src), "<combine>", "exec"), env2)
    df = __import__("pandas").DataFrame([
        {"Model": "M", "Benchmark": "SynthWildx-DALLE3", "bAcc": 80.0},
        {"Model": "M", "Benchmark": "SynthWildx-Firefly", "bAcc": 90.0},
        {"Model": "M", "Benchmark": "Chameleon", "bAcc": 50.0},
        {"Model": "M", "Benchmark": "CO-SPY-Civitai", "bAcc": 100.0},
        {"Model": "M", "Benchmark": "CO-SPY-Lexica", "bAcc": 0.0},
    ])
    avg = env2["combine_parent_dataset_average"](df)
    assert abs(avg["M"] - ((80 + 90) / 2 + 50 + (100 + 0) / 2) / 3) < 1e-9
    print(f"NB03 combine_parent_dataset_average OK: {avg['M']:.2f}")


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(dir=HERE) as tmp:
        run_bundle_helpers(tmp)
        run_nb02_pair_dataset(tmp)
        run_nb03_eval(tmp)
    print("\nSMOKE TESTS ALL PASS")
