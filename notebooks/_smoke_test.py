"""Smoke-test notebook cell logic locally (CPU) with mocked data.

Executes the pure-Python parts of the notebook cells (PairDataset, scoring
kernels, run_subset metrics, combine_parent_dataset_average) against tiny
synthetic image folders. Run: python notebooks/_smoke_test.py
"""

import ast
import hashlib
import json
import math
import os
import random
import shutil
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


def run_nb02_pair_dataset(tmp):
    # --- build fake dataset tree: 8 real ids, 6 fake variants ---
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
           "random": random, "np": np, "torch": torch, "Image": Image,
           "T": T,
           "TF": TF,
           "DegradationPipeline": __import__("datasets.bfree_dataset", fromlist=["x"]).DegradationPipeline,
           "get_list_norm": __import__("utils.normalization", fromlist=["x"]).get_list_norm,
           "Dataset": torch.utils.data.Dataset,
           "CONFIG": {"val_md5_percentile": 3, "img_size": 504, "pair_real_prob": 0.5,
                      "val_fake_variant": "SD2.1_selfconditioned", "num_workers": 0,
                      "seed": 42},
           "DATA_ROOT": data_root, "REAL_DIR": real_dir, "FAKE_DIRS": fake_dirs}
    src = cells[4]  # dataset cell (build_stem_index + PairDataset)
    tree = ast.parse(strip_magics(src))
    exec(compile(tree, "<pair-cell>", "exec"), env)

    train_ds, val_ds = env["train_ds"], env["val_ds"]
    a, b, c = train_ds[0]
    assert tuple(a.shape) == (3, 504, 504) and tuple(b.shape) == (3, 504, 504)
    assert c in (0, 1)
    v0, v1 = val_ds[0], val_ds[1]
    assert v0[2] == 0 and v1[2] == 1, f"val not balanced: {v0[2]}, {v1[2]}"

    # md5 split determinism
    stems = env["ALL_STEMS"]
    assert stems == sorted(stems)
    n_val = sum(1 for s in stems
                if int(hashlib.md5(s.encode()).hexdigest(), 16) % 100 < 3)
    assert n_val == len(env["VAL_IDS"])

    # per-ID pairing statistics (over many draws each id must give real & fakes)
    labels_seen = {0: 0, 1: 0}
    for i in range(len(train_ds) * 3):
        _, _, lab = train_ds[i % len(train_ds)]
        labels_seen[lab] += 1
    assert labels_seen[1] > 0 and labels_seen[0] > 0
    print(f"NB02 PairDataset OK: train={len(train_ds)}, val={len(val_ds)}, "
          f"draw labels={labels_seen}, val_ids={n_val}")


def run_nb03_eval(tmp):
    cells = get_code_cells("03_bfree_kaggle_eval.ipynb")
    # fake DATA_ROOT with one subset
    data_root = os.path.join(tmp, "wild")
    for sub in ["Chameleon/0_real", "Chameleon/1_fake"]:
        d = os.path.join(data_root, sub)
        os.makedirs(d, exist_ok=True)
    for i in range(3):
        Image.new("RGB", (300, 300), (i * 40, 20, 20)).save(os.path.join(data_root, "Chameleon/0_real", f"r{i}.jpg"))
        Image.new("RGB", (700, 500), (10, i * 40, 20)).save(os.path.join(data_root, "Chameleon/1_fake", f"f{i}.png"))

    dmetrics = __import__("utils.dmetrics", fromlist=["x"])
    env = {"__name__": "nb", "os": os, "glob": __import__("glob"), "np": np,
           "torch": torch, "tqdm": __import__("tqdm"), "pd": __import__("pandas"),
           "Image": Image, "Path": Path, "io": __import__("io"),
           "T": T,
           "TF": TF,
           "UnidentifiedImageError": __import__("PIL").UnidentifiedImageError,
           "dmetrics": dmetrics, "get_list_norm": __import__("utils.normalization", fromlist=["x"]).get_list_norm,
           "DEVICE": "cpu", "BFREE_NORM": "resnet"}

    # scoring kernels cell
    exec(compile(ast.parse(strip_magics(cells[6])), "<kernels>", "exec"), env)

    # replace real models with a stub scorer: use replicate_pad_504 + five_crop_boxes on tiny images
    def fake_scorer_integrated(p):
        img = env["replicate_pad_504"](Image.open(p).convert("RGB"))
        boxes = env["five_crop_boxes"](img)
        return float(len(boxes))

    def fake_scorer_gf(p):
        img = Image.open(p).convert("RGB")
        w, h = img.size
        if min(w, h) > 1296:
            scale = 1296 / min(w, h)
            img = env["TF"].resize(img, [round(h * scale), round(w * scale)])
        img = env["TF"].center_crop(img, [224, 224])
        if p.lower().endswith(".png"):
            import io as _io
            buf = _io.BytesIO()
            img.save(buf, format="JPEG", quality=100)
            buf.seek(0)
            img = Image.open(buf).convert("RGB").copy()
        return 1.0

    env["SCORERS"] = {"Integrated-LoRA": fake_scorer_integrated,
                      "B-Free-baseline": fake_scorer_integrated,
                      "GlobalForge-REM": fake_scorer_gf}
    env["DATA_ROOT"] = data_root

    # registry cell (adjust WILD_SUBSETS availability check to pass)
    reg_src = strip_magics(cells[7])
    reg_src = reg_src.replace('MAX_IMAGES = None', 'MAX_IMAGES = None')
    exec(compile(ast.parse(reg_src), "<registry>", "exec"), env)

    # run_subset cell (only the function def + wild loop) — extract just function def
    kernel_src = strip_magics(cells[8])
    # cut everything from 'all_results = []' onwards (loop over real DATA_ROOT)
    kernel_src = kernel_src.split("all_results = []")[0]
    exec(compile(ast.parse(kernel_src), "<run_subset>", "exec"), env)

    r = env["run_subset"]("Integrated-LoRA", "Chameleon/0_real", "Chameleon/1_fake",
                          data_root, fake_only=False, max_images=None)
    assert r["n_images"] == 6 and 0 <= r["bAcc"] <= 100 and 0 <= r["AUC"] <= 1
    assert math.isfinite(r["NLL"]) and math.isfinite(r["EER"])
    rf = env["run_subset"]("GlobalForge-REM", "Chameleon/0_real", "Chameleon/1_fake",
                           data_root, fake_only=True, max_images=None)
    assert rf["fake_only"] and rf["n_images"] == 3 and math.isnan(rf["AUC"])
    print(f"NB03 run_subset OK: pair={r['bAcc']:.1f} bAcc, fake_only n={rf['n_images']}")

    # combine_parent_dataset_average
    combine_src = strip_magics(cells[10]).split("results_df = pd.concat")[0]
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
        run_nb02_pair_dataset(tmp)
        run_nb03_eval(tmp)
    print("\nSMOKE TESTS ALL PASS")
