"""Stage-resume gate (CPU): 2 epochs in one go == 1 epoch + resume 1 epoch.

Synthetic COCO_real_512 + 6 SD2.1_* dirs, random-init DINOv2 weights, multi-severity ON.
Passes when the resumed run reproduces the uninterrupted per-epoch metrics (csv, 4 decimals)
and the stage bookkeeping is right. Prints RESUME PASS.
"""
import hashlib
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import timm
import yaml
from PIL import Image
from safetensors.torch import save_file

import _common

IMG = _common.IMG
N_TRAIN, N_VAL, VAL_PCT, BS = 4, 2, 40, 4  # exactly 1 train step (batch 4, drop_last) per epoch


def pick_ids():
    """Stems with the script's own md5 split rule -> exactly N_TRAIN train / N_VAL val ids."""
    tr, va, i = [], [], 0
    while len(tr) < N_TRAIN or len(va) < N_VAL:
        stem = f"id{i:03d}"
        i += 1
        is_val = int(hashlib.md5(stem.encode()).hexdigest(), 16) % 100 < VAL_PCT
        if is_val and len(va) < N_VAL:
            va.append(stem)
        elif not is_val and len(tr) < N_TRAIN:
            tr.append(stem)
    return tr + va
FAKES = ["SD2.1_a", "SD2.1_b", "SD2.1_c", "SD2.1_d", "SD2.1_e", "SD2.1_selfconditioned"]


def make_data(root):
    g = np.random.default_rng(0)
    for d in ["COCO_real_512"] + FAKES:
        (root / d).mkdir(parents=True)
        for stem in pick_ids():
            arr = g.integers(0, 256, (IMG + 16, IMG + 16, 3), dtype=np.uint8)
            Image.fromarray(arr).save(root / d / f"{stem}.png")


def make_weights(path):
    m = timm.create_model("vit_base_patch14_reg4_dinov2.lvd142m", pretrained=False, num_classes=0)
    save_file({k: v.contiguous() for k, v in m.state_dict().items()}, str(path))


def run(tmp, out, extra):
    cmd = [sys.executable, f"{_common.CODE}/train_lora.py",
           "--train_data_root", str(tmp / "data"), "--dinov2_sd", str(tmp / "dino.safetensors"),
           "--output_dir", str(out), "--config", str(tmp / "ms.yaml"),
           "--epochs", "2", "--img_size", str(IMG), "--batch_size", str(BS), "--num_workers", "0",
           "--lora_rank", "4", "--lora_alpha", "8", "--val_md5_percentile", str(VAL_PCT),
           "--seed", "7"] + extra
    out.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(cmd, cwd=_common.CODE, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-3000:]


def rows(out):
    return (out / "train_log.csv").read_text().strip().splitlines()


def main():
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        make_data(tmp / "data")
        make_weights(tmp / "dino.safetensors")
        ms = yaml.safe_load(open(f"{_common.CODE}/configs/bfree_dcs.yaml"))
        ms["multi_severity"]["enabled"] = True
        yaml.safe_dump(ms, open(tmp / "ms.yaml", "w"))

        A, B, C = tmp / "A", tmp / "B", tmp / "C"
        run(tmp, A, [])
        run(tmp, B, ["--end_epoch", "1"])
        assert len(rows(B)) == 2, rows(B)                      # header + epoch 1
        assert (B / "resume_state.pth").is_file()
        assert not (B / "selected_ckpt.txt").exists(), "stage 1 must not select a ckpt"
        run(tmp, C, ["--resume_from", str(B / "resume_state.pth")])
        assert len(rows(C)) == 3, rows(C)                      # header + epoch 1 (carried) + 2
        assert (C / "selected_ckpt.txt").is_file()
        assert rows(C)[1] == rows(B)[1] == rows(A)[1], (rows(A)[1], rows(B)[1], rows(C)[1])
        print("uninterrupted:", rows(A)[2])
        print("resumed      :", rows(C)[2])
        assert rows(A)[2] == rows(C)[2], "resumed epoch 2 differs from uninterrupted run"

        # config drift must be refused
        try:
            run(tmp, tmp / "D", ["--resume_from", str(B / "resume_state.pth"), "--lr", "5e-5"])
        except AssertionError as e:
            assert "config khác stage trước" in str(e), str(e)[-500:]
        else:
            raise SystemExit("FAIL: lr drift not detected")
    print("RESUME PASS")


if __name__ == "__main__":
    main()
