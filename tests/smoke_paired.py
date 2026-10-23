"""Tiny CPU smoke: paired sampling + online aug + class-aware DCS + time budget.

Not part of the gate suite (slow: subprocess ViT-B). Run manually:
    cd tests && ..\\.venv\\Scripts\\python.exe smoke_paired.py
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

IMG, VAL_PCT = _common.IMG, 40
FAKES = ["SD2.1_a", "SD2.1_b", "SD2.1_c", "SD2.1_d", "SD2.1_e", "SD2.1_selfconditioned"]


def pick_ids():
    tr, va, i = [], [], 0
    while len(tr) < 4 or len(va) < 2:
        stem = f"id{i:03d}"
        i += 1
        is_val = int(hashlib.md5(stem.encode()).hexdigest(), 16) % 100 < VAL_PCT
        (va if is_val else tr).append(stem)
    return tr[:4] + va[:2]


def main():
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        g = np.random.default_rng(0)
        for d in ["COCO_real_512"] + FAKES:
            (tmp / "data" / d).mkdir(parents=True)
            for stem in pick_ids():
                Image.fromarray(g.integers(0, 256, (IMG + 16, IMG + 16, 3), dtype=np.uint8)).save(
                    tmp / "data" / d / f"{stem}.png")
        m = timm.create_model("vit_base_patch14_reg4_dinov2.lvd142m", pretrained=False, num_classes=0)
        save_file({k: v.contiguous() for k, v in m.state_dict().items()}, str(tmp / "dino.safetensors"))
        yaml.safe_dump(yaml.safe_load(open(f"{_common.CODE}/configs/bfree_dcs.yaml", encoding="utf-8")),
                       open(tmp / "cfg.yaml", "w", encoding="utf-8"))
        out = tmp / "run"
        cmd = [sys.executable, f"{_common.CODE}/train_lora.py",
               "--train_data_root", str(tmp / "data"), "--dinov2_sd", str(tmp / "dino.safetensors"),
               "--output_dir", str(out), "--config", str(tmp / "cfg.yaml"),
               "--epochs", "1", "--img_size", str(IMG), "--batch_size", "4", "--num_workers", "0",
               "--lora_rank", "4", "--lora_alpha", "8", "--val_md5_percentile", str(VAL_PCT),
               "--paired", "--other_frac", "0.0", "--online_aug", "light", "--val_aug",
               "--dcs_mode", "class_aware", "--head_mode", "cls_plus_patch",
               "--gsr_zero_init", "--label_smoothing", "0.0", "--time_budget_hours", "0.0001"]
        r = subprocess.run(cmd, cwd=_common.CODE, capture_output=True, text=True,
                           encoding="utf-8", errors="replace")
        print(r.stdout[-1500:])
        if r.returncode != 0:
            print(r.stderr[-4000:])
            raise SystemExit(1)
        log = (out / "train_log.csv").read_text(encoding="utf-8").splitlines()
        print(log[0])
        print(log[1])
        assert log[0].endswith("gamma") and len(log) == 2
        # gamma bat dau tu 0 va phai roi khoi 0 (dau tuy loss; chi kiem do lon).
        assert 0.0 < abs(float(log[1].split(",")[7])) < 1e-2, log[1]
        assert (out / "selected_ckpt.txt").is_file(), "1/1 epoch completed -> selected ckpt expected"
        assert Path(out / "bfree_globalforge_lora_r16.pth").is_file()
        print("SMOKE PASS")


if __name__ == "__main__":
    main()
