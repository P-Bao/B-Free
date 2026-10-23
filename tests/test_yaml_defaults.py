"""YAML -> CFG threading: YAML values must win unless the flag is passed on the CLI.

Regression guard: S5 writes `backbone.gsr_window: 7` into run_config.yaml, but
argparse defaults used to win -> the run silently used 3.
"""

import pathlib
import sys
import tempfile

import yaml

import _common  # noqa: F401  (adds code/ to sys.path)
import train_lora as TL

CFG_PATH = pathlib.Path(_common.ROOT) / "code" / "configs" / "bfree_dcs.yaml"


def _parse(argv):
    old = sys.argv
    sys.argv = ["train_lora.py"] + argv
    try:
        TL.parse_args()
    finally:
        sys.argv = old


def _run_cfg(td, base, extra=()):
    cfg = td / "run.yaml"
    cfg.write_text(yaml.safe_dump(base, sort_keys=False), encoding="utf-8")
    sd = td / "dummy.safetensors"
    sd.write_bytes(b"")
    _parse(["--config", str(cfg), "--train_data_root", str(td), "--dinov2_sd", str(sd),
            "--output_dir", str(td), "--num_workers", "0", *extra])


def test_yaml_values_reach_cfg():
    base = yaml.safe_load(CFG_PATH.read_text(encoding="utf-8"))
    base["backbone"].update(dict(use_lib=True, use_gsr=True, head_mode="cls_plus_patch",
                                 gsr_window=7, gsr_mask_prob=0.5, gsr_zero_init=True,
                                 lib_beta_init=-10.0, lib_kernel=5, lib_tau=0.25))
    base["loss"].update(dict(use_dcs=True, dcs_mode="class_aware", dcs_tau=0.11,
                             lambda_dcs=0.2, label_smoothing=0.0))
    base["data"].update(dict(paired=True, other_frac=0.25, val_aug=True))
    base["online_aug"].update(dict(enabled=True, preset="light"))
    with tempfile.TemporaryDirectory() as tmp:
        _run_cfg(pathlib.Path(tmp), base)
    expect = dict(gsr_window=7, gsr_mask_prob=0.5, gsr_zero_init=True, lib_beta_init=-10.0,
                  lib_kernel=5, lib_tau=0.25, head_mode="cls_plus_patch",
                  dcs_mode="class_aware", dcs_tau=0.11, lambda_dcs=0.2,
                  label_smoothing=0.0, paired=True, other_frac=0.25, val_aug=True)
    for k, v in expect.items():
        assert TL.CFG[k] == v, (k, TL.CFG[k], v)
    assert TL.CFG["online_aug"]["preset"] == "light"


def test_cli_overrides_yaml():
    base = yaml.safe_load(CFG_PATH.read_text(encoding="utf-8"))
    base["backbone"]["gsr_window"] = 7
    base["backbone"]["head_mode"] = "cls_plus_patch"
    with tempfile.TemporaryDirectory() as tmp:
        _run_cfg(pathlib.Path(tmp), base,
                 extra=["--gsr_window", "3", "--head_mode", "cls",
                        "--label_smoothing", "0.05"])
    assert TL.CFG["gsr_window"] == 3
    assert TL.CFG["head_mode"] == "cls"
    assert TL.CFG["label_smoothing"] == 0.05


if __name__ == "__main__":
    test_yaml_values_reach_cfg()
    test_cli_overrides_yaml()
    print("OK yaml defaults")
