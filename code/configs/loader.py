"""Loader YAML tối giản cho config bfree_dcs.yaml.

B-Free gốc không có loader YAML riêng (config nằm trong weights/<model>/config.yaml
với 4 key: weights_file, model_name, arch, norm_type — chỉ phục vụ inference).
Nên loader này tự viết, không động vào code gốc B-Free.
"""

import yaml


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_model_kwargs(cfg: dict) -> dict:
    """Trích dict kwargs cho BFreeGlobalForgeViT.__init__ từ config YAML."""
    b = cfg["backbone"]
    l = cfg["loss"]
    return {
        "arch": b["arch"],
        "num_classes": b["num_classes"],
        "img_size": b["img_size"],
        "pretrained": b["pretrained"],
        "use_lib": b["use_lib"],
        "use_gsr": b["use_gsr"],
        "lib_kernel": b["lib_kernel"],
        "lib_tau": b["lib_tau"],
        "gsr_window": b["gsr_window"],
        "gsr_mask_prob": b["gsr_mask_prob"],
        "use_dcs": l["use_dcs"],
        "dcs_tau": l["dcs_tau"],
        "lambda_dcs": l["lambda_dcs"],
        "label_smoothing": l["label_smoothing"],
    }
