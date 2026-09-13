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


def build_degradation_kwargs(cfg: dict) -> dict:
    """Trích dict kwargs cho DegradationPipeline.__init__ (datasets/bfree_dataset.py)
    từ config YAML — đối xứng với build_model_kwargs(). Dùng .get(..., default) cho
    resize_*/noise_* vì đây là key mới bổ sung, để các YAML cũ (chưa có 2 mục này)
    vẫn nạp được mà không lỗi, rơi về đúng default của DegradationPipeline."""
    d = cfg["degradation"]
    return {
        "jpeg_quality_min": d["jpeg_quality_min"],
        "jpeg_quality_max": d["jpeg_quality_max"],
        "blur_kernel": d["blur_kernel"],
        "blur_sigma_min": d["blur_sigma_min"],
        "blur_sigma_max": d["blur_sigma_max"],
        "blur_prob": d["blur_prob"],
        "color_prob": d["color_prob"],
        "resize_scale_min": d.get("resize_scale_min", 0.2),
        "resize_scale_max": d.get("resize_scale_max", 0.9),
        "resize_prob": d.get("resize_prob", 0.5),
        "noise_std_min": d.get("noise_std_min", 0.02),
        "noise_std_max": d.get("noise_std_max", 0.1),
        "noise_prob": d.get("noise_prob", 0.5),
    }