import os
import io
import random
from typing import Tuple, Dict, Any, Optional

import pandas as pd
from PIL import Image, ImageFilter
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

from utils.normalization import get_list_norm


class DegradationPipeline:
    """Pipeline suy hao mô phỏng GlobalForge (JPEG + Blur + Color Distortion)."""

    def __init__(
        self,
        jpeg_quality_min: int = 20,
        jpeg_quality_max: int = 80,
        blur_kernel: int = 7,
        blur_sigma_min: float = 0.5,
        blur_sigma_max: float = 1.5,
        blur_prob: float = 0.8,
        color_prob: float = 0.8,
    ):
        self.jpeg_min = jpeg_quality_min
        self.jpeg_max = jpeg_quality_max
        self.blur_kernel = blur_kernel
        self.blur_sigma_min = blur_sigma_min
        self.blur_sigma_max = blur_sigma_max
        self.blur_prob = blur_prob
        self.color_prob = color_prob
        self.color_jitter = transforms.ColorJitter(
            brightness=0.4, contrast=0.4, saturation=0.4, hue=0.06
        )

    def _apply_jpeg(self, img: Image.Image) -> Image.Image:
        quality = random.randint(self.jpeg_min, self.jpeg_max)
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        return Image.open(buffer).convert("RGB")

    def _apply_blur(self, img: Image.Image) -> Image.Image:
        if random.random() < self.blur_prob:
            sigma = random.uniform(self.blur_sigma_min, self.blur_sigma_max)
            return img.filter(ImageFilter.GaussianBlur(radius=sigma))
        return img

    def _apply_color(self, img: Image.Image) -> Image.Image:
        if random.random() < self.color_prob:
            return self.color_jitter(img)
        return img

    def __call__(self, img: Image.Image) -> Image.Image:
        # Áp dụng tuần tự JPEG -> GaussianBlur -> ColorDistortion
        img = self._apply_jpeg(img)
        img = self._apply_blur(img)
        img = self._apply_color(img)
        return img


class BFreeDataset(Dataset):
    """Dataset nạp cặp ảnh clean / degraded cho B-Free + GlobalForge."""

    def __init__(
        self,
        csv_file: str,
        data_root: str,
        img_size: int = 504,
        is_train: bool = True,
        degradation_cfg: Optional[Dict[str, Any]] = None,
        norm_type: str = "resnet",
    ):
        self.data_root = data_root
        self.df = pd.read_csv(csv_file)
        self.img_size = img_size
        self.is_train = is_train

        deg_cfg = degradation_cfg or {}
        self.degradation = DegradationPipeline(
            jpeg_quality_min=deg_cfg.get("jpeg_quality_min", 20),
            jpeg_quality_max=deg_cfg.get("jpeg_quality_max", 80),
            blur_kernel=deg_cfg.get("blur_kernel", 7),
            blur_sigma_min=deg_cfg.get("blur_sigma_min", 0.5),
            blur_sigma_max=deg_cfg.get("blur_sigma_max", 1.5),
            blur_prob=deg_cfg.get("blur_prob", 0.8),
            color_prob=deg_cfg.get("color_prob", 0.8),
        )

        self.normalize = transforms.Compose(get_list_norm(norm_type))

    def __len__(self) -> int:
        return len(self.df)

    def _load_image(self, rel_path: str) -> Image.Image:
        full_path = os.path.join(self.data_root, rel_path)
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Image not found: {full_path}")
        return Image.open(full_path).convert("RGB")

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        row = self.df.iloc[idx]
        img_clean = self._load_image(row["filename"])
        
        # Nhãn: 0 nếu là REAL/0, 1 nếu là FAKE/Synthetic
        label_raw = row.get("label", 0)
        label = 0 if (label_raw == "REAL" or label_raw == 0 or label_raw == "0") else 1

        # Cắt ngẫu nhiên (train) hoặc crop trung tâm (val) về img_size
        if self.is_train:
            i, j, h, w = transforms.RandomCrop.get_params(
                img_clean, output_size=(self.img_size, self.img_size)
            )
            img_clean = TF.crop(img_clean, i, j, h, w)
            if random.random() > 0.5:
                img_clean = TF.hflip(img_clean)
        else:
            img_clean = TF.center_crop(img_clean, (self.img_size, self.img_size))

        # View suy hao cho DCS loss
        img_deg = self.degradation(img_clean)

        tensor_clean = self.normalize(img_clean)
        tensor_deg = self.normalize(img_deg)

        return tensor_clean, tensor_deg, label