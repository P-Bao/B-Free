import os
import io
import random
from typing import Tuple, Dict, Any, Optional

import numpy as np
import pandas as pd
from PIL import Image, ImageFilter
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

from utils.normalization import get_list_norm


class DegradationPipeline:
    """Pipeline suy hao mô phỏng GlobalForge (JPEG + Blur + Color Distortion), bổ sung
    thêm Resize + Gaussian Noise.

    Lý do bổ sung: JPEG/Blur/Color là nguyên bản từ pipeline GlobalForge
    (datasets_real.py:378-383) — KHÔNG đổi mặc định của 3 toán tử này để giữ đúng
    recipe gốc của paper. Resize và Noise KHÔNG có trong pipeline gốc, và đối chiếu với
    eval RealDeg-Bench-style (notebook 02) cho thấy đây chính xác là 2 toán tử tệ nhất
    (bAcc 0.6464 / 0.6402, so với brightness/contrast/saturation ~0.999 vì ColorJitter
    hiện có đã đủ mạnh) — khớp với việc L_DCS chưa từng thấy 2 phép biến đổi này lúc
    train nên model không học được bất biến với chúng. Compound degradation (nhiều toán
    tử nối tiếp) vì vậy cũng rớt rất nặng (0.5715 ở 5 bước) do gần như luôn dính ít nhất
    1 trong 2 toán tử còn thiếu này.
    """

    def __init__(
        self,
        jpeg_quality_min: int = 20,
        jpeg_quality_max: int = 80,
        blur_kernel: int = 7,
        blur_sigma_min: float = 0.5,
        blur_sigma_max: float = 1.5,
        blur_prob: float = 0.8,
        color_prob: float = 0.8,
        resize_scale_min: float = 0.2,
        resize_scale_max: float = 0.9,
        resize_prob: float = 0.5,
        noise_std_min: float = 0.02,
        noise_std_max: float = 0.1,
        noise_prob: float = 0.5,
    ):
        self.jpeg_min = jpeg_quality_min
        self.jpeg_max = jpeg_quality_max
        self.blur_kernel = blur_kernel
        self.blur_sigma_min = blur_sigma_min
        self.blur_sigma_max = blur_sigma_max
        self.blur_prob = blur_prob
        self.color_prob = color_prob
        self.resize_scale_min = resize_scale_min
        self.resize_scale_max = resize_scale_max
        self.resize_prob = resize_prob
        self.noise_std_min = noise_std_min
        self.noise_std_max = noise_std_max
        self.noise_prob = noise_prob
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

    def _apply_resize(self, img: Image.Image) -> Image.Image:
        """Thu nhỏ rồi phóng to lại về đúng kích thước gốc — mô phỏng ảnh bị resize
        xuống độ phân giải thấp rồi phóng lại (thường gặp khi ảnh đi qua nhiều lần
        chia sẻ/re-upload). KHÔNG có trong pipeline GlobalForge gốc."""
        if random.random() < self.resize_prob:
            w, h = img.size
            scale = random.uniform(self.resize_scale_min, self.resize_scale_max)
            small = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.BICUBIC)
            return small.resize((w, h), Image.BICUBIC)
        return img

    def _apply_noise(self, img: Image.Image) -> Image.Image:
        """Gaussian noise cộng vào ảnh (chuẩn hoá [0,1] trước khi cộng nhiễu, clip lại
        sau đó). KHÔNG có trong pipeline GlobalForge gốc."""
        if random.random() < self.noise_prob:
            std = random.uniform(self.noise_std_min, self.noise_std_max)
            arr = np.asarray(img).astype(np.float32) / 255.0
            arr = arr + np.random.normal(0.0, std, arr.shape)
            arr = np.clip(arr, 0.0, 1.0)
            return Image.fromarray((arr * 255).astype(np.uint8))
        return img

    def _apply_color(self, img: Image.Image) -> Image.Image:
        if random.random() < self.color_prob:
            return self.color_jitter(img)
        return img

    def __call__(self, img: Image.Image) -> Image.Image:
        # Thứ tự: JPEG -> Blur -> Resize -> Noise -> Color.
        # Giữ nguyên JPEG/Blur ở đầu và Color ở cuối như pipeline gốc; Resize/Noise (mới
        # thêm) chèn xen giữa Blur và Color vì đây cũng là các phép biến dạng không gian/
        # tần số như Blur, hợp lý đứng cùng nhóm trước bước chỉnh màu toàn cục.
        img = self._apply_jpeg(img)
        img = self._apply_blur(img)
        img = self._apply_resize(img)
        img = self._apply_noise(img)
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
            resize_scale_min=deg_cfg.get("resize_scale_min", 0.2),
            resize_scale_max=deg_cfg.get("resize_scale_max", 0.9),
            resize_prob=deg_cfg.get("resize_prob", 0.5),
            noise_std_min=deg_cfg.get("noise_std_min", 0.02),
            noise_std_max=deg_cfg.get("noise_std_max", 0.1),
            noise_prob=deg_cfg.get("noise_prob", 0.5),
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