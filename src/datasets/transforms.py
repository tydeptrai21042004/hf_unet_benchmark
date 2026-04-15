"""Segmentation transforms without external augmentation dependencies.

These transforms operate on a dictionary with at least:
    {"image": PIL.Image, "mask": PIL.Image}

The final transform should convert them into tensors.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, List, MutableMapping, Sequence

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter

Sample = MutableMapping[str, object]


class SegCompose:
    """Compose segmentation transforms that jointly modify image and mask."""

    def __init__(self, transforms: Sequence[Callable[[Sample], Sample]]) -> None:
        self.transforms = list(transforms)

    def __call__(self, sample: Sample) -> Sample:
        for transform in self.transforms:
            sample = transform(sample)
        return sample

    def __repr__(self) -> str:
        inner = ",\n  ".join(repr(t) for t in self.transforms)
        return f"{self.__class__.__name__}([\n  {inner}\n])"


@dataclass
class SegResize:
    size: int | Sequence[int]
    image_resample: int = Image.BILINEAR
    mask_resample: int = Image.NEAREST

    def __call__(self, sample: Sample) -> Sample:
        image = sample["image"]
        mask = sample["mask"]
        target_size = _normalize_size(self.size)
        sample["image"] = image.resize(target_size, resample=self.image_resample)
        sample["mask"] = mask.resize(target_size, resample=self.mask_resample)
        return sample


@dataclass
class SegRandomHorizontalFlip:
    p: float = 0.5

    def __call__(self, sample: Sample) -> Sample:
        if random.random() < self.p:
            sample["image"] = sample["image"].transpose(Image.FLIP_LEFT_RIGHT)
            sample["mask"] = sample["mask"].transpose(Image.FLIP_LEFT_RIGHT)
        return sample


@dataclass
class SegRandomVerticalFlip:
    p: float = 0.5

    def __call__(self, sample: Sample) -> Sample:
        if random.random() < self.p:
            sample["image"] = sample["image"].transpose(Image.FLIP_TOP_BOTTOM)
            sample["mask"] = sample["mask"].transpose(Image.FLIP_TOP_BOTTOM)
        return sample


@dataclass
class SegRandomRotate90:
    p: float = 0.5

    def __call__(self, sample: Sample) -> Sample:
        if random.random() < self.p:
            k = random.randint(1, 3)
            angle = 90 * k
            sample["image"] = sample["image"].rotate(angle, resample=Image.BILINEAR)
            sample["mask"] = sample["mask"].rotate(angle, resample=Image.NEAREST)
        return sample


@dataclass
class SegRandomRotate:
    degrees: float = 15.0
    p: float = 0.5

    def __call__(self, sample: Sample) -> Sample:
        if random.random() >= self.p:
            return sample
        angle = random.uniform(-self.degrees, self.degrees)
        sample["image"] = sample["image"].rotate(angle, resample=Image.BILINEAR)
        sample["mask"] = sample["mask"].rotate(angle, resample=Image.NEAREST)
        return sample


@dataclass
class SegRandomBrightnessContrast:
    brightness: float = 0.15
    contrast: float = 0.15
    p: float = 0.5

    def __call__(self, sample: Sample) -> Sample:
        if random.random() >= self.p:
            return sample

        image = sample["image"]
        if self.brightness > 0:
            factor = 1.0 + random.uniform(-self.brightness, self.brightness)
            image = ImageEnhance.Brightness(image).enhance(factor)
        if self.contrast > 0:
            factor = 1.0 + random.uniform(-self.contrast, self.contrast)
            image = ImageEnhance.Contrast(image).enhance(factor)

        sample["image"] = image
        return sample


@dataclass
class SegRandomGaussianBlur:
    radius: float = 1.5
    p: float = 0.2

    def __call__(self, sample: Sample) -> Sample:
        if random.random() < self.p:
            radius = random.uniform(0.0, self.radius)
            sample["image"] = sample["image"].filter(ImageFilter.GaussianBlur(radius=radius))
        return sample


@dataclass
class SegNormalize:
    mean: Sequence[float] = (0.485, 0.456, 0.406)
    std: Sequence[float] = (0.229, 0.224, 0.225)

    def __call__(self, sample: Sample) -> Sample:
        image = sample["image"]
        if not isinstance(image, torch.Tensor):
            raise TypeError("SegNormalize expects image to be a torch.Tensor. Put SegToTensor before it.")

        mean = torch.tensor(self.mean, dtype=image.dtype, device=image.device).view(-1, 1, 1)
        std = torch.tensor(self.std, dtype=image.dtype, device=image.device).view(-1, 1, 1)
        sample["image"] = (image - mean) / std
        return sample


class SegToTensor:
    """Convert image and mask to tensors.

    Output:
        image: float tensor in [0, 1], shape [3, H, W]
        mask: float tensor in {0, 1}, shape [1, H, W]
    """

    def __call__(self, sample: Sample) -> Sample:
        image = sample["image"]
        mask = sample["mask"]

        if not isinstance(image, Image.Image) or not isinstance(mask, Image.Image):
            raise TypeError("SegToTensor expects PIL.Image inputs for both image and mask.")

        image_np = np.asarray(image, dtype=np.float32)
        if image_np.ndim != 3 or image_np.shape[2] != 3:
            raise ValueError(f"Expected RGB image with shape [H, W, 3], got {image_np.shape}")
        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).contiguous() / 255.0

        mask_np = np.asarray(mask, dtype=np.float32)
        if mask_np.ndim == 3:
            mask_np = mask_np[..., 0]
        mask_tensor = torch.from_numpy(mask_np).unsqueeze(0).contiguous() / 255.0
        mask_tensor = (mask_tensor > 0.5).float()

        sample["image"] = image_tensor
        sample["mask"] = mask_tensor
        return sample


def _normalize_size(size: int | Sequence[int]) -> tuple[int, int]:
    if isinstance(size, int):
        return (size, size)
    if len(size) != 2:
        raise ValueError(f"size must be int or sequence of length 2, got {size}")
    height, width = int(size[0]), int(size[1])
    return (width, height)


def build_train_transforms(
    image_size: int | Sequence[int] = 352,
    normalize: bool = True,
    mean: Sequence[float] = (0.485, 0.456, 0.406),
    std: Sequence[float] = (0.229, 0.224, 0.225),
    preset: str = "baseline",
) -> SegCompose:
    preset_key = preset.lower()
    if preset_key == "strong":
        transforms: List[Callable[[Sample], Sample]] = [
            SegResize(image_size),
            SegRandomHorizontalFlip(0.5),
            SegRandomVerticalFlip(0.2),
            SegRandomRotate90(0.4),
            SegRandomRotate(degrees=15.0, p=0.35),
            SegRandomBrightnessContrast(brightness=0.2, contrast=0.2, p=0.6),
            SegRandomGaussianBlur(radius=1.5, p=0.2),
            SegToTensor(),
        ]
    else:
        transforms = [
            SegResize(image_size),
            SegRandomHorizontalFlip(0.5),
            SegRandomVerticalFlip(0.2),
            SegRandomRotate90(0.3),
            SegRandomBrightnessContrast(brightness=0.15, contrast=0.15, p=0.5),
            SegToTensor(),
        ]
    if normalize:
        transforms.append(SegNormalize(mean=mean, std=std))
    return SegCompose(transforms)


def build_eval_transforms(
    image_size: int | Sequence[int] = 352,
    normalize: bool = True,
    mean: Sequence[float] = (0.485, 0.456, 0.406),
    std: Sequence[float] = (0.229, 0.224, 0.225),
) -> SegCompose:
    transforms: List[Callable[[Sample], Sample]] = [
        SegResize(image_size),
        SegToTensor(),
    ]
    if normalize:
        transforms.append(SegNormalize(mean=mean, std=std))
    return SegCompose(transforms)
