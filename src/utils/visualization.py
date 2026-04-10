from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
from PIL import Image
import torch


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def tensor_image_to_numpy(image: torch.Tensor, *, denormalize: bool = True) -> np.ndarray:
    if image.ndim != 3:
        raise ValueError(f"Expected image tensor [C, H, W], got {tuple(image.shape)}")
    array = image.detach().cpu().float().permute(1, 2, 0).numpy()
    if denormalize and array.shape[2] == 3:
        array = array * IMAGENET_STD + IMAGENET_MEAN
    array = np.clip(array, 0.0, 1.0)
    return (array * 255.0).astype(np.uint8)


def tensor_mask_to_numpy(mask: torch.Tensor, *, threshold: float = 0.5, from_logits: bool = True) -> np.ndarray:
    if mask.ndim == 3 and mask.shape[0] == 1:
        mask = mask[0]
    elif mask.ndim != 2:
        raise ValueError(f"Expected mask tensor [1, H, W] or [H, W], got {tuple(mask.shape)}")
    array = mask.detach().cpu().float()
    if from_logits:
        array = torch.sigmoid(array)
    array = (array >= threshold).to(torch.uint8).numpy() * 255
    return array


def overlay_mask(image: np.ndarray, mask: np.ndarray, color: Sequence[int] = (255, 0, 0), alpha: float = 0.4) -> np.ndarray:
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("overlay_mask expects an RGB image array")
    if mask.ndim != 2:
        raise ValueError("overlay_mask expects a single-channel mask array")
    overlay = image.copy().astype(np.float32)
    color_array = np.array(color, dtype=np.float32).reshape(1, 1, 3)
    mask_bool = mask > 0
    overlay[mask_bool] = (1.0 - alpha) * overlay[mask_bool] + alpha * color_array
    return np.clip(overlay, 0, 255).astype(np.uint8)


def save_prediction_triplet(
    image: torch.Tensor,
    pred_mask: torch.Tensor,
    gt_mask: Optional[torch.Tensor],
    save_path: str | Path,
    *,
    threshold: float = 0.5,
    from_logits: bool = True,
) -> Path:
    image_np = tensor_image_to_numpy(image)
    pred_np = tensor_mask_to_numpy(pred_mask, threshold=threshold, from_logits=from_logits)
    pred_overlay = overlay_mask(image_np, pred_np, color=(255, 0, 0), alpha=0.45)

    tiles = [image_np, pred_overlay]
    if gt_mask is not None:
        gt_np = tensor_mask_to_numpy(gt_mask, threshold=0.5, from_logits=False)
        gt_overlay = overlay_mask(image_np, gt_np, color=(0, 255, 0), alpha=0.45)
        tiles.append(gt_overlay)

    canvas = np.concatenate(tiles, axis=1)
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(canvas).save(save_path)
    return save_path


__all__ = [
    "tensor_image_to_numpy",
    "tensor_mask_to_numpy",
    "overlay_mask",
    "save_prediction_triplet",
]
