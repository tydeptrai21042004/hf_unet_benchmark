from __future__ import annotations

from typing import Dict

import torch

from .dice import DiceMeter, compute_dice
from .iou import IoUMeter, compute_iou
from .mae import MAEMeter, compute_mae
from .precision_recall import PrecisionRecallMeter, compute_precision_recall


def compute_segmentation_metrics(
    preds: torch.Tensor,
    targets: torch.Tensor,
    *,
    from_logits: bool = True,
    threshold: float = 0.5,
) -> Dict[str, float]:
    dice = compute_dice(preds, targets, from_logits=from_logits, threshold=threshold).item()
    iou = compute_iou(preds, targets, from_logits=from_logits, threshold=threshold).item()
    precision, recall = compute_precision_recall(preds, targets, from_logits=from_logits, threshold=threshold)
    mae = compute_mae(preds, targets, from_logits=from_logits).item()
    return {
        "dice": dice,
        "iou": iou,
        "precision": precision.item(),
        "recall": recall.item(),
        "mae": mae,
    }


__all__ = [
    "DiceMeter",
    "IoUMeter",
    "PrecisionRecallMeter",
    "MAEMeter",
    "compute_dice",
    "compute_iou",
    "compute_precision_recall",
    "compute_mae",
    "compute_segmentation_metrics",
]
