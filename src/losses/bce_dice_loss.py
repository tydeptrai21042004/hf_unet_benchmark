from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .dice_loss import DiceLoss


class BCEDiceLoss(nn.Module):
    """Weighted sum of BCEWithLogits and Dice loss for binary segmentation."""

    def __init__(
        self,
        bce_weight: float = 0.5,
        dice_weight: float = 0.5,
        pos_weight: Optional[float] = None,
        smooth: float = 1.0,
        eps: float = 1e-7,
    ) -> None:
        super().__init__()
        if bce_weight < 0 or dice_weight < 0:
            raise ValueError("bce_weight and dice_weight must be non-negative")
        if (bce_weight + dice_weight) == 0:
            raise ValueError("At least one of bce_weight or dice_weight must be positive")

        if pos_weight is None:
            self.bce = nn.BCEWithLogitsLoss()
        else:
            self.register_buffer("_pos_weight", torch.tensor([float(pos_weight)], dtype=torch.float32))
            self.bce = nn.BCEWithLogitsLoss(pos_weight=self._pos_weight)
        self.dice = DiceLoss(from_logits=True, smooth=smooth, eps=eps)
        self.bce_weight = float(bce_weight)
        self.dice_weight = float(dice_weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float()
        if targets.ndim == 3:
            targets = targets.unsqueeze(1)
        bce_term = self.bce(logits, targets)
        dice_term = self.dice(logits, targets)
        return self.bce_weight * bce_term + self.dice_weight * dice_term


__all__ = ["BCEDiceLoss"]
