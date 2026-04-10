from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class StructureLoss(nn.Module):
    """PraNet-style weighted BCE + weighted IoU loss.

    This loss emphasizes pixels around object boundaries by using a weight map
    derived from local mask contrast.
    """

    def __init__(self, kernel_size: int = 31, weight_factor: float = 5.0, eps: float = 1e-7) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd")
        self.kernel_size = kernel_size
        self.weight_factor = weight_factor
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if targets.ndim == 3:
            targets = targets.unsqueeze(1)
        targets = targets.float()

        pooled = F.avg_pool2d(targets, kernel_size=self.kernel_size, stride=1, padding=self.kernel_size // 2)
        weights = 1.0 + self.weight_factor * torch.abs(pooled - targets)

        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        wbce = (weights * bce).sum(dim=(2, 3)) / (weights.sum(dim=(2, 3)) + self.eps)

        probs = torch.sigmoid(logits)
        intersection = ((probs * targets) * weights).sum(dim=(2, 3))
        union = ((probs + targets) * weights).sum(dim=(2, 3))
        wiou = 1.0 - (intersection + 1.0) / (union - intersection + 1.0 + self.eps)

        return (wbce + wiou).mean()


__all__ = ["StructureLoss"]
