from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


def _validate_shapes(logits: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if logits.ndim != 4:
        raise ValueError(f"Expected logits with shape [B, C, H, W], got {tuple(logits.shape)}")
    if targets.ndim == 3:
        targets = targets.unsqueeze(1)
    if targets.ndim != 4:
        raise ValueError(f"Expected targets with shape [B, 1, H, W] or [B, H, W], got {tuple(targets.shape)}")
    if logits.shape[0] != targets.shape[0] or logits.shape[-2:] != targets.shape[-2:]:
        raise ValueError(
            f"Logits and targets batch/spatial dims must match, got {tuple(logits.shape)} vs {tuple(targets.shape)}"
        )
    return logits, targets.float()


def soft_dice_score(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    from_logits: bool = True,
    smooth: float = 1.0,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Compute mean soft Dice score for binary or multi-label segmentation.

    Parameters
    ----------
    logits:
        Tensor of shape [B, C, H, W].
    targets:
        Tensor of shape [B, 1, H, W] or [B, H, W] for binary segmentation.
        If C > 1, targets may also be [B, C, H, W].
    """
    logits, targets = _validate_shapes(logits, targets)

    if logits.shape[1] > 1 and targets.shape[1] == 1:
        raise ValueError(
            "Dice loss received multi-channel logits but single-channel targets. "
            "For multi-class use one-hot/multi-label targets or keep num_classes=1."
        )

    probs = torch.sigmoid(logits) if from_logits else logits
    if targets.shape[1] == 1 and probs.shape[1] > 1:
        targets = targets.expand(-1, probs.shape[1], -1, -1)

    probs = probs.reshape(probs.shape[0], probs.shape[1], -1)
    targets = targets.reshape(targets.shape[0], targets.shape[1], -1)

    intersection = (probs * targets).sum(dim=-1)
    denominator = probs.sum(dim=-1) + targets.sum(dim=-1)
    dice = (2.0 * intersection + smooth) / (denominator + smooth + eps)
    return dice.mean()


def soft_dice_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    from_logits: bool = True,
    smooth: float = 1.0,
    eps: float = 1e-7,
) -> torch.Tensor:
    return 1.0 - soft_dice_score(logits, targets, from_logits=from_logits, smooth=smooth, eps=eps)


@dataclass
class DiceLossConfig:
    from_logits: bool = True
    smooth: float = 1.0
    eps: float = 1e-7


class DiceLoss(nn.Module):
    def __init__(self, from_logits: bool = True, smooth: float = 1.0, eps: float = 1e-7) -> None:
        super().__init__()
        self.from_logits = from_logits
        self.smooth = smooth
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return soft_dice_loss(
            logits,
            targets,
            from_logits=self.from_logits,
            smooth=self.smooth,
            eps=self.eps,
        )


__all__ = [
    "DiceLoss",
    "DiceLossConfig",
    "soft_dice_score",
    "soft_dice_loss",
]
