from __future__ import annotations

from dataclasses import dataclass

import torch


def _prepare_predictions_and_targets(
    preds: torch.Tensor,
    targets: torch.Tensor,
    *,
    from_logits: bool = True,
    threshold: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    if preds.ndim == 3:
        preds = preds.unsqueeze(1)
    if targets.ndim == 3:
        targets = targets.unsqueeze(1)
    preds = preds.float()
    targets = targets.float()
    if from_logits:
        preds = torch.sigmoid(preds)
    preds = (preds >= threshold).float()
    targets = (targets >= 0.5).float()
    return preds, targets


def compute_dice(
    preds: torch.Tensor,
    targets: torch.Tensor,
    *,
    from_logits: bool = True,
    threshold: float = 0.5,
    smooth: float = 1.0,
    eps: float = 1e-7,
    reduction: str = "mean",
) -> torch.Tensor:
    preds, targets = _prepare_predictions_and_targets(preds, targets, from_logits=from_logits, threshold=threshold)
    preds = preds.reshape(preds.shape[0], -1)
    targets = targets.reshape(targets.shape[0], -1)
    intersection = (preds * targets).sum(dim=1)
    denominator = preds.sum(dim=1) + targets.sum(dim=1)
    dice = (2.0 * intersection + smooth) / (denominator + smooth + eps)
    if reduction == "none":
        return dice
    if reduction == "sum":
        return dice.sum()
    return dice.mean()


@dataclass
class DiceMeter:
    threshold: float = 0.5
    smooth: float = 1.0
    total: float = 0.0
    count: int = 0

    def update(self, preds: torch.Tensor, targets: torch.Tensor, *, from_logits: bool = True) -> float:
        value = compute_dice(
            preds,
            targets,
            from_logits=from_logits,
            threshold=self.threshold,
            smooth=self.smooth,
            reduction="mean",
        ).item()
        self.total += value
        self.count += 1
        return value

    @property
    def avg(self) -> float:
        return self.total / max(self.count, 1)

    def reset(self) -> None:
        self.total = 0.0
        self.count = 0


__all__ = ["compute_dice", "DiceMeter"]
