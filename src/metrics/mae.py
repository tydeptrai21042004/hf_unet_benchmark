from __future__ import annotations

from dataclasses import dataclass

import torch


def compute_mae(
    preds: torch.Tensor,
    targets: torch.Tensor,
    *,
    from_logits: bool = True,
    reduction: str = "mean",
) -> torch.Tensor:
    if preds.ndim == 3:
        preds = preds.unsqueeze(1)
    if targets.ndim == 3:
        targets = targets.unsqueeze(1)
    preds = preds.float()
    targets = targets.float()
    if from_logits:
        preds = torch.sigmoid(preds)
    mae = torch.abs(preds - targets)
    dims = tuple(range(1, mae.ndim))
    mae = mae.mean(dim=dims)
    if reduction == "none":
        return mae
    if reduction == "sum":
        return mae.sum()
    return mae.mean()


@dataclass
class MAEMeter:
    total: float = 0.0
    count: int = 0

    def update(self, preds: torch.Tensor, targets: torch.Tensor, *, from_logits: bool = True) -> float:
        value = compute_mae(preds, targets, from_logits=from_logits, reduction="mean").item()
        self.total += value
        self.count += 1
        return value

    @property
    def avg(self) -> float:
        return self.total / max(self.count, 1)

    def reset(self) -> None:
        self.total = 0.0
        self.count = 0


__all__ = ["compute_mae", "MAEMeter"]
