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


def compute_precision_recall(
    preds: torch.Tensor,
    targets: torch.Tensor,
    *,
    from_logits: bool = True,
    threshold: float = 0.5,
    eps: float = 1e-7,
    reduction: str = "mean",
) -> tuple[torch.Tensor, torch.Tensor]:
    preds, targets = _prepare_predictions_and_targets(preds, targets, from_logits=from_logits, threshold=threshold)
    preds = preds.reshape(preds.shape[0], -1)
    targets = targets.reshape(targets.shape[0], -1)

    tp = (preds * targets).sum(dim=1)
    fp = (preds * (1.0 - targets)).sum(dim=1)
    fn = ((1.0 - preds) * targets).sum(dim=1)

    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)

    if reduction == "none":
        return precision, recall
    if reduction == "sum":
        return precision.sum(), recall.sum()
    return precision.mean(), recall.mean()


@dataclass
class PrecisionRecallMeter:
    threshold: float = 0.5
    precision_total: float = 0.0
    recall_total: float = 0.0
    count: int = 0

    def update(self, preds: torch.Tensor, targets: torch.Tensor, *, from_logits: bool = True) -> tuple[float, float]:
        precision, recall = compute_precision_recall(
            preds,
            targets,
            from_logits=from_logits,
            threshold=self.threshold,
            reduction="mean",
        )
        p = precision.item()
        r = recall.item()
        self.precision_total += p
        self.recall_total += r
        self.count += 1
        return p, r

    @property
    def avg(self) -> tuple[float, float]:
        denom = max(self.count, 1)
        return self.precision_total / denom, self.recall_total / denom

    def reset(self) -> None:
        self.precision_total = 0.0
        self.recall_total = 0.0
        self.count = 0


__all__ = ["compute_precision_recall", "PrecisionRecallMeter"]
