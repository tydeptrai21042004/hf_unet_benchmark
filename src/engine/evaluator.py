from __future__ import annotations

from collections import defaultdict
from typing import Callable, Dict, Optional

import torch
from torch.utils.data import DataLoader

from ..metrics import compute_segmentation_metrics


class Evaluator:
    def __init__(
        self,
        device: str | torch.device = "cuda",
        *,
        threshold: float = 0.5,
        logger=None,
    ) -> None:
        self.device = torch.device(device)
        self.threshold = threshold
        self.logger = logger

    @torch.no_grad()
    def evaluate(
        self,
        model: torch.nn.Module,
        dataloader: DataLoader,
        *,
        loss_fn: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
    ) -> Dict[str, float]:
        model.eval()
        totals = defaultdict(float)
        steps = 0
        pixels = 0

        for batch in dataloader:
            images = batch["image"].to(self.device, non_blocking=True)
            masks = batch["mask"].to(self.device, non_blocking=True)
            logits = model(images)

            if loss_fn is not None:
                loss = loss_fn(logits, masks)
                totals["loss"] += float(loss.item())

            batch_metrics = compute_segmentation_metrics(
                logits,
                masks,
                from_logits=True,
                threshold=self.threshold,
            )
            batch_size = images.shape[0]
            pixels += batch_size
            for key, value in batch_metrics.items():
                totals[key] += float(value) * batch_size
            steps += 1

        if steps == 0:
            return {"loss": 0.0, "dice": 0.0, "iou": 0.0, "precision": 0.0, "recall": 0.0, "mae": 0.0}

        metrics = {key: value / max(pixels, 1) for key, value in totals.items() if key != "loss"}
        if "loss" in totals:
            metrics["loss"] = totals["loss"] / steps
        return metrics


__all__ = ["Evaluator"]
