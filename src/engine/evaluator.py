from __future__ import annotations

from collections import defaultdict
from typing import Callable, Dict, Optional, Sequence

import torch
from torch.utils.data import DataLoader

from ..engine.output_utils import compute_supervised_loss, parse_model_output
from ..metrics import compute_segmentation_metrics


class Evaluator:
    def __init__(
        self,
        device: str | torch.device = "cuda",
        *,
        threshold: float = 0.5,
        logger=None,
        loss_fn: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        aux_loss_fn: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        aux_weights: float | Sequence[float] | None = None,
        boundary_loss_fn: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        boundary_weight: float = 0.0,
    ) -> None:
        self.device = torch.device(device)
        self.threshold = threshold
        self.logger = logger
        self.loss_fn = loss_fn
        self.aux_loss_fn = aux_loss_fn
        self.aux_weights = aux_weights
        self.boundary_loss_fn = boundary_loss_fn
        self.boundary_weight = float(boundary_weight)

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
        effective_loss_fn = loss_fn or self.loss_fn

        for batch in dataloader:
            images = batch["image"].to(self.device, non_blocking=True)
            masks = batch["mask"].to(self.device, non_blocking=True)
            model_output = model(images)
            parsed = parse_model_output(model_output)

            if effective_loss_fn is not None:
                total_loss, _, _ = compute_supervised_loss(
                    model_output,
                    masks,
                    main_loss_fn=effective_loss_fn,
                    aux_loss_fn=self.aux_loss_fn,
                    aux_weights=self.aux_weights,
                    boundary_loss_fn=self.boundary_loss_fn,
                    boundary_weight=self.boundary_weight,
                )
                totals["loss"] += float(total_loss.item())

            batch_metrics = compute_segmentation_metrics(
                parsed.main,
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
