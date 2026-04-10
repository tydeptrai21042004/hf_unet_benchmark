from __future__ import annotations

from typing import Callable, Dict, Optional

import torch
from torch.utils.data import DataLoader

from ..engine.evaluator import Evaluator
from ..utils.logger import AverageMeter


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        *,
        device: str | torch.device = "cuda",
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        mixed_precision: bool = True,
        grad_clip: Optional[float] = None,
        aux_loss_weight: float = 1.0,
        log_interval: int = 20,
        threshold: float = 0.5,
        logger=None,
    ) -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.scheduler = scheduler
        self.grad_clip = grad_clip
        self.aux_loss_weight = aux_loss_weight
        self.log_interval = log_interval
        self.threshold = threshold
        self.logger = logger

        use_amp = mixed_precision and self.device.type == "cuda"
        self.mixed_precision = use_amp
        self.scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
        self.evaluator = Evaluator(device=self.device, threshold=threshold, logger=logger)

    def _log(self, message: str) -> None:
        if self.logger is not None:
            self.logger.info(message)

    def _compute_total_loss(self, logits: torch.Tensor, masks: torch.Tensor) -> tuple[torch.Tensor, Dict[str, float]]:
        seg_loss = self.loss_fn(logits, masks)
        total_loss = seg_loss
        log_items = {"seg_loss": float(seg_loss.detach().item())}

        if hasattr(self.model, "auxiliary_regularization"):
            aux_loss = self.model.auxiliary_regularization()
            if not torch.is_tensor(aux_loss):
                aux_loss = torch.tensor(float(aux_loss), device=self.device)
            total_loss = total_loss + self.aux_loss_weight * aux_loss
            log_items["aux_loss"] = float(aux_loss.detach().item())

        log_items["loss"] = float(total_loss.detach().item())
        return total_loss, log_items

    def train_one_epoch(self, dataloader: DataLoader, epoch: int) -> Dict[str, float]:
        self.model.train()
        loss_meter = AverageMeter()
        seg_meter = AverageMeter()
        aux_meter = AverageMeter()

        for step, batch in enumerate(dataloader, start=1):
            images = batch["image"].to(self.device, non_blocking=True)
            masks = batch["mask"].to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=self.device.type, enabled=self.mixed_precision):
                logits = self.model(images)
                total_loss, log_items = self._compute_total_loss(logits, masks)

            self.scaler.scale(total_loss).backward()
            self.scaler.unscale_(self.optimizer)
            if self.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            loss_meter.update(log_items["loss"], n=images.size(0))
            seg_meter.update(log_items["seg_loss"], n=images.size(0))
            aux_meter.update(log_items.get("aux_loss", 0.0), n=images.size(0))

            if step % self.log_interval == 0:
                lr = self.optimizer.param_groups[0]["lr"]
                self._log(
                    f"epoch={epoch} step={step}/{len(dataloader)} "
                    f"loss={loss_meter.avg:.4f} seg={seg_meter.avg:.4f} aux={aux_meter.avg:.4f} lr={lr:.2e}"
                )

        if self.scheduler is not None:
            try:
                self.scheduler.step()
            except TypeError:
                # ReduceLROnPlateau should be stepped externally after validation.
                pass

        return {
            "loss": loss_meter.avg,
            "seg_loss": seg_meter.avg,
            "aux_loss": aux_meter.avg,
            "lr": float(self.optimizer.param_groups[0]["lr"]),
        }

    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        return self.evaluator.evaluate(self.model, dataloader, loss_fn=self.loss_fn)

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        *,
        epochs: int,
        metric_for_plateau: str = "loss",
    ) -> list[Dict[str, float]]:
        history: list[Dict[str, float]] = []
        for epoch in range(1, epochs + 1):
            train_metrics = self.train_one_epoch(train_loader, epoch)
            record: Dict[str, float] = {f"train/{k}": v for k, v in train_metrics.items()}

            if val_loader is not None:
                val_metrics = self.validate(val_loader)
                record.update({f"val/{k}": v for k, v in val_metrics.items()})
                self._log(
                    f"epoch={epoch} train_loss={train_metrics['loss']:.4f} "
                    f"val_loss={val_metrics.get('loss', 0.0):.4f} val_dice={val_metrics.get('dice', 0.0):.4f}"
                )
                if self.scheduler is not None and isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics.get(metric_for_plateau, val_metrics.get("loss", 0.0)))
            else:
                self._log(f"epoch={epoch} train_loss={train_metrics['loss']:.4f}")

            history.append(record)
        return history


__all__ = ["Trainer"]
