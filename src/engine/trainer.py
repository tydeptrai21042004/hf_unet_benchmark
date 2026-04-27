from __future__ import annotations

from typing import Callable, Dict, Optional

import torch
from torch.utils.data import DataLoader

from ..engine.evaluator import Evaluator
from ..engine.output_utils import compute_supervised_loss
from ..utils.logger import AverageMeter


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        *,
        aux_loss_fn: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        aux_weights: float | list[float] | tuple[float, ...] | None = None,
        boundary_loss_fn: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        boundary_weight: float = 0.0,
        device: str | torch.device = "cuda",
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        mixed_precision: bool = True,
        grad_clip: Optional[float] = None,
        aux_loss_weight: float = 1.0,
        aux_warmup_epochs: int = 0,
        aux_ramp_epochs: int = 0,
        log_interval: int = 20,
        threshold: float = 0.5,
        logger=None,
    ) -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.aux_loss_fn = aux_loss_fn
        self.aux_weights = aux_weights
        self.boundary_loss_fn = boundary_loss_fn
        self.boundary_weight = float(boundary_weight)
        self.scheduler = scheduler
        self.grad_clip = grad_clip
        self.aux_loss_weight = float(aux_loss_weight)
        self.aux_warmup_epochs = int(aux_warmup_epochs)
        self.aux_ramp_epochs = int(aux_ramp_epochs)
        self.log_interval = log_interval
        self.threshold = threshold
        self.logger = logger

        use_amp = mixed_precision and self.device.type == "cuda"
        self.mixed_precision = use_amp
        self.scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
        self.evaluator = Evaluator(
            device=self.device,
            threshold=threshold,
            logger=logger,
            loss_fn=self.loss_fn,
            aux_loss_fn=self.aux_loss_fn,
            aux_weights=self.aux_weights,
            boundary_loss_fn=self.boundary_loss_fn,
            boundary_weight=self.boundary_weight,
        )

    def _log(self, message: str) -> None:
        if self.logger is not None:
            self.logger.info(message)

    def _current_aux_weight(self, epoch: int) -> float:
        if self.aux_loss_weight <= 0.0:
            return 0.0
        if epoch <= self.aux_warmup_epochs:
            return 0.0
        if self.aux_ramp_epochs <= 0:
            return self.aux_loss_weight
        progress = min(max(epoch - self.aux_warmup_epochs, 0) / max(self.aux_ramp_epochs, 1), 1.0)
        return self.aux_loss_weight * progress

    def _compute_total_loss(self, model_output, masks: torch.Tensor, *, epoch: int) -> tuple[torch.Tensor, Dict[str, float]]:
        total_loss, log_items, _ = compute_supervised_loss(
            model_output,
            masks,
            main_loss_fn=self.loss_fn,
            aux_loss_fn=self.aux_loss_fn,
            aux_weights=self.aux_weights,
            boundary_loss_fn=self.boundary_loss_fn,
            boundary_weight=self.boundary_weight,
        )

        if hasattr(self.model, "auxiliary_regularization"):
            aux_reg = self.model.auxiliary_regularization()
            if not torch.is_tensor(aux_reg):
                aux_reg = torch.tensor(float(aux_reg), device=self.device)
            aux_weight = self._current_aux_weight(epoch)
            total_loss = total_loss + aux_weight * aux_reg
            log_items["model_aux_loss"] = float(aux_reg.detach().item())
            log_items["model_aux_weight"] = float(aux_weight)
            log_items["loss"] = float(total_loss.detach().item())

        return total_loss, log_items

    def train_one_epoch(self, dataloader: DataLoader, epoch: int) -> Dict[str, float]:
        self.model.train()
        if hasattr(self.model, "set_epoch"):
            self.model.set_epoch(epoch)
        loss_meter = AverageMeter()
        seg_meter = AverageMeter()
        aux_meter = AverageMeter()
        aux_w_meter = AverageMeter()
        boundary_meter = AverageMeter()
        boundary_w_meter = AverageMeter()
        model_aux_meter = AverageMeter()
        model_aux_w_meter = AverageMeter()

        for step, batch in enumerate(dataloader, start=1):
            images = batch["image"].to(self.device, non_blocking=True)
            masks = batch["mask"].to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=self.device.type, enabled=self.mixed_precision):
                model_output = self.model(images)
                total_loss, log_items = self._compute_total_loss(model_output, masks, epoch=epoch)

            self.scaler.scale(total_loss).backward()
            self.scaler.unscale_(self.optimizer)
            if self.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            loss_meter.update(log_items["loss"], n=images.size(0))
            seg_meter.update(log_items["seg_loss"], n=images.size(0))
            aux_meter.update(log_items.get("aux_loss", 0.0), n=images.size(0))
            aux_w_meter.update(log_items.get("aux_weight", 0.0), n=images.size(0))
            boundary_meter.update(log_items.get("boundary_loss", 0.0), n=images.size(0))
            boundary_w_meter.update(log_items.get("boundary_weight", 0.0), n=images.size(0))
            model_aux_meter.update(log_items.get("model_aux_loss", 0.0), n=images.size(0))
            model_aux_w_meter.update(log_items.get("model_aux_weight", 0.0), n=images.size(0))

            if step % self.log_interval == 0:
                lr = self.optimizer.param_groups[0]["lr"]
                self._log(
                    f"epoch={epoch} step={step}/{len(dataloader)} "
                    f"loss={loss_meter.avg:.4f} seg={seg_meter.avg:.4f} aux={aux_meter.avg:.4f} "
                    f"aux_w={aux_w_meter.avg:.4f} boundary={boundary_meter.avg:.4f} "
                    f"boundary_w={boundary_w_meter.avg:.4f} reg={model_aux_meter.avg:.4f} "
                    f"reg_w={model_aux_w_meter.avg:.4f} lr={lr:.2e}"
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
            "aux_weight": aux_w_meter.avg,
            "boundary_loss": boundary_meter.avg,
            "boundary_weight": boundary_w_meter.avg,
            "model_aux_loss": model_aux_meter.avg,
            "model_aux_weight": model_aux_w_meter.avg,
            "lr": float(self.optimizer.param_groups[0]["lr"]),
        }

    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        return self.evaluator.evaluate(self.model, dataloader)

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
