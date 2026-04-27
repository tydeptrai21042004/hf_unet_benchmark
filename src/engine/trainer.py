from __future__ import annotations

from copy import deepcopy
from typing import Callable, Dict, Optional

import torch
from torch.utils.data import DataLoader

from ..engine.evaluator import Evaluator
from ..engine.output_utils import compute_supervised_loss, parse_model_output
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
        debug_logits: bool = False,
        debug_logits_interval: int = 1,
        include_aux_loss_in_eval: bool = False,
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
        self.debug_logits = bool(debug_logits)
        self.debug_logits_interval = max(int(debug_logits_interval), 1)

        # Best-checkpoint state.  These are updated inside fit() as soon as a
        # monitored validation metric improves.  This prevents the common bug of
        # reporting best_epoch while saving the final epoch weights as best.pt.
        self.best_state_dict: dict[str, torch.Tensor] | None = None
        self.best_optimizer_state: dict | None = None
        self.best_record: Dict[str, float] | None = None
        self.best_epoch: int | None = None
        self.best_score: float | None = None

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
            include_aux_loss=include_aux_loss_in_eval,
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

    @staticmethod
    def _state_dict_to_cpu(module: torch.nn.Module) -> dict[str, torch.Tensor]:
        return {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}

    def _capture_best(self, *, epoch: int, record: Dict[str, float], score: float) -> None:
        self.best_score = float(score)
        self.best_epoch = int(epoch)
        self.best_record = dict(record)
        self.best_state_dict = self._state_dict_to_cpu(self.model)
        # deepcopy is important because optimizer.state_dict() contains mutable
        # tensors/lists that will keep changing after this epoch.
        self.best_optimizer_state = deepcopy(self.optimizer.state_dict())

    def _is_better(self, score: float, mode: str) -> bool:
        if self.best_score is None:
            return True
        if mode == "min":
            return score < self.best_score
        return score > self.best_score

    def _maybe_log_tensor_sanity(self, model_output, masks: torch.Tensor, *, epoch: int, step: int) -> None:
        if not self.debug_logits:
            return
        if step != 1 and step % self.debug_logits_interval != 0:
            return
        try:
            parsed = parse_model_output(model_output)
            logits = parsed.main.detach()
            message = (
                f"tensor_sanity epoch={epoch} step={step} "
                f"logits[min={logits.min().item():.4f}, max={logits.max().item():.4f}, "
                f"mean={logits.mean().item():.4f}, std={logits.std().item():.4f}] "
                f"mask[min={masks.min().item():.4f}, max={masks.max().item():.4f}, "
                f"mean={masks.mean().item():.4f}]"
            )
            if parsed.aux:
                aux_ranges = []
                for idx, aux in enumerate(parsed.aux):
                    aux_detached = aux.detach()
                    aux_ranges.append(
                        f"aux{idx + 1}[min={aux_detached.min().item():.4f}, "
                        f"max={aux_detached.max().item():.4f}]"
                    )
                message += " " + " ".join(aux_ranges)
            self._log(message)
        except Exception as exc:  # pragma: no cover - debug aid only
            self._log(f"tensor_sanity logging failed: {exc}")

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
                self._maybe_log_tensor_sanity(model_output, masks, epoch=epoch, step=step)
                total_loss, log_items = self._compute_total_loss(model_output, masks, epoch=epoch)

            if not torch.isfinite(total_loss):
                self._maybe_log_tensor_sanity(model_output, masks, epoch=epoch, step=step)
                raise FloatingPointError(
                    f"Non-finite training loss at epoch={epoch}, step={step}: {float(total_loss.detach().cpu())}"
                )

            self.scaler.scale(total_loss).backward()
            self.scaler.unscale_(self.optimizer)
            if self.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(self.grad_clip))
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
        monitor: str = "dice",
        monitor_mode: str = "max",
    ) -> list[Dict[str, float]]:
        monitor = str(monitor).strip()
        monitor_mode = str(monitor_mode).lower().strip()
        if monitor_mode not in {"min", "max"}:
            raise ValueError("monitor_mode must be either 'min' or 'max'.")

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
                monitor_value = val_metrics.get(monitor)
                if monitor_value is None:
                    # Allow monitor='val/dice' or fallback to loss if a custom key
                    # is unavailable.
                    monitor_value = record.get(monitor, val_metrics.get("loss", train_metrics["loss"]))
                score = float(monitor_value)
                if torch.isfinite(torch.tensor(score)) and self._is_better(score, monitor_mode):
                    self._capture_best(epoch=epoch, record=record, score=score)
                    self._log(f"new_best epoch={epoch} {monitor}={score:.6f}")

                if self.scheduler is not None and isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics.get(metric_for_plateau, val_metrics.get("loss", 0.0)))
            else:
                self._log(f"epoch={epoch} train_loss={train_metrics['loss']:.4f}")
                monitor_value = record.get(f"train/{monitor}", record.get(monitor, train_metrics["loss"]))
                score = float(monitor_value)
                if torch.isfinite(torch.tensor(score)) and self._is_better(score, monitor_mode):
                    self._capture_best(epoch=epoch, record=record, score=score)

            history.append(record)

        # Safety fallback: even if the monitored metric was missing or NaN for all
        # epochs, still save a valid checkpoint state rather than the caller having
        # to guess what happened.
        if self.best_state_dict is None and history:
            self._capture_best(epoch=len(history), record=history[-1], score=float("nan"))
        return history


__all__ = ["Trainer"]
