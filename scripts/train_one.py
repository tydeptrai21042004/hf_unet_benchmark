#!/usr/bin/env python3
"""Train one segmentation model inside the unified benchmark framework."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets import build_eval_transforms, build_train_transforms, KvasirSegDataset
from src.engine import Trainer
from src.losses import BCEDiceLoss, DiceLoss, StructureLoss
from src.models import build_model
from src.utils import ExperimentPaths, get_logger, load_yaml, seed_everything, dump_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train one model.")
    parser.add_argument("--model", type=str, required=True, help="Registered model name, e.g. unet or proposal_hf_unet")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config")
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output-root", type=str, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--resume", type=str, default=None)
    return parser.parse_args()


def _deep_update(base: Dict[str, Any], updates: Mapping[str, Any]) -> Dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_config_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    cfg: Dict[str, Any] = {
        "experiment": {"output_root": str(PROJECT_ROOT), "name": None},
        "data": {"root": "data", "image_size": 352, "batch_size": 8, "num_workers": 4, "pin_memory": True},
        "train": {
            "epochs": 30,
            "lr": 1e-3,
            "weight_decay": 1e-4,
            "optimizer": "adamw",
            "scheduler": "cosine",
            "mixed_precision": True,
            "grad_clip": None,
            "seed": 42,
            "loss": "bce_dice",
            "threshold": 0.5,
            "aux_loss_weight": 1.0,
            "save_metric": "dice",
            "save_metric_mode": "max",
        },
        "model": {"name": args.model},
    }
    if args.config:
        _deep_update(cfg, load_yaml(args.config))
    overrides: Dict[str, Any] = {}
    if args.data_root is not None:
        overrides.setdefault("data", {})["root"] = args.data_root
    if args.image_size is not None:
        overrides.setdefault("data", {})["image_size"] = args.image_size
    if args.batch_size is not None:
        overrides.setdefault("data", {})["batch_size"] = args.batch_size
    if args.num_workers is not None:
        overrides.setdefault("data", {})["num_workers"] = args.num_workers
    if args.epochs is not None:
        overrides.setdefault("train", {})["epochs"] = args.epochs
    if args.lr is not None:
        overrides.setdefault("train", {})["lr"] = args.lr
    if args.device is not None:
        overrides.setdefault("train", {})["device"] = args.device
    if args.output_root is not None:
        overrides.setdefault("experiment", {})["output_root"] = args.output_root
    if args.seed is not None:
        overrides.setdefault("train", {})["seed"] = args.seed
    _deep_update(cfg, overrides)
    cfg["model"]["name"] = args.model
    return cfg


def build_dataloaders(cfg: Dict[str, Any]) -> tuple[DataLoader, DataLoader]:
    data_cfg = cfg["data"]
    image_size = int(data_cfg.get("image_size", 352))
    train_ds = KvasirSegDataset(
        root=data_cfg.get("root", "data"),
        split="train",
        transform=build_train_transforms(
            image_size=image_size,
            preset=str(data_cfg.get("augmentation", "baseline")),
        ),
    )
    val_ds = KvasirSegDataset(
        root=data_cfg.get("root", "data"),
        split="val",
        transform=build_eval_transforms(image_size=image_size),
    )
    batch_size = int(data_cfg.get("batch_size", 8))
    num_workers = int(data_cfg.get("num_workers", 4))
    pin_memory = bool(data_cfg.get("pin_memory", True))
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    return train_loader, val_loader


def build_loss(loss_name: str):
    name = loss_name.lower()
    if name == "dice":
        return DiceLoss(from_logits=True)
    if name in {"bce_dice", "bcedice"}:
        return BCEDiceLoss()
    if name == "structure":
        return StructureLoss()
    raise ValueError(f"Unsupported loss: {loss_name}")


def build_optimizer(model: torch.nn.Module, cfg: Dict[str, Any]) -> torch.optim.Optimizer:
    train_cfg = cfg["train"]
    lr = float(train_cfg.get("lr", 1e-3))
    weight_decay = float(train_cfg.get("weight_decay", 0.0))
    opt_name = str(train_cfg.get("optimizer", "adamw")).lower()
    if opt_name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if opt_name == "sgd":
        momentum = float(train_cfg.get("momentum", 0.9))
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=True)
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)


def build_scheduler(optimizer: torch.optim.Optimizer, cfg: Dict[str, Any]):
    train_cfg = cfg["train"]
    name = str(train_cfg.get("scheduler", "cosine")).lower()
    epochs = int(train_cfg.get("epochs", 30))
    if name == "none":
        return None
    if name == "step":
        step_size = int(train_cfg.get("step_size", max(epochs // 3, 1)))
        gamma = float(train_cfg.get("gamma", 0.1))
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    if name == "plateau":
        factor = float(train_cfg.get("gamma", 0.5))
        patience = int(train_cfg.get("plateau_patience", 3))
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=factor, patience=patience)
    t_max = int(train_cfg.get("t_max", epochs))
    eta_min = float(train_cfg.get("eta_min", 1e-6))
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(t_max, 1), eta_min=eta_min)


def unwrap_state_dict(checkpoint: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    if "state_dict" in checkpoint:
        return checkpoint["state_dict"]
    if "model" in checkpoint and isinstance(checkpoint["model"], dict):
        return checkpoint["model"]
    return checkpoint  # already a state_dict


def save_json(data: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def save_history_csv(history: list[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not history:
        return
    fieldnames = list(history[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)


def main() -> None:
    args = parse_args()
    cfg = load_config_from_args(args)
    seed_everything(int(cfg["train"].get("seed", 42)))

    model_name = cfg["model"].get("name", args.model)
    exp_name = cfg["experiment"].get("name") or model_name
    output_root = Path(cfg["experiment"].get("output_root", PROJECT_ROOT))
    exp_paths = ExperimentPaths.create(output_root, experiment_name=exp_name)
    logger = get_logger(name=f"train_{model_name}", log_file=exp_paths.logs / "train.log")
    logger.info(f"Training model: {model_name}")
    dump_yaml(cfg, exp_paths.root / "resolved_train_config.yaml")

    train_loader, val_loader = build_dataloaders(cfg)
    model_cfg = {k: v for k, v in cfg["model"].items() if k != "name"}
    model = build_model(model_name, config=model_cfg)
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)
    loss_fn = build_loss(str(cfg["train"].get("loss", "bce_dice")))
    device = cfg["train"].get("device") or ("cuda" if torch.cuda.is_available() else "cpu")

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
        scheduler=scheduler,
        mixed_precision=bool(cfg["train"].get("mixed_precision", True)),
        grad_clip=cfg["train"].get("grad_clip"),
        aux_loss_weight=float(cfg["train"].get("aux_loss_weight", 1.0)),
        aux_warmup_epochs=int(cfg["train"].get("aux_warmup_epochs", 0)),
        aux_ramp_epochs=int(cfg["train"].get("aux_ramp_epochs", 0)),
        threshold=float(cfg["train"].get("threshold", 0.5)),
        logger=logger,
    )

    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(unwrap_state_dict(ckpt), strict=True)
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        logger.info(f"Resumed from checkpoint: {args.resume}")

    epochs = int(cfg["train"].get("epochs", 30))
    save_metric = str(cfg["train"].get("save_metric", "dice"))
    save_mode = str(cfg["train"].get("save_metric_mode", "max")).lower()
    best_score = -math.inf if save_mode == "max" else math.inf
    history: list[Dict[str, Any]] = []

    for epoch in range(1, epochs + 1):
        train_metrics = trainer.train_one_epoch(train_loader, epoch)
        val_metrics = trainer.validate(val_loader)
        record = {"epoch": epoch}
        record.update({f"train/{k}": v for k, v in train_metrics.items()})
        record.update({f"val/{k}": v for k, v in val_metrics.items()})
        history.append(record)

        score = float(val_metrics.get(save_metric, val_metrics.get("loss", 0.0)))
        improved = score > best_score if save_mode == "max" else score < best_score
        if improved:
            best_score = score
            best_path = exp_paths.checkpoints / "best.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "state_dict": trainer.model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "config": cfg,
                    "best_metric_name": save_metric,
                    "best_metric_value": best_score,
                    "val_metrics": val_metrics,
                },
                best_path,
            )
            logger.info(f"Saved best checkpoint to {best_path} with {save_metric}={best_score:.6f}")

        last_path = exp_paths.checkpoints / "last.pt"
        torch.save(
            {
                "epoch": epoch,
                "state_dict": trainer.model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "config": cfg,
                "val_metrics": val_metrics,
            },
            last_path,
        )
        logger.info(
            f"epoch={epoch}/{epochs} train_loss={train_metrics['loss']:.4f} "
            f"val_loss={val_metrics.get('loss', 0.0):.4f} val_dice={val_metrics.get('dice', 0.0):.4f}"
        )

    save_history_csv(history, exp_paths.results / "train_history.csv")
    save_json({"history": history, "best_metric": {"name": save_metric, "value": best_score}}, exp_paths.results / "train_history.json")
    logger.info(f"Training complete. Results saved under: {exp_paths.root}")


if __name__ == "__main__":
    main()
