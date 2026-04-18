#!/usr/bin/env python3
"""Train one segmentation model inside the unified benchmark framework."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Mapping

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets import build_eval_transforms, build_train_transforms, KvasirSegDataset, normalize_dataset_name
from src.engine import Trainer
from src.losses import BCEDiceLoss, DiceLoss, StructureLoss
from src.models import build_model
from src.utils import (
    ExperimentPaths,
    dump_yaml,
    get_logger,
    load_yaml,
    resolve_device,
    seed_everything,
    should_pin_memory,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train one model.")
    parser.add_argument("--model", type=str, required=True, help="Registered model name, e.g. unet or proposal_hf_unet")
    parser.add_argument("--dataset", type=str, default="kvasir_seg", help="Dataset key. Currently supports kvasir_seg and custom.")
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
        "data": {"dataset": normalize_dataset_name(args.dataset), "root": "data", "image_size": 352, "batch_size": 8, "num_workers": 4, "pin_memory": True},
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
    overrides.setdefault("data", {})["dataset"] = normalize_dataset_name(args.dataset)
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
    cfg["data"]["dataset"] = normalize_dataset_name(cfg["data"].get("dataset", args.dataset))
    return cfg


def build_dataloaders(cfg: Dict[str, Any]) -> tuple[DataLoader, DataLoader]:
    data_cfg = cfg["data"]
    image_size = int(data_cfg.get("image_size", 352))
    dataset_name = normalize_dataset_name(data_cfg.get("dataset", "kvasir_seg"))
    if dataset_name not in {"kvasir_seg", "custom"}:
        raise ValueError(f"Unsupported dataset for training: {dataset_name}")

    train_ds = KvasirSegDataset(
        root=data_cfg.get("root", "data"),
        split="train",
        image_size=image_size,
        transform=build_train_transforms(
            image_size=image_size,
            preset=str(data_cfg.get("augmentation", "baseline")),
        ),
    )
    val_ds = KvasirSegDataset(
        root=data_cfg.get("root", "data"),
        split="val",
        image_size=image_size,
        transform=build_eval_transforms(image_size=image_size),
    )
    batch_size = int(data_cfg.get("batch_size", 8))
    num_workers = int(data_cfg.get("num_workers", 4))
    pin_memory = bool(data_cfg.get("pin_memory", should_pin_memory(cfg["train"].get("device"))))
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
    return checkpoint


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
    resolved_device = resolve_device(cfg["train"].get("device"))
    cfg["train"]["device"] = resolved_device
    cfg["data"]["pin_memory"] = bool(cfg["data"].get("pin_memory", should_pin_memory(resolved_device)))
    logger.info(f"Training model: {model_name}")
    logger.info(f"Dataset: {cfg['data'].get('dataset')} | Device: {resolved_device}")
    dump_yaml(cfg, exp_paths.root / "resolved_train_config.yaml")

    train_loader, val_loader = build_dataloaders(cfg)
    model_cfg = {k: v for k, v in cfg["model"].items() if k != "name"}
    model = build_model(model_name, config=model_cfg)
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)
    loss_fn = build_loss(str(cfg["train"].get("loss", "bce_dice")))
    aux_loss_name = str(cfg["train"].get("aux_loss", cfg["train"].get("loss", "bce_dice")))
    aux_loss_fn = build_loss(aux_loss_name) if cfg["train"].get("aux_output_weights") is not None else None
    boundary_loss_name = cfg["train"].get("boundary_loss")
    boundary_loss_fn = build_loss(str(boundary_loss_name)) if boundary_loss_name else None

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        aux_loss_fn=aux_loss_fn,
        aux_weights=cfg["train"].get("aux_output_weights"),
        boundary_loss_fn=boundary_loss_fn,
        boundary_weight=float(cfg["train"].get("boundary_weight", 0.0)),
        device=resolved_device,
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
        checkpoint = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(unwrap_state_dict(checkpoint), strict=True)
        logger.info(f"Resumed from checkpoint: {args.resume}")

    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=int(cfg["train"].get("epochs", 30)),
        metric_for_plateau=str(cfg["train"].get("plateau_metric", "loss")),
    )

    if not history:
        raise RuntimeError("Training produced an empty history.")

    save_history_csv(history, exp_paths.results / "history.csv")
    best_metric_name = str(cfg["train"].get("save_metric", "dice"))
    best_metric_mode = str(cfg["train"].get("save_metric_mode", "max")).lower()

    def metric_value(record: Dict[str, Any]) -> float:
        key = f"val/{best_metric_name}"
        if key in record:
            return float(record[key])
        fallback = record.get("val/loss", record.get("train/loss", math.nan))
        return float(fallback)

    valid_records = [r for r in history if not math.isnan(metric_value(r))]
    if best_metric_mode == "min":
        best_record = min(valid_records or history, key=metric_value)
    else:
        best_record = max(valid_records or history, key=metric_value)
    best_epoch = history.index(best_record) + 1

    ckpt_payload = {
        "epoch": best_epoch,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": cfg,
        "history": history,
    }
    torch.save(ckpt_payload, exp_paths.checkpoints / "best.pt")

    summary = {
        "model": model_name,
        "dataset": cfg["data"].get("dataset"),
        "device": resolved_device,
        "best_epoch": best_epoch,
        "best_record": best_record,
        "num_train_batches": len(train_loader),
        "num_val_batches": len(val_loader),
    }
    save_json(summary, exp_paths.results / "summary.json")
    logger.info(f"Saved best checkpoint to {exp_paths.checkpoints / 'best.pt'}")
    logger.info(f"Best epoch={best_epoch} metric={metric_value(best_record):.4f}")


if __name__ == "__main__":
    main()
