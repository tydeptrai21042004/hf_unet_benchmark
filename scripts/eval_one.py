#!/usr/bin/env python3
"""Evaluate one model checkpoint on val/test split and optionally save predictions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Mapping

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets import build_eval_transforms, KvasirSegDataset
from src.engine import Evaluator, Inferencer
from src.losses import BCEDiceLoss, DiceLoss, StructureLoss
from src.models import build_model
from src.utils import ExperimentPaths, get_logger, load_yaml, dump_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate one checkpoint.")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint. Defaults to <output-root>/<model>/checkpoints/best.pt")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output-root", type=str, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--save-visualizations", action="store_true")
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
        "eval": {"loss": "bce_dice", "threshold": 0.5},
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
    if args.device is not None:
        overrides.setdefault("eval", {})["device"] = args.device
    if args.output_root is not None:
        overrides.setdefault("experiment", {})["output_root"] = args.output_root
    _deep_update(cfg, overrides)
    cfg["model"]["name"] = args.model
    return cfg


def build_loss(loss_name: str):
    name = loss_name.lower()
    if name == "dice":
        return DiceLoss(from_logits=True)
    if name in {"bce_dice", "bcedice"}:
        return BCEDiceLoss()
    if name == "structure":
        return StructureLoss()
    raise ValueError(f"Unsupported loss: {loss_name}")


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


def main() -> None:
    args = parse_args()
    cfg = load_config_from_args(args)
    model_name = cfg["model"].get("name", args.model)
    exp_name = cfg["experiment"].get("name") or model_name
    output_root = Path(cfg["experiment"].get("output_root", PROJECT_ROOT))
    exp_paths = ExperimentPaths.create(output_root, experiment_name=exp_name)
    logger = get_logger(name=f"eval_{model_name}", log_file=exp_paths.logs / "eval.log")
    dump_yaml(cfg, exp_paths.root / f"resolved_eval_config_{args.split}.yaml")

    checkpoint_path = Path(args.checkpoint) if args.checkpoint else exp_paths.checkpoints / "best.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    data_cfg = cfg["data"]
    image_size = int(data_cfg.get("image_size", 352))
    dataset = KvasirSegDataset(
        root=data_cfg.get("root", "data"),
        split=args.split,
        transform=build_eval_transforms(image_size=image_size),
        return_paths=args.save_predictions,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=int(data_cfg.get("batch_size", 8)),
        shuffle=False,
        num_workers=int(data_cfg.get("num_workers", 4)),
        pin_memory=bool(data_cfg.get("pin_memory", True)),
        drop_last=False,
    )

    model_cfg = {k: v for k, v in cfg["model"].items() if k != "name"}
    model = build_model(model_name, config=model_cfg)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(unwrap_state_dict(checkpoint), strict=True)
    device = cfg.get("eval", {}).get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    threshold = float(cfg.get("eval", {}).get("threshold", 0.5))
    loss_fn = build_loss(str(cfg.get("eval", {}).get("loss", cfg.get("train", {}).get("loss", "bce_dice"))))

    evaluator = Evaluator(device=device, threshold=threshold, logger=logger)
    metrics = evaluator.evaluate(model.to(device), dataloader, loss_fn=loss_fn)
    payload = {
        "model": model_name,
        "split": args.split,
        "checkpoint": str(checkpoint_path),
        "metrics": metrics,
        "num_samples": len(dataset),
    }

    save_json(payload, exp_paths.results / f"metrics_{args.split}.json")
    tables_dir = output_root / "results" / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    save_json(payload, tables_dir / f"{model_name}_{args.split}_metrics.json")
    logger.info(f"Evaluation metrics: {metrics}")

    if args.save_predictions:
        inferencer = Inferencer(model=model.to(device), device=device, threshold=threshold)
        pred_dir = output_root / "results" / "predictions" / model_name / args.split
        inferencer.save_predictions(dataloader, pred_dir, save_visualizations=args.save_visualizations)
        logger.info(f"Saved predictions to {pred_dir}")


if __name__ == "__main__":
    main()
