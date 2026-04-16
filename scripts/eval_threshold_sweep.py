#!/usr/bin/env python3
"""Tune the segmentation threshold on validation and optionally evaluate on test.

This keeps inference policy fair by applying the same threshold-selection rule to
all models.
"""

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

from src.datasets import build_eval_transforms, KvasirSegDataset, normalize_dataset_name
from src.engine import Evaluator
from src.losses import BCEDiceLoss, DiceLoss, StructureLoss
from src.models import build_model
from src.utils import ExperimentPaths, get_logger, load_yaml, resolve_device, should_pin_memory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune threshold on val and evaluate on test.")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="kvasir_seg", help="Dataset key. Currently supports kvasir_seg and custom.")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output-root", type=str, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--metric", type=str, default="dice", choices=["dice", "iou", "precision", "recall"])
    parser.add_argument("--min-threshold", type=float, default=0.3)
    parser.add_argument("--max-threshold", type=float, default=0.7)
    parser.add_argument("--step", type=float, default=0.02)
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
        "eval": {"loss": "bce_dice", "threshold": 0.5},
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
    if args.device is not None:
        overrides.setdefault("eval", {})["device"] = args.device
    if args.output_root is not None:
        overrides.setdefault("experiment", {})["output_root"] = args.output_root
    _deep_update(cfg, overrides)
    cfg["model"]["name"] = args.model
    cfg["data"]["dataset"] = normalize_dataset_name(cfg["data"].get("dataset", args.dataset))
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


def build_loader(cfg: Dict[str, Any], split: str, device: str) -> DataLoader:
    data_cfg = cfg["data"]
    image_size = int(data_cfg.get("image_size", 352))
    dataset_name = normalize_dataset_name(data_cfg.get("dataset", "kvasir_seg"))
    if dataset_name not in {"kvasir_seg", "custom"}:
        raise ValueError(f"Unsupported dataset for threshold sweep: {dataset_name}")
    dataset = KvasirSegDataset(
        root=data_cfg.get("root", "data"),
        split=split,
        image_size=image_size,
        transform=build_eval_transforms(image_size=image_size),
    )
    return DataLoader(
        dataset,
        batch_size=int(data_cfg.get("batch_size", 8)),
        shuffle=False,
        num_workers=int(data_cfg.get("num_workers", 4)),
        pin_memory=bool(data_cfg.get("pin_memory", should_pin_memory(device))),
        drop_last=False,
    )


def main() -> None:
    args = parse_args()
    cfg = load_config_from_args(args)
    model_name = cfg["model"].get("name", args.model)
    exp_name = cfg["experiment"].get("name") or model_name
    output_root = Path(cfg["experiment"].get("output_root", PROJECT_ROOT))
    exp_paths = ExperimentPaths.create(output_root, experiment_name=exp_name)
    logger = get_logger(name=f"threshold_sweep_{model_name}", log_file=exp_paths.logs / "threshold_sweep.log")

    checkpoint_path = Path(args.checkpoint) if args.checkpoint else exp_paths.checkpoints / "best.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model_cfg = {k: v for k, v in cfg["model"].items() if k != "name"}
    model = build_model(model_name, config=model_cfg)
    model.load_state_dict(unwrap_state_dict(checkpoint), strict=True)
    device = resolve_device(cfg.get("eval", {}).get("device"))
    model = model.to(device)
    loss_fn = build_loss(str(cfg.get("eval", {}).get("loss", cfg.get("train", {}).get("loss", "bce_dice"))))

    val_loader = build_loader(cfg, "val", device)
    test_loader = build_loader(cfg, "test", device)

    thresholds = []
    t = args.min_threshold
    while t <= args.max_threshold + 1e-8:
        thresholds.append(round(t, 4))
        t += args.step

    best_threshold = thresholds[0]
    best_metrics = None
    best_score = float("-inf")
    sweep_results = []
    for threshold in thresholds:
        evaluator = Evaluator(device=device, threshold=threshold, logger=logger)
        metrics = evaluator.evaluate(model, val_loader, loss_fn=loss_fn)
        score = float(metrics.get(args.metric, metrics.get("dice", 0.0)))
        sweep_results.append({"threshold": threshold, **metrics})
        if score > best_score:
            best_score = score
            best_threshold = threshold
            best_metrics = metrics

    test_metrics = Evaluator(device=device, threshold=best_threshold, logger=logger).evaluate(model, test_loader, loss_fn=loss_fn)
    payload = {
        "model": model_name,
        "dataset": normalize_dataset_name(cfg["data"].get("dataset")),
        "device": device,
        "checkpoint": str(checkpoint_path),
        "metric": args.metric,
        "best_threshold": best_threshold,
        "best_val_metrics": best_metrics,
        "test_metrics": test_metrics,
        "sweep": sweep_results,
    }
    out_path = exp_paths.results / "threshold_sweep.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info(f"Best threshold={best_threshold:.3f} val_{args.metric}={best_score:.4f}")
    logger.info(f"Test metrics @ best threshold: {test_metrics}")


if __name__ == "__main__":
    main()
