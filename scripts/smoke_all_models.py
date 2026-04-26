#!/usr/bin/env python3
"""Run a quick local smoke check for registered segmentation models.

This script is intentionally lightweight: it does not download datasets and does
not train for epochs. It builds each selected model, runs a dummy forward pass,
computes the configured supervised loss, and optionally runs one backward step.
Use it before launching expensive multi-seed benchmarks.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.engine.output_utils import compute_supervised_loss, parse_model_output
from src.losses import BCEDiceLoss, DiceLoss, StructureLoss
from src.models import build_model

DEFAULT_MODELS = "unet,unet_cbam,unetpp,pranet,acsnet,hardnet_mseg,polyp_pvt,caranet,cfanet,hsnet,csca_unet,proposal_hf_unet"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-test model build/forward/loss/backward without a real dataset.")
    parser.add_argument("--models", type=str, default=DEFAULT_MODELS, help="Comma-separated model names")
    parser.add_argument("--config-dir", type=str, default="configs/paper_fair", help="Directory containing <model>.yaml")
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--no-backward", action="store_true", help="Skip backward/optimizer step")
    return parser.parse_args()


def _loss(name: str):
    name = str(name).lower()
    if name == "structure":
        return StructureLoss()
    if name == "dice":
        return DiceLoss(from_logits=True)
    if name in {"bce_dice", "bcedice"}:
        return BCEDiceLoss()
    raise ValueError(f"Unsupported loss: {name}")


def _load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    config_dir = PROJECT_ROOT / args.config_dir
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    x = torch.randn(args.batch_size, 3, args.image_size, args.image_size, device=device)
    masks = torch.randint(0, 2, (args.batch_size, 1, args.image_size, args.image_size), device=device).float()

    for model_name in models:
        cfg_path = config_dir / f"{model_name}.yaml"
        if not cfg_path.is_file():
            raise FileNotFoundError(f"Missing config for {model_name}: {cfg_path}")
        cfg = _load_yaml(cfg_path)
        torch.manual_seed(123)
        model = build_model(model_name, cfg["model"]).to(device)
        model.train()

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
        optimizer.zero_grad(set_to_none=True)
        output = model(x)
        parsed = parse_model_output(output)
        if parsed.main.shape != masks.shape:
            raise AssertionError(f"{model_name}: main output shape {tuple(parsed.main.shape)} != mask shape {tuple(masks.shape)}")

        total_loss, log_items, parsed = compute_supervised_loss(
            output,
            masks,
            main_loss_fn=_loss(cfg["train"].get("loss", "bce_dice")),
            aux_loss_fn=_loss(cfg["train"].get("aux_loss", cfg["train"].get("loss", "bce_dice")))
            if cfg["train"].get("aux_output_weights") is not None
            else None,
            aux_weights=cfg["train"].get("aux_output_weights"),
            boundary_loss_fn=_loss(cfg["train"].get("boundary_loss")) if cfg["train"].get("boundary_loss") else None,
            boundary_weight=float(cfg["train"].get("boundary_weight", 0.0)),
        )
        if not torch.isfinite(total_loss):
            raise AssertionError(f"{model_name}: non-finite loss")

        if not args.no_backward:
            total_loss.backward()
            finite_grad = any(
                p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
                for p in model.parameters()
                if p.requires_grad
            )
            if not finite_grad:
                raise AssertionError(f"{model_name}: no finite non-zero gradient found")
            optimizer.step()

        print(f"[OK] {model_name}: main={tuple(parsed.main.shape)} aux={len(parsed.aux)} boundary={parsed.boundary is not None} loss={log_items['loss']:.4f}")


if __name__ == "__main__":
    main()
