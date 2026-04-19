from __future__ import annotations

from pathlib import Path
import sys

import pytest
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.engine.output_utils import compute_supervised_loss
from src.losses import BCEDiceLoss, DiceLoss, StructureLoss
from src.models import build_model

FAITHFUL_DIR = PROJECT_ROOT / "configs" / "faithful"
MODELS = [
    "unet",
    "unet_cbam",
    "unetpp",
    "pranet",
    "acsnet",
    "hardnet_mseg",
    "polyp_pvt",
    "caranet",
    "cfanet",
    "hsnet",
    "proposal_hf_unet_lite",
    "proposal_hf_unet",
]


def _load_cfg(name: str):
    return yaml.safe_load((FAITHFUL_DIR / f"{name}.yaml").read_text(encoding="utf-8"))


def _build_loss(name: str):
    name = str(name).lower()
    if name == "structure":
        return StructureLoss()
    if name == "dice":
        return DiceLoss(from_logits=True)
    if name in {"bce_dice", "bcedice"}:
        return BCEDiceLoss()
    raise ValueError(name)


@pytest.mark.parametrize("model_name", MODELS)
def test_all_faithful_baselines_support_one_training_step(model_name: str):
    torch.manual_seed(0)
    cfg = _load_cfg(model_name)
    model = build_model(model_name, cfg["model"])
    model.train()

    x = torch.randn(2, 3, 32, 32)
    masks = torch.randint(0, 2, (2, 1, 32, 32), dtype=torch.float32)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    optimizer.zero_grad(set_to_none=True)
    output = model(x)
    total_loss, log_items, parsed = compute_supervised_loss(
        output,
        masks,
        main_loss_fn=_build_loss(cfg["train"].get("loss", "bce_dice")),
        aux_loss_fn=_build_loss(cfg["train"].get("aux_loss", cfg["train"].get("loss", "bce_dice"))) if cfg["train"].get("aux_output_weights") is not None else None,
        aux_weights=cfg["train"].get("aux_output_weights"),
        boundary_loss_fn=_build_loss(cfg["train"].get("boundary_loss")) if cfg["train"].get("boundary_loss") else None,
        boundary_weight=float(cfg["train"].get("boundary_weight", 0.0)),
    )
    assert torch.isfinite(total_loss), model_name
    assert parsed.main.shape == masks.shape, model_name
    assert torch.isfinite(torch.tensor(float(log_items["loss"]))), model_name
    total_loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
    assert grads, model_name
    assert any(torch.isfinite(g).all() and g.abs().sum() > 0 for g in grads), model_name
    optimizer.step()
