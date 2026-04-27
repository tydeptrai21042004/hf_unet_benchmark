from __future__ import annotations

from pathlib import Path
import sys

import pytest
import torch
import yaml
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.engine import Evaluator, Trainer
from src.engine.output_utils import compute_supervised_loss, parse_model_output
from src.losses import BCEDiceLoss, DiceLoss, StructureLoss
from src.models import build_model

PAPER_FAIR_DIR = PROJECT_ROOT / "configs" / "paper_fair"
FAITHFUL_DIR = PROJECT_ROOT / "configs" / "faithful"

ALL_RUNNABLE_MODELS = [
    "unet",
    "unet_cbam",
    "unetpp",
    "resunetpp",
    "pranet",
    "acsnet",
    "hardnet_mseg",
    "polyp_pvt",
    "caranet",
    "cfanet",
    "hsnet",
    "csca_unet",
    "proposal_hf_unet",
]

GRAD_RUNNABLE_MODELS = [m for m in ALL_RUNNABLE_MODELS if m != "csca_unet"]

REPRESENTATIVE_TRAINER_EVAL_MODELS = [
    "unet",              # plain tensor output
    "pranet",            # multi-stage auxiliary output
    "cfanet",            # auxiliary + boundary output
    "proposal_hf_unet",  # proposal regularizer path
]


class TinySegmentationDataset(Dataset):
    def __init__(self, length: int = 2, image_size: int = 64) -> None:
        self.length = length
        self.image_size = image_size

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int):
        g = torch.Generator().manual_seed(index)
        image = torch.randn(3, self.image_size, self.image_size, generator=g)
        mask = (torch.rand(1, self.image_size, self.image_size, generator=g) > 0.5).float()
        return {"image": image, "mask": mask}


def _load_cfg(config_dir: Path, name: str):
    return yaml.safe_load((config_dir / f"{name}.yaml").read_text(encoding="utf-8"))


def _fast_test_cfg(model_name: str, cfg: dict):
    cfg = dict(cfg)
    model_cfg = dict(cfg["model"])
    if model_name == "csca_unet":
        model_cfg["channels"] = [2, 4, 8, 16, 32, 64]
        model_cfg["attention_mode"] = "efficient"
    cfg["model"] = model_cfg
    return cfg


def _loss(name: str):
    name = str(name).lower()
    if name == "structure":
        return StructureLoss()
    if name == "dice":
        return DiceLoss(from_logits=True)
    if name in {"bce_dice", "bcedice"}:
        return BCEDiceLoss()
    raise ValueError(f"Unsupported loss in test: {name}")


@pytest.mark.parametrize("model_name", ALL_RUNNABLE_MODELS)
def test_paper_fair_model_runs_forward_parse_and_loss(model_name: str):
    torch.manual_seed(123)
    cfg = _fast_test_cfg(model_name, _load_cfg(PAPER_FAIR_DIR, model_name))
    model = build_model(model_name, cfg["model"])
    model.eval()

    x = torch.randn(1, 3, 32, 32)
    masks = torch.randint(0, 2, (1, 1, 32, 32), dtype=torch.float32)
    with torch.no_grad():
        output = model(x)
    parsed = parse_model_output(output)

    assert parsed.main.shape == masks.shape, model_name
    assert torch.isfinite(parsed.main).all(), model_name

    total_loss, log_items, _ = compute_supervised_loss(
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
    assert torch.isfinite(total_loss), model_name
    assert log_items["loss"] == pytest.approx(float(total_loss.detach().item()))


@pytest.mark.parametrize("model_name", GRAD_RUNNABLE_MODELS)
def test_paper_fair_model_runs_backward_with_finite_gradients(model_name: str):
    torch.manual_seed(123)
    cfg = _load_cfg(PAPER_FAIR_DIR, model_name)
    model = build_model(model_name, cfg["model"])
    model.train()

    x = torch.randn(2, 3, 32, 32)
    masks = torch.randint(0, 2, (2, 1, 32, 32), dtype=torch.float32)
    output = model(x)
    total_loss, _, _ = compute_supervised_loss(
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
    assert torch.isfinite(total_loss), model_name
    total_loss.backward()

    finite_nonzero_grad = False
    for param in model.parameters():
        if param.requires_grad and param.grad is not None:
            assert torch.isfinite(param.grad).all(), model_name
            finite_nonzero_grad = finite_nonzero_grad or bool(param.grad.abs().sum().item() > 0)
    assert finite_nonzero_grad, model_name


@pytest.mark.parametrize("model_name", REPRESENTATIVE_TRAINER_EVAL_MODELS)
def test_faithful_model_can_run_tiny_trainer_and_evaluator_loop(model_name: str):
    """Exercise the same Trainer/Evaluator path used by train_one.py/eval_one.py.

    This catches bugs that pure forward tests miss, especially dict/list outputs,
    auxiliary losses, boundary losses, gradient clipping, and metric computation.
    """
    torch.manual_seed(321)
    cfg = _load_cfg(FAITHFUL_DIR, model_name)
    model = build_model(model_name, cfg["model"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    loader = DataLoader(TinySegmentationDataset(length=2, image_size=32), batch_size=2)

    trainer = Trainer(
        model,
        optimizer,
        _loss(cfg["train"].get("loss", "bce_dice")),
        aux_loss_fn=_loss(cfg["train"].get("aux_loss", cfg["train"].get("loss", "bce_dice")))
        if cfg["train"].get("aux_output_weights") is not None
        else None,
        aux_weights=cfg["train"].get("aux_output_weights"),
        boundary_loss_fn=_loss(cfg["train"].get("boundary_loss")) if cfg["train"].get("boundary_loss") else None,
        boundary_weight=float(cfg["train"].get("boundary_weight", 0.0)),
        device="cpu",
        mixed_precision=False,
        grad_clip=cfg["train"].get("grad_clip"),
        log_interval=999,
    )

    train_metrics = trainer.train_one_epoch(loader, epoch=1)
    assert train_metrics["loss"] > 0.0
    assert torch.isfinite(torch.tensor(train_metrics["loss"]))

    val_metrics = trainer.validate(loader)
    for key in ["loss", "dice", "iou", "precision", "recall", "mae"]:
        assert key in val_metrics, (model_name, key, val_metrics)
        assert torch.isfinite(torch.tensor(float(val_metrics[key]))), (model_name, key, val_metrics[key])


def test_evaluator_standalone_handles_dict_aux_and_boundary_outputs():
    masks = torch.randint(0, 2, (2, 1, 32, 32), dtype=torch.float32)
    output = {
        "main": torch.randn(2, 1, 32, 32),
        "aux": [torch.randn(2, 1, 32, 32), torch.randn(2, 1, 32, 32)],
        "boundary": torch.randn(2, 1, 32, 32),
    }
    loss, logs, parsed = compute_supervised_loss(
        output,
        masks,
        main_loss_fn=BCEDiceLoss(),
        aux_loss_fn=BCEDiceLoss(),
        aux_weights=[0.25, 0.5],
        boundary_loss_fn=BCEDiceLoss(),
        boundary_weight=0.25,
    )
    assert torch.isfinite(loss)
    assert parsed.main.shape == masks.shape
    assert len(parsed.aux) == 2
    assert "boundary_loss" in logs
