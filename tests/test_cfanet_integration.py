from __future__ import annotations

from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import benchmark_all, benchmark_fair, eval_all, train_all
from src.models import build_model


def test_build_cfanet_from_registry():
    model = build_model(
        "cfanet",
        config={
            "in_channels": 3,
            "num_classes": 1,
            "channels": [16, 32, 64, 128, 256],
            "aggregation_channels": 32,
            "boundary_channels": 16,
        },
    )
    assert model is not None


def test_cfanet_forward_shape_matches_binary_segmentation_contract():
    model = build_model(
        "cfanet",
        config={
            "in_channels": 3,
            "num_classes": 1,
            "channels": [16, 32, 64, 128, 256],
            "aggregation_channels": 32,
            "boundary_channels": 16,
        },
    )
    model.eval()
    x = torch.randn(2, 3, 128, 128)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (2, 1, 128, 128)


def test_cli_model_lists_include_cfanet():
    assert "cfanet" in train_all.DEFAULT_MODELS
    assert "cfanet" in eval_all.DEFAULT_MODELS
    assert "cfanet" in benchmark_fair.MODELS
    assert "cfanet" in benchmark_all.DEFAULT_MODELS.split(",")
