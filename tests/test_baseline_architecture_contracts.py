from __future__ import annotations

from pathlib import Path
import sys

import pytest
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models import build_model
from src.models.common.paper_baselines import (
    AdaptiveSelectionModule,
    AxialReverseAttention,
    BoundaryAggregationModule,
    BoundaryPredictionNetwork,
    CFPModule,
    CamouflageIdentificationModule,
    CascadedFusionModule,
    CrossFeatureFusion,
    DenseAggregation,
    GlobalContextModule,
    HarDBlock,
    LocalContextAttention,
    OverlapPatchEmbed,
    PVTTransformerBlock,
    RFBModified,
    SimilarityAggregationModule,
)

CONFIG_DIR = PROJECT_ROOT / "configs"


MODEL_MODULE_CONTRACTS = {
    "pranet": (RFBModified, DenseAggregation),
    "acsnet": (LocalContextAttention, GlobalContextModule, AdaptiveSelectionModule),
    "cfanet": (BoundaryPredictionNetwork, CrossFeatureFusion, BoundaryAggregationModule),
    "caranet": (CFPModule, AxialReverseAttention, DenseAggregation),
    "hardnet_mseg": (HarDBlock, RFBModified, DenseAggregation),
    "polyp_pvt": (OverlapPatchEmbed, PVTTransformerBlock, CascadedFusionModule, CamouflageIdentificationModule, SimilarityAggregationModule),
}


@pytest.mark.parametrize("model_name", [
    "unet",
    "unetpp",
    "unet_cbam",
    "pranet",
    "acsnet",
    "cfanet",
    "caranet",
    "hardnet_mseg",
    "polyp_pvt",
    "proposal_hf_unet",
])
def test_default_config_builds_and_preserves_segmentation_shape(model_name: str):
    with (CONFIG_DIR / f"{model_name}.yaml").open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    model = build_model(model_name, cfg["model"])
    model.eval()
    x = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        y = model(x)
    assert isinstance(y, torch.Tensor)
    assert y.shape == (1, 1, 64, 64)


@pytest.mark.parametrize("model_name,module_types", MODEL_MODULE_CONTRACTS.items())
def test_advanced_baselines_expose_paper_modules(model_name: str, module_types: tuple[type, ...]):
    with (CONFIG_DIR / f"{model_name}.yaml").open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    model = build_model(model_name, cfg["model"])
    modules = list(model.modules())
    for module_type in module_types:
        assert any(isinstance(m, module_type) for m in modules), (model_name, module_type.__name__)


def test_strict_fairness_audit_no_longer_reports_lite_baseline_warning():
    import json
    import subprocess

    proc = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "tools" / "audit_fairness.py")],
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads(proc.stdout)
    joined = "\n".join(report.get("warnings", []))
    assert 'simplified "Lite"' not in joined
