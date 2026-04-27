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
from src.models.baselines.csca_unet import CSCABasicBlock, CSCADecoderBlock, CSCASpatialAttention, DoubleSqueezeExcitation
from src.models.baselines.resunetpp import ResUNetPPAttentionGate, ResUNetPPDecoderBlock
from src.models.common.blocks import ASPP, ResidualBlock, SqueezeExcitation
from src.models.common.official_backbones import OfficialHarDNetEncoder, OfficialPVTv2Backbone, OfficialRes2NetEncoder
from src.models.common.paper_baselines import (
    AdaptiveSelectionModule,
    AxialReverseAttention,
    BoundaryAggregationModule,
    BoundaryPredictionNetwork,
    CFPModule,
    CamouflageIdentificationModule,
    CascadedFusionModule,
    CrossFeatureFusion,
    CrossSemanticAttention,
    DenseAggregation,
    HybridSemanticComplementaryModule,
    GlobalContextModule,
    HarDBlock,
    LocalContextAttention,
    MultiScalePredictionModule,
    OverlapPatchEmbed,
    PVTTransformerBlock,
    RFBModified,
    SimilarityAggregationModule,
)

CONFIG_DIR = PROJECT_ROOT / "configs"


def _fast_test_model_cfg(model_name: str, cfg: dict):
    cfg = dict(cfg)
    if model_name == "csca_unet":
        cfg["channels"] = [2, 4, 8, 16, 32, 64]
        cfg["attention_mode"] = "efficient"
    return cfg


MODEL_MODULE_CONTRACTS = {
    "pranet": (RFBModified, DenseAggregation, OfficialRes2NetEncoder),
    "acsnet": (LocalContextAttention, GlobalContextModule, AdaptiveSelectionModule, OfficialRes2NetEncoder),
    "cfanet": (BoundaryPredictionNetwork, CrossFeatureFusion, BoundaryAggregationModule),
    "caranet": (CFPModule, AxialReverseAttention, DenseAggregation, OfficialRes2NetEncoder),
    "hardnet_mseg": (RFBModified, DenseAggregation, OfficialHarDNetEncoder),
    "polyp_pvt": (CascadedFusionModule, CamouflageIdentificationModule, SimilarityAggregationModule, OfficialPVTv2Backbone),
    "hsnet": (CrossSemanticAttention, HybridSemanticComplementaryModule, MultiScalePredictionModule, OfficialRes2NetEncoder, OfficialPVTv2Backbone),
    "csca_unet": (CSCABasicBlock, DoubleSqueezeExcitation, CSCASpatialAttention, CSCADecoderBlock),
    "resunetpp": (ResidualBlock, SqueezeExcitation, ASPP, ResUNetPPAttentionGate, ResUNetPPDecoderBlock),
}


@pytest.mark.parametrize("model_name", [
    "unet",
    "unetpp",
    "resunetpp",
    "unet_cbam",
    "pranet",
    "acsnet",
    "cfanet",
    "caranet",
    "hardnet_mseg",
    "polyp_pvt",
    "hsnet",
    "csca_unet",
    "proposal_hf_unet",
])
def test_default_config_builds_and_preserves_segmentation_shape(model_name: str):
    with (CONFIG_DIR / f"{model_name}.yaml").open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    model = build_model(model_name, _fast_test_model_cfg(model_name, cfg["model"]))
    model.eval()
    x = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        y = model(x)
    assert isinstance(y, torch.Tensor)
    assert y.shape == (1, 1, 32, 32)


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
