from __future__ import annotations

from pathlib import Path
import sys

import pytest
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.engine.output_utils import parse_model_output
from src.models import build_model
from src.models.baselines.csca_unet import (
    CSCABasicBlock,
    CSCADecoderBlock,
    CSCASpatialAttention,
    DoubleSqueezeExcitation,
)
from src.models.baselines.resunetpp import ResUNetPPAttentionGate, ResUNetPPDecoderBlock
from src.models.common.blocks import ASPP, ResidualBlock, SqueezeExcitation
from src.models.common.official_backbones import (
    OfficialHarDNetEncoder,
    OfficialPVTv2Backbone,
    OfficialRes2NetEncoder,
)
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
    GlobalContextModule,
    HybridSemanticComplementaryModule,
    LocalContextAttention,
    MultiScalePredictionModule,
    RFBModified,
    ReverseAttentionBranch,
    SimilarityAggregationModule,
)
from src.models.proposal.hf_bottleneck import HFBottleneck

FAITHFUL_DIR = PROJECT_ROOT / "configs" / "faithful"


def _load_cfg(model_name: str):
    return yaml.safe_load((FAITHFUL_DIR / f"{model_name}.yaml").read_text(encoding="utf-8"))


def _fast_test_cfg(model_name: str, cfg: dict):
    cfg = dict(cfg)
    model_cfg = dict(cfg["model"])
    if model_name == "csca_unet":
        model_cfg["channels"] = [2, 4, 8, 16, 32, 64]
        model_cfg["attention_mode"] = "efficient"
    cfg["model"] = model_cfg
    return cfg


PAPER_MODULE_CONTRACTS: dict[str, tuple[type, ...]] = {
    # PraNet: Res2Net backbone + RFB aggregation + parallel reverse attention.
    "pranet": (OfficialRes2NetEncoder, RFBModified, DenseAggregation, ReverseAttentionBranch),
    # ACSNet: local context attention + global context module + adaptive selection.
    "acsnet": (OfficialRes2NetEncoder, LocalContextAttention, GlobalContextModule, AdaptiveSelectionModule),
    # HarDNet-MSEG: HarDNet encoder with RFB and aggregation decoder pieces.
    "hardnet_mseg": (OfficialHarDNetEncoder, RFBModified, DenseAggregation),
    # Polyp-PVT: PVT backbone + cascaded fusion / camouflage identification / similarity aggregation.
    "polyp_pvt": (OfficialPVTv2Backbone, CascadedFusionModule, CamouflageIdentificationModule, SimilarityAggregationModule),
    # CaraNet: Res2Net backbone + CFP + axial reverse attention + aggregation.
    "caranet": (OfficialRes2NetEncoder, CFPModule, AxialReverseAttention, DenseAggregation),
    # CFA-Net: boundary branch + cross-feature fusion + boundary aggregation.
    "cfanet": (BoundaryPredictionNetwork, CrossFeatureFusion, BoundaryAggregationModule),
    # HSNet: hybrid Res2Net/PVT encoders + cross-semantic attention + hybrid semantic complementary module.
    "hsnet": (
        OfficialRes2NetEncoder,
        OfficialPVTv2Backbone,
        CrossSemanticAttention,
        HybridSemanticComplementaryModule,
        MultiScalePredictionModule,
    ),
    # CSCA U-Net: residual basic blocks + DSE bottleneck/attention + CSCA decoder attention + deep supervision.
    "csca_unet": (CSCABasicBlock, DoubleSqueezeExcitation, CSCASpatialAttention, CSCADecoderBlock),
    # ResUNet++: residual units + SE + ASPP bridge + attention-gated decoder.
    "resunetpp": (ResidualBlock, SqueezeExcitation, ASPP, ResUNetPPAttentionGate, ResUNetPPDecoderBlock),
    # Proposed method: HF bottleneck must exist as the named architectural contribution.
    "proposal_hf_unet": (HFBottleneck,),
}


EXPECTED_FAITHFUL_AUX_COUNTS = {
    "unetpp": 3,
    "pranet": 3,
    "acsnet": 3,
    "polyp_pvt": 1,
    "caranet": 3,
    "cfanet": 4,
    "hsnet": 4,
    "csca_unet": 5,
}


@pytest.mark.parametrize("model_name,module_types", PAPER_MODULE_CONTRACTS.items())
def test_baseline_contains_named_modules_from_official_paper_design(model_name: str, module_types: tuple[type, ...]):
    """Detect accidental replacement by a generic U-Net-like placeholder.

    This test is not a mathematical proof of full paper equivalence. It is a
    strict architecture contract: each baseline must expose the key modules that
    justify comparing it under the paper name.
    """
    cfg = _load_cfg(model_name)
    model = build_model(model_name, cfg["model"])
    modules = list(model.modules())

    for module_type in module_types:
        assert any(isinstance(module, module_type) for module in modules), (
            model_name,
            f"missing expected module {module_type.__name__}",
        )


@pytest.mark.parametrize("model_name,expected_aux", EXPECTED_FAITHFUL_AUX_COUNTS.items())
def test_faithful_output_contract_matches_paper_style_deep_supervision(model_name: str, expected_aux: int):
    cfg = _fast_test_cfg(model_name, _load_cfg(model_name))
    model = build_model(model_name, cfg["model"]).eval()
    x = torch.randn(1, 3, 64, 64)

    with torch.no_grad():
        parsed = parse_model_output(model(x))

    assert parsed.main.shape == (1, 1, 64, 64), model_name
    assert len(parsed.aux) == expected_aux, (model_name, len(parsed.aux), expected_aux)
    for aux in parsed.aux:
        assert aux.shape == parsed.main.shape, (model_name, aux.shape, parsed.main.shape)
        assert torch.isfinite(aux).all(), model_name


def test_cfanet_faithful_output_exposes_boundary_prediction_branch():
    cfg = _load_cfg("cfanet")
    model = build_model("cfanet", cfg["model"]).eval()
    x = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        parsed = parse_model_output(model(x))
    assert parsed.boundary is not None
    assert parsed.boundary.shape == parsed.main.shape
    assert torch.isfinite(parsed.boundary).all()


def test_csca_unet_official_faithful_config_uses_six_stage_encoder_and_five_deep_supervision_maps():
    cfg = _load_cfg("csca_unet")
    model_cfg = cfg["model"]
    assert model_cfg["channels"] == [32, 64, 128, 256, 512, 1024]
    assert model_cfg["deep_supervision"] is True
    assert model_cfg["faithful_output"] is True
    assert len(cfg["train"]["aux_output_weights"]) == 5

    model = build_model("csca_unet", model_cfg)
    assert isinstance(model.down_conv6[-1], DoubleSqueezeExcitation)
    assert all(hasattr(model, name) for name in ["dp6", "dp5", "dp4", "dp3", "dp2"])
    assert all(hasattr(model, name) for name in ["up_conv5", "up_conv4", "up_conv3", "up_conv2", "up_conv1"])
