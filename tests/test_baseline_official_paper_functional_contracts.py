from __future__ import annotations

from pathlib import Path
import sys

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.engine.output_utils import compute_supervised_loss, parse_model_output
from src.losses import BCEDiceLoss, StructureLoss
from src.models.baselines.acsnet import ACSNet
from src.models.baselines.caranet import CaraNet
from src.models.baselines.cfanet import CFANet
from src.models.baselines.csca_unet import CSCAUNet, DoubleSqueezeExcitation
from src.models.baselines.hardnet_mseg import HarDNetMSEG
from src.models.baselines.hsnet import HSNet
from src.models.baselines.polyp_pvt import PolypPVT
from src.models.baselines.pranet import PraNet
from src.models.baselines.resunetpp import ResUNetPlusPlus
from src.models.common.paper_baselines import (
    AdaptiveSelectionModule,
    AxialReverseAttention,
    CamouflageIdentificationModule,
    CascadedFusionModule,
    CFPModule,
    CrossFeatureFusion,
    DenseAggregation,
    GlobalContextModule,
    HybridSemanticComplementaryModule,
    LocalContextAttention,
    MultiScalePredictionModule,
    RFBModified,
    ReverseAttentionBranch,
    SimilarityAggregationModule,
)


def _finite_tensor(t: torch.Tensor) -> bool:
    return bool(torch.isfinite(t).all().item())


def _seg_mask(batch: int = 2, image_size: int = 64) -> torch.Tensor:
    mask = torch.zeros(batch, 1, image_size, image_size)
    mask[:, :, image_size // 4 : image_size * 3 // 4, image_size // 3 : image_size * 2 // 3] = 1.0
    return mask


def _lite_res2_channels() -> tuple[int, int, int, int, int]:
    return (8, 16, 32, 64, 128)


def _lite_pvt_channels() -> tuple[int, int, int, int]:
    return (16, 32, 64, 128)


def test_pranet_paper_contract_reverse_attention_chain_and_supervision_order():
    """PraNet paper contract: RFB aggregation feeds a coarse map, then RA4 -> RA3 -> RA2."""
    torch.manual_seed(11)
    model = PraNet(channels=_lite_res2_channels(), backbone_impl="lite", faithful_output=True).eval()
    assert isinstance(model.rfb2_1, RFBModified)
    assert isinstance(model.rfb3_1, RFBModified)
    assert isinstance(model.rfb4_1, RFBModified)
    assert isinstance(model.agg1, DenseAggregation)
    assert isinstance(model.ra4, ReverseAttentionBranch)
    assert isinstance(model.ra3, ReverseAttentionBranch)
    assert isinstance(model.ra2, ReverseAttentionBranch)

    coarse_inputs: list[torch.Size] = []

    def capture_ra_input(_module, inputs):
        feat, coarse = inputs
        assert feat.shape[-2:] == coarse.shape[-2:]
        coarse_inputs.append(coarse.shape)

    handles = [m.register_forward_pre_hook(capture_ra_input) for m in (model.ra4, model.ra3, model.ra2)]
    with torch.no_grad():
        out = model(torch.randn(1, 3, 64, 64))
    for h in handles:
        h.remove()

    parsed = parse_model_output(out)
    assert parsed.main.shape == (1, 1, 64, 64)
    assert len(parsed.aux) == 3
    assert len(coarse_inputs) == 3
    # The three RA stages should proceed from deeper/smaller to shallower/larger features.
    spatial_sizes = [shape[-1] for shape in coarse_inputs]
    assert spatial_sizes == sorted(spatial_sizes)
    assert all(_finite_tensor(t) for t in [parsed.main, *parsed.aux])


def test_acsnet_paper_contract_lca_gcm_asm_and_multistage_predictions():
    """ACSNet paper contract: LCA + GCM + ASM at multiple decoder levels."""
    torch.manual_seed(12)
    model = ACSNet(channels=_lite_res2_channels(), backbone_impl="lite", faithful_output=True).eval()
    assert isinstance(model.gcm, GlobalContextModule)
    assert all(isinstance(getattr(model, name), LocalContextAttention) for name in ["lca3", "lca2", "lca1", "lca0"])
    assert all(isinstance(getattr(model, name), AdaptiveSelectionModule) for name in ["asm3", "asm2", "asm1", "asm0"])

    asm_shapes: list[torch.Size] = []

    def capture_asm_output(_module, _inputs, output):
        asm_shapes.append(output.shape)

    handles = [getattr(model, name).register_forward_hook(capture_asm_output) for name in ["asm3", "asm2", "asm1", "asm0"]]
    with torch.no_grad():
        out = model(torch.randn(1, 3, 64, 64))
    for h in handles:
        h.remove()

    parsed = parse_model_output(out)
    assert parsed.main.shape == (1, 1, 64, 64)
    assert len(parsed.aux) == 3
    assert len(asm_shapes) == 4
    assert [s[1] for s in asm_shapes] == [64, 32, 16, 8]
    assert all(_finite_tensor(t) for t in [parsed.main, *parsed.aux])


def test_hardnet_mseg_paper_contract_hardnet_encoder_plus_cascaded_partial_decoder_style():
    """HarDNet-MSEG paper contract: high-level encoder features pass through RFBs and a dense/cascaded decoder."""
    torch.manual_seed(13)
    model = HarDNetMSEG(base_channels=8, backbone_impl="lite").eval()
    assert isinstance(model.rfb2, RFBModified)
    assert isinstance(model.rfb3, RFBModified)
    assert isinstance(model.rfb4, RFBModified)
    assert isinstance(model.decoder, DenseAggregation)

    decoder_input_shapes: list[torch.Size] = []

    def capture_decoder_input(_module, inputs):
        decoder_input_shapes.extend([x.shape for x in inputs])

    handle = model.decoder.register_forward_pre_hook(capture_decoder_input)
    with torch.no_grad():
        out = model(torch.randn(1, 3, 64, 64))
    handle.remove()

    assert out.shape == (1, 1, 64, 64)
    assert _finite_tensor(out)
    assert len(decoder_input_shapes) == 3
    assert decoder_input_shapes[0][-1] < decoder_input_shapes[1][-1] < decoder_input_shapes[2][-1]


def test_polyp_pvt_paper_contract_cfm_cim_sam_and_two_prediction_maps():
    """Polyp-PVT paper contract: PVT features are processed by CFM, CIM, and SAM with coarse+refined maps."""
    torch.manual_seed(14)
    model = PolypPVT(channels=_lite_pvt_channels(), backbone_impl="lite", faithful_output=True, image_size=64).eval()
    assert isinstance(model.cfm, CascadedFusionModule)
    assert isinstance(model.cim, CamouflageIdentificationModule)
    assert isinstance(model.sam, SimilarityAggregationModule)

    module_hits = {"cfm": 0, "cim": 0, "sam": 0}
    handles = [
        model.cfm.register_forward_hook(lambda *_: module_hits.__setitem__("cfm", module_hits["cfm"] + 1)),
        model.cim.register_forward_hook(lambda *_: module_hits.__setitem__("cim", module_hits["cim"] + 1)),
        model.sam.register_forward_hook(lambda *_: module_hits.__setitem__("sam", module_hits["sam"] + 1)),
    ]
    with torch.no_grad():
        out = model(torch.randn(1, 3, 64, 64))
    for h in handles:
        h.remove()

    parsed = parse_model_output(out)
    assert parsed.main.shape == (1, 1, 64, 64)
    assert len(parsed.aux) == 1
    assert module_hits == {"cfm": 1, "cim": 1, "sam": 1}
    assert not torch.allclose(parsed.main, parsed.aux[0]), "Refined SAM output should not collapse to the coarse CFM map."
    assert all(_finite_tensor(t) for t in [parsed.main, *parsed.aux])


def test_caranet_paper_contract_cfp_then_axial_reverse_attention_chain():
    """CaraNet paper contract: CFP enriches the deepest feature before axial reverse attention refinement."""
    torch.manual_seed(15)
    model = CaraNet(channels=_lite_res2_channels(), backbone_impl="lite", faithful_output=True).eval()
    assert isinstance(model.cfp, CFPModule)
    assert isinstance(model.agg, DenseAggregation)
    assert all(isinstance(getattr(model, name), AxialReverseAttention) for name in ["ara4", "ara3", "ara2"])

    ara_shapes: list[torch.Size] = []

    def capture_ara_input(_module, inputs):
        feat, coarse = inputs
        assert feat.shape[-2:] == coarse.shape[-2:]
        ara_shapes.append(feat.shape)

    handles = [getattr(model, name).register_forward_pre_hook(capture_ara_input) for name in ["ara4", "ara3", "ara2"]]
    with torch.no_grad():
        out = model(torch.randn(1, 3, 64, 64))
    for h in handles:
        h.remove()

    parsed = parse_model_output(out)
    assert parsed.main.shape == (1, 1, 64, 64)
    assert len(parsed.aux) == 3
    assert len(ara_shapes) == 3
    assert [s[-1] for s in ara_shapes] == sorted([s[-1] for s in ara_shapes])
    assert all(_finite_tensor(t) for t in [parsed.main, *parsed.aux])


def test_cfanet_paper_contract_boundary_branch_two_stream_fusion_and_boundary_gradients():
    """CFA-Net paper contract: boundary prediction supervises a boundary-aware two-stream segmentation path."""
    torch.manual_seed(16)
    model = CFANet(channels=_lite_res2_channels(), aggregation_channels=16, boundary_channels=8, faithful_output=True).train()
    assert isinstance(model.boundary, type(model.boundary))
    assert all(isinstance(getattr(model, name), CrossFeatureFusion) for name in ["stream1_43", "stream1_32", "stream1_21", "stream1_10"])

    x = torch.randn(2, 3, 64, 64)
    masks = _seg_mask(batch=2, image_size=64)
    out = model(x)
    parsed = parse_model_output(out)
    assert parsed.main.shape == masks.shape
    assert parsed.boundary is not None and parsed.boundary.shape == masks.shape
    assert len(parsed.aux) == 4

    loss, logs, _ = compute_supervised_loss(
        out,
        masks,
        main_loss_fn=BCEDiceLoss(),
        aux_loss_fn=BCEDiceLoss(),
        aux_weights=[0.4, 0.3, 0.2, 0.1],
        boundary_loss_fn=BCEDiceLoss(),
        boundary_weight=0.5,
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert "boundary_loss" in logs
    assert model.boundary.edge_head.weight.grad is not None
    assert model.boundary.edge_head.weight.grad.abs().sum() > 0
    assert model.seg_head[-1].weight.grad is not None
    assert model.seg_head[-1].weight.grad.abs().sum() > 0


def test_hsnet_paper_contract_dual_backbone_cross_semantic_hsc_and_multiscale_fusion():
    """HSNet contract: CNN/PVT branches are fused by CSA, HSC decoder, and trainable multi-scale prediction."""
    torch.manual_seed(17)
    model = HSNet(
        channels=_lite_res2_channels(),
        transformer_channels=_lite_pvt_channels(),
        decoder_channels=16,
        backbone_impl="lite",
        faithful_output=True,
        image_size=64,
    ).eval()
    assert all(isinstance(getattr(model, name), type(model.csa1)) for name in ["csa1", "csa2", "csa3", "csa4"])
    assert all(isinstance(getattr(model, name), HybridSemanticComplementaryModule) for name in ["hsc1", "hsc2", "hsc3", "hsc4"])
    assert isinstance(model.msp, MultiScalePredictionModule)

    with torch.no_grad():
        out = model(torch.randn(1, 3, 64, 64))
    parsed = parse_model_output(out)
    assert parsed.main.shape == (1, 1, 64, 64)
    assert len(parsed.aux) == 4
    assert len(out["stage_logits"]) == 5
    assert torch.allclose(out["msp_weights"].sum(), torch.tensor(1.0), atol=1e-6)
    assert not torch.allclose(parsed.main, out["stage_logits"][-1]), "MSP main output should be fused/refined, not only the shallow stage."
    assert all(_finite_tensor(t) for t in [parsed.main, *parsed.aux, *out["stage_logits"]])


def test_csca_unet_paper_contract_six_stage_dse_csca_decoder_and_all_deep_outputs_train():
    """CSCA U-Net contract: six stages, DSE bottleneck, CSCA decoder, and five deep-supervision maps."""
    torch.manual_seed(18)
    model = CSCAUNet(
        channels=(4, 8, 16, 32, 64, 128),
        deep_supervision=True,
        faithful_output=True,
        attention_mode="efficient",
    ).train()
    assert isinstance(model.down_conv6[-1], DoubleSqueezeExcitation)
    assert all(hasattr(model, name) for name in ["dp6", "dp5", "dp4", "dp3", "dp2", "out"])

    x = torch.randn(2, 3, 64, 64)
    masks = _seg_mask(batch=2, image_size=64)
    out = model(x)
    parsed = parse_model_output(out)
    assert parsed.main.shape == masks.shape
    assert len(parsed.aux) == 5

    loss, logs, _ = compute_supervised_loss(
        out,
        masks,
        main_loss_fn=BCEDiceLoss(),
        aux_loss_fn=BCEDiceLoss(),
        aux_weights=[0.1, 0.2, 0.3, 0.4, 0.5],
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert logs["aux_weight"] == pytest.approx(0.3)
    for head in [model.dp6, model.dp5, model.dp4, model.dp3, model.dp2, model.out]:
        assert head.weight.grad is not None
        assert head.weight.grad.abs().sum() > 0


def test_resunetpp_paper_contract_residual_se_aspp_attention_decoder_and_backward():
    """ResUNet++ paper contract: residual encoder, SE gates, ASPP bridge, and attention-gated decoder."""
    torch.manual_seed(19)
    model = ResUNetPlusPlus(channels=(8, 16, 32, 64, 128)).train()
    assert hasattr(model, "aspp_bridge")
    assert hasattr(model.dec4, "attention")
    assert hasattr(model.dec4, "se")

    x = torch.randn(2, 3, 64, 64)
    masks = _seg_mask(batch=2, image_size=64)
    out = model(x)
    assert out.shape == masks.shape
    loss = BCEDiceLoss()(out, masks)
    loss.backward()
    assert torch.isfinite(loss)
    assert model.aspp_bridge.project.conv.weight.grad is not None
    assert model.aspp_bridge.project.conv.weight.grad.abs().sum() > 0
    assert model.dec4.attention.psi[0].weight.grad is not None
    assert model.dec4.attention.psi[0].weight.grad.abs().sum() > 0


@pytest.mark.parametrize(
    "module,inputs",
    [
        (
            CamouflageIdentificationModule(low_channels=4, guide_channels=8, out_channels=8).eval(),
            (torch.ones(1, 4, 8, 8), torch.ones(1, 8, 4, 4), torch.zeros(1, 1, 4, 4), torch.full((1, 1, 4, 4), 10.0)),
        ),
        (
            SimilarityAggregationModule(channels=8, max_tokens_hw=4).eval(),
            (torch.ones(1, 8, 8, 8), torch.zeros(1, 1, 8, 8), torch.full((1, 1, 8, 8), 8.0)),
        ),
    ],
)
def test_polyp_pvt_conditioning_modules_do_not_ignore_coarse_prediction(module, inputs):
    """CIM/SAM must be conditioned by the coarse prediction map, as in the Polyp-PVT refinement design."""
    with torch.no_grad():
        if isinstance(module, CamouflageIdentificationModule):
            low, guide, uncertain_logits, confident_logits = inputs
            out_uncertain = module(low, guide, uncertain_logits)
            out_confident = module(low, guide, confident_logits)
        else:
            feat, low_logits, high_logits = inputs
            out_uncertain = module(feat, low_logits)
            out_confident = module(feat, high_logits)

    assert out_uncertain.shape == out_confident.shape
    assert _finite_tensor(out_uncertain) and _finite_tensor(out_confident)
    assert not torch.allclose(out_uncertain, out_confident), f"{type(module).__name__} ignored the coarse prediction map."
