from __future__ import annotations

import torch
import torch.nn as nn

from ..common.official_backbones import OfficialPVTv2Backbone, OfficialRes2NetEncoder
from ..common.paper_baselines import BasicConv2d, CrossSemanticAttention, HybridSemanticComplementaryModule, MultiScalePredictionModule, PVTLikeBackbone, Res2NetLikeEncoder
from ..common.utils import resize_to
from ..registry import register_model


@register_model("hsnet")
class HSNet(nn.Module):
    """HSNet baseline with optional official Res2Net + PVTv2 backbones."""

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1,
        channels: tuple[int, ...] = (32, 64, 128, 256, 512),
        transformer_channels: tuple[int, ...] | None = None,
        decoder_channels: int = 64,
        faithful_output: bool = False,
        norm: str = "bn",
        act: str = "relu",
        backbone_impl: str = "official",
        res2net_variant: str = "res2net50_v1b_26w_4s",
        pvt_variant: str = "pvt_v2_b2",
        backbone_pretrained: bool = False,
        cnn_backbone_checkpoint: str | None = None,
        transformer_backbone_checkpoint: str | None = None,
        cnn_backbone_checkpoint_url: str | None = None,
        transformer_backbone_checkpoint_url: str | None = None,
        image_size: int = 352,
    ) -> None:
        super().__init__()
        if num_classes != 1:
            raise ValueError("HSNet currently supports binary segmentation only.")
        if len(channels) != 5:
            raise ValueError("HSNet expects five CNN encoder stages.")
        c0, c1, c2, c3, c4 = channels
        t1, t2, t3, t4 = transformer_channels or channels[:4]
        self.faithful_output = faithful_output

        if backbone_impl.lower() in {"official", "official_backbone"}:
            self.cnn_encoder = OfficialRes2NetEncoder(in_channels=in_channels, channels=channels, variant=res2net_variant, pretrained=backbone_pretrained, checkpoint=cnn_backbone_checkpoint, checkpoint_url=cnn_backbone_checkpoint_url)
            self.transformer_encoder = OfficialPVTv2Backbone(in_channels=in_channels, embed_dims=(t1, t2, t3, t4), variant=pvt_variant, pretrained=backbone_pretrained, checkpoint=transformer_backbone_checkpoint, checkpoint_url=transformer_backbone_checkpoint_url, image_size=image_size)
        else:
            self.cnn_encoder = Res2NetLikeEncoder(in_channels=in_channels, channels=channels)
            self.transformer_encoder = PVTLikeBackbone(in_channels=in_channels, embed_dims=(t1, t2, t3, t4))

        self.csa1 = CrossSemanticAttention(c1, t1, decoder_channels)
        self.csa2 = CrossSemanticAttention(c2, t2, decoder_channels)
        self.csa3 = CrossSemanticAttention(c3, t3, decoder_channels)
        self.csa4 = CrossSemanticAttention(c4, t4, decoder_channels)
        self.hsc4 = HybridSemanticComplementaryModule(decoder_channels, t4, 0, decoder_channels)
        self.hsc3 = HybridSemanticComplementaryModule(decoder_channels, t3, decoder_channels, decoder_channels)
        self.hsc2 = HybridSemanticComplementaryModule(decoder_channels, t2, decoder_channels, decoder_channels)
        self.hsc1 = HybridSemanticComplementaryModule(decoder_channels, t1, decoder_channels, decoder_channels)
        self.shallow = nn.Sequential(
            BasicConv2d(c0, decoder_channels, 3, padding=1),
            BasicConv2d(decoder_channels, decoder_channels, 3, padding=1),
        )
        self.fuse0 = BasicConv2d(decoder_channels * 2, decoder_channels, 3, padding=1)
        self.pred4 = nn.Conv2d(decoder_channels, 1, 1)
        self.pred3 = nn.Conv2d(decoder_channels, 1, 1)
        self.pred2 = nn.Conv2d(decoder_channels, 1, 1)
        self.pred1 = nn.Conv2d(decoder_channels, 1, 1)
        self.pred0 = nn.Conv2d(decoder_channels, 1, 1)
        self.msp = MultiScalePredictionModule(num_scales=5, refine=True)

    def forward(self, x: torch.Tensor):
        x0, x1, x2, x3, x4 = self.cnn_encoder(x)
        t1, t2, t3, t4 = self.transformer_encoder(x)
        s4 = self.csa4(x4, t4)
        s3 = self.csa3(x3, t3)
        s2 = self.csa2(x2, t2)
        s1 = self.csa1(x1, t1)
        d4 = self.hsc4(s4, t4)
        d3 = self.hsc3(s3, t3, resize_to(d4, s3))
        d2 = self.hsc2(s2, t2, resize_to(d3, s2))
        d1 = self.hsc1(s1, t1, resize_to(d2, s1))
        d0 = self.fuse0(torch.cat([self.shallow(x0), resize_to(d1, x0)], dim=1))
        p4 = resize_to(self.pred4(d4), x)
        p3 = resize_to(self.pred3(d3), x)
        p2 = resize_to(self.pred2(d2), x)
        p1 = resize_to(self.pred1(d1), x)
        p0 = resize_to(self.pred0(d0), x)
        fused, weights = self.msp([p4, p3, p2, p1, p0], x)
        if self.faithful_output:
            return {"main": fused, "aux": [p4, p3, p2, p1], "msp_weights": weights.detach(), "stage_logits": [p4, p3, p2, p1, p0]}
        return fused


HSNetLite = HSNet
