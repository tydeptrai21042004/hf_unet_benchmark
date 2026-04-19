from __future__ import annotations

import torch
import torch.nn as nn

from ..common.official_backbones import OfficialRes2NetEncoder
from ..common.paper_baselines import AxialReverseAttention, CFPModule, Res2NetLikeEncoder, RFBModified, DenseAggregation
from ..common.utils import resize_to
from ..registry import register_model


@register_model("caranet")
class CaraNet(nn.Module):
    """CaraNet-style baseline with optional official Res2Net backbone."""

    def __init__(self, in_channels: int = 3, num_classes: int = 1, channels: tuple[int, ...] = (32, 64, 128, 256, 512), faithful_output: bool = False, norm: str = "bn", act: str = "relu", backbone_impl: str = "official", res2net_variant: str = "res2net50_v1b_26w_4s", backbone_pretrained: bool = False, backbone_checkpoint: str | None = None, backbone_checkpoint_url: str | None = None) -> None:
        super().__init__()
        if num_classes != 1:
            raise ValueError("CaraNet currently supports binary segmentation only.")
        if len(channels) != 5:
            raise ValueError("CaraNet expects exactly five encoder stages.")
        agg_channels = max(channels[0], 16)
        self.faithful_output = faithful_output
        if backbone_impl.lower() in {"official", "official_backbone"}:
            self.encoder = OfficialRes2NetEncoder(in_channels=in_channels, channels=channels, variant=res2net_variant, pretrained=backbone_pretrained, checkpoint=backbone_checkpoint, checkpoint_url=backbone_checkpoint_url)
        else:
            self.encoder = Res2NetLikeEncoder(in_channels=in_channels, channels=channels)
        self.cfp = CFPModule(channels[4], dilation=8)
        self.rfb2 = RFBModified(channels[2], agg_channels)
        self.rfb3 = RFBModified(channels[3], agg_channels)
        self.rfb4 = RFBModified(channels[4], agg_channels)
        self.agg = DenseAggregation(agg_channels)
        self.ara4 = AxialReverseAttention(channels[4], hidden_channels=max(channels[4] // 2, 32))
        self.ara3 = AxialReverseAttention(channels[3], hidden_channels=max(channels[3] // 2, 32))
        self.ara2 = AxialReverseAttention(channels[2], hidden_channels=max(channels[2] // 2, 32))

    def forward(self, x: torch.Tensor):
        _, _, x2, x3, x4 = self.encoder(x)
        x4 = self.cfp(x4)
        coarse = self.agg(self.rfb4(x4), self.rfb3(x3), self.rfb2(x2))
        y4 = self.ara4(x4, resize_to(coarse, x4))
        y3 = self.ara3(x3, resize_to(y4, x3))
        y2 = self.ara2(x2, resize_to(y3, x2))
        lateral_map_5 = resize_to(coarse, x)
        lateral_map_4 = resize_to(y4, x)
        lateral_map_3 = resize_to(y3, x)
        lateral_map_2 = resize_to(y2, x)
        if self.faithful_output:
            return {"main": lateral_map_2, "aux": [lateral_map_5, lateral_map_4, lateral_map_3]}
        return lateral_map_2


CaraNetLite = CaraNet
