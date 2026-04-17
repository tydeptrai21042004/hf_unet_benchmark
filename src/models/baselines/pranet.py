from __future__ import annotations

import torch
import torch.nn as nn

from ..common.paper_baselines import DenseAggregation, RFBModified, Res2NetLikeEncoder, ReverseAttentionBranch
from ..common.utils import init_weights, resize_to
from ..registry import register_model


@register_model("pranet")
class PraNet(nn.Module):
    """PraNet with RFB reduction, dense partial decoder, and three reverse-attention stages."""

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1,
        channels: tuple[int, ...] = (32, 64, 128, 256, 512),
        norm: str = "bn",
        act: str = "relu",
    ) -> None:
        super().__init__()
        if num_classes != 1:
            raise ValueError("PraNet currently supports binary segmentation only.")
        if len(channels) != 5:
            raise ValueError("PraNet expects exactly five encoder stages.")
        self.encoder = Res2NetLikeEncoder(in_channels=in_channels, channels=channels)
        agg_channels = max(channels[0], 16)
        self.rfb2_1 = RFBModified(channels[2], agg_channels)
        self.rfb3_1 = RFBModified(channels[3], agg_channels)
        self.rfb4_1 = RFBModified(channels[4], agg_channels)
        self.agg1 = DenseAggregation(agg_channels)
        self.ra4 = ReverseAttentionBranch(channels[4], mid_channels=max(channels[4] // 2, 32), depth=4, kernel_size=5)
        self.ra3 = ReverseAttentionBranch(channels[3], mid_channels=max(channels[3] // 4, 32), depth=3, kernel_size=3)
        self.ra2 = ReverseAttentionBranch(channels[2], mid_channels=max(channels[2] // 2, 32), depth=3, kernel_size=3)
        init_weights(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, x2, x3, x4 = self.encoder(x)
        x2_rfb = self.rfb2_1(x2)
        x3_rfb = self.rfb3_1(x3)
        x4_rfb = self.rfb4_1(x4)
        ra5_feat = self.agg1(x4_rfb, x3_rfb, x2_rfb)
        crop_4 = resize_to(ra5_feat, x4)
        x4_refine = self.ra4(x4, crop_4)
        crop_3 = resize_to(x4_refine, x3)
        x3_refine = self.ra3(x3, crop_3)
        crop_2 = resize_to(x3_refine, x2)
        x2_refine = self.ra2(x2, crop_2)
        return resize_to(x2_refine, x)


PraNetLite = PraNet
