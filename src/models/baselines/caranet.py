from __future__ import annotations

import torch
import torch.nn as nn

from ..common.paper_baselines import AxialReverseAttention, CFPModule, Res2NetLikeEncoder, RFBModified, DenseAggregation
from ..common.utils import init_weights, resize_to
from ..registry import register_model


@register_model("caranet")
class CaraNet(nn.Module):
    """CaraNet with CFP context modeling and axial reverse attention refinement."""

    def __init__(self, in_channels: int = 3, num_classes: int = 1, channels: tuple[int, ...] = (32, 64, 128, 256, 512), norm: str = "bn", act: str = "relu") -> None:
        super().__init__()
        if num_classes != 1:
            raise ValueError("CaraNet currently supports binary segmentation only.")
        if len(channels) != 5:
            raise ValueError("CaraNet expects exactly five encoder stages.")
        agg_channels = max(channels[0], 16)
        self.encoder = Res2NetLikeEncoder(in_channels=in_channels, channels=channels)
        self.cfp = CFPModule(channels[4], dilation=8)
        self.rfb2 = RFBModified(channels[2], agg_channels)
        self.rfb3 = RFBModified(channels[3], agg_channels)
        self.rfb4 = RFBModified(channels[4], agg_channels)
        self.agg = DenseAggregation(agg_channels)
        self.ara4 = AxialReverseAttention(channels[4], hidden_channels=max(channels[4] // 2, 32))
        self.ara3 = AxialReverseAttention(channels[3], hidden_channels=max(channels[3] // 2, 32))
        self.ara2 = AxialReverseAttention(channels[2], hidden_channels=max(channels[2] // 2, 32))
        init_weights(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, x2, x3, x4 = self.encoder(x)
        x4 = self.cfp(x4)
        coarse = self.agg(self.rfb4(x4), self.rfb3(x3), self.rfb2(x2))
        y4 = self.ara4(x4, resize_to(coarse, x4))
        y3 = self.ara3(x3, resize_to(y4, x3))
        y2 = self.ara2(x2, resize_to(y3, x2))
        return resize_to(y2, x)


CaraNetLite = CaraNet
