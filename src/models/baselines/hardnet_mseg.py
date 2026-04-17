from __future__ import annotations

import torch
import torch.nn as nn

from ..common.paper_baselines import DenseAggregation, HarDNetLikeEncoder, RFBModified
from ..common.utils import init_weights, resize_to
from ..registry import register_model


@register_model("hardnet_mseg")
class HarDNetMSEG(nn.Module):
    """HarDNet-MSEG with HarDNet-style encoder and cascaded partial decoder."""

    def __init__(self, in_channels: int = 3, num_classes: int = 1, base_channels: int = 32, norm: str = "bn", act: str = "relu") -> None:
        super().__init__()
        if num_classes != 1:
            raise ValueError("HarDNet-MSEG currently supports binary segmentation only.")
        self.encoder = HarDNetLikeEncoder(in_channels=in_channels, base_channels=base_channels)
        # The transition layers of HarDNetLikeEncoder are deterministic from base_channels.
        ch2 = base_channels * 4
        ch3 = base_channels * 8
        ch4 = base_channels * 16
        agg_channels = max(base_channels, 16)
        self.rfb2 = RFBModified(ch2, agg_channels)
        self.rfb3 = RFBModified(ch3, agg_channels)
        self.rfb4 = RFBModified(ch4, agg_channels)
        self.decoder = DenseAggregation(agg_channels)
        init_weights(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, x2, x3, x4 = self.encoder(x)
        logits = self.decoder(self.rfb4(x4), self.rfb3(x3), self.rfb2(x2))
        return resize_to(logits, x)


HarDNetMSEGLite = HarDNetMSEG
