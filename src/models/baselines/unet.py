from __future__ import annotations

import torch
import torch.nn as nn

from ..common.decoder import UNetDecoder
from ..common.encoder import PyramidEncoder
from ..common.utils import init_weights
from ..registry import register_model


@register_model("unet")
class UNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1,
        channels: tuple[int, ...] = (32, 64, 128, 256, 512),
        norm: str = "bn",
        act: str = "relu",
    ) -> None:
        super().__init__()
        self.encoder = PyramidEncoder(in_channels=in_channels, channels=channels, block="double", norm=norm, act=act)
        self.decoder = UNetDecoder(channels=channels, norm=norm, act=act)
        self.seg_head = nn.Conv2d(channels[0], num_classes, kernel_size=1)
        init_weights(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.encoder(x)
        dec = self.decoder(feats)
        return self.seg_head(dec)
