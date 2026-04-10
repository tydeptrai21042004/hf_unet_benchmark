from __future__ import annotations

import torch
import torch.nn as nn

from ..common.blocks import UpBlock
from ..common.encoder import PyramidEncoder
from ..common.utils import init_weights
from ..registry import register_model
from .hf_bottleneck import HFBottleneck
from .hf_regularizer import HFRegularizer


@register_model("proposal_hf_unet")
class HFUNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1,
        channels: tuple[int, ...] = (32, 64, 128, 256, 512),
        hf_alpha: float = 0.5,
        hf_expansion: float = 1.5,
        hf_dropout: float = 0.0,
        use_hf_regularizer: bool = True,
        norm: str = "bn",
        act: str = "relu",
    ) -> None:
        super().__init__()
        self.encoder = PyramidEncoder(in_channels=in_channels, channels=channels, block="double", norm=norm, act=act)
        self.hf_bottleneck = HFBottleneck(channels[-1], expansion=hf_expansion, alpha=hf_alpha, dropout=hf_dropout)
        self.up4 = UpBlock(channels[4], channels[3], channels[3], norm=norm, act=act, use_cbam=True)
        self.up3 = UpBlock(channels[3], channels[2], channels[2], norm=norm, act=act, use_cbam=True)
        self.up2 = UpBlock(channels[2], channels[1], channels[1], norm=norm, act=act, use_cbam=True)
        self.up1 = UpBlock(channels[1], channels[0], channels[0], norm=norm, act=act, use_cbam=True)
        self.head = nn.Conv2d(channels[0], num_classes, 1)
        self.regularizer = HFRegularizer() if use_hf_regularizer else None
        init_weights(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0, x1, x2, x3, x4 = self.encoder(x)
        x4 = self.hf_bottleneck(x4)
        d3 = self.up4(x4, x3)
        d2 = self.up3(d3, x2)
        d1 = self.up2(d2, x1)
        d0 = self.up1(d1, x0)
        return self.head(d0)

    def auxiliary_regularization(self) -> torch.Tensor:
        if self.regularizer is None:
            device = next(self.parameters()).device
            return torch.zeros((), device=device)
        return self.regularizer.from_module(self.hf_bottleneck)
