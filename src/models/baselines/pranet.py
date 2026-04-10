from __future__ import annotations

import torch
import torch.nn as nn

from ..common.blocks import ASPP, ConvNormAct, ReverseAttentionRefine
from ..common.encoder import PyramidEncoder
from ..common.utils import init_weights, resize_to
from ..registry import register_model


class PartialDecoder(nn.Module):
    def __init__(self, channels: tuple[int, int, int], out_channels: int = 64, norm: str = "bn", act: str = "relu") -> None:
        super().__init__()
        c2, c3, c4 = channels
        self.p4 = ConvNormAct(c4, out_channels, 1, padding=0, norm=norm, act=act)
        self.p3 = ConvNormAct(c3, out_channels, 1, padding=0, norm=norm, act=act)
        self.p2 = ConvNormAct(c2, out_channels, 1, padding=0, norm=norm, act=act)
        self.fuse = nn.Sequential(
            ConvNormAct(out_channels * 3, out_channels, 3, norm=norm, act=act),
            ConvNormAct(out_channels, out_channels, 3, norm=norm, act=act),
        )
        self.head = nn.Conv2d(out_channels, 1, 1)

    def forward(self, x2: torch.Tensor, x3: torch.Tensor, x4: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        p2 = self.p2(x2)
        p3 = resize_to(self.p3(x3), p2)
        p4 = resize_to(self.p4(x4), p2)
        fused = self.fuse(torch.cat([p2, p3, p4], dim=1))
        return self.head(fused), fused


@register_model("pranet")
class PraNetLite(nn.Module):
    """A benchmark-friendly simplified PraNet-style model.

    This is not a paper-faithful reproduction. It keeps the key ideas:
    multi-scale aggregation + reverse-attention refinement.
    """

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
            raise ValueError("PraNetLite currently supports binary segmentation only.")
        self.encoder = PyramidEncoder(in_channels=in_channels, channels=channels, block="res", norm=norm, act=act)
        self.context = ASPP(channels[-1], channels[-1], norm=norm, act=act)
        self.partial_decoder = PartialDecoder((channels[2], channels[3], channels[4]), out_channels=64, norm=norm, act=act)
        self.ra4 = ReverseAttentionRefine(channels[4], out_channels=1, norm=norm, act=act)
        self.ra3 = ReverseAttentionRefine(channels[3], out_channels=1, norm=norm, act=act)
        self.ra2 = ReverseAttentionRefine(channels[2], out_channels=1, norm=norm, act=act)
        init_weights(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2, x3, x4, x5 = self.encoder(x)
        x5 = self.context(x5)
        coarse, _ = self.partial_decoder(x3, x4, x5)
        logit4 = self.ra4(x5, coarse)
        logit3 = self.ra3(x4, logit4)
        logit2 = self.ra2(x3, logit3)
        return resize_to(logit2, x)
