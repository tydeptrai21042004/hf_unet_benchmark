from __future__ import annotations

import torch
import torch.nn as nn

from ..common.paper_baselines import AdaptiveSelectionModule, GlobalContextModule, LocalContextAttention, Res2NetLikeEncoder
from ..common.paper_baselines import BasicConv2d
from ..common.utils import init_weights, resize_to
from ..registry import register_model


@register_model("acsnet")
class ACSNet(nn.Module):
    """ACSNet with Local Context Attention, Global Context Module, and Adaptive Selection Modules."""

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
            raise ValueError("ACSNet currently supports binary segmentation only.")
        if len(channels) != 5:
            raise ValueError("ACSNet expects exactly five encoder stages.")
        c0, c1, c2, c3, c4 = channels
        self.encoder = Res2NetLikeEncoder(in_channels=in_channels, channels=channels)
        self.context_seed = BasicConv2d(c4, c3, 3, padding=1)
        self.gcm = GlobalContextModule(c4, decoder_channels=(c3, c2, c1, c0))
        self.lca3 = LocalContextAttention(c3)
        self.lca2 = LocalContextAttention(c2)
        self.lca1 = LocalContextAttention(c1)
        self.lca0 = LocalContextAttention(c0)
        self.asm3 = AdaptiveSelectionModule(c3, c3, c3, c3)
        self.asm2 = AdaptiveSelectionModule(c2, c3, c2, c2)
        self.asm1 = AdaptiveSelectionModule(c1, c2, c1, c1)
        self.asm0 = AdaptiveSelectionModule(c0, c1, c0, c0)
        self.pred3 = nn.Conv2d(c3, 1, 1)
        self.pred2 = nn.Conv2d(c2, 1, 1)
        self.pred1 = nn.Conv2d(c1, 1, 1)
        self.pred0 = nn.Conv2d(c0, 1, 1)
        init_weights(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0, x1, x2, x3, x4 = self.encoder(x)
        g3, g2, g1, g0 = self.gcm(x4, refs=(x3, x2, x1, x0))
        d3 = self.context_seed(x4)
        p3 = self.pred3(resize_to(d3, x3))
        d3 = self.asm3(self.lca3(x3, p3), d3, g3)

        p2 = self.pred3(resize_to(d3, x2))
        d2 = self.asm2(self.lca2(x2, p2), d3, g2)

        p1 = self.pred2(resize_to(d2, x1))
        d1 = self.asm1(self.lca1(x1, p1), d2, g1)

        p0 = self.pred1(resize_to(d1, x0))
        d0 = self.asm0(self.lca0(x0, p0), d1, g0)
        return resize_to(self.pred0(d0), x)


ACSNetLite = ACSNet
