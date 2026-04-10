from __future__ import annotations

import torch
import torch.nn as nn

from ..common.blocks import DoubleConv
from ..common.encoder import PyramidEncoder
from ..common.utils import init_weights, resize_to
from ..registry import register_model


@register_model("unetpp")
class UNetPlusPlus(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1,
        channels: tuple[int, ...] = (32, 64, 128, 256, 512),
        deep_supervision: bool = False,
        norm: str = "bn",
        act: str = "relu",
    ) -> None:
        super().__init__()
        self.deep_supervision = deep_supervision
        self.encoder = PyramidEncoder(in_channels=in_channels, channels=channels, block="double", norm=norm, act=act)

        c0, c1, c2, c3, c4 = channels
        self.x01 = DoubleConv(c0 + c1, c0, norm=norm, act=act)
        self.x11 = DoubleConv(c1 + c2, c1, norm=norm, act=act)
        self.x21 = DoubleConv(c2 + c3, c2, norm=norm, act=act)
        self.x31 = DoubleConv(c3 + c4, c3, norm=norm, act=act)

        self.x02 = DoubleConv(c0 * 2 + c1, c0, norm=norm, act=act)
        self.x12 = DoubleConv(c1 * 2 + c2, c1, norm=norm, act=act)
        self.x22 = DoubleConv(c2 * 2 + c3, c2, norm=norm, act=act)

        self.x03 = DoubleConv(c0 * 3 + c1, c0, norm=norm, act=act)
        self.x13 = DoubleConv(c1 * 3 + c2, c1, norm=norm, act=act)

        self.x04 = DoubleConv(c0 * 4 + c1, c0, norm=norm, act=act)

        self.head1 = nn.Conv2d(c0, num_classes, 1)
        self.head2 = nn.Conv2d(c0, num_classes, 1)
        self.head3 = nn.Conv2d(c0, num_classes, 1)
        self.head4 = nn.Conv2d(c0, num_classes, 1)
        init_weights(self)

    def _upcat(self, left: torch.Tensor, right: torch.Tensor, *extra: torch.Tensor) -> torch.Tensor:
        xs = [left]
        for t in extra:
            xs.append(t)
        xs.append(resize_to(right, left))
        return torch.cat(xs, dim=1)

    def forward(self, x: torch.Tensor):
        x00, x10, x20, x30, x40 = self.encoder(x)
        x01 = self.x01(self._upcat(x00, x10))
        x11 = self.x11(self._upcat(x10, x20))
        x21 = self.x21(self._upcat(x20, x30))
        x31 = self.x31(self._upcat(x30, x40))

        x02 = self.x02(self._upcat(x00, x11, x01))
        x12 = self.x12(self._upcat(x10, x21, x11))
        x22 = self.x22(self._upcat(x20, x31, x21))

        x03 = self.x03(self._upcat(x00, x12, x01, x02))
        x13 = self.x13(self._upcat(x10, x22, x11, x12))

        x04 = self.x04(self._upcat(x00, x13, x01, x02, x03))

        if self.deep_supervision:
            return [self.head1(x01), self.head2(x02), self.head3(x03), self.head4(x04)]
        return self.head4(x04)
