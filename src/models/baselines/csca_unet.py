from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..common.utils import init_weights
from ..registry import register_model


class CSCABasicBlock(nn.Module):
    """Residual two-convolution block used by CSCA U-Net."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv1 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        y = self.conv1(x)
        y = self.conv2(y)
        return y + x


class DoubleSqueezeExcitation(nn.Module):
    """Double squeeze-and-excitation block from CSCA U-Net."""

    def __init__(self, channels: int, decay: int = 2) -> None:
        super().__init__()
        hidden = max(channels // decay, 1)
        self.avg_gate = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.max_gate = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * self.avg_gate(self.avg_pool(x))
        return x * self.max_gate(self.max_pool(x))


class CSCASpatialAttention(nn.Module):
    """Spatial part of the CSCA decoder attention.

    ``attention_mode='paper'`` mirrors the public implementation style, where
    the attention tensor and value tensor are multiplied with ``torch.matmul``
    over the last two dimensions. This is closest to the released code but can
    be slow at 352x352.

    ``attention_mode='efficient'`` keeps the same Q/K/V branches but replaces
    the large spatial matrix multiplication by element-wise gated fusion. This
    is more robust for repeated benchmark runs.
    """

    def __init__(self, channels: int, decay: int = 2, attention_mode: Literal["paper", "efficient"] = "efficient") -> None:
        super().__init__()
        hidden = max(channels // decay, 1)
        self.attention_mode = attention_mode
        self.q = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.Conv2d(hidden, 1, kernel_size=1),
            nn.Sigmoid(),
        )
        self.k = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, bias=False),
            DoubleSqueezeExcitation(hidden, decay=decay),
        )
        self.v = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, bias=False),
            DoubleSqueezeExcitation(hidden, decay=decay),
        )
        self.out = nn.Sequential(
            nn.Conv2d(hidden, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, low: torch.Tensor, high: torch.Tensor) -> torch.Tensor:
        q = self.q(low)
        k = self.k(low)
        v = self.v(high)
        att = q * k
        if self.attention_mode == "paper":
            att = torch.matmul(att, v)
        elif self.attention_mode == "efficient":
            att = att * v
        else:
            raise ValueError(f"Unsupported attention_mode: {self.attention_mode}")
        return self.out(att)


class CSCADecoderBlock(nn.Module):
    """CSCA decoder block with transposed upsampling, DSE, and spatial attention."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        decay: int = 2,
        attention_mode: Literal["paper", "efficient"] = "efficient",
    ) -> None:
        super().__init__()
        if out_channels % 2 != 0:
            raise ValueError("CSCADecoderBlock expects an even out_channels value.")
        mid_channels = out_channels // 2
        self.upsample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.fuse = CSCABasicBlock(in_channels=out_channels * 2, out_channels=mid_channels)
        self.channel_att = DoubleSqueezeExcitation(mid_channels, decay=decay)
        self.spatial_att = CSCASpatialAttention(mid_channels, decay=decay, attention_mode=attention_mode)

    def forward(self, high: torch.Tensor, low: torch.Tensor) -> torch.Tensor:
        up = self.upsample(high)
        if up.shape[-2:] != low.shape[-2:]:
            up = F.interpolate(up, size=low.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([up, low], dim=1)
        point = self.fuse(x)
        channel_feat = self.channel_att(point)
        spatial_feat = self.spatial_att(point, channel_feat)
        fused = channel_feat * spatial_feat
        return torch.cat([fused, channel_feat], dim=1)


@register_model("csca_unet")
class CSCAUNet(nn.Module):
    """CSCA U-Net baseline adapted to the benchmark registry.

    The implementation follows the released CSCA-U-Net architecture: six encoder
    levels, a DSE-enhanced bottleneck, CSCA decoder blocks, cross-layer feature
    fusion, and optional deep supervision.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1,
        channels: tuple[int, int, int, int, int, int] = (32, 64, 128, 256, 512, 1024),
        decay: int = 2,
        deep_supervision: bool = False,
        faithful_output: bool = False,
        attention_mode: Literal["paper", "efficient"] = "efficient",
        logit_clip: float | None = None,
        **_: object,
    ) -> None:
        super().__init__()
        if len(channels) != 6:
            raise ValueError("CSCAUNet expects six channel values, e.g. (32, 64, 128, 256, 512, 1024).")
        c1, c2, c3, c4, c5, c6 = channels
        self.deep_supervision = bool(deep_supervision)
        self.faithful_output = bool(faithful_output)
        self.logit_clip = None if logit_clip is None else float(logit_clip)
        self.pool = nn.MaxPool2d(2)

        self.down_conv1 = CSCABasicBlock(in_channels, c1)
        self.down_conv2 = CSCABasicBlock(c1, c2)
        self.down_conv3 = CSCABasicBlock(c2, c3)
        self.down_conv4 = CSCABasicBlock(c3, c4)
        self.down_conv5 = CSCABasicBlock(c4, c5)
        self.down_conv6 = nn.Sequential(CSCABasicBlock(c5, c6), DoubleSqueezeExcitation(c6, decay=decay))

        self.up_conv5 = CSCADecoderBlock(c6, c5, decay=decay, attention_mode=attention_mode)
        self.up_conv4 = CSCADecoderBlock(c5, c4, decay=decay, attention_mode=attention_mode)
        self.up_conv3 = CSCADecoderBlock(c4, c3, decay=decay, attention_mode=attention_mode)
        self.up_conv2 = CSCADecoderBlock(c3, c2, decay=decay, attention_mode=attention_mode)
        self.up_conv1 = CSCADecoderBlock(c2, c1, decay=decay, attention_mode=attention_mode)

        self.dp6 = nn.Conv2d(c6, num_classes, kernel_size=1)
        self.dp5 = nn.Conv2d(c5, num_classes, kernel_size=1)
        self.dp4 = nn.Conv2d(c4, num_classes, kernel_size=1)
        self.dp3 = nn.Conv2d(c3, num_classes, kernel_size=1)
        self.dp2 = nn.Conv2d(c2, num_classes, kernel_size=1)
        self.out = nn.Conv2d(c1, num_classes, kernel_size=3, padding=1)

        self.center5 = nn.Conv2d(c6, c5, kernel_size=1)
        self.decodeup4 = nn.Conv2d(c5, c4, kernel_size=1)
        self.decodeup3 = nn.Conv2d(c4, c3, kernel_size=1)
        self.decodeup2 = nn.Conv2d(c3, c2, kernel_size=1)

        init_weights(self)

    @staticmethod
    def _resize_like(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        return F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)

    @staticmethod
    def _resize_to_input(x: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        return F.interpolate(x, size=input_tensor.shape[-2:], mode="bilinear", align_corners=False)

    def _stabilize_logits(self, x: torch.Tensor) -> torch.Tensor:
        # BCEWithLogitsLoss is stable, but CSCA's multiplicative attention can
        # occasionally create very large logits on tiny splits. Optional clipping
        # keeps the BCE scale meaningful without changing the raw-logit API.
        if self.logit_clip is None or self.logit_clip <= 0:
            return x
        return torch.clamp(x, min=-self.logit_clip, max=self.logit_clip)

    def forward(self, inputs: torch.Tensor):
        down1 = self.down_conv1(inputs)
        down2 = self.down_conv2(self.pool(down1))
        down3 = self.down_conv3(self.pool(down2))
        down4 = self.down_conv4(self.pool(down3))
        down5 = self.down_conv5(self.pool(down4))
        center = self.down_conv6(self.pool(down5))

        out6 = self._stabilize_logits(self._resize_to_input(self.dp6(center), inputs))

        deco5 = self.up_conv5(center, down5)
        out5 = self._stabilize_logits(self._resize_to_input(self.dp5(deco5), inputs))
        deco5 = deco5 + self._resize_like(self.center5(center), deco5)

        deco4 = self.up_conv4(deco5, down4)
        out4 = self._stabilize_logits(self._resize_to_input(self.dp4(deco4), inputs))
        deco4 = deco4 + self._resize_like(self.decodeup4(deco5), deco4)

        deco3 = self.up_conv3(deco4, down3)
        out3 = self._stabilize_logits(self._resize_to_input(self.dp3(deco3), inputs))
        deco3 = deco3 + self._resize_like(self.decodeup3(deco4), deco3)

        deco2 = self.up_conv2(deco3, down2)
        out2 = self._stabilize_logits(self._resize_to_input(self.dp2(deco2), inputs))
        deco2 = deco2 + self._resize_like(self.decodeup2(deco3), deco2)

        deco1 = self.up_conv1(deco2, down1)
        out = self._stabilize_logits(self.out(deco1))

        if self.deep_supervision and self.faithful_output:
            return {"main": out, "aux": [out2, out3, out4, out5, out6]}
        if self.deep_supervision:
            return [out2, out3, out4, out5, out6, out]
        return out


CSCAUNetLite = CSCAUNet
