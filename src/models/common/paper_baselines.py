from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import CBAM, ConvNormAct, DoubleConv, SqueezeExcitation
from .utils import ensure_tuple_channels, init_weights, resize_to


class BasicConv2d(nn.Module):
    """Conv-BN with optional ReLU, matching the style used by several official baselines."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        relu: bool = True,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True) if relu else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class RFBModified(nn.Module):
    """Receptive Field Block used by PraNet and HarDNet-MSEG decoders."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.branch0 = nn.Sequential(BasicConv2d(in_channels, out_channels, 1))
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channels, out_channels, 1),
            BasicConv2d(out_channels, out_channels, (1, 3), padding=(0, 1)),
            BasicConv2d(out_channels, out_channels, (3, 1), padding=(1, 0)),
            BasicConv2d(out_channels, out_channels, 3, padding=3, dilation=3),
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels, out_channels, 1),
            BasicConv2d(out_channels, out_channels, (1, 5), padding=(0, 2)),
            BasicConv2d(out_channels, out_channels, (5, 1), padding=(2, 0)),
            BasicConv2d(out_channels, out_channels, 3, padding=5, dilation=5),
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channels, out_channels, 1),
            BasicConv2d(out_channels, out_channels, (1, 7), padding=(0, 3)),
            BasicConv2d(out_channels, out_channels, (7, 1), padding=(3, 0)),
            BasicConv2d(out_channels, out_channels, 3, padding=7, dilation=7),
        )
        self.conv_cat = BasicConv2d(4 * out_channels, out_channels, 3, padding=1)
        self.conv_res = BasicConv2d(in_channels, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), dim=1))
        return self.relu(x_cat + self.conv_res(x))


class DenseAggregation(nn.Module):
    """PraNet/HarDNet-MSEG dense aggregation block."""

    def __init__(self, channel: int) -> None:
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample3 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample4 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample5 = BasicConv2d(2 * channel, 2 * channel, 3, padding=1)
        self.conv_concat2 = BasicConv2d(2 * channel, 2 * channel, 3, padding=1)
        self.conv_concat3 = BasicConv2d(3 * channel, 3 * channel, 3, padding=1)
        self.conv4 = BasicConv2d(3 * channel, 3 * channel, 3, padding=1)
        self.conv5 = nn.Conv2d(3 * channel, 1, 1)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, x3: torch.Tensor) -> torch.Tensor:
        x1_1 = x1
        x2_1 = self.conv_upsample1(self.upsample(x1)) * x2
        x3_1 = (
            self.conv_upsample2(self.upsample(self.upsample(x1)))
            * self.conv_upsample3(self.upsample(x2))
            * x3
        )
        x2_2 = torch.cat((x2_1, self.conv_upsample4(self.upsample(x1_1))), dim=1)
        x2_2 = self.conv_concat2(x2_2)
        x3_2 = torch.cat((x3_1, self.conv_upsample5(self.upsample(x2_2))), dim=1)
        x3_2 = self.conv_concat3(x3_2)
        x = self.conv4(x3_2)
        return self.conv5(x)


class ReverseAttentionBranch(nn.Module):
    def __init__(self, in_channels: int, mid_channels: int, depth: int = 3, kernel_size: int = 3) -> None:
        super().__init__()
        padding = kernel_size // 2
        layers = [BasicConv2d(in_channels, mid_channels, 1)]
        for _ in range(depth - 1):
            layers.append(BasicConv2d(mid_channels, mid_channels, kernel_size, padding=padding))
        self.pre = nn.Sequential(*layers)
        self.out = BasicConv2d(mid_channels, 1, kernel_size if kernel_size > 1 else 1, padding=padding if kernel_size > 1 else 0, relu=False)

    def forward(self, feat: torch.Tensor, coarse_logits: torch.Tensor) -> torch.Tensor:
        coarse = resize_to(coarse_logits, feat)
        reverse = 1.0 - torch.sigmoid(coarse)
        x = feat * reverse.expand(-1, feat.shape[1], -1, -1)
        x = self.pre(x)
        x = self.out(x)
        return x + coarse


class Bottle2neck(nn.Module):
    """A lightweight Res2Net-style bottleneck used to keep paper baselines closer to their original backbones."""

    def __init__(self, in_channels: int, out_channels: int, scale: int = 4) -> None:
        super().__init__()
        if scale < 2:
            raise ValueError("scale must be >= 2")
        width = max(out_channels // scale, 8)
        self.scale = scale
        self.width = width
        self.conv1 = BasicConv2d(in_channels, width * scale, 1, relu=True)
        self.convs = nn.ModuleList(
            [BasicConv2d(width, width, 3, stride=1, padding=1, relu=True) for _ in range(scale - 1)]
        )
        self.pool = nn.Identity()
        self.conv3 = BasicConv2d(width * scale, out_channels, 1, relu=False)
        self.shortcut = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        out = self.conv1(x)
        spx = torch.split(out, self.width, dim=1)
        outputs = []
        for i in range(self.scale - 1):
            if i == 0:
                sp = spx[i]
            else:
                sp = spx[i] + outputs[-1]
            outputs.append(self.convs[i](sp))
        outputs.append(self.pool(spx[-1]))
        out = torch.cat(outputs, dim=1)
        out = self.conv3(out)
        return self.relu(out + residual)


class Res2Stage(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, blocks: int = 2, stride: int = 1) -> None:
        super().__init__()
        self.down = nn.AvgPool2d(kernel_size=2, stride=2) if stride > 1 else nn.Identity()
        layers = [Bottle2neck(in_channels, out_channels)]
        for _ in range(blocks - 1):
            layers.append(Bottle2neck(out_channels, out_channels))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(self.down(x))


class Res2NetLikeEncoder(nn.Module):
    def __init__(self, in_channels: int = 3, channels: Sequence[int] = (32, 64, 128, 256, 512), blocks_per_stage: Sequence[int] = (2, 2, 2, 2)) -> None:
        super().__init__()
        channels = ensure_tuple_channels(channels)
        if len(channels) != 5:
            raise ValueError("Res2NetLikeEncoder expects five channel values.")
        c0, c1, c2, c3, c4 = channels
        self.stem = nn.Sequential(
            BasicConv2d(in_channels, max(c0 // 2, 16), 3, stride=2, padding=1),
            BasicConv2d(max(c0 // 2, 16), max(c0 // 2, 16), 3, padding=1),
            BasicConv2d(max(c0 // 2, 16), c0, 3, padding=1),
        )
        self.pool = nn.MaxPool2d(3, stride=2, padding=1)
        self.layer1 = Res2Stage(c0, c1, blocks=blocks_per_stage[0], stride=1)
        self.layer2 = Res2Stage(c1, c2, blocks=blocks_per_stage[1], stride=2)
        self.layer3 = Res2Stage(c2, c3, blocks=blocks_per_stage[2], stride=2)
        self.layer4 = Res2Stage(c3, c4, blocks=blocks_per_stage[3], stride=2)
        init_weights(self)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        x0 = self.stem(x)   # H/2
        x = self.pool(x0)   # H/4
        x1 = self.layer1(x) # H/4
        x2 = self.layer2(x1) # H/8
        x3 = self.layer3(x2) # H/16
        x4 = self.layer4(x3) # H/32
        return [x0, x1, x2, x3, x4]


class LocalContextAttention(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.refine = nn.Sequential(
            BasicConv2d(channels, channels, 3, padding=1),
            BasicConv2d(channels, channels, 3, padding=1),
        )

    def forward(self, feat: torch.Tensor, prev_logits: torch.Tensor) -> torch.Tensor:
        pred = resize_to(prev_logits, feat)
        prob = torch.sigmoid(pred)
        hard_region = 1.0 - (2.0 * torch.abs(prob - 0.5))
        return feat + self.refine(feat * hard_region)


class GlobalContextModule(nn.Module):
    def __init__(self, in_channels: int, decoder_channels: Sequence[int]) -> None:
        super().__init__()

        def _pool_proj(pool_size: int) -> nn.Module:
            return nn.Sequential(
                nn.AdaptiveAvgPool2d(pool_size),
                nn.Conv2d(in_channels, in_channels // 2, 1, bias=False),
                nn.ReLU(inplace=True),
            )

        self.pool_branches = nn.ModuleList([_pool_proj(1), _pool_proj(3), _pool_proj(5)])
        self.global_proj = BasicConv2d(in_channels, in_channels // 2, 1)
        fused_channels = in_channels // 2 * 4
        self.out_projs = nn.ModuleList([BasicConv2d(fused_channels, c, 3, padding=1) for c in decoder_channels])

    def forward(self, x: torch.Tensor, refs: Sequence[torch.Tensor]) -> list[torch.Tensor]:
        feats = [resize_to(branch(x), x) for branch in self.pool_branches]
        feats.append(self.global_proj(x))
        fused = torch.cat(feats, dim=1)
        return [resize_to(proj(fused), ref) for proj, ref in zip(self.out_projs, refs)]


class AdaptiveSelectionModule(nn.Module):
    def __init__(self, local_channels: int, decoder_channels: int, global_channels: int, out_channels: int) -> None:
        super().__init__()
        total = local_channels + decoder_channels + global_channels
        self.fuse = BasicConv2d(total, out_channels, 3, padding=1)
        self.attn = SqueezeExcitation(out_channels)
        self.out = BasicConv2d(out_channels, out_channels, 3, padding=1)

    def forward(self, local_feat: torch.Tensor, decoder_feat: torch.Tensor, global_feat: torch.Tensor) -> torch.Tensor:
        decoder_feat = resize_to(decoder_feat, local_feat)
        global_feat = resize_to(global_feat, local_feat)
        x = torch.cat([local_feat, decoder_feat, global_feat], dim=1)
        x = self.fuse(x)
        x = self.attn(x)
        return self.out(x)


class BoundaryPredictionNetwork(nn.Module):
    def __init__(self, channels: Sequence[int], boundary_channels: int) -> None:
        super().__init__()
        c0, c1, c2, c3, _ = channels
        self.proj0 = BasicConv2d(c0, boundary_channels, 3, padding=1)
        self.proj1 = BasicConv2d(c1, boundary_channels, 3, padding=1)
        self.proj2 = BasicConv2d(c2, boundary_channels, 3, padding=1)
        self.proj3 = BasicConv2d(c3, boundary_channels, 3, padding=1)
        self.refine2 = BasicConv2d(boundary_channels * 2, boundary_channels, 3, padding=1)
        self.refine1 = BasicConv2d(boundary_channels * 2, boundary_channels, 3, padding=1)
        self.refine0 = BasicConv2d(boundary_channels * 2, boundary_channels, 3, padding=1)
        self.edge_head = nn.Conv2d(boundary_channels, 1, 1)

    def forward(self, feats: Sequence[torch.Tensor]) -> tuple[list[torch.Tensor], torch.Tensor]:
        x0, x1, x2, x3, _ = feats
        b3 = self.proj3(x3)
        b2 = self.refine2(torch.cat([self.proj2(x2), resize_to(b3, x2)], dim=1))
        b1 = self.refine1(torch.cat([self.proj1(x1), resize_to(b2, x1)], dim=1))
        b0 = self.refine0(torch.cat([self.proj0(x0), resize_to(b1, x0)], dim=1))
        edge = self.edge_head(b0)
        return [b0, b1, b2, b3], edge


class CrossFeatureFusion(nn.Module):
    def __init__(self, high_channels: int, low_channels: int, out_channels: int) -> None:
        super().__init__()
        self.high = BasicConv2d(high_channels, out_channels, 3, padding=1)
        self.low = BasicConv2d(low_channels, out_channels, 3, padding=1)
        self.out = BasicConv2d(out_channels * 2, out_channels, 3, padding=1)

    def forward(self, high_feat: torch.Tensor, low_feat: torch.Tensor) -> torch.Tensor:
        high_feat = resize_to(self.high(high_feat), low_feat)
        low_feat = self.low(low_feat)
        fused = torch.cat([high_feat * low_feat, high_feat + low_feat], dim=1)
        return self.out(fused)


class BoundaryAggregationModule(nn.Module):
    def __init__(self, feat_channels: int, boundary_channels: int) -> None:
        super().__init__()
        self.gate = nn.Sequential(
            BasicConv2d(feat_channels + boundary_channels, feat_channels, 3, padding=1),
            nn.Conv2d(feat_channels, feat_channels, 1),
            nn.Sigmoid(),
        )
        self.out = BasicConv2d(feat_channels, feat_channels, 3, padding=1)

    def forward(self, feat: torch.Tensor, boundary_feat: torch.Tensor) -> torch.Tensor:
        boundary_feat = resize_to(boundary_feat, feat)
        gate = self.gate(torch.cat([feat, boundary_feat], dim=1))
        return self.out(feat * gate + feat)


class CFPModule(nn.Module):
    """Context feature pyramid used by CaraNet."""

    def __init__(self, channels: int, dilation: int = 8) -> None:
        super().__init__()
        branch_channels = max(channels // 4, 16)
        dilations = [1, max(dilation // 4, 1), max(dilation // 2, 1), dilation]
        self.pre = BasicConv2d(channels, branch_channels, 1)
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    BasicConv2d(branch_channels, branch_channels, 3, padding=d, dilation=d),
                    BasicConv2d(branch_channels, branch_channels, 3, padding=d, dilation=d),
                )
                for d in dilations
            ]
        )
        self.fuse = BasicConv2d(branch_channels * len(dilations), channels, 1)
        self.res = BasicConv2d(channels, channels, 1, relu=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        stem = self.pre(x)
        feats = [branch(stem) for branch in self.branches]
        return self.relu(self.fuse(torch.cat(feats, dim=1)) + self.res(x))


class AxialAttention2d(nn.Module):
    def __init__(self, channels: int, heads: int = 4) -> None:
        super().__init__()
        self.row_attn = nn.MultiheadAttention(channels, heads, batch_first=True)
        self.col_attn = nn.MultiheadAttention(channels, heads, batch_first=True)
        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)
        self.ffn = nn.Sequential(
            nn.Linear(channels, channels * 2),
            nn.GELU(),
            nn.Linear(channels * 2, channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        rows = x.permute(0, 2, 3, 1).reshape(b * h, w, c)
        rows2, _ = self.row_attn(rows, rows, rows, need_weights=False)
        rows = self.norm1(rows + rows2)
        rows = rows + self.ffn(rows)
        x = rows.reshape(b, h, w, c).permute(0, 3, 1, 2)

        cols = x.permute(0, 3, 2, 1).reshape(b * w, h, c)
        cols2, _ = self.col_attn(cols, cols, cols, need_weights=False)
        cols = self.norm2(cols + cols2)
        cols = cols + self.ffn(cols)
        return cols.reshape(b, w, h, c).permute(0, 3, 2, 1)


class AxialReverseAttention(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int) -> None:
        super().__init__()
        self.pre = BasicConv2d(in_channels, hidden_channels, 1)
        self.axial = AxialAttention2d(hidden_channels, heads=max(1, min(4, hidden_channels // 16)))
        self.refine = nn.Sequential(
            BasicConv2d(hidden_channels, hidden_channels, 3, padding=1),
            BasicConv2d(hidden_channels, hidden_channels, 3, padding=1),
        )
        self.out = nn.Conv2d(hidden_channels, 1, 1)

    def forward(self, feat: torch.Tensor, coarse_logits: torch.Tensor) -> torch.Tensor:
        coarse = resize_to(coarse_logits, feat)
        reverse = 1.0 - torch.sigmoid(coarse)
        x = feat * reverse.expand(-1, feat.shape[1], -1, -1)
        x = self.pre(x)
        x = self.axial(x)
        x = self.refine(x)
        return self.out(x) + coarse


class AttentionGate2d(nn.Module):
    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gate(x)


class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_channels: int, embed_dim: int, patch_size: int, stride: int) -> None:
        super().__init__()
        padding = patch_size // 2
        self.proj = nn.Conv2d(in_channels, embed_dim, patch_size, stride=stride, padding=padding)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        x = self.proj(x)
        h, w = x.shape[-2:]
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, h, w


class DWConv(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.dw = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)

    def forward(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        b, n, c = x.shape
        x = x.transpose(1, 2).reshape(b, c, h, w)
        x = self.dw(x)
        return x.flatten(2).transpose(1, 2)


class MLPWithDWConv(nn.Module):
    def __init__(self, channels: int, ratio: float = 4.0) -> None:
        super().__init__()
        hidden = int(channels * ratio)
        self.fc1 = nn.Linear(channels, hidden)
        self.dw = DWConv(hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, channels)

    def forward(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        x = self.fc1(x)
        x = self.dw(x, h, w)
        x = self.act(x)
        return self.fc2(x)


class SpatialReductionAttention(nn.Module):
    def __init__(self, channels: int, heads: int = 4, sr_ratio: int = 1) -> None:
        super().__init__()
        if channels % heads != 0:
            raise ValueError("channels must be divisible by heads")
        self.channels = channels
        self.heads = heads
        self.head_dim = channels // heads
        self.scale = self.head_dim ** -0.5
        self.q = nn.Linear(channels, channels)
        self.kv = nn.Linear(channels, channels * 2)
        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(channels, channels, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(channels)
        else:
            self.sr = None
            self.norm = nn.Identity()
        self.proj = nn.Linear(channels, channels)

    def forward(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        b, n, c = x.shape
        q = self.q(x).reshape(b, n, self.heads, self.head_dim).permute(0, 2, 1, 3)
        if self.sr is not None:
            x_ = x.transpose(1, 2).reshape(b, c, h, w)
            x_ = self.sr(x_).reshape(b, c, -1).transpose(1, 2)
            x_ = self.norm(x_)
        else:
            x_ = x
        kv = self.kv(x_).reshape(b, -1, 2, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(b, n, c)
        return self.proj(out)


class PVTTransformerBlock(nn.Module):
    def __init__(self, channels: int, heads: int, sr_ratio: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(channels)
        self.attn = SpatialReductionAttention(channels, heads=heads, sr_ratio=sr_ratio)
        self.norm2 = nn.LayerNorm(channels)
        self.mlp = MLPWithDWConv(channels)

    def forward(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), h, w)
        x = x + self.mlp(self.norm2(x), h, w)
        return x


class PVTLikeBackbone(nn.Module):
    def __init__(self, in_channels: int = 3, embed_dims: Sequence[int] = (32, 64, 128, 256), depths: Sequence[int] = (2, 2, 2, 2), heads: Sequence[int] = (1, 2, 4, 8), sr_ratios: Sequence[int] = (8, 4, 2, 1)) -> None:
        super().__init__()
        embed_dims = ensure_tuple_channels(embed_dims)
        if len(embed_dims) != 4:
            raise ValueError("PVTLikeBackbone expects four stage channel values.")
        self.patch_embeds = nn.ModuleList(
            [
                OverlapPatchEmbed(in_channels, embed_dims[0], patch_size=7, stride=4),
                OverlapPatchEmbed(embed_dims[0], embed_dims[1], patch_size=3, stride=2),
                OverlapPatchEmbed(embed_dims[1], embed_dims[2], patch_size=3, stride=2),
                OverlapPatchEmbed(embed_dims[2], embed_dims[3], patch_size=3, stride=2),
            ]
        )
        self.stages = nn.ModuleList(
            [
                nn.ModuleList([PVTTransformerBlock(embed_dims[i], heads[i], sr_ratios[i]) for _ in range(depths[i])])
                for i in range(4)
            ]
        )
        init_weights(self)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        outs = []
        for patch, blocks in zip(self.patch_embeds, self.stages):
            x, h, w = patch(x)
            for blk in blocks:
                x = blk(x, h, w)
            feat = x.transpose(1, 2).reshape(x.shape[0], -1, h, w)
            outs.append(feat)
            x = feat
        return outs


class CascadedFusionModule(nn.Module):
    def __init__(self, c2: int, c3: int, c4: int, out_channels: int) -> None:
        super().__init__()
        self.p2 = BasicConv2d(c2, out_channels, 1)
        self.p3 = BasicConv2d(c3, out_channels, 1)
        self.p4 = BasicConv2d(c4, out_channels, 1)
        self.fuse3 = BasicConv2d(out_channels * 2, out_channels, 3, padding=1)
        self.fuse2 = BasicConv2d(out_channels * 2, out_channels, 3, padding=1)
        self.out = BasicConv2d(out_channels, out_channels, 3, padding=1)

    def forward(self, x2: torch.Tensor, x3: torch.Tensor, x4: torch.Tensor) -> torch.Tensor:
        p4 = self.p4(x4)
        p3 = self.fuse3(torch.cat([self.p3(x3), resize_to(p4, x3)], dim=1))
        p2 = self.fuse2(torch.cat([self.p2(x2), resize_to(p3, x2)], dim=1))
        return self.out(p2)


class CamouflageIdentificationModule(nn.Module):
    def __init__(self, low_channels: int, guide_channels: int, out_channels: int) -> None:
        super().__init__()
        self.low = BasicConv2d(low_channels, out_channels, 3, padding=1)
        self.guide = BasicConv2d(guide_channels, out_channels, 3, padding=1)
        self.out = BasicConv2d(out_channels * 2, out_channels, 3, padding=1)

    def forward(self, low_feat: torch.Tensor, guide_feat: torch.Tensor, coarse_logits: torch.Tensor) -> torch.Tensor:
        low_feat = self.low(low_feat)
        guide_feat = resize_to(self.guide(guide_feat), low_feat)
        guide_logits = resize_to(coarse_logits, low_feat)
        uncertainty = 1.0 - (2.0 * torch.abs(torch.sigmoid(guide_logits) - 0.5))
        x = torch.cat([low_feat * uncertainty, guide_feat], dim=1)
        return self.out(x)


class SimilarityAggregationModule(nn.Module):
    def __init__(self, channels: int, max_tokens_hw: int = 22) -> None:
        super().__init__()
        self.query = nn.Conv2d(channels, channels, 1, bias=False)
        self.key = nn.Conv2d(channels, channels, 1, bias=False)
        self.value = nn.Conv2d(channels, channels, 1, bias=False)
        self.max_tokens_hw = max_tokens_hw
        self.proj = BasicConv2d(channels, channels, 3, padding=1)

    def forward(self, feat: torch.Tensor, coarse_logits: torch.Tensor) -> torch.Tensor:
        prob = torch.sigmoid(resize_to(coarse_logits, feat))
        feat = feat * (1.0 + prob)
        b, c, h, w = feat.shape
        pooled_h = min(h, self.max_tokens_hw)
        pooled_w = min(w, self.max_tokens_hw)
        pooled = F.adaptive_avg_pool2d(feat, (pooled_h, pooled_w))
        q = self.query(pooled).reshape(b, c, pooled_h * pooled_w).transpose(1, 2)
        k = self.key(pooled).reshape(b, c, pooled_h * pooled_w)
        v = self.value(pooled).reshape(b, c, pooled_h * pooled_w).transpose(1, 2)
        attn = torch.softmax((q @ k) / math.sqrt(c), dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(b, c, pooled_h, pooled_w)
        out = F.interpolate(out, size=(h, w), mode="bilinear", align_corners=False)
        return self.proj(out + feat)


class HarDBlock(nn.Module):
    """HarDNet block with harmonic dense connectivity."""

    def __init__(self, in_channels: int, growth_rate: int, grmul: float, n_layers: int, keep_base: bool = False) -> None:
        super().__init__()
        self.keep_base = keep_base
        self.links: list[list[int]] = []
        self.out_partitions: list[int] = []
        layers = []
        out_channels_list = [in_channels]
        for layer in range(1, n_layers + 1):
            out_ch, in_ch, link = self.get_link(layer, in_channels, growth_rate, grmul)
            self.links.append(link)
            self.out_partitions.append(out_ch)
            layers.append(BasicConv2d(in_ch, out_ch, 3, padding=1))
            out_channels_list.append(out_ch)
        self.layers = nn.ModuleList(layers)
        self.out_channels = 0
        for i in range(n_layers + 1):
            if (i == 0 and self.keep_base) or (i == n_layers) or (i % 2 == 1):
                self.out_channels += out_channels_list[i]

    def get_link(self, layer: int, base_ch: int, growth_rate: int, grmul: float) -> tuple[int, int, list[int]]:
        if layer == 0:
            return base_ch, 0, []
        out_channels = growth_rate
        link = []
        for i in range(10):
            dv = 2 ** i
            if layer % dv == 0:
                k = layer - dv
                link.append(k)
                if i > 0:
                    out_channels *= grmul
        out_channels = int(int(out_channels + 1) / 2) * 2
        in_channels = 0
        for i in link:
            ch, _, _ = self.get_link(i, base_ch, growth_rate, grmul)
            in_channels += ch
        return out_channels, in_channels, link

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        layers_out = [x]
        for layer, links in zip(self.layers, self.links):
            tensors = [layers_out[i] for i in links]
            out = layer(torch.cat(tensors, dim=1) if len(tensors) > 1 else tensors[0])
            layers_out.append(out)
        outputs = []
        for i, t in enumerate(layers_out):
            if (i == 0 and self.keep_base) or (i == len(layers_out) - 1) or (i % 2 == 1):
                outputs.append(t)
        return torch.cat(outputs, dim=1)


class HarDNetLikeEncoder(nn.Module):
    def __init__(self, in_channels: int = 3, base_channels: int = 32) -> None:
        super().__init__()
        stem = max(base_channels // 2, 16)
        self.stem = nn.Sequential(
            BasicConv2d(in_channels, stem, 3, stride=2, padding=1),
            BasicConv2d(stem, base_channels, 3, padding=1),
        )
        self.pool = nn.MaxPool2d(2)
        self.blk1 = HarDBlock(base_channels, growth_rate=max(base_channels // 4, 8), grmul=1.7, n_layers=4, keep_base=True)
        self.tr1 = BasicConv2d(self.blk1.out_channels, base_channels * 2, 1)
        self.blk2 = HarDBlock(base_channels * 2, growth_rate=max(base_channels // 3, 10), grmul=1.7, n_layers=4, keep_base=True)
        self.tr2 = BasicConv2d(self.blk2.out_channels, base_channels * 4, 1)
        self.blk3 = HarDBlock(base_channels * 4, growth_rate=max(base_channels // 2, 12), grmul=1.7, n_layers=8, keep_base=True)
        self.tr3 = BasicConv2d(self.blk3.out_channels, base_channels * 8, 1)
        self.blk4 = HarDBlock(base_channels * 8, growth_rate=max(base_channels // 2, 12), grmul=1.7, n_layers=8, keep_base=True)
        self.tr4 = BasicConv2d(self.blk4.out_channels, base_channels * 16, 1)
        init_weights(self)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        s0 = self.stem(x)                  # H/2
        s1 = self.blk1(self.pool(s0))      # H/4
        s1t = self.tr1(s1)
        s2 = self.blk2(self.pool(s1t))     # H/8
        s2t = self.tr2(s2)
        s3 = self.blk3(self.pool(s2t))     # H/16
        s3t = self.tr3(s3)
        s4 = self.blk4(self.pool(s3t))     # H/32
        s4t = self.tr4(s4)
        return [s0, s1t, s2t, s3t, s4t]


__all__ = [
    "BasicConv2d",
    "RFBModified",
    "DenseAggregation",
    "ReverseAttentionBranch",
    "Bottle2neck",
    "Res2NetLikeEncoder",
    "LocalContextAttention",
    "GlobalContextModule",
    "AdaptiveSelectionModule",
    "BoundaryPredictionNetwork",
    "CrossFeatureFusion",
    "BoundaryAggregationModule",
    "CFPModule",
    "AxialReverseAttention",
    "OverlapPatchEmbed",
    "PVTTransformerBlock",
    "PVTLikeBackbone",
    "CascadedFusionModule",
    "CamouflageIdentificationModule",
    "SimilarityAggregationModule",
    "HarDBlock",
    "HarDNetLikeEncoder",
]
