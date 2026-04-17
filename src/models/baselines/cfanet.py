from __future__ import annotations

import torch
import torch.nn as nn

from ..common.paper_baselines import (
    BasicConv2d,
    BoundaryAggregationModule,
    BoundaryPredictionNetwork,
    CrossFeatureFusion,
    Res2NetLikeEncoder,
)
from ..common.utils import init_weights, resize_to
from ..registry import register_model


@register_model("cfanet")
class CFANet(nn.Module):
    """CFANet with boundary prediction network, two-stream cross-level fusion, and BAM refinement."""

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1,
        channels: tuple[int, ...] = (32, 64, 128, 256, 512),
        aggregation_channels: int = 64,
        boundary_channels: int = 32,
        norm: str = "bn",
        act: str = "relu",
    ) -> None:
        super().__init__()
        if num_classes != 1:
            raise ValueError("CFANet currently supports binary segmentation only.")
        if len(channels) != 5:
            raise ValueError("CFANet expects exactly five encoder stages.")
        c0, c1, c2, c3, c4 = channels
        self.encoder = Res2NetLikeEncoder(in_channels=in_channels, channels=channels)
        self.boundary = BoundaryPredictionNetwork(channels=channels, boundary_channels=boundary_channels)

        self.stream1_43 = CrossFeatureFusion(c4, c3, aggregation_channels)
        self.stream1_32 = CrossFeatureFusion(aggregation_channels, c2, aggregation_channels)
        self.stream1_21 = CrossFeatureFusion(aggregation_channels, c1, aggregation_channels)
        self.stream1_10 = CrossFeatureFusion(aggregation_channels, c0, aggregation_channels)

        self.stream2_01 = CrossFeatureFusion(aggregation_channels, aggregation_channels, aggregation_channels)
        self.stream2_12 = CrossFeatureFusion(aggregation_channels, aggregation_channels, aggregation_channels)
        self.stream2_23 = CrossFeatureFusion(aggregation_channels, aggregation_channels, aggregation_channels)

        self.bam0 = BoundaryAggregationModule(aggregation_channels, boundary_channels)
        self.bam1 = BoundaryAggregationModule(aggregation_channels, boundary_channels)
        self.bam2 = BoundaryAggregationModule(aggregation_channels, boundary_channels)
        self.bam3 = BoundaryAggregationModule(aggregation_channels, boundary_channels)

        self.seg_head = nn.Sequential(
            BasicConv2d(aggregation_channels * 2 + boundary_channels, aggregation_channels, 3, padding=1),
            BasicConv2d(aggregation_channels, aggregation_channels, 3, padding=1),
            nn.Conv2d(aggregation_channels, 1, 1),
        )
        init_weights(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0, x1, x2, x3, x4 = self.encoder(x)
        boundary_feats, boundary_logits = self.boundary((x0, x1, x2, x3, x4))
        b0, b1, b2, b3 = boundary_feats

        s3 = self.bam3(self.stream1_43(x4, x3), b3)
        s2 = self.bam2(self.stream1_32(s3, x2), b2)
        s1 = self.bam1(self.stream1_21(s2, x1), b1)
        s0 = self.bam0(self.stream1_10(s1, x0), b0)

        u1 = self.stream2_01(s0, resize_to(s1, s0))
        u2 = self.stream2_12(s1, resize_to(s2, s1))
        u3 = self.stream2_23(s2, resize_to(s3, s2))
        fused = torch.cat([u1, resize_to(u2, u1), resize_to(b0, u1)], dim=1)
        return resize_to(self.seg_head(fused), x)


CFANetLite = CFANet
