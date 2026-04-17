from __future__ import annotations

import torch
import torch.nn as nn

from ..common.paper_baselines import (
    BasicConv2d,
    CamouflageIdentificationModule,
    CascadedFusionModule,
    PVTLikeBackbone,
    SimilarityAggregationModule,
)
from ..common.utils import init_weights, resize_to
from ..registry import register_model


@register_model("polyp_pvt")
class PolypPVT(nn.Module):
    """Polyp-PVT with PVT-style encoder, CFM, CIM, and SAM modules."""

    def __init__(self, in_channels: int = 3, num_classes: int = 1, channels: tuple[int, ...] = (32, 64, 128, 256), norm: str = "bn", act: str = "gelu") -> None:
        super().__init__()
        if num_classes != 1:
            raise ValueError("Polyp-PVT currently supports binary segmentation only.")
        if len(channels) != 4:
            raise ValueError("Polyp-PVT expects four backbone channel values.")
        c1, c2, c3, c4 = channels
        self.backbone = PVTLikeBackbone(in_channels=in_channels, embed_dims=channels)
        fusion_channels = max(c2, 32)
        self.cfm = CascadedFusionModule(c2, c3, c4, fusion_channels)
        self.coarse_head = nn.Conv2d(fusion_channels, 1, 1)
        self.cim = CamouflageIdentificationModule(low_channels=c1, guide_channels=fusion_channels, out_channels=fusion_channels)
        self.sam = SimilarityAggregationModule(fusion_channels)
        self.refine = nn.Sequential(
            BasicConv2d(fusion_channels * 2, fusion_channels, 3, padding=1),
            BasicConv2d(fusion_channels, fusion_channels, 3, padding=1),
            nn.Conv2d(fusion_channels, 1, 1),
        )
        init_weights(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2, x3, x4 = self.backbone(x)
        cfm = self.cfm(x2, x3, x4)
        coarse = self.coarse_head(cfm)
        cim = self.cim(x1, cfm, coarse)
        sam = self.sam(cim, coarse)
        logits = self.refine(torch.cat([sam, resize_to(cfm, sam)], dim=1))
        return resize_to(logits, x)


PolypPVTLite = PolypPVT
