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
    """Polyp-PVT-style baseline with PVT-like encoder, CFM, CIM, and SAM modules."""

    def __init__(self, in_channels: int = 3, num_classes: int = 1, channels: tuple[int, ...] = (32, 64, 128, 256), faithful_output: bool = False, norm: str = "bn", act: str = "gelu") -> None:
        super().__init__()
        if num_classes != 1:
            raise ValueError("Polyp-PVT currently supports binary segmentation only.")
        if len(channels) != 4:
            raise ValueError("Polyp-PVT expects four backbone channel values.")
        c1, c2, c3, c4 = channels
        self.faithful_output = faithful_output
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

    def forward(self, x: torch.Tensor):
        x1, x2, x3, x4 = self.backbone(x)
        cfm = self.cfm(x2, x3, x4)
        p1 = self.coarse_head(cfm)
        cim = self.cim(x1, cfm, p1)
        sam = self.sam(cim, p1)
        p2 = self.refine(torch.cat([sam, resize_to(cfm, sam)], dim=1))
        p1 = resize_to(p1, x)
        p2 = resize_to(p2, x)
        if self.faithful_output:
            return {"main": p2, "aux": [p1]}
        return p2


PolypPVTLite = PolypPVT
