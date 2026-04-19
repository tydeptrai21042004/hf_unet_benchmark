from __future__ import annotations

import torch
import torch.nn as nn

from ..common.official_backbones import OfficialPVTv2Backbone
from ..common.paper_baselines import BasicConv2d, CamouflageIdentificationModule, CascadedFusionModule, PVTLikeBackbone, SimilarityAggregationModule
from ..common.utils import resize_to
from ..registry import register_model


@register_model("polyp_pvt")
class PolypPVT(nn.Module):
    """Polyp-PVT-style baseline with optional official PVTv2 backbone."""

    def __init__(self, in_channels: int = 3, num_classes: int = 1, channels: tuple[int, ...] = (32, 64, 128, 256), faithful_output: bool = False, norm: str = "bn", act: str = "gelu", backbone_impl: str = "official", pvt_variant: str = "pvt_v2_b2", backbone_pretrained: bool = False, backbone_checkpoint: str | None = None, backbone_checkpoint_url: str | None = None, image_size: int = 352) -> None:
        super().__init__()
        if num_classes != 1:
            raise ValueError("Polyp-PVT currently supports binary segmentation only.")
        if len(channels) != 4:
            raise ValueError("Polyp-PVT expects four backbone channel values.")
        c1, c2, c3, c4 = channels
        self.faithful_output = faithful_output
        if backbone_impl.lower() in {"official", "official_backbone"}:
            self.backbone = OfficialPVTv2Backbone(in_channels=in_channels, embed_dims=channels, variant=pvt_variant, pretrained=backbone_pretrained, checkpoint=backbone_checkpoint, checkpoint_url=backbone_checkpoint_url, image_size=image_size)
        else:
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
