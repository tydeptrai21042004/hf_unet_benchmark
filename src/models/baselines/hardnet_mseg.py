from __future__ import annotations

import torch
import torch.nn as nn

from ..common.official_backbones import OfficialHarDNetEncoder
from ..common.paper_baselines import DenseAggregation, HarDNetLikeEncoder, RFBModified
from ..common.utils import resize_to
from ..registry import register_model


@register_model("hardnet_mseg")
class HarDNetMSEG(nn.Module):
    """HarDNet-MSEG with optional official HarDNet-68 backbone."""

    def __init__(self, in_channels: int = 3, num_classes: int = 1, base_channels: int = 32, faithful_output: bool = False, norm: str = "bn", act: str = "relu", backbone_impl: str = "official", hardnet_arch: int = 68, backbone_pretrained: bool = False, backbone_checkpoint: str | None = None, backbone_checkpoint_url: str | None = None) -> None:
        super().__init__()
        if num_classes != 1:
            raise ValueError("HarDNet-MSEG currently supports binary segmentation only.")
        self.faithful_output = faithful_output
        channels = (base_channels, base_channels * 2, base_channels * 4, base_channels * 8, base_channels * 16)
        if backbone_impl.lower() in {"official", "official_backbone"}:
            self.encoder = OfficialHarDNetEncoder(in_channels=in_channels, channels=channels, arch=hardnet_arch, pretrained=backbone_pretrained, checkpoint=backbone_checkpoint, checkpoint_url=backbone_checkpoint_url)
        else:
            self.encoder = HarDNetLikeEncoder(in_channels=in_channels, base_channels=base_channels)
        ch2 = base_channels * 4
        ch3 = base_channels * 8
        ch4 = base_channels * 16
        agg_channels = max(base_channels, 16)
        self.rfb2 = RFBModified(ch2, agg_channels)
        self.rfb3 = RFBModified(ch3, agg_channels)
        self.rfb4 = RFBModified(ch4, agg_channels)
        self.decoder = DenseAggregation(agg_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, x2, x3, x4 = self.encoder(x)
        logits = self.decoder(self.rfb4(x4), self.rfb3(x3), self.rfb2(x2))
        return resize_to(logits, x)


HarDNetMSEGLite = HarDNetMSEG
