from __future__ import annotations

import torch
import torch.nn as nn

from ..common.decoder import UNetDecoder
from ..common.encoder import PyramidEncoder
from ..common.utils import init_weights
from ..registry import register_model
from .hf_bottleneck import HFBottleneck
from .hf_regularizer import HFRegularizer


@register_model("proposal_hf_unet")
class HFUNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1,
        channels: tuple[int, ...] = (32, 64, 128, 256, 512),
        hf_alpha: float = 0.5,
        hf_alpha_start: float = 0.0,
        hf_alpha_warmup_epochs: int = 10,
        hf_expansion: float = 1.5,
        hf_dropout: float = 0.0,
        use_hf_regularizer: bool = True,
        norm: str = "bn",
        act: str = "relu",
        hf_block_norm: str | None = None,
        hf_block_act: str | None = None,
        mixer_act: str = "identity",
        use_se: bool = False,
        use_gate: bool = True,
        gate_init_bias: float = -2.0,
        decoder_use_cbam: bool = False,
        identity_init: bool = True,
    ) -> None:
        super().__init__()
        self.encoder = PyramidEncoder(in_channels=in_channels, channels=channels, block="double", norm=norm, act=act)
        self.hf_alpha_target = float(hf_alpha)
        self.hf_alpha_start = float(hf_alpha_start)
        self.hf_alpha_warmup_epochs = int(hf_alpha_warmup_epochs)
        block_norm = hf_block_norm or norm
        block_act = hf_block_act or act
        self.hf_bottleneck = HFBottleneck(
            channels[-1],
            expansion=hf_expansion,
            alpha=hf_alpha,
            dropout=hf_dropout,
            use_se=use_se,
            use_gate=use_gate,
            norm=block_norm,
            act=block_act,
            mixer_act=mixer_act,
            gate_init_bias=gate_init_bias,
            identity_init=identity_init,
        )
        self.decoder = UNetDecoder(channels=channels, norm=norm, act=act, use_cbam=decoder_use_cbam)
        self.seg_head = nn.Conv2d(channels[0], num_classes, 1)
        self.regularizer = HFRegularizer() if use_hf_regularizer else None
        init_weights(self)
        self.set_epoch(0)

    def set_epoch(self, epoch: int) -> None:
        if self.hf_alpha_warmup_epochs <= 0:
            alpha = self.hf_alpha_target
        else:
            progress = min(max(float(epoch), 0.0) / float(self.hf_alpha_warmup_epochs), 1.0)
            alpha = self.hf_alpha_start + (self.hf_alpha_target - self.hf_alpha_start) * progress
        self.hf_bottleneck.set_alpha(alpha)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.encoder(x)
        feats[-1] = self.hf_bottleneck(feats[-1])
        dec = self.decoder(feats)
        return self.seg_head(dec)

    def auxiliary_regularization(self) -> torch.Tensor:
        if self.regularizer is None:
            device = next(self.parameters()).device
            return torch.zeros((), device=device)
        return self.regularizer.from_module(self.hf_bottleneck)
