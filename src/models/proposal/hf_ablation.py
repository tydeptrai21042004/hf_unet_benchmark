from __future__ import annotations

from typing import Literal, Optional, Sequence

import torch
import torch.nn as nn

from ..common.blocks import ConvNormAct, DoubleConv
from ..common.decoder import UNetDecoder
from ..common.encoder import PyramidEncoder
from ..common.utils import init_weights
from ..registry import register_model
from .hf_bottleneck import HFBottleneck, HartleyTransform2d
from .hf_regularizer import HFRegularizer


class IdentityTransform2d(nn.Module):
    """Identity signal transform used for the w/o-Hartley ablation."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class ConvBottleneck(nn.Module):
    """Parameter-control bottleneck: local residual convolution only.

    This variant is intentionally non-spectral. It answers whether improvement
    comes merely from inserting an extra trainable block at the U-Net bottleneck.
    """

    def __init__(self, channels: int, norm: str = "bn", act: str = "relu", residual: bool = True) -> None:
        super().__init__()
        self.residual = bool(residual)
        self.block = DoubleConv(channels, channels, norm=norm, act=act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.block(x)
        return x + y if self.residual else y


class FFTGFNetLikeBottleneck(nn.Module):
    """Small FFT/GFNet-like spectral bottleneck for controlled comparison.

    It uses a generic Fourier-domain channel mixer. It is a fair spectral rival
    to HF because it is global and frequency-domain, but it does not use the
    Hartley-on-signal / Fourier-on-kernel factorization of HF-U-Net.
    """

    def __init__(
        self,
        channels: int,
        expansion: float = 1.0,
        alpha: float = 0.5,
        dropout: float = 0.0,
        use_gate: bool = True,
        gate_init_bias: float = -2.0,
        identity_init: bool = True,
    ) -> None:
        super().__init__()
        self.alpha = float(alpha)
        hidden = max(int(channels * expansion), channels)
        self.real_imag_mixer = nn.Sequential(
            nn.Conv2d(2 * channels, 2 * hidden, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(2 * hidden, 2 * channels, kernel_size=1, bias=False),
        )
        self.use_gate = bool(use_gate)
        if self.use_gate:
            self.gate = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(channels, channels, 1),
                nn.Sigmoid(),
            )
        else:
            self.gate = nn.Identity()
        if identity_init:
            last = self.real_imag_mixer[-1]
            if isinstance(last, nn.Conv2d):
                nn.init.normal_(last.weight, mean=0.0, std=1.0e-3)
            if self.use_gate:
                gate_conv = self.gate[1]
                nn.init.zeros_(gate_conv.weight)
                if gate_conv.bias is not None:
                    nn.init.constant_(gate_conv.bias, gate_init_bias)

    def set_alpha(self, alpha: float) -> None:
        self.alpha = float(alpha)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        orig_dtype = x.dtype
        fft_input = x.float() if x.is_cuda and x.dtype in (torch.float16, torch.bfloat16) else x
        z = torch.fft.fft2(fft_input, norm="ortho")
        ri = torch.cat([z.real, z.imag], dim=1)
        mixed_ri = self.real_imag_mixer(ri)
        real, imag = torch.chunk(mixed_ri, 2, dim=1)
        restored = torch.fft.ifft2(torch.complex(real, imag), norm="ortho").real
        if restored.dtype != orig_dtype and orig_dtype in (torch.float16, torch.bfloat16):
            restored = restored.to(orig_dtype)
        gate = self.gate(restored) if self.use_gate else torch.ones_like(restored)
        return identity + self.alpha * gate * restored


class HFAblationUNet(nn.Module):
    """U-Net wrapper used to run compact HF-bottleneck ablations.

    Variants are controlled through `ablation` and `placement`:
      - conv_bottleneck: local parameter-control block.
      - fft_bottleneck: generic FFT/GFNet-like spectral rival.
      - full_hf: proposed HF bottleneck.
      - wo_hartley: same HF block but the Hartley signal transform is replaced by identity.
      - wo_fourier_kernel: HF signal transform is kept but frequency-domain mixer is disabled.
      - wo_residual: HF block output does not add the input identity path.
      - encoder_stage4: HF is applied to encoder feature index -2 instead of deepest bottleneck.
      - decoder_stage: HF is applied after the decoder and before the segmentation head.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1,
        channels: Sequence[int] = (32, 64, 128, 256, 512),
        ablation: Literal[
            "conv_bottleneck",
            "fft_bottleneck",
            "full_hf",
            "wo_hartley",
            "wo_fourier_kernel",
            "wo_residual",
            "encoder_stage4",
            "decoder_stage",
        ] = "full_hf",
        hf_alpha: float = 0.5,
        hf_alpha_start: float = 0.0,
        hf_alpha_warmup_epochs: int = 10,
        hf_expansion: float = 1.5,
        hf_dropout: float = 0.0,
        use_hf_regularizer: bool = True,
        norm: str = "bn",
        act: str = "relu",
        hf_block_norm: Optional[str] = None,
        hf_block_act: Optional[str] = None,
        mixer_act: str = "identity",
        use_se: bool = False,
        use_gate: bool = True,
        gate_init_bias: float = -2.0,
        decoder_use_cbam: bool = False,
        identity_init: bool = True,
    ) -> None:
        super().__init__()
        self.ablation = ablation
        self.encoder = PyramidEncoder(in_channels=in_channels, channels=channels, block="double", norm=norm, act=act)
        self.decoder = UNetDecoder(channels=channels, norm=norm, act=act, use_cbam=decoder_use_cbam)
        self.seg_head = nn.Conv2d(channels[0], num_classes, 1)

        self.hf_alpha_target = float(hf_alpha)
        self.hf_alpha_start = float(hf_alpha_start)
        self.hf_alpha_warmup_epochs = int(hf_alpha_warmup_epochs)
        block_norm = hf_block_norm or norm
        block_act = hf_block_act or act

        if ablation == "encoder_stage4":
            target_channels = channels[-2]
            self.placement = "encoder_stage4"
        elif ablation == "decoder_stage":
            target_channels = channels[0]
            self.placement = "decoder_stage"
        else:
            target_channels = channels[-1]
            self.placement = "bottleneck"

        self.block = self._make_block(
            ablation=ablation,
            channels=int(target_channels),
            hf_alpha=hf_alpha,
            hf_expansion=hf_expansion,
            hf_dropout=hf_dropout,
            use_hf_regularizer=use_hf_regularizer,
            norm=block_norm,
            act=block_act,
            mixer_act=mixer_act,
            use_se=use_se,
            use_gate=use_gate,
            gate_init_bias=gate_init_bias,
            identity_init=identity_init,
        )
        self.regularizer = HFRegularizer() if use_hf_regularizer and isinstance(self.block, HFBottleneck) else None
        init_weights(self)
        # Re-apply identity-friendly initialization because init_weights initializes all modules.
        self._refresh_identity_friendly_init(gate_init_bias=gate_init_bias, identity_init=identity_init)
        self.set_epoch(0)

    def _make_block(
        self,
        *,
        ablation: str,
        channels: int,
        hf_alpha: float,
        hf_expansion: float,
        hf_dropout: float,
        use_hf_regularizer: bool,
        norm: str,
        act: str,
        mixer_act: str,
        use_se: bool,
        use_gate: bool,
        gate_init_bias: float,
        identity_init: bool,
    ) -> nn.Module:
        if ablation == "conv_bottleneck":
            return ConvBottleneck(channels, norm=norm, act=act, residual=True)
        if ablation == "fft_bottleneck":
            return FFTGFNetLikeBottleneck(
                channels,
                expansion=hf_expansion,
                alpha=hf_alpha,
                dropout=hf_dropout,
                use_gate=use_gate,
                gate_init_bias=gate_init_bias,
                identity_init=identity_init,
            )

        residual = ablation != "wo_residual"
        block = HFBottleneck(
            channels,
            expansion=hf_expansion,
            alpha=hf_alpha,
            dropout=hf_dropout,
            use_se=use_se,
            use_gate=use_gate,
            norm=norm,
            act=act,
            mixer_act=mixer_act,
            gate_init_bias=gate_init_bias,
            identity_init=identity_init,
        )
        if ablation == "wo_hartley":
            block.hartley = IdentityTransform2d()
        if ablation == "wo_fourier_kernel":
            block.mixer = nn.Identity()
        if not residual:
            block.forward = self._make_no_residual_forward(block)  # type: ignore[method-assign]
        return block

    @staticmethod
    def _make_no_residual_forward(block: HFBottleneck):
        def forward_no_residual(x: torch.Tensor) -> torch.Tensor:
            x_pre = block.pre(x)
            freq = block.hartley(x_pre)
            mixed = block.mixer(freq)
            restored = block.hartley(mixed)
            restored = block.post(restored)
            restored = block.se(restored)
            if block.use_gate:
                gate = block.gate(restored)
            else:
                gate = torch.ones_like(restored)
            out = block.alpha * gate * restored

            smoothness = (mixed[:, :, 1:, :] - mixed[:, :, :-1, :]).abs().mean() + (mixed[:, :, :, 1:] - mixed[:, :, :, :-1]).abs().mean()
            if block.use_gate:
                near_identity = (gate.mean() - block.gate_identity_target).abs()
            else:
                near_identity = restored.new_zeros(())
            energy_penalty = (restored.pow(2).mean() - x.pow(2).mean()).abs()
            from .hf_bottleneck import HFRegularizationTerms

            block._last_terms = HFRegularizationTerms(
                spectral_smoothness=smoothness,
                near_identity=near_identity,
                energy_penalty=energy_penalty,
            )
            return out

        return forward_no_residual

    def _refresh_identity_friendly_init(self, *, gate_init_bias: float, identity_init: bool) -> None:
        if not identity_init:
            return
        if isinstance(self.block, HFBottleneck):
            if hasattr(self.block.mixer, "in_proj") and hasattr(self.block.mixer, "out_proj"):
                self.block._apply_identity_friendly_init(gate_init_bias=gate_init_bias)
            elif self.block.use_gate:
                gate_conv = self.block.gate[1]
                nn.init.zeros_(gate_conv.weight)
                if gate_conv.bias is not None:
                    nn.init.constant_(gate_conv.bias, gate_init_bias)
        elif isinstance(self.block, FFTGFNetLikeBottleneck):
            last = self.block.real_imag_mixer[-1]
            if isinstance(last, nn.Conv2d):
                nn.init.normal_(last.weight, mean=0.0, std=1.0e-3)
            if self.block.use_gate:
                gate_conv = self.block.gate[1]
                nn.init.zeros_(gate_conv.weight)
                if gate_conv.bias is not None:
                    nn.init.constant_(gate_conv.bias, gate_init_bias)

    def set_epoch(self, epoch: int) -> None:
        if self.hf_alpha_warmup_epochs <= 0:
            alpha = self.hf_alpha_target
        else:
            progress = min(max(float(epoch), 0.0) / float(self.hf_alpha_warmup_epochs), 1.0)
            alpha = self.hf_alpha_start + (self.hf_alpha_target - self.hf_alpha_start) * progress
        if hasattr(self.block, "set_alpha"):
            self.block.set_alpha(alpha)  # type: ignore[attr-defined]

    def _apply_block_to_features(self, feats: list[torch.Tensor]) -> list[torch.Tensor]:
        if self.placement == "bottleneck":
            feats[-1] = self.block(feats[-1])
        elif self.placement == "encoder_stage4":
            feats[-2] = self.block(feats[-2])
        return feats

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.encoder(x)
        feats = self._apply_block_to_features(feats)
        dec = self.decoder(feats)
        if self.placement == "decoder_stage":
            dec = self.block(dec)
        return self.seg_head(dec)

    def auxiliary_regularization(self) -> torch.Tensor:
        if self.regularizer is None or not isinstance(self.block, HFBottleneck):
            device = next(self.parameters()).device
            return torch.zeros((), device=device)
        return self.regularizer.from_module(self.block)


@register_model("unet_conv_bottleneck")
class UNetConvBottleneck(HFAblationUNet):
    def __init__(self, **kwargs) -> None:
        kwargs.pop("ablation", None)
        kwargs.pop("use_hf_regularizer", None)
        super().__init__(ablation="conv_bottleneck", use_hf_regularizer=False, **kwargs)


@register_model("unet_fft_bottleneck")
class UNetFFTGFNetBottleneck(HFAblationUNet):
    def __init__(self, **kwargs) -> None:
        kwargs.pop("ablation", None)
        kwargs.pop("use_hf_regularizer", None)
        super().__init__(ablation="fft_bottleneck", use_hf_regularizer=False, **kwargs)


@register_model("hf_unet_wo_hartley")
class HFUNetWithoutHartley(HFAblationUNet):
    def __init__(self, **kwargs) -> None:
        kwargs.pop("ablation", None)
        super().__init__(ablation="wo_hartley", **kwargs)


@register_model("hf_unet_wo_fourier_kernel")
class HFUNetWithoutFourierKernel(HFAblationUNet):
    def __init__(self, **kwargs) -> None:
        kwargs.pop("ablation", None)
        super().__init__(ablation="wo_fourier_kernel", **kwargs)


@register_model("hf_unet_wo_residual")
class HFUNetWithoutResidual(HFAblationUNet):
    def __init__(self, **kwargs) -> None:
        kwargs.pop("ablation", None)
        super().__init__(ablation="wo_residual", **kwargs)


@register_model("hf_unet_encoder_stage4")
class HFUNetEncoderStage4(HFAblationUNet):
    def __init__(self, **kwargs) -> None:
        kwargs.pop("ablation", None)
        super().__init__(ablation="encoder_stage4", **kwargs)


@register_model("hf_unet_decoder_stage")
class HFUNetDecoderStage(HFAblationUNet):
    def __init__(self, **kwargs) -> None:
        kwargs.pop("ablation", None)
        super().__init__(ablation="decoder_stage", **kwargs)


__all__ = [
    "IdentityTransform2d",
    "ConvBottleneck",
    "FFTGFNetLikeBottleneck",
    "HFAblationUNet",
    "UNetConvBottleneck",
    "UNetFFTGFNetBottleneck",
    "HFUNetWithoutHartley",
    "HFUNetWithoutFourierKernel",
    "HFUNetWithoutResidual",
    "HFUNetEncoderStage4",
    "HFUNetDecoderStage",
]
