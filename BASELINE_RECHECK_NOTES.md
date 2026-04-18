# Baseline recheck notes

This benchmark keeps the paper-defining modules of each baseline inside a unified training and evaluation framework.
The goal is stronger reproducibility and fairer comparison on the same dataset/problem setting, while avoiding large codebase forks.

## Rechecked baseline mapping

- `unet`: classic encoder-decoder with skip connections.
- `unet_cbam`: internal ablation baseline using U-Net plus CBAM attention modules.
- `unetpp`: nested skip pathways with deep supervision support.
- `pranet`: reverse-attention style decoder with dense aggregation / partial-decoder style supervision.
- `acsnet`: adaptive context selection using local/global context and fusion.
- `hardnet_mseg`: HarDNet-style encoder-decoder with RFB-like receptive-field refinement and dense aggregation.
- `polyp_pvt`: PVT-style transformer backbone with camouflage/similarity-style fusion blocks and faithful auxiliary prediction.
- `caranet`: axial reverse attention with CFP-style feature processing.
- `cfanet`: cross-level feature aggregation with boundary prediction and boundary-aware supervision.
- `hsnet`: cross-semantic attention, hybrid semantic complementary module, and multi-scale prediction.
- `proposal_hf_unet`: proposed HF bottleneck benchmark model.

## Reproducibility policy used here

- Epoch budget kept at 30 for paper-facing comparison configs.
- Cosine scheduler horizon (`t_max`) is matched to the 30-epoch budget in `fair`, `faithful`, and `paper_fair` configs.
- Faithful configs preserve model-specific supervision when the paper design clearly relies on it.
- Additional smoke tests verify build, forward, loss computation, backward, and one optimizer step for all baselines.

## Important limitation

These implementations are benchmark-integrated reproductions, not byte-for-byte copies of every original training repository.
They aim to preserve paper-defining architectural behavior while using one shared experimental framework.
