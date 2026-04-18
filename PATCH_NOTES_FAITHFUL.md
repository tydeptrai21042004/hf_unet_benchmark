# Faithful benchmark patch notes

This patch adds a second benchmark track aimed at **core-faithful adapted reproductions**.

## Main additions

- Added `configs/faithful/` for model-specific supervision while keeping a unified dataset and optimizer recipe.
- Added `proposal_hf_unet_lite` as a parameter-matched control against plain U-Net.
- Added multi-output training/evaluation support for:
  - UNet++ deep supervision
  - PraNet multi-stage outputs
  - ACSNet side outputs
  - CaraNet multi-stage outputs
  - Polyp-PVT two-output supervision
  - CFANet auxiliary segmentation outputs plus boundary supervision
- Added `scripts/benchmark_faithful.py`.
- Added tests for faithful output contracts, multi-output losses, and parameter matching.

## Important scope note

These baselines are now **more faithful to the paper-defining heads and supervision patterns**, but they are still **adapted reproductions** rather than byte-for-byte ports of every official repository. In particular:

- The shared benchmark still uses the repo's local backbone implementations rather than downloading every official pretrained backbone.
- This patch is intended for fair comparison on your dataset while preserving the identity of each baseline more faithfully.

## Recommended benchmark usage

### Primary table
- `unet`
- `unet_cbam`
- `unetpp`
- `proposal_hf_unet_lite`
- `proposal_hf_unet`

### Secondary adapted faithful table
- `pranet`
- `acsnet`
- `hardnet_mseg`
- `polyp_pvt`
- `caranet`
- `cfanet`

Run:

```bash
python scripts/benchmark_faithful.py
```
