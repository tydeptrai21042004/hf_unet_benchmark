# HF-U-Net benchmark

## Paper-fair training recipe

Use `configs/paper_fair/` for strict paper comparisons. This preset aligns the training recipe across all models:

- same image size: 352
- same batch size: 6
- same augmentation: `strong`
- same optimizer / scheduler: AdamW + cosine
- same learning rate: 3e-4
- same weight decay: 1e-4
- same epochs: 30
- same gradient clipping: 1.0
- same segmentation loss: BCE+Dice
- same threshold: 0.5
- same mixed precision policy: disabled for every model
- no trainer-level auxiliary loss weighting for any model

The proposal model still differs structurally, but proposal-only training-side extras are disabled in `configs/paper_fair/` so the comparison is easier to defend in a paper.

### Run all models with the strict paper setup

```bash
CONFIG_DIR=configs/paper_fair OUTPUT_ROOT=outputs_paper_fair DEVICE=auto \
MODELS="unet,unet_cbam,unetpp,pranet,acsnet,hardnet_mseg,polyp_pvt,caranet,proposal_hf_unet" \
bash run.sh benchmark
```

### Kaggle TLS workaround

If auto-download fails because of certificate validation on the dataset host:

```bash
ALLOW_INSECURE_DOWNLOAD=1 CONFIG_DIR=configs/paper_fair OUTPUT_ROOT=outputs_paper_fair DEVICE=auto bash run.sh benchmark
```


Added baseline: HSNet (adapted faithful implementation with CSA, HSC, and MSP modules).
