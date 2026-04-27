# CSCA U-Net loss/checkpoint fix notes

This patch fixes the abnormal CSCA U-Net loss behavior observed in repeated runs.

## Fixed issues

1. **Best checkpoint bug**
   - Previous behavior: `best_epoch` was selected from validation history, but `best.pt` stored the final epoch `model.state_dict()`.
   - New behavior: `Trainer.fit()` captures the CPU copy of the model state whenever the monitored validation metric improves. `scripts/train_one.py` saves that state as `best.pt` and also saves the final epoch as `last.pt`.

2. **CSCA logit explosion guard**
   - CSCA U-Net now accepts `model.logit_clip`.
   - The CSCA configs set `logit_clip: 20.0` to prevent very large raw logits from producing meaningless BCE loss values on tiny splits.

3. **Auxiliary-output evaluation loss control**
   - `Evaluator` now has `include_aux_loss=False` by default.
   - This keeps validation/test loss comparable by reporting the main-output loss unless `eval.include_aux_loss: true` is explicitly set.

4. **Debug support**
   - Training supports `train.debug_logits: true` to log main-logit and mask ranges.
   - This is useful for verifying whether loss spikes are caused by raw-logit explosion or mask scaling errors.

## Recommended rerun

Rerun CSCA U-Net and ResUNet++ after this patch because the checkpoint-saving bug affects all models:

```bash
python scripts/benchmark_all.py \
  --dataset custom \
  --config-dir configs/paper_fair \
  --data-root data_isbi2012 \
  --image-size 352 \
  --device auto \
  --output-root outputs_paper_fair_isbi2012_fixed \
  --models "csca_unet,resunetpp" \
  --skip-prepare \
  --skip-splits \
  --seed 1
```

Repeat with `--seed 2` and `--seed 42`.
