# Baseline implementation audit

This project is best described as a **benchmark-integrated baseline suite**, not a byte-for-byte copy of every official repository. The code is suitable for fair same-framework comparison when the paper clearly states that all methods are reimplemented or adapted under the same training/evaluation protocol.

## Overall judgement

The current implementations preserve the main paper-defining architectural ideas for the included baselines:

- **U-Net / U-Net++**: encoder-decoder skip connections and nested decoder pathways; U-Net++ supports deep supervision.
- **ResUNet++**: residual blocks, squeeze-and-excitation, ASPP bridge/output refinement, and attention-gated skip fusion.
- **PraNet**: Res2Net-style backbone adapter, RFB feature reduction, dense/partial decoder style aggregation, and reverse-attention refinement branches.
- **ACSNet**: local context attention, global context module, and adaptive selection module.
- **HarDNet-MSEG**: HarDNet-style/official HarDNet adapter plus RFB and dense/cascaded aggregation decoder pieces.
- **Polyp-PVT**: PVTv2 backbone adapter plus cascaded fusion, camouflage-identification, and similarity aggregation modules.
- **CaraNet**: Res2Net backbone adapter, context feature pyramid, and axial reverse-attention refinement.
- **CFA-Net**: boundary prediction, cross-level feature fusion, boundary aggregation, auxiliary segmentation heads, and boundary-aware loss compatibility.
- **HSNet**: CNN + PVT hybrid encoders, cross-semantic attention, hybrid semantic complementary modules, and multi-scale prediction fusion.
- **CSCA U-Net**: six-stage U-Net, residual CSCA basic blocks, DSE bottleneck/channel attention, CSCA spatial attention, decoder cross-layer fusion, and five auxiliary maps in faithful mode.

## Important limitations to disclose in the paper

1. **Not exact official repository replication.** These are unified-framework reproductions. They preserve defining modules but do not guarantee identical layer naming, pretrained weights, image preprocessing, optimizer schedules, or post-processing.
2. **Pretrained checkpoints are not bundled by default.** Several configs set `backbone_impl: official` but `backbone_pretrained: false`. This is fair if all models are trained from scratch, but it should be stated explicitly.
3. **Fast official-backbone variants are used in test/fair configs.** For example, Res2Net/PVT fast variants reduce runtime and make the suite easier to run. Use the `official_faithful` configs or provide checkpoints when you want a closer official-repository reproduction.
4. **CSCA U-Net uses `attention_mode: efficient` in most runnable configs.** The project also keeps `attention_mode: paper` in `configs/official_faithful/csca_unet.yaml`, but the paper-style spatial matmul is much heavier and is practical mainly on square resized inputs.
5. **Post-processing is not included unless explicitly coded.** For example, ResUNet++ is implemented as the neural architecture only; CRF/TTA-style extras should not be claimed unless added and tested.
6. **Training protocol is unified.** Losses, epoch count, scheduler, augmentation, and split generation are controlled by this benchmark. This improves fairness but differs from many original repositories.

## New tests added in this patch

File added:

```text
tests/test_baseline_paper_behavior_contracts.py
```

The new tests go beyond simple “module exists” checks and validate behavior-level contracts:

- reverse-attention output changes when the previous/coarse prediction map changes;
- ACSNet LCA emphasizes uncertain prediction regions;
- ACSNet GCM returns one context tensor per decoder level with correct channels and size;
- CSCA U-Net faithful mode returns one main map plus five auxiliary supervision maps;
- CSCA paper-style spatial attention executes on small square feature maps;
- HSNet multi-scale prediction weights are softmax-normalized and trainable;
- the unified loss path supports auxiliary outputs and boundary outputs simultaneously.

## Recommended commands

Fast checks:

```bash
python -m pytest -q \
  tests/test_baseline_paper_behavior_contracts.py \
  tests/test_model_registry_config_coverage.py \
  tests/test_paper_fair_configs.py \
  tests/test_scheduler_tmax_alignment.py \
  tests/test_runtime_and_data_cli.py
```

Heavier checks:

```bash
python -m pytest -q \
  tests/test_official_paper_alignment_contracts.py \
  tests/test_faithful_outputs_and_losses.py \
  tests/test_baseline_train_step_smoke.py \
  tests/test_all_models_runnability_contracts.py
```

For paper wording, use a sentence like:

> All compared methods are benchmark-integrated reproductions implemented in a unified PyTorch training framework. For each named baseline, we preserve the paper-defining modules and supervision pattern where applicable, while using the same preprocessing, loss family, split protocol, optimizer, scheduler, and epoch budget for fair comparison.
