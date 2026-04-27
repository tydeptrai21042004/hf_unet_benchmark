# Baseline paper-alignment and runnability tests

This patch adds tests for two different goals.

## 1. Paper-alignment contract tests

File:

```text
tests/test_official_paper_alignment_contracts.py
```

These tests check that each named baseline still exposes the architectural modules that justify using the official paper name. Examples:

- PraNet: Res2Net adapter, RFB, aggregation, reverse-attention branches.
- ACSNet: local-context attention, global-context module, adaptive-selection modules.
- Polyp-PVT: PVT backbone adapter, cascaded fusion, camouflage-identification, similarity aggregation.
- CaraNet: CFP and axial reverse-attention blocks.
- CFA-Net: boundary prediction, cross-feature fusion, boundary aggregation.
- HSNet: hybrid Res2Net/PVT encoders, cross-semantic attention, hybrid semantic complementary module.
- CSCA U-Net: residual basic blocks, DSE, CSCA spatial attention, CSCA decoder blocks, five deep-supervision maps.
- HF-U-Net: HF bottleneck.

Important: these tests are architecture-contract tests. They cannot prove that a reimplementation is numerically identical to an official repository, but they catch the common failure case where a baseline is silently replaced by a generic U-Net-like placeholder.

## 2. Runnability tests

Files:

```text
tests/test_all_models_runnability_contracts.py
tests/test_baseline_train_step_smoke.py
tests/test_faithful_outputs_and_losses.py
tests/test_model_registry_config_coverage.py
```

These tests check that:

- all reported models are registered;
- all reported models have configs in the expected config suites;
- default benchmark scripts include every reported model, including `csca_unet`;
- every model can run forward, parse outputs, compute loss, backpropagate, and produce finite gradients;
- representative output families work through `Trainer` and `Evaluator`;
- auxiliary-output and boundary-output contracts remain compatible with the unified loss code.

## Quick manual smoke test

Run:

```bash
python scripts/smoke_all_models.py \
  --config-dir configs/paper_fair \
  --image-size 32 \
  --batch-size 2 \
  --device cpu
```

For a faster no-gradient check:

```bash
python scripts/smoke_all_models.py \
  --config-dir configs/paper_fair \
  --image-size 32 \
  --batch-size 2 \
  --device cpu \
  --no-backward
```

## Full pytest commands

Recommended targeted checks:

```bash
python -m pytest \
  tests/test_model_registry_config_coverage.py \
  tests/test_official_paper_alignment_contracts.py \
  tests/test_faithful_outputs_and_losses.py \
  tests/test_baseline_train_step_smoke.py \
  tests/test_all_models_runnability_contracts.py
```

For only CSCA U-Net:

```bash
python -m pytest -q -k "csca"
```

## 3. Behavior-level paper-contract tests

Additional file:

```text
tests/test_baseline_paper_behavior_contracts.py
```

These tests check behavioral properties that are closer to the papers than a pure module-existence test:

- reverse attention must depend on the coarse/previous prediction map;
- ACSNet local context attention must emphasize uncertain regions;
- ACSNet global context must return decoder-level context maps with matching sizes/channels;
- CSCA U-Net faithful mode must expose five deep-supervision maps;
- CSCA paper-style attention must run on a small square feature map;
- HSNet multi-scale prediction weights must be normalized and trainable;
- the common loss path must support auxiliary + boundary outputs together.

## Added official-paper functional contract tests

A stronger test file has been added:

```bash
python -m pytest -q tests/test_baseline_official_paper_functional_contracts.py
```

These tests verify functional paper contracts for PraNet, ACSNet, HarDNet-MSEG, Polyp-PVT, CaraNet, CFA-Net, HSNet, CSCA U-Net, and ResUNet++: required branches are instantiated, called in the forward pass, expose the expected auxiliary/boundary outputs, and receive gradients where applicable.
