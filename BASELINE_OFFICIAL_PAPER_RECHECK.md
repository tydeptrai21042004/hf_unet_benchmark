# Baseline official-paper recheck and added test coverage

This project uses **faithful reimplementations under one unified PyTorch training/evaluation framework**. The tests below are designed to prevent accidental replacement of paper baselines by generic U-Net variants. They do not prove bit-for-bit equivalence to official repositories or pretrained checkpoints.

## Recheck summary

| Baseline | Official-paper architectural contract checked in code | Added / strengthened tests |
|---|---|---|
| PraNet | Res2Net-style/official backbone option, RFB high-level feature reduction, dense aggregation/parallel partial decoder, RA4 -> RA3 -> RA2 reverse-attention refinement, main + three lateral outputs in faithful mode. | `test_pranet_paper_contract_reverse_attention_chain_and_supervision_order` verifies RFB/DenseAggregation/RA modules, RA coarse-map chaining, output order, finite logits, and 3 auxiliary maps. |
| ACSNet | Encoder-decoder with Local Context Attention, Global Context Module, and Adaptive Selection Module at multiple decoder levels. | `test_acsnet_paper_contract_lca_gcm_asm_and_multistage_predictions` verifies LCA/GCM/ASM modules are used, four ASM stages run, channel schedule is correct, and three auxiliary decoder predictions are exposed. |
| HarDNet-MSEG | HarDNet encoder option plus RFB reductions and cascaded partial/dense decoder over the high three encoder features. | `test_hardnet_mseg_paper_contract_hardnet_encoder_plus_cascaded_partial_decoder_style` verifies RFB2/RFB3/RFB4, dense decoder input ordering from deep to shallow features, single final segmentation output, and finite logits. |
| Polyp-PVT | PVT-style/official PVTv2 backbone option, CFM coarse prediction, CIM low-level camouflage identification, SAM refinement, coarse + refined outputs. | `test_polyp_pvt_paper_contract_cfm_cim_sam_and_two_prediction_maps` verifies CFM/CIM/SAM are called exactly once, faithful output exposes one coarse auxiliary map, and refined output does not collapse to coarse output. |
| CaraNet | Res2Net-style/official backbone option, CFP on deepest feature, dense aggregation, axial reverse attention chain. | `test_caranet_paper_contract_cfp_then_axial_reverse_attention_chain` verifies CFP, DenseAggregation, ARA4 -> ARA3 -> ARA2 chaining, main + three auxiliary maps, and finite logits. |
| CFA-Net | Boundary prediction network, two-stream cross-level feature fusion, boundary aggregation modules, segmentation branch, boundary-supervised branch. | `test_cfanet_paper_contract_boundary_branch_two_stream_fusion_and_boundary_gradients` verifies boundary output shape, four aux maps, boundary loss logging, and nonzero gradients through both boundary and segmentation heads. |
| HSNet | CNN + transformer dual branches, cross-semantic attention, hybrid semantic complementary decoder, trainable multi-scale prediction fusion. | `test_hsnet_paper_contract_dual_backbone_cross_semantic_hsc_and_multiscale_fusion` verifies CSA/HSC/MSP modules, five stage logits, four auxiliary maps, normalized MSP weights, and fused main output. |
| CSCA U-Net | Six-stage encoder, DSE bottleneck, CSCA decoder blocks, five deep-supervision outputs. | `test_csca_unet_paper_contract_six_stage_dse_csca_decoder_and_all_deep_outputs_train` verifies DSE bottleneck, five deep-supervision heads, finite supervised loss, and nonzero gradients through all deep-supervision heads. |
| ResUNet++ | Residual blocks, SE gates, ASPP bridge, attention-gated decoder skip fusion. | `test_resunetpp_paper_contract_residual_se_aspp_attention_decoder_and_backward` verifies ASPP/attention decoder path and nonzero gradients through ASPP and attention gate parameters. |

## New test file

```bash
python -m pytest -q tests/test_baseline_official_paper_functional_contracts.py
```

## Recommended full local verification command

Run this on your Kaggle/Colab/local environment after installing the project requirements:

```bash
python -m pytest -q \
  tests/test_baseline_official_paper_functional_contracts.py \
  tests/test_official_paper_alignment_contracts.py \
  tests/test_baseline_paper_behavior_contracts.py \
  tests/test_faithful_outputs_and_losses.py \
  tests/test_baseline_train_step_smoke.py \
  tests/test_all_models_runnability_contracts.py
```

## Important paper wording

Use this wording in the manuscript or rebuttal:

> We compare against faithful PyTorch reimplementations of the baselines under a unified training and evaluation framework. For fairness, the same preprocessing, optimizer, scheduler, loss interface, image size, data splits, and evaluation scripts are used across methods. For paper-faithful settings, the implementation preserves each baseline's key architectural components and output-supervision contract.

Avoid claiming:

> We use the official implementation for every baseline.

unless you actually train and evaluate directly from each official repository with the official released checkpoints/configurations.
