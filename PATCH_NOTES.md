# HF-UNet benchmark baseline patch notes

## What was changed

This patch rewrites the advanced baseline implementations to be much closer to the official paper architectures while keeping the existing YAML configs unchanged.

### Reworked baselines
- PraNet: added RFB reduction blocks, dense partial decoder aggregation, and staged reverse attention refinement.
- ACSNet: added Local Context Attention (LCA), Global Context Module (GCM), and Adaptive Selection Module (ASM).
- CFANet: added boundary prediction network, two-stream cross-level fusion, and boundary aggregation modules.
- CaraNet: added CFP context module and axial reverse attention refinement.
- HarDNet-MSEG: replaced the toy HarDBlock with a harmonic-dense HarDBlock and used an RFB + dense aggregation decoder.
- Polyp-PVT: replaced the toy attention encoder with a PVT-like backbone and added CFM, CIM, and SAM modules.

### Compatibility
- Existing config files were kept as-is.
- Legacy class aliases such as `PraNetLite` and `ACSNetLite` are preserved for compatibility.
- Existing training / evaluation entry points remain unchanged.

## Added tests
- Default-config forward-contract tests for all models.
- Architecture-contract tests for paper-specific modules in each advanced baseline.
- Audit test to ensure the repo no longer flags the advanced baselines as simplified Lite-only implementations.

## Validation
- `pytest -q` passes for the patched repo.
