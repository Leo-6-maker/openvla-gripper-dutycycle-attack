# Codex Next Experiment Harness Summary

Date: 2026-05-31

## Code Added

- VIS API helper: `get_adv_inputs_from_attack_result`
- VIS interface tests: `tests/v4/test_token_prefix_pgd_interface.py`
- Token-flip diagnostic skeleton: `scripts/diagnostics/vis_token_flip_threshold.py`
- Arm-drift diagnostic skeleton: `scripts/diagnostics/vis_arm_drift_sweep.py`
- Cross-suite relative feature audit: `scripts/diagnostics/crosssuite_feature_transform_audit.py`
- Cross-suite metadata index builder: `scripts/diagnostics/build_crosssuite_proprio_dataset_index.py`

## Tests Added

- lightweight dtype tests for bf16/fp16/fp32 fake models
- TokenPrefixPGD `action_adv=None` / `debug["adv_inputs"]` interface tests
- helper rejection tests for missing or incomplete adversarial inputs

## Diagnostics Now Available

- dry-run CSV schema for token-flip threshold diagnostics
- dry-run CSV schema for arm-drift diagnostics
- offline feature-transform audit for raw versus causal-relative EEF features
- metadata-only CrossSuite-Proprio dataset index builder

## VIS Rollout Gates

Do not run VIS rollout until:

- decoded gripper token flips at acceptable epsilon, or decoded gripper action changes meaningfully
- arm drift is controlled
- random same-norm baseline is weaker
- re-decode uses `debug["adv_inputs"]`
- no caller treats `action_adv=None` as zero action

## CrossSuite sus30 Gates

Do not run cross-suite sus30 until:

- CrossSuite-ProprioNoStep-v2 is trained separately from Object production
- only clean teacher labels are used
- relative features reduce suite shift
- Object retention gate passes
- Spatial/Goal eligible subset gates pass

## Valid Claims

- Object ProprioNoStep remains production.
- VIS engineering path has dtype/tokenization/re-decode clarified.
- Small-epsilon VIS is not rollout-ready unless decoded token flip and arm-drift gates pass.
- Cross-suite transfer is limited and needs v2 or calibration.

## Forbidden Claims

- VIS attack successful.
- Command-layer sus30 equals VIS.
- Cross-suite attack ready.
- ProprioNoStep universal.
- Detector oracle-optimal.
- Universal attack.

## Boundary

This branch adds experiment harnesses, diagnostics, and tests only. It does not run rollout, train a detector, or change production attack semantics.
