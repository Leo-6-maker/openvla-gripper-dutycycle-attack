# Codex Autonomous Phase2 Re-decode + CrossSuite Status

Date: 2026-05-31

## Branch / Commit

- Branch: `exp/codex-autonomous-vis-crosssuite-20260531`
- Starting commit: `3561c20394e332d6521a0f8571e0dd5b7bc2ca49`
- Final pushed commit: see PR branch tip and final Codex response
- PR: https://github.com/Leo-6-maker/openvla-gripper-dutycycle-attack/pull/5

## Tests

Passed on server in `openvla_official_libero_20260525`:

```bash
PYTHONPATH=src python -m py_compile \
  src/gripper_attack/attack_adapter.py \
  src/gripper_attack/openvla_redecode.py \
  scripts/run_official_eval_artifact_rich.py \
  scripts/diagnostics/vis_token_flip_threshold.py \
  scripts/diagnostics/vis_arm_drift_sweep.py \
  scripts/diagnostics/crosssuite_feature_transform_audit.py \
  scripts/diagnostics/build_crosssuite_proprio_dataset_index.py

PYTHONPATH=src python -m pytest \
  tests/v4/test_success_predicate_regression.py \
  tests/v4/test_sustained_proxy_burst.py \
  tests/v4/test_token_prefix_pgd_interface.py \
  tests/v4/test_openvla_redecode.py

bash -n scripts/*.sh
```

Result: `30 passed`.

## Re-decode Helper Status

Implemented:

```text
src/gripper_attack/openvla_redecode.py
```

The helper decodes continuous OpenVLA actions from `debug["adv_inputs"]` using `input_ids` and `pixel_values`. It validates required tensors, preserves dtype/device, uses OpenVLA generation, decodes action tokens through bin centers and action stats, and rejects missing inputs, missing stats, dimension mismatch, and NaN/Inf actions.

It never uses `action_adv` and never falls back to zeros.

## One-frame VIS Token-flip Status

Status: blocked-safe.

The diagnostic script now imports the real re-decode helper, but a real one-frame token-flip smoke was not run because there is still no concrete model/frame/attack-result loader that creates `debug["adv_inputs"]` from a saved contact frame.

The script was probed in real mode and wrote:

```text
tables/vis_token_flip_threshold_diagnostic.csv
```

with an explicit error row instead of fabricated decoded actions.

## VIS Status

VIS remains blocked before rollout.

Reason:

- decoded gripper token flip was not evaluated on a real frame
- decoded gripper action movement was not evaluated on a real frame
- arm-drift gate was not run
- no random same-norm comparison exists for decoded actions

No VIS rollout, forced-window VIS micro, or detector-triggered VIS was launched.

## Richer CrossSuite Index Status

Generated:

```text
tables/crosssuite_proprio_dataset_index.csv
tables/crosssuite_relative_feature_audit.csv
```

The index combines:

- Milestone 2B student scaffold metadata
- artifact-rich official clean run directories where available
- `milestone_3a_crosssuite_proprio_shadow_20260531` Spatial/Goal shadow artifacts
- clean-only teacher window labels for mechanism eligibility

Counts:

| Suite | Rows | Full EEF xyz | Full EEF velocity | Teacher labels | Mechanism eligible | Full split candidates | Partial EEF-z candidates |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| libero_spatial | 131 | 20 | 20 | 120 | 99 | 20 | 100 |
| libero_object | 112 | 0 | 0 | 100 | 68 | 0 | 100 |
| libero_goal | 122 | 12 | 12 | 112 | 84 | 12 | 100 |
| libero_10 | 110 | 0 | 0 | 100 | 53 | 0 | 100 |

## Gate XS-2

Result: fail / blocked for full CrossSuite-v2 training.

Reason:

- Object does not have full EEF x/y and velocity in the available production-reference clean artifacts.
- Spatial/Goal shadow artifacts have full EEF xyz/velocity for a limited subset.
- All four suites remain usable only for `partial_eef_z_only` indexing.
- CrossSuite-v2 full relative-EEF-xyz training cannot satisfy the Object retention gate from this index.

CrossSuite-v2 training is not allowed from this branch.

## Production Semantics

Unchanged:

- ProprioNoStep production semantics
- `sustained_command_open_proxy_30`
- `attack_burst_steps` guard
- success predicate / LIBERO done handling
- production runner defaults

## Artifact Hygiene

No checkpoints, videos, frames, rollout outputs, model files, or large artifacts were committed. The generated index is small enough for review (`~321 KB`).

## Next Recommended Action

1. Wire a real one-frame diagnostic loader:
   - load one saved Object contact frame
   - run `TokenPrefixPGDAttacker`
   - pass `attack_result.debug["adv_inputs"]` into `redecode_openvla_action_from_adv_inputs`
   - record decoded clean/adv token and action
2. If token-flip gate passes, run the limited arm-drift sweep.
3. For CrossSuite-v2, either:
   - generate artifact-rich Object clean data with full EEF xyz/velocity, or
   - explicitly narrow the next smoke to EEF-z-only with separate claim boundaries.

## Valid Claims

- VIS re-decode helper is implemented and mock-tested.
- VIS token-flip diagnostic remains blocked-safe until a real frame/model loader exists.
- CrossSuite index is richer than before but still insufficient for full CrossSuite-v2 training.
- Production Object line remains unchanged.

## Forbidden Claims

- VIS attack successful.
- CrossSuite attack ready.
- ProprioNoStep universal.
- Detector oracle-optimal.
- Command-layer sus30 equals VIS.
