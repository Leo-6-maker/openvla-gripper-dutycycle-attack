# Codex Autonomous Phase3 Final Status

Date: 2026-05-31

## Branch / Commit

- Branch: `exp/codex-autonomous-vis-crosssuite-20260531`
- Starting commit: `e23398108223885fc35e96d6b83f14bad17fb1a6`
- Final pushed commit: see PR branch tip and final Codex response
- PR: https://github.com/Leo-6-maker/openvla-gripper-dutycycle-attack/pull/5

## Tests

Run on server in `openvla_official_libero_20260525`:

```bash
PYTHONPATH=src python -m py_compile \
  src/gripper_attack/attack_adapter.py \
  src/gripper_attack/openvla_redecode.py \
  scripts/run_official_eval_artifact_rich.py \
  scripts/diagnostics/vis_token_flip_threshold.py \
  scripts/diagnostics/vis_arm_drift_sweep.py \
  scripts/diagnostics/crosssuite_feature_transform_audit.py \
  scripts/diagnostics/build_crosssuite_proprio_dataset_index.py \
  scripts/diagnostics/vis_one_frame_loader.py

PYTHONPATH=src python -m pytest \
  tests/v4/test_success_predicate_regression.py \
  tests/v4/test_sustained_proxy_burst.py \
  tests/v4/test_token_prefix_pgd_interface.py \
  tests/v4/test_openvla_redecode.py

bash -n scripts/*.sh
```

Expected/observed result before final commit: 30 tests passed.

## One-frame VIS Loader Status

Implemented:

```text
scripts/diagnostics/vis_one_frame_loader.py
```

The loader performs a real no-rollout path from saved RGB frame to clean OpenVLA decode, TokenPrefixPGD attack, `debug["adv_inputs"]`, and adversarial OpenVLA re-decode.

Smoke status:

- single visible GPU2 attempt failed with CUDA OOM
- visible GPUs `2,6` succeeded
- no physical GPU0 was used

## VIS Token-flip Status

Output:

```text
tables/vis_token_flip_threshold_diagnostic.csv
tables/vis_one_frame_loader_smoke.csv
```

Result:

- clean gripper token: `31872`
- adversarial gripper token: `31872`
- token flip: `false`
- clean gripper action: `0.0`
- adversarial gripper action: `0.0`
- target CE improved: `32.0000 -> 15.9500`
- perturbation Linf after budget fix: `0.0078125` under requested `eps=4/255`
- arm L2: `0.184442`

Gate VIS-1: FAIL.

VIS remains blocked before rollout because there is no decoded token/action flip and arm drift is nontrivial despite a valid processor-pixel Linf budget.

## VIS Arm-drift Status

Not run.

Reason: VIS-1 failed, so the arm-drift diagnostic is gated off.

## VIS Rollout Status

No rollout was launched.

Blocked:

- full VIS token sweep
- arm-drift sweep
- forced-window VIS micro
- detector-triggered VIS

## CrossSuite Rich Index Status

Generated:

```text
tables/crosssuite_proprio_dataset_index.csv
tables/crosssuite_relative_feature_audit.csv
```

The index now includes full-feature Object clean rows from visual-fusion clean pilot artifacts and full-feature Spatial/Goal shadow rows.

Summary:

| Suite | Rows | Full EEF xyz | Full EEF velocity | Teacher labels | Mechanism eligible | Full split candidates | Partial EEF-z candidates |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| libero_spatial | 131 | 20 | 20 | 120 | 99 | 20 | 100 |
| libero_object | 136 | 24 | 24 | 124 | 88 | 24 | 100 |
| libero_goal | 122 | 12 | 12 | 112 | 84 | 12 | 100 |
| libero_10 | 110 | 0 | 0 | 100 | 53 | 0 | 100 |

## XS-2

XS-2 passes only for a limited offline smoke proposal.

It does not pass for broad CrossSuite-v2 production training because full-feature Object coverage is limited, LIBERO-10 lacks full-feature rows, and source roots are heterogeneous.

## Object Artifact-rich Clean Data

Additional Object artifact-rich clean data is not required for a tiny smoke proposal, but it is still required for any broad/full Object retention claim.

Recommended future collection, if approved:

- Object Full10 tasks
- states 0-4
- clean only
- artifact-rich full EEF xyz/velocity + gripper/action fields
- no attack, no VIS, no sus30

No new Object collection was launched in this phase.

## Tiny Smoke

No rollout/data-generation smoke was launched.

The only real model diagnostic was the no-rollout one-frame VIS loader.

## Production Semantics

Unchanged:

- ProprioNoStep
- `sustained_command_open_proxy_30`
- `attack_burst_steps`
- success predicate / LIBERO done handling
- production defaults

## Artifact Hygiene

No checkpoints, videos, frames, rollout outputs, or model files were committed.

## Next Recommended Action

VIS:

1. Fix bf16 multi-step budget accounting in TokenPrefixPGD before another real sweep.
2. Re-run a small no-rollout threshold diagnostic only after nominal budget checks pass.
3. Treat rollout as blocked unless a valid-budget sweep shows decoded gripper token/action movement without dominant arm drift.

CrossSuite:

1. Review the smoke proposal in `reports/CROSSSUITE_PROPRIO_V2_SMOKE_PROPOSAL.md`.
2. If approved, run offline-only CrossSuite-v2 smoke training with task-only and label-shuffle baselines.
3. Do not run cross-suite sus30 attack.

## Valid Claims

- VIS re-decode helper exists.
- One-frame VIS loader is implemented and can decode clean/adversarial actions from real model execution.
- VIS token-flip diagnostic failed on the first real smoke and remains blocked.
- CrossSuite index is now sufficient for a limited offline smoke proposal.
- Object production line is unchanged.

## Forbidden Claims

- VIS attack successful.
- CrossSuite attack ready.
- ProprioNoStep universal.
- detector oracle-optimal.
- command-layer sus30 equals VIS.
