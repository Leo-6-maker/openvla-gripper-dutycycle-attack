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

Additional local validation after contact-frame collection planning:

```bash
python -m py_compile \
  src/gripper_attack/attack_adapter.py \
  scripts/run_official_eval_artifact_rich.py \
  src/gripper_attack/openvla_redecode.py \
  scripts/diagnostics/vis_token_flip_threshold.py \
  scripts/diagnostics/vis_arm_drift_sweep.py \
  scripts/diagnostics/crosssuite_feature_transform_audit.py \
  scripts/diagnostics/build_crosssuite_proprio_dataset_index.py \
  scripts/diagnostics/select_vis_contact_frames.py \
  scripts/diagnostics/build_vis_contact_frame_collection_plan.py

pytest \
  tests/v4/test_success_predicate_regression.py \
  tests/v4/test_sustained_proxy_burst.py \
  tests/v4/test_token_prefix_pgd_interface.py \
  tests/v4/test_openvla_redecode.py \
  tests/v4/test_vis_arm_drift_sweep.py \
  tests/v4/test_select_vis_contact_frames.py \
  tests/v4/test_build_vis_contact_frame_collection_plan.py
```

Observed local result: 39 tests passed.

`bash -n scripts/*.sh` passed under Git Bash. The default `bash` entrypoint on this Windows host failed because no default WSL distribution is installed.

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

## Contact-frame Collection Plan

Generated a proposal only:

```text
scripts/diagnostics/build_vis_contact_frame_collection_plan.py
tables/vis_contact_frame_collection_plan.csv
reports/VIS_CONTACT_FRAME_COLLECTION_PROPOSAL.md
```

Planned clean-only frame dump rows:

| Task | State | Target policy step | Requested frames |
| --- | ---: | ---: | --- |
| ketchup | 0 | 98 | 96..100 |
| tomato_sauce | 0 | 134 | 132..136 |
| cream_cheese | 0 | 143 | 141..145 |

The plan is not executed. It uses `scripts/run_official_eval_artifact_rich.py` with clean-only settings, `attack_condition=clean`, no detector, no VIS, no sus30, and a maximum of three state-0 clean episodes if explicitly approved later.

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

1. If approved, execute only the three-row clean contact-frame collection plan in `tables/vis_contact_frame_collection_plan.csv`.
2. Verify that `frames/step_0098.png`, `frames/step_0134.png`, and `frames/step_0143.png` equivalents exist in the new clean artifact-rich output.
3. Re-run no-rollout VIS confirmation on those verified contact/carry frames.
4. Do not run forced-window VIS micro until verified contact-frame evidence passes and explicit approval is given.

CrossSuite:

1. Review the smoke proposal in `reports/CROSSSUITE_PROPRIO_V2_SMOKE_PROPOSAL.md`.
2. If approved, run offline-only CrossSuite-v2 smoke training with task-only and label-shuffle baselines.
3. Do not run cross-suite sus30 attack.

## Valid Claims

- VIS re-decode helper exists.
- One-frame VIS loader is implemented and can decode clean/adversarial actions from real model execution.
- VIS token-flip is now observed on one frame after bf16-safe budget accounting.
- One-frame arm-drift/random baseline diagnostic passed for that frame.
- Four-frame no-rollout confirmation is partial: ketchup and tomato pass; cream_cheese s0/s1 fail.
- Contact-frame audit shows previous saved VIS frames are wait/pre-policy frames; selected contact/carry steps currently lack frame images.
- A clean-only contact-frame collection plan exists, but it has not been executed.
- CrossSuite index is now sufficient for a limited offline smoke proposal.
- Object production line is unchanged.

## Forbidden Claims

- VIS attack successful.
- CrossSuite attack ready.
- ProprioNoStep universal.
- detector oracle-optimal.
- command-layer sus30 equals VIS.
