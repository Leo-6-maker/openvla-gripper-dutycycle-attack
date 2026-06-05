# Qpos-Verified Phase Features For Detector V3

Phase E exposed a concrete mismatch between proxy phase labels and physical gripper state. A window can look like a near-closed phase by schedule or phase proxy while the gripper qpos is already naturally open. Detector v3 should therefore separate phase proxy from qpos-verified physical state.

## Definitions

- `true_closed`: clean-rollout MuJoCo gripper qpos is at or above the calibrated closed threshold.
- `transitional_pre_open`: clean-rollout MuJoCo qpos is between open and closed thresholds.
- `natural_open`: clean-rollout MuJoCo qpos is at or below the open threshold.
- `missing_qpos`: MuJoCo qpos unavailable. Obs-only qpos is audit-only by default.

## Candidate Features

- `qpos_phase_class` one-hot.
- `true_closed_score`.
- `natural_open_score`.
- `phase_proxy_mismatch`.
- `qpos_source_confidence`, with MuJoCo qpos as high confidence and obs-only as manual-review/audit-only.

## Leakage Risks

- Only clean rollout qpos is allowed.
- No attacked qpos.
- No task outcome, done/success outcome, VIS result, manual audit outcome, or Phase D/E proxy label as detector input.

## Expected Benefit

Qpos-verified features should reduce false positives caused by natural-open or phase-proxy-mismatched windows and improve negative recall.

## Required Ablations

- `D_causal_safe`.
- `D_causal_safe + qpos_verified`.
- `phase_only`.
- `task_key_only`.
