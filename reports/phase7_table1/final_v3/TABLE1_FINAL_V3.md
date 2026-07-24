# Table 1 — Final Freeze V3

**Status**: `RNAD_MECHANISM_FREEZE_V3 = PASS`
**Previous**: `c82e85c` (v1, mixed-space), `94c0e57` (v2, wrong env grip range)

## Corrected rNAD V3 (representation-aware, same-space only)

Action stats loaded at runtime from `dataset_statistics.json` (SHA: `a4b953c2`).
Policy gripper range = model Q99-Q01. Env gripper range = 2.0 (policy [0,1] -> postprocess -> env [-1,+1]).

| Metric | TMA no-lock | TMA ArmLock | Prefix no-lock | Prefix ArmLock |
|--------|:----------:|:----------:|:--------------:|:--------------:|
| rNAD_pol_prelock_arm | 0.0719 | 0.0757 | 0.0491 | 0.0480 |
| rNAD_pol_exec_arm | 0.0719 | **0.0000** | 0.0491 | **0.0000** |
| rNAD_env_exec_arm | 0.0719 | **0.000000** | 0.0491 | **0.000000** |
| rNAD_pol_prelock_grip | 0.7747 | 0.7563 | 0.8006 | 0.7895 |
| rNAD_env_exec_grip | **0.7778** | **0.7593** | **0.8037** | **0.7926** |

540 ArmLock attack frames, 0 violations. 108/108 parsed, 0 skipped, 0 nonfinite.

### Paired rNAD_env_exec_grip Deltas (ArmLock - NoLock, N=27)

| Objective | Mean Delta | 95% Bootstrap CI | pos/neg/zero |
|-----------|:---:|:---:|:---:|
| TMA | -0.0185 | [-0.048, +0.011] | 3/8/16 |
| Prefix | -0.0111 | [-0.041, +0.022] | 3/8/16 |

Both CIs cross zero. 16/27 pairs show zero difference.
ArmLock produces no clear paired directional effect on env exec grip discrepancy.

### Provenance

- `dataset_statistics.json` SHA: `a4b953c2ab889176a019ce92b86b855bc5990312d1a46cd7a2d9da1abbece861`
- `ACTION_STATS_SOURCE.json` SHA: `25b98ddb4a8b60c4c6e88424e72cde566fcdebc14bb96c5aa511f53b23b217c9`
- Analysis script: `tmp/rnad_v3_final.py`

## CQFR V3

- 108 runs -> 68 unique video hashes (12 duplicate groups)
- Duplicate-group consistency: 0 task conflicts, 0 instruction conflicts, 0 success conflicts
- Global shuffle (seed 42), mtime scrubbed to 2000-01-01
- Actual video metadata from ffprobe
- 108-row private mapping for per-condition CQFR
- Public ZIP: `24191cafe3ebdb1d18d81638b5288438c4f88a41685503836833ec9b95774035` (2.9MB, verified)
- All public files SHA256SUMS included

## Frozen Claims

1. TMA and Prefix produce high task failure rates in LIBERO-Object.
2. Attack transfers to 8 new state slots with state-dependent effectiveness.
3. Prefix ArmLock 100% FR limited to reference cells.
4. tomato_sauce_s1: state-specific detector non-emission.
5. ArmLock zeros executed arm discrepancy (540 frames, 0 violations).
6. High failure rates persist after arm perturbation removal.
7. rNAD_env_exec_grip is large (0.76-0.80) with small ArmLock-noLock paired deltas and CIs crossing zero.
8. RAND and Adapted Untargeted Clean-Token CE PGD produce zero observed failures.
9. Prefix ~2.9-3.5x (mean) / ~2.9-3.1x (median) TMA attack-preparation latency.
10. Legacy timing evidence: Random-Time < Early-Shift < Student Trigger (provenance reconstruction pending).
