# D7 Table1 — Four-Suite Attack Main Table Completion Report

**Date**: 2026-07-08 | **Commit**: `59c33df` | **Branch**: `plan/codex-gated-experiment-v1-c2e0`

## Pipeline Status

| Gate | Status | Key Metric |
|---|---|---|
| D7A Manifest | PASS | 179 parents × 4 conditions = 716 episodes |
| D7B2 Rollout | PASS | 716/716 episodes, 32 workers (4/GPU) |
| D7C Postrun Audit | **PASS_D7_POSTRUN_AUDIT** | 0 missing, 0 unpaired, 0 violations |
| D7D Aggregate | PASS | Panel A built |
| D7E Render | PASS | Markdown table rendered |

## D7C Audit Details

```json
{
  "gate": "D7_TABLE1_POSTRUN_AUDIT",
  "status": "PASS_D7_POSTRUN_AUDIT",
  "completed": 716, "missing": 0, "unpaired": 0,
  "condition_violations": 0,
  "runtime_contract_violations": 0,
  "runtime_error_violations": 0,
  "sha_violations": 0,
  "runtime_contract_status": "PASS",
  "d7d_aggregation_blocked": false
}
```

## Panel A — Main Results

| Suite | Condition | Success/N | SR | 95% CI | Attack Frames | Trigger Rate |
|---|---|---|---|---|---|---|
| **Object** | CLEAN | 50/50 | **1.000** | [0.929, 1.000] | 0 | 1.000 |
| | TRUE_T10 | 30/50 | 0.600 | [0.462, 0.724] | 330 | 0.660 |
| | RAND_T10 | 49/50 | 0.980 | [0.895, 0.997] | 490 | 0.980 |
| | CMD_OPEN | 25/50 | 0.500 | [0.366, 0.634] | 500 | 1.000 |
| **Goal** | CLEAN | 28/33 | 0.848 | [0.691, 0.934] | 0 | 1.000 |
| | TRUE_T10 | 17/33 | 0.515 | [0.352, 0.675] | 250 | 0.758 |
| | RAND_T10 | 26/33 | 0.788 | [0.623, 0.893] | 330 | 1.000 |
| | CMD_OPEN | 26/33 | 0.788 | [0.623, 0.893] | 330 | 1.000 |
| **L10** | CLEAN | 18/50 | 0.360 | [0.241, 0.499] | 0 | 1.000 |
| | TRUE_T10 | 15/50 | 0.300 | [0.191, 0.438] | 330 | 0.660 |
| | RAND_T10 | 6/50 | 0.120 | [0.056, 0.238] | 500 | 1.000 |
| | CMD_OPEN | 13/50 | 0.260 | [0.159, 0.396] | 500 | 1.000 |
| **Spatial** | CLEAN | 37/46 | 0.804 | [0.668, 0.894] | 0 | 1.000 |
| | TRUE_T10 | 27/46 | 0.587 | [0.443, 0.717] | 440 | 0.957 |
| | RAND_T10 | 39/46 | 0.848 | [0.718, 0.924] | 460 | 1.000 |
| | CMD_OPEN | 41/46 | 0.891 | [0.770, 0.953] | 460 | 1.000 |

## Key Findings

1. **Direction Specificity**: TRUE_T10 SR < RAND_T10 SR in Object (60→98%, Δ=-38pp),
   Goal (51.5→78.8%, Δ=-27pp), Spatial (58.7→84.8%, Δ=-26pp). Gripper-open attack
   is direction-specific, not just perturbation.

2. **L10 Anomaly**: RAND_T10 (12%) < TRUE_T10 (30%) < CLEAN (36%). L10 multi-object
   tasks show RAND perturbations more destructive than targeted gripper-open, likely
   due to poor detector timing and low baseline.

3. **Object CLEAN=100%** — perfect baseline, ideal contrast for attack effect measurement.

4. **Detector Trigger Rate**: TRUE_T10 trigger rate 66-96% across suites. L10 at 66%
   reflects known C2e3 GRU recall limitation (45.6%).

5. **Artifact Integrity**: All 716 episodes pass runtime contract audit.
   normalization_applied=True, detector_checkpoint_sha256 consistent,
   context_policy=lookup_from_c2e1_dataset.

## Detector Configuration

- Model: C2e3 GRU W=16 H=128
- Checkpoint SHA: 3283f9492902f8cb...
- τ_emit=0.33, τ_suppress=0.67
- ε=6/255, K=10, MAX_STEPS=300

## Output Roots

| Artifact | Path |
|---|---|
| Rollout | `/mnt/sdc/.../d7b2_table1_normalized_rollout/` |
| Audit | `/mnt/sdc/.../d7b2_audit/` |
| Aggregate | `/mnt/sdc/.../d7b2_aggregate/` |
| Render | `/mnt/sdc/.../d7b2_render/` |
