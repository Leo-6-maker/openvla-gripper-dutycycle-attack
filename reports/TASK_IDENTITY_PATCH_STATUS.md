# Task Identity + Metadata Standardization Patch Status

**Branch**: `fix/task-identity-metadata-20260523`
**Date**: 2026-05-23
**Mode**: Read-only audit + metadata patch. No rollout, no GPU, no VIS/random/attack.

## Patch Summary

### Files Created
| File | Purpose |
|------|---------|
| `src/utils/__init__.py` | Package init |
| `src/utils/task_identity.py` | Canonical task identity mapping + dict-based condition configs |
| `tests/v4/test_task_identity.py` | Test task identity mapping and condition schemas |
| `tests/v4/test_metadata_schema.py` | Test run_manifest/summary schema requirements |
| `tests/v4/test_clean_protocol.py` | Test clean/autowindow protocol invariants |

### Files Modified
| File | Change |
|------|--------|
| `docs/claim_and_evidence.md` | Added Task Identity section with runner_task_id vs semantic_task_name clarification |

## Key Changes

### 1. Task Identity Mapping
- `runner_task_id`: `libero_spatial_black_bowl` (passed to runner --task_id)
- `semantic_task_name`: `goal_put_the_bowl_on_the_plate` (used in run_ids, reports, Table1 joins)
- `suite`: `libero_spatial` (NOT `libero_goal`)
- `is_black_bowl_related`: true (current task IS black bowl)
- `is_non_black_bowl_claim`: false

### 2. Dict-based Condition Configs
Replaces fragile tuple-based `matched_conditions` with dict-based configs.
Each condition has explicit keys: `condition_name`, `attack_objective`, `temporal_init`,
`force_open_raw_gripper`, `rho`, `cw_margin`, `epsilon`, `step_size`, `attack_steps`,
`is_attack`, `is_control`.

This eliminates the `ValueError: expected 9, got 8` bug permanently.

### 3. Evidence Docs Clarification
Claim boundary now explicitly states:
- Bowl-on-plate spatial diagnostic experiments use the black-bowl task
- "nonbb" directory names are legacy; task identity corrected in this patch
- A true non-Black-Bowl claim requires a separate task with non-black-bowl object semantics

### 4. Test Coverage
- 20+ assertions on task identity consistency
- Condition schema validation (all fields present, no empty names)
- Control vs attack condition discrimination
- Run_id construction format verification

## What This Patch Does NOT Do
- Does NOT run new experiments
- Does NOT modify existing artifacts
- Does NOT change claim conclusions
- Does NOT fix the driver_v2.py matched_conditions tuple bug in-place (dict-based fix is in task_identity.py; driver_v2.py on server still needs update)
- Does NOT start Table2 / 142 / broad rollout

## Status

```
final_state=task_identity_metadata_patch_ready
go_to_rollout=false
go_to_table2=false
go_to_142=false
next_action=review_patch_then_select_true_non_black_bowl_candidate
```
