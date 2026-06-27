# RAND OBJECTIVE METADATA ERRATUM

**Date**: 2026-06-27
**Affected runs**: 24 (rand/ condition, all seeds 42/123/456, all 8 state-slots)
**Severity**: Metadata only — no impact on attack generation or scientific results

## Issue

The sealed manifest (`object_breadth_120.jsonl`, SHA256: `6aedde8048...`) records:

```json
{"method": "RAND", "objective": "random_linf_control", "condition": "RAND_T10"}
```

But the Stage C launcher (`tmp/launch_stage_c.sh`) passed:

```bash
OBJ=""
```

empty string for the RAND condition. The bridge received no explicit `--attack_objective`, so `episode_summary.json` records:

```json
{"objective_id": "", "condition": "RAND_T10"}
```

The `condition` field correctly records `RAND_T10`. Only `objective_id` is empty (vs. manifest's `random_linf_control`).

## Actual Routing

The attack perturbation routing is controlled by the `--condition` CLI argument, which was set to `RAND_T10` in the launcher for these runs. The `objective_id` being empty does not affect the random perturbation implementation because:

1. The bridge checks `condition == "RAND_T10"` to select the random perturbation path
2. The `objective_id` is used only for the targeted attack path (TMA/Prefix)
3. The random perturbation does not reference `objective_id`

## Confirmation

All 24 RAND runs show:
- `token_open_duty = 0.0` (random perturbation does not produce open-token outputs)
- `open_tokens = 0`
- `arm_duty = 0.0`
- `attack_frames = 10` (except tomato_s1 which has 0 — detector no-emit)
- `task_success = True` (24/24)

This confirms the random perturbation was correctly applied.

## Resolution

1. The manifest `objective: random_linf_control` is the canonical record.
2. The launcher's empty `OBJ=""` is a metadata-only deviation.
3. RAND condition identity is established by `condition: RAND_T10` per launcher, which the bridge routes correctly.
4. No re-run needed.
5. For future panels, explicitly pass `--attack_objective random_linf_control` to the RAND bridge.

## Impact on Table 1

None. RAND FR = 0/24 remains valid. The RAND control demonstrates that matched-budget random perturbation (ε=6/255, K=10) produces zero open-token commands and zero task failures.

The `condition` field in episode_summary.json correctly reads `RAND_T10`. The only metadata gap is `objective_id=""` vs. manifest `random_linf_control`. Condition identity can be verified from either `condition` field or `token_open_duty` (0.0 for RAND vs ~1.0 for TMA/Prefix).
