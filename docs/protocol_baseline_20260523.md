# Protocol Baseline 2026-05-23

**Branch**: `fix/protocol-schema-and-condition-config-20260523`
**Commit**: `f5acd9b`
**Status**: Clean baseline consolidation

## Purpose

This document establishes the protocol baseline for all future experiments
on the OpenVLA gripper duty-cycle attack project. It supersedes all prior
protocol definitions that were found to contain drift errors.

## What Changed From DeepSeek Drift Config

The previous `MATCHED_CONDITIONS` (now `DEPRECATED_DEEPSEEK_DRIFT_MATCHED_CONDITIONS`)
contained multiple protocol errors discovered through Codex r0/r1/r2 raw
step_records audit:

| Parameter | DeepSeek (wrong) | Codex (correct) | Fixed? |
|-----------|-----------------|-----------------|--------|
| command_open rho | 0.0 | 1.0 | YES |
| targeted attack_objective | gripper_logit_margin_cw | force_gripper_open_token_ce | YES |
| targeted epsilon | 0.10 | 0.25 | YES |
| targeted step_size | 0.020 | 0.050 | YES |
| targeted attack_steps | 20 | 60 | YES |
| targeted force_open_raw_gripper | None | 1.0 | YES |
| random rho | 0.0 (is_attack=false) | 1.0 (is_attack=true) | YES |
| VIS_current rho | 0.0 (is_attack=false) | 1.0 (is_attack=true) | YES |
| matched seed protocol | seed = repeat_id + 100 | seed = repeat_id | YES |

## Current Protocol Definitions

All protocol definitions are in `src/utils/condition_protocols.py`:

| Protocol | Variable | Category |
|----------|----------|----------|
| Clean Detect | `CLEAN_DETECT_PROTOCOL` | clean_baseline |
| Random Same-Window | `CODEX_LEGACY_RANDOM_SAME_WINDOW` | legacy_codex_attack |
| VIS-Current Same-Window | `CODEX_LEGACY_VIS_CURRENT_SAME_WINDOW` | legacy_codex_attack |
| Targeted Force-Gripper-Open Token CE | `CODEX_LEGACY_TARGETED_FORCE_GRIPPER_OPEN` | main_targeted_attack |
| Command-Open Oracle | `COMMAND_OPEN_ORACLE_PROTOCOL` | oracle_upper_bound |
| Gripper Logit Margin CW | `DIAGNOSTIC_GRIPPER_MARGIN_PROTOCOL` | diagnostic |
| Open Region CE | `DIAGNOSTIC_OPEN_REGION_CE_PROTOCOL` | diagnostic |

The convenience list `LEGACY_CODEX_STATE5_MATCHED_CONDITIONS` provides the
four main Codex conditions in run order:
random → VIS_current → targeted → command_open.

## Fail-Fast Guards

### Deprecated Condition Access

Importing `MATCHED_CONDITIONS` or `TRIAGE_MATCHED_CONDITIONS` from `task_identity.py`
returns a `_DeprecatedProtocolSentinel` that raises `RuntimeError` on any
iteration, indexing, len(), or bool() access. The error message points to
`LEGACY_CODEX_STATE5_MATCHED_CONDITIONS` as the correct replacement.

### Protocol Validators

`src/utils/protocol_validation.py` provides runtime fail-fast guards:

| Validator | Checks |
|-----------|--------|
| `validate_command_open_protocol` | rho>0, objective, env_extra, attack_steps>=1 |
| `validate_same_seed_protocol` | matched_seed == clean_seed |
| `validate_window_source` | Not table1_prior_window |
| `validate_codex_targeted_protocol` | Exact params (eps=0.25, ss=0.050, asteps=60, fo=1.0) |
| `validate_condition_config_schema` | Required + recommended fields, objective-specific checks |

All validators use `ProtocolValidationError(ValueError)` — NOT `assert` —
so they survive `python -O`.

## Claim Boundary

- **Black-Bowl Spatial evidence**: `libero_spatial_black_bowl`, state5, bowl-on-plate
- **NOT True Non-BB evidence**: Requires separate Object-suite calibration
- **NOT Full Table2**: State5 confirmatory repeats only
- **NOT Broad benchmark**: Single task, single state, single mechanism
- **Human review**: Pending

## Task Identity

- `runner_task_id`: `libero_spatial_black_bowl` (passed to `--task_id`)
- `semantic_task_name`: `goal_put_the_bowl_on_the_plate` (used in run_ids and Table1 joins)
- `suite`: `libero_spatial`
- `is_black_bowl_related`: true
- `is_non_black_bowl_claim`: false

## Driver Integration

Future experiment drivers should:

1. Import protocols from `src.utils.condition_protocols`
2. Call validators from `src.utils.protocol_validation` at config-load time
3. Handle `attack_objective=None` + `omit_attack_objective_cli_arg=True` by OMITTING
   the `--attack_objective` CLI flag entirely
4. Use `LEGACY_CODEX_STATE5_MATCHED_CONDITIONS` for matched condition runs
5. Never use `MATCHED_CONDITIONS` or `TRIAGE_MATCHED_CONDITIONS` (fail-fast sentinel)

## Tests

All tests in `tests/v4/`. Run with `python -m pytest tests/v4 -q`.
No GPU required. Smoke test without pytest:

```python
python -c "
from src.utils.condition_protocols import *
from src.utils.protocol_validation import *
from src.utils.task_identity import make_run_id
validate_command_open_protocol(COMMAND_OPEN_ORACLE_PROTOCOL)
validate_codex_targeted_protocol(CODEX_LEGACY_TARGETED_FORCE_GRIPPER_OPEN)
for c in LEGACY_CODEX_STATE5_MATCHED_CONDITIONS:
    validate_condition_config_schema(c)
print('smoke ok')
"
```
