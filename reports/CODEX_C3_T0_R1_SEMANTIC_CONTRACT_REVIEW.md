# C3-T0-R1 semantic contract review

## Decision

```text
C3-T0-R1 contract             = PASS (synthetic contract only)
local C3-T0 tests             = 9 PASS / 0 FAIL / 0 ERROR
official A800 C3-T0 tests     = 9 PASS / 0 FAIL / 0 ERROR
official A800 py_compile      = PASS
real episode consumption      = 0
V23 pilot implementation      = NOT AVAILABLE
```

The implementation and configuration were frozen in commit `c923b7b`.
`C3-T0-R1` is a semantic contract gate; it is not a claim that V23 labels
already exist.

## Contract changes

- Each head projects through an explicit input allowlist.
- Forbidden aliases and outcome/future/action fields fail closed.
- `safe_release` is computed once as
  `placement AND release AND stability` and the result is passed to K10.
- K10 consumes only `protocol_steps_remaining` plus that computed safe-release
  result. Null protocol horizon is `UNKNOWN`; 0--9 is `FALSE`; at least 10 is
  `TRUE` only when safe release is `FALSE`.
- Cross-head invariants reject `safe_release=TRUE` with `k10=TRUE`, and reject
  K10 `TRUE` unless safe release is known `FALSE`.
- Unknown is never converted to `FALSE`.
- Physical gripper closing uses finite physical qpos only.
- Persistence, right-censoring, q/-q equivalence, NaN/Inf rejection, and
  non-head alias isolation are covered by tests.

## Execution boundary

The existing `run_pilot_12_v3.py` is a legacy V22 pipeline. It uses the old
V22 label adapter, terminal/task-success-dependent safe-release logic, and a
fixed `state_35` manifest. It is not a valid V23 pilot entrypoint and was not
run. Therefore the 40-episode `V23_DEV_PILOT` remains stopped until a runner
that consumes the C3-T0-R1 contract is implemented and independently checked.

No protected or real episode content was consumed for this gate. No model
inference, Student training, rollout, CAL, CHECK, or attack was run.

