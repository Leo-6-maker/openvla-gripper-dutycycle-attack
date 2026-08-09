# C3-T0 semantic contract review

> Historical pre-R1 review. Superseded by
> `CODEX_C3_T0_R1_SEMANTIC_CONTRACT_REVIEW.md`; the current contract test
> count is 9.

## Decision

```text
C3-T0 semantic contract       = PASS (synthetic contract only)
five-head semantic tests      = 8 PASS / 0 FAIL
official A800 Python check    = PASS
real episode consumption      = 0
Teacher materialization       = NOT STARTED
Student training/inference   = NOT STARTED
rollout/attack               = NOT STARTED
```

The contract is frozen in
`configs/C3_T0_TEACHER_SEMANTIC_CONTRACT_V1.json`. The implementation is
`n5/phase2_labels/c3_t0_semantic_contract.py`. It is a contract/test module,
not a claim that V23 labels have been produced.

## Frozen semantics

- Every head returns `TRUE`, `FALSE`, or `UNKNOWN`; `UNKNOWN` always has
  `mask=false` and is never converted to a negative label.
- `physical_criticality` consumes physical grasp/transport evidence only and
  rejects task success, terminal, reward, outcome, attack, and future fields.
- `k10_feasible` is right-censor conservative and requires a known remaining
  horizon and known safe-release state.
- `safe_release` is exactly placement AND release AND stability.
- `instability` requires physical slip/regrasp/contact-loss evidence.
- `gripper_closing_state` uses finite physical qpos and a physical threshold;
  policy/action command is not an input.
- Positive persistence requires two contiguous confirmed steps; unconfirmed
  positives remain `UNKNOWN`.
- Quaternion sign equivalence is accepted; nonfinite values are rejected.

## Verification

The synthetic test file contains the causal, perturbation, mask, persistence,
right-censor, q/-q, NaN/Inf, and safe-release conjunction checks. It passed
both locally and with the official environment:

```text
python -m unittest discover -s n5/phase2_labels/tests -p test_c3_t0_semantic_contract.py -v
7 tests passed, 0 failed
python -m py_compile ...
PASS
```

No protected or real episode content was read for this gate.
