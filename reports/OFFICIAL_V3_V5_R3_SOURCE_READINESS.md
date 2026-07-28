# Official V3 V5 R3 Source Readiness

## Current source

```text
branch = codex/v5-student-learnability-r3-20260728
HEAD   = 002007e27fdde641e8d9c9364fdbf0f6de5a9999
tree   = e3d0700dae65eaeddb8f736d87a5724490210652
base   = 6504e94567d9f6bc6394185daf26a60eccf0bb19
R2 execution source = f6f619b4fc6b1706aff1cf1967c73e8cc10b8c28
```

The R3 protocol remains frozen at `R3_DEV_PROTOCOL.json`; no label or episode
payload was read while preparing this source bundle.

## Implemented, source-only

- `audit_r3_contact_input.py`: exact eight-episode manifest gate, recursive
  seal/closure, source commit/tree/SHA validation, target/object/entity checks,
  complete contact record checks, and fail-closed protected boundary.
- `v5_r3_teacher.py`: independent five-head causal TRUE/FALSE/UNKNOWN labels,
  contact-complete geometry, sign-invariant quaternion distance, safe-release
  conjunction, K10 right-censor rule, and no outcome/future fallback.
- `run_r3_v23_teacher.py`: consumes only a canary with the exact consumable
  status; refuses old Fresh40 proxy roots and existing output roots.
- `v5_r3_student.py`: per-head UNKNOWN masking, canonical five-head loss,
  deterministic label-shuffle control, and finite-output checks.
- `run_r3_learnability_smoke.py`: synthetic-only, explicitly
  `ENGINEERING_NONCONSUMABLE`; it cannot consume real episode roots.

## Official environment verification

```text
server       = pm-364c0001 / dty_user
environment  = /mnt/sdc/dty_user/openvla_attack/envs/openvla-official-a800
worktree     = /mnt/sdc/dty_user/worktrees/codex-v5-student-r3-20260728
HEAD         = 002007e27fdde641e8d9c9364fdbf0f6de5a9999
worktree     = clean
py_compile   = PASS
R3 tests     = 31 passed, 0 failed, 0 critical skip
synthetic smoke = finite, nonconsumable
GPU tasks    = 0
```

## Gate state

```text
R3_0_SOURCE_AND_PROTOCOL_CLOSURE = PASS
R3_1_AUDITOR_AND_RUNNER_READY    = PASS_SOURCE_ONLY
R3_1_INPUT_GATE                  = HOLD_INPUT
R3_2_TEACHER_LABELS              = NOT RUN
R3_3_COVERAGE                    = NOT RUN
R3_4_STUDENT_LEARNABILITY        = NOT RUN ON REAL DATA
R3_5_GENERALIZATION              = NOT RUN
R3_6_SHADOW                      = NOT RUN
```

R3-1 may start immediately when DeepSeek provides a separately sealed,
identity-frozen eight-episode root whose manifest status is exactly
`PASS_ENGINEERING_CONSUMABLE_INPUT_GATE`. The auditor will require all eight
episodes; it will not select rows from a directory still being written.

```text
protected reads = 0
new Teacher labels = 0
Student training on real data = 0
OpenVLA / rollout / attack = 0
```
