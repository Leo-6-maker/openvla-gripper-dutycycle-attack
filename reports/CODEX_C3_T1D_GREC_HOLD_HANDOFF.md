# C3-T1D G-REC hold handoff

## Decision

```text
C3_T1D_G_REC_MATERIALIZATION = PASS_ENGINEERING_ONLY
C3_T1D_G_REC_INDEPENDENT_REVIEW = HOLD_GEOMETRY_THRESHOLD
C3_T1D_G_REC = HOLD
T_PILOT = NOT_STARTED
```

The gate stopped fail-closed after the independent review of `run_A`. No
Teacher pilot, Student training, model inference, rollout, or attack was
started.

## Source and inputs

```text
source_commit = 33e6b11a111dee4fc4b6b1943c6ca227457250c3
source_tree = 50bc06753ffcc9e19a797bfa312a011098f4d9c2
official_environment = /mnt/sdc/dty_user/openvla_attack/envs/openvla-official-a800
libero_head = 8f1084e3132a39270c3a13ebe37270a43ece2a01
libero_tree = 99f4ada3f1d62e026fc9ff2390eb4ff8a1760e60
```

The source worktree was detached and clean. Inputs were the sealed 40-episode
DEV pilot, sealed D0 receipt/DEV_POOL manifest, sealed object-state index map,
sealed task registry, official LIBERO root, and the frozen one-entry alias
ledger. The run manifest records `protected_payload_read=false`,
`action_replay=false`, `model_inference=false`, and `teacher_labeling=false`.

## Evidence root

The failed outer staging root is intentionally preserved and is not renamed
over an output root:

```text
/mnt/sdc/dty_user/openvla_attack_outputs/n5/phase3_student/.grec_v1_33e6b11_20260727_1355.staging.1000738
```

Both materializers completed and produced independently sealed run roots:

| root | SHA256SUMS SHA | SHA256SUMS.sha256 SHA |
|---|---|---|
| `run_A` | `6223b3d26aa0338590a8239623c70dc43e5a250d94f5c2bbc276192ce8619d36` | `4274b4d45d9d65e53ce436de44e70d11a09624c454fafb17c922cf2c93d09a49` |
| `run_B` | `356e3da1aa3a286b32bdc67cd0a827d35a162fb6eaa26452c70e5ff5e0bbe1f9` | `0f1576cf82abf982b14ab295b9fb9cd981059debf597d44861af7eaa853c270d` |

Each materialization reported 40 episodes. The `run_A` independent review
receipt SHA is
`0a357fd022b4b8a139d848ec6fdd513e6799e93260363c90990022dade5460d5`.
The outer evidence was not sealed because the first required independent
review did not pass.

## Independent review result

```text
episodes                 = 40
steps                    = 9422
relation_rows            = 11880
alias_rows               = 217
supported_unknown_rows   = 0
errors                   = []
```

Frozen thresholds and observed maxima:

| metric | observed | threshold | result |
|---|---:|---:|---|
| body-origin position | 0.0 m | 1e-8 m | PASS |
| body-origin rotation | 5.16191365590357e-8 rad | 1e-8 rad | FAIL |
| geometry position | 0.01713032199612712 m | 1e-6 m | FAIL |
| geometry rotation | 5.16191365590357e-8 rad | 1e-6 rad | PASS |
| geometry extents | 0.0 m | 1e-6 m | PASS |

Because the independent source-only recomputation did not meet the frozen
body-origin rotation and geometry-position thresholds, it returned `HOLD`.
The runner therefore did not execute the `run_B` independent review and did
not generate A/B comparison or an outer sealed evidence root.

## Tests

In the official environment:

```text
py_compile = PASS
13 recorded-geometry contract tests = PASS
```

## Boundary declaration

```text
protected payload read = false
model inference        = false
action replay          = false
Teacher labeling       = false
Student training       = false
rollout                = false
attack                 = false
```

The next authorized action is remediation/audit of the independent geometry
comparison. This handoff does not authorize reuse of either run as final V23
labels or entry into T-PILOT.
