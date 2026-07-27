# C3-T1D G-REC R1 handoff

Status: `G-REC-R1 = PASS_LOCAL_DIAGNOSTIC_AND_CANARY`

This is not the final G-REC decision.  The old sealed `run_A` and `run_B`
remain unchanged and are not reclassified as final evidence until a fresh R2
double run passes.

## R1A read-only localization

Diagnostic source commit: `474dca525ce04537994ff6a02412249bf2b3beb5`

Diagnostic root:

`/mnt/sdc/dty_user/openvla_attack_outputs/n5/phase3_student/grec_r1a_diag_474dca5_20260727_1600`

Root `SHA256SUMS` SHA: `2c977f9002476df30f7b5c05e9201e019fbe86484b793badd75139d817fad020`

The independent verifier found all position errors above `1e-6 m` on the
target side with source `FROZEN_MODEL_SITE`; no high-error row was one of the
217 alias rows.  The complete high-error entity census was:

| target entity | rows |
|---|---:|
| `desk_caddy_1_back_contain_region` | 217 |
| `microwave_1_heating_region` | 297 |
| `white_cabinet_1_bottom_region` | 520 |
| `wooden_cabinet_1_top_region` | 191 |
| `flat_stove_1_cook_region` | 1496 |
| `wine_rack_1_top_region` | 165 |
| `wooden_cabinet_1_top_side` | 399 |

The largest recorded row was `libero_10/task_03/state_18`, step 0,
`In -> white_cabinet_1_bottom_region`, with position L2 error
`0.022695981058470777 m`.  Its ancestor chain contains the articulated
`white_cabinet_1_bottom_level` joint.  A/B comparison changed 10 geometry
case files and 6,570 fields; the changes are reset-time target pose fields.

Conclusion: R1A localized the discrepancy to reset-time target geometry and
proved that the alias mapping is not the high-error source.

## R1B numerical amendment

Frozen before R1C in:

`configs/V23_G_REC_NUMERICAL_PROTOCOL_AMENDMENT_V1.json`

The quaternion metric is the stable sign-invariant geodesic
`2*atan2(sqrt(max(0,1-dot^2)),dot)`.  The frozen limits are:

| quantity | limit |
|---|---:|
| body-origin position | `1e-8 m` |
| body-origin rotation | `1e-7 rad` |
| geometry position | `1e-6 m` |
| geometry rotation | `1e-6 rad` |
| extent | `1e-6 m` |

Geometry position was not relaxed.  The official environment contract tests
passed: `21 passed, 0 failed, 0 errors`.

## R1C geometry source correction

The materializer and independent verifier now:

- compose fixed entities through the complete world-to-entity model ancestor
  chain;
- never use `sim.data` reset-time world pose for fixed site/body targets;
- use recorded object-state body origin for dynamic rigid objects;
- return `UNKNOWN_UNOBSERVED_JOINTED` when a jointed fixed target lacks a
  recorded parent pose;
- return `ARTICULATED_UNKNOWN` for articulated entities;
- enforce model entity identity before accepting a target or alias;
- use the same world-chain logic for articulated-joint detection.

The fresh verifier path now requires and records the frozen numerical
amendment; it does not silently fall back to old constants.

## R1D direct calibration canary

Execution source commit: `dc9ec711ec0b0b8a674dd9b30673b55991ec5bb3`

Canary root:

`/mnt/sdc/dty_user/openvla_attack_outputs/n5/phase3_student/grec_r1d_calibration_dc9ec71_20260727_1730`

Root `SHA256SUMS` SHA: `095ba45039bc05374a9171fa177ebaca54a3381e68a62b100578a99614f536fd`

The strict canary reported:

```text
status                         PASS
tasks                          40
relations                      44
side mappings                  88
direct checks                  88
MODEL_FIXED_CHAIN mappings     8
DYNAMIC_RECORDED mappings      74
ALIAS mappings                 1
ARTICULATED_UNKNOWN mappings   5
fixed-chain position max       0.0 m
fixed-chain rotation max       0.0 rad
q/-q equivalence               PASS
identity mutation              FAIL-CLOSED
descendant-joint mutation      FAIL-CLOSED
protected payload              false
model inference                false
action replay                  false
```

The first, broader canary root
`grec_r1d_calibration_05f948e_20260727_1700` is retained as a diagnostic
history root.  It is not used for the strict result because it did not bind
the index map and alias ledger.

## Boundary

The next permitted action is a fresh R2 source commit and clean detached
worktree producing new `run_A`, `run_B`, independent reviews, and comparison.
The old G-REC staging and both old sealed runs remain read-only.  No protected
payload, CAL/G10/T2R-D data, OpenVLA inference, Student training, rollout, or
attack was executed in R1.

`G-REC-R2 = NOT RUN`
`T-PILOT = BLOCKED UNTIL R2 PASS`
