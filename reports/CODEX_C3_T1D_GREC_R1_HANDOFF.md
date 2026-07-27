# C3-T1D G-REC R1 handoff

## Current decision

```text
G-REC-R1A discrepancy localization = PASS
G-REC-R1B numerical amendment      = PASS
G-REC-R1C geometry correction       = HOLD_RESET_VARYING_FIXED_TARGET
G-REC-R1D direct calibration        = HOLD_RESET_INVARIANCE
G-REC-R2 fresh A/B                 = HOLD_INDEPENDENT_REVIEW
G-REC final                        = HOLD_DATA_GAP
T-PILOT                            = BLOCKED
```

No old root was modified or re-sealed.  No protected payload, CAL/G10/T2R-D
data, OpenVLA inference, Student training, rollout, or attack was executed.

## R1A — read-only discrepancy localization

Diagnostic root:
`/mnt/sdc/dty_user/openvla_attack_outputs/n5/phase3_student/grec_r1a_diag_474dca5_20260727_1600`

`SHA256SUMS` SHA:
`2c977f9002476df30f7b5c05e9201e019fbe86484b793badd75139d817fad020`

All rows above `1e-6 m` were target-side `FROZEN_MODEL_SITE`; none of the 217
alias rows was a high-error row.  The complete high-error census was:

| target entity | rows |
|---|---:|
| desk_caddy_1_back_contain_region | 217 |
| microwave_1_heating_region | 297 |
| white_cabinet_1_bottom_region | 520 |
| wooden_cabinet_1_top_region | 191 |
| flat_stove_1_cook_region | 1496 |
| wine_rack_1_top_region | 165 |
| wooden_cabinet_1_top_side | 399 |

The largest old diagnostic row was
`libero_10/task_03/state_18`, step 0, `In` to
`white_cabinet_1_bottom_region`, L2 position error
`0.022695981058470777 m`.  The A/B comparison changed 10 geometry files and
6,570 fields.  This localized the problem to target geometry, but did not by
itself establish the target pose source.

## R1B — frozen numerical amendment

Configuration:
`configs/V23_G_REC_NUMERICAL_PROTOCOL_AMENDMENT_V1.json`

The frozen sign-invariant quaternion metric is
`2*atan2(sqrt(max(0,1-dot^2)),dot)`.  The limits are:

| quantity | limit |
|---|---:|
| body-origin position | `1e-8 m` |
| body-origin rotation | `1e-7 rad` |
| geometry position | `1e-6 m` |
| geometry rotation | `1e-6 rad` |
| extent | `1e-6 m` |

Geometry position was not relaxed.  The official-environment contract test
run before the fresh replay passed `21 passed, 0 failed, 0 errors`.

## R1C — source correction and remaining semantic failure

The materializer and independent verifier now compose fixed entities through the
complete model ancestor chain and no longer use `sim.data` reset-time poses for
fixed geometry.  Dynamic rigid entities use recorded object-state body origins;
jointed entities without recorded parent telemetry fail closed as
`UNKNOWN_UNOBSERVED_JOINTED`, and articulated entities remain
`ARTICULATED_UNKNOWN`.

The corrected cross-reset canary exposed the remaining problem: LIBERO's
placement sampler changes fixture body transforms during reset.  A fixed-chain
reference captured before the extra reset did not remain invariant after the
reset.  This is a geometry-source/data-gap finding, not a reason to widen the
position threshold.

Corrected canary root:
`/mnt/sdc/dty_user/openvla_attack_outputs/n5/phase3_student/grec_r1d_reset_audit_20260727_1820`

`SHA256SUMS` SHA:
`e827e82d481e424a64f4127db78d518f3a1b978126197ab4f934f6de5498f711`

`SHA256SUMS.sha256` SHA:
`0cd01edf84721de6d2e81ea519d58cc1cc5e1bc2ec930282eb23bdd21279d9f9`

Result: `HOLD`, 44 relations, 88 mappings, 8 fixed mappings, 74 recorded
dynamic mappings, 1 alias mapping, 5 articulated-unknown mappings, and 10
reset-invariance failures.  The older strict canary root
`grec_r1d_calibration_dc9ec71_20260727_1730` is retained as a historical
diagnostic only; its same-reset comparison was tautological and is not a
formal R1D PASS.

## R2 — fresh replay result

Source commit: `4194f47efa58cd8f57d79cf19aa2701d3d6ead5f`.

The fresh R2 materialization reached 40/40 episodes, 9,422/9,422 steps,
11,880 relation rows, and 217 alias rows, but the independent verifier failed
closed before the B-side review could complete.  The unsealed staging root is
preserved at:

`/mnt/sdc/dty_user/openvla_attack_outputs/n5/phase3_student/.grec_r2_4194f47_20260727_1800.staging.1219803`

Observed verifier summary:

```text
body-origin position max       = 0.0 m
body-origin rotation max       = 5.16191365590357e-08 rad
geometry position max           = 0.013316277407554006 m
geometry rotation max           = 5.16191365590357e-08 rad
supported UNKNOWN rows         = 711
high position rows              = 2574
high rows source                = MODEL_FIXED_CHAIN target geometry
```

The largest R2 row was `libero_goal/task_09/state_16`, step 0,
`wine_rack_1_top_region`, with position L2 error
`0.01799183047 m`.  This is not numerical noise and is not converted to a
negative label.

## Required data-gap recovery

The next permitted action is a read-only telemetry/schema audit over the
already bound 40 DEV payloads.  It must bind all three files per episode,
count collector-source variants, and establish whether target/site/body world
poses or only an `initial_state_sha256` digest are present.  If target fixture
poses are absent, G-REC-DATA-FALLBACK is required: new FIT-only telemetry must
be collected directly, with static targets from frozen model/site transforms
and dynamic targets from recorded object-state body origins.  Action replay
geometry cannot be used as original-trajectory truth.

The audit has now been executed without model inference or replay:

```text
root: /mnt/sdc/dty_user/openvla_attack_outputs/n5/phase3_student/grec_r1c_telemetry_audit_33d4886_20260727_1910
SHA256SUMS: f36c572d09f630d682b35eddbbf4e9c54ddaea0551cd5a26ff63897de1353165
SHA256SUMS.sha256: 607def6a438dac63c17b1f80a00bf7bb0edf4628ae3f6b6c9f3ac3b935056aab
episodes: 40/40
steps: 9422/9422
target/site/body pose paths in payload: 0 episode-level matches
```

Four collector-source variants are present, with `official_clean_worker.py`
episode counts `7, 3, 11, 19`; the other collector component hashes also
split into two variants.  Every episode has an `initial_state_sha256` digest,
but no initial-state payload and no per-step target/site/body world-pose
field.  This establishes the data gap; it does not prove the four collector
variants semantically equivalent.

No new R2 run is authorized until this source decision is sealed.  If the
missing target pose cannot be recovered or collector variants cannot be proven
equivalent, the result remains `DATA_GAP` and T-PILOT stays blocked.
