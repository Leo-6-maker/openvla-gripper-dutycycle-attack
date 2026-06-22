# Real CLEAN Restore Goal Known-Emitter Report

Status: `STOP_AT_ONLINE_EMIT_GATE`

This report documents the authorized clean-only Goal restore follow-up. No VIS,
RAND, shuffled, oracle, PGD, or attack path was run.

## Code Changes

- Runtime thresholds are now bound to checkpoint-selected values.
- Silent threshold override is rejected unless explicitly diagnostic.
- Runtime normalization now matches the evaluator exactly: `(x - mean) / std`.
- The restore runner can consume a single explicit candidate manifest, avoiding
  implicit candidate-pool expansion.

Latest code head:

```text
49ca314b62e0df203d5089a147bff6529d9b520d
```

CPU checks:

```text
46 passed
```

## Frozen Goal Runtime Parity

Output:

```text
/data/liuyu/layer3_outputs/sc5_goal_runtime_parity_62aa583_normfix_tol1e6_20260622_191505
```

Result:

```text
SC5_RUNTIME_EVALUATOR_PARITY: PASS
threshold_source: checkpoint
tau_corridor: 0.9
tau_release: 0.1
rows: 16326
episodes: 90
phase_mismatch_count: 0
prob_mismatch_count: 0
emit_mismatch_count: 0
evaluator_emit_positive_count: 55
runtime_emit_positive_count: 55
max_corridor_abs_diff: 5.364418029785156e-07
max_release_abs_diff: 8.940696716308594e-08
tolerance: 1e-6
```

The previous probability drift was traced to the runtime adding a second
epsilon to the checkpoint `std`. After aligning normalization with the
training evaluator, row-level predictions match within float tolerance and
episode-level emit decisions match exactly.

## Diagnostic R1 Candidate Scan

The earlier 40-candidate Goal states20-23 run is retained only as diagnostic
evidence because it used pre-repair threshold behavior.

Output:

```text
/data/liuyu/layer3_outputs/real_clean_restore_r1_goal_d362f52_gpu13_20260622_142520
/data/liuyu/layer3_outputs/real_clean_restore_r1_goal_d362f52_gpu13_20260622_142520_audit_summary_190021
```

Observed:

```text
candidate_count: 40
selected_count: 0
ineligible_count: 40
reason_bucket_counts.no_natural_student_emit: 40
```

Allowed interpretation:

```text
GOAL_STATES20_23_ZERO_EMIT_UNDER_DIAGNOSTIC_RUNTIME: OBSERVED
```

Forbidden interpretation:

```text
FROZEN_GOAL_M2_R1_QUALIFICATION: not established by this diagnostic run
```

## Known Offline-Emitter Canary

Preselected episode:

```text
libero_goal|4|1|0|CLEAN
offline arm_step: 46
offline emit_step: 51
selection rule: SHA-sorted frozen predictions_test emit-positive episodes
```

First attempt:

```text
/data/liuyu/layer3_outputs/real_clean_restore_known_goal_t4_s1_701aba8_gpu13_20260622_191946
```

Result:

```text
INFRA_INVALID
reason: NameError:name 'detector' is not defined
```

This attempt is excluded from scientific interpretation.

Second attempt after engineering repair:

```text
/data/liuyu/layer3_outputs/real_clean_restore_known_goal_t4_s1_49ca314_gpu13_20260622_192212
```

Result:

```text
NO_ELIGIBLE_GOAL_RESTORE_PARENT
candidate: libero_goal|4|1|0|CLEAN
reason: candidate did not produce eligible natural Student emit
```

Therefore:

```text
KNOWN_OFFLINE_EMITTER_ONLINE_STUDENT_EMIT: FAIL
EXACT_RESTORE_3X: NOT_RUN
RESTORE_REPLAY_MISMATCH: NOT_EVALUATED
```

## Current Gate

The system now has a repaired checkpoint-threshold runtime and a passing
offline evaluator/runtime parity audit. However, the preselected offline
Goal emitter did not reproduce as an online natural Student emit in the real
LIBERO clean replay. Because the snapshot boundary was never reached, the
3/3 exact restore replay gate was not entered.

## Allowed Claims

- The Goal M2 checkpoint is now loaded with frozen checkpoint thresholds
  `0.9 / 0.1`.
- Runtime/evaluator parity passes on the frozen Goal test predictions at
  row-level phase/probability tolerance and episode-level emit decisions.
- The previous 40-candidate R1 scan remains diagnostic only.
- One known offline-emitter Goal episode was tested online and did not produce
  an eligible natural Student emit.

## Forbidden Claims

- Do not claim exact restore qualification passed.
- Do not claim online Goal Student trigger is established.
- Do not claim restore replay failed; replay was not entered.
- Do not claim VIS/RAND/shuffled/oracle/attack evidence.
- Do not expand the candidate pool or start R2 without a new gate.

## Next Gate

```text
HUMAN_GATE_REQUIRED
```

Recommended review question:

```text
Should the next step be an online/offline feature-trajectory mismatch audit
for libero_goal|4|1|0|CLEAN, or should the Goal exact-restore line stop here?
```
