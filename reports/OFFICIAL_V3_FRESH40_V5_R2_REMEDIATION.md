# Official V3 Fresh40 V5 R2 Remediation

## Scope and provenance

This report supersedes the provisional R1 summary at report commit `4041b0ad6a1af4f47c47db63ad33681799f2449d`. The valid R2 execution code commit is:

```text
f6f619b4fc6b1706aff1cf1967c73e8cc10b8c28
```

The earlier report contained one 41-character rendering of `404c5279...`; that historical typo is not used as a source binding. All R2 roots below bind the actual `f6f619b4...` commit and record runtime chronology. Directory timestamp labels are explicitly non-authoritative.

No historical root was modified. No protected split, Fresh670, OpenVLA, rollout, or attack input was read.

## R2-R1 contract corrections

The event label is now the frozen three-value OR:

```text
any TRUE              -> TRUE
all FALSE             -> FALSE
otherwise             -> UNKNOWN
```

Thus `[TRUE, UNKNOWN]` is a positive event, `[FALSE, UNKNOWN]` is UNKNOWN, and `[FALSE, FALSE]` is negative. UNKNOWN is never converted to negative.

Candidate-independent Teacher critical events are also built from the complete development timeline before candidate gating. Candidate overlap is then computed as an interval relation, separately from candidate-event labels. The output now distinguishes:

```text
total_dev_steps
candidate_steps_processed
known_critical_steps
unknown_critical_steps
teacher_critical_steps
teacher_critical_candidate_steps
candidate_events.{total,known_positive,known_negative,unknown}
teacher_critical_events.{total,overlapped_by_candidate,missed_by_candidate,event_recall_ceiling}
```

Runtime manifests now include `source_commit`, `source_commit_time`, `actual_started_at`, `actual_ended_at`, and `directory_label_timestamp_is_authoritative=false`.

The test suite now exercises the real `_publish()` non-overwrite guard, quaternion `q/-q` geodesic equivalence, three-value event semantics, partial-UNKNOWN scoring, and candidate-independent critical event construction.

## Official test gate

Using `/mnt/sdc/dty_user/openvla_attack/envs/openvla-official-a800`:

```text
py_compile = PASS
pytest = 11 passed, 0 failed
```

## Corrected R2 development replay

The replay used the existing sealed dataset and existing checkpoints only:

```text
development identities       = 8
total development steps      = 2,363
candidate steps              = 558
known critical steps         = 153
unknown critical steps       = 2,210
Teacher TRUE steps           = 106
Teacher TRUE candidate steps = 87
step candidate coverage      = 87/106 = 0.8207547169811321
candidate events              = 85
known positive candidate events = 3
known negative candidate events = 6
UNKNOWN candidate events      = 76
```

Candidate-independent event coverage:

```text
Teacher critical events       = 2
overlapped by candidate       = 2
missed by candidate           = 0
event candidate ceiling       = 2/2 = 1.0
```

The step coverage and event coverage are intentionally separate. The event ceiling is not inferred from `87/106`.

### Oracle ladder

| Stage | Definition | Positive event recall | Known-negative FPR | Mean selected latency |
|---|---|---:|---:|---:|
| A | independent Teacher event / candidate overlap | 2/2 ceiling | n/a | n/a |
| B | candidate + Teacher critical | 3/3 | 0/6 | 7.33 steps |
| C | candidate + predicted critical | 2/3 | 0/6 | 6 steps |
| D | predicted critical + Teacher auxiliary veto | 0/3 | 0/6 | n/a |
| E | predicted three-head | 1/3 | 0/6 | 12 steps |
| F | predicted full-five one-shot | 1/3 | 0/6 | 12 steps |

These are development diagnostics over a proxy Teacher with only three known positive candidate events. They are not a scientific promotion gate. Stage B is not a counterfactual vulnerability measurement; it is a Teacher-label diagnostic.

## Head diagnostics

All values below are development-only and use the sealed dev split. The prevalence column is the positive fraction among known labels.

| Checkpoint / head | Known + / - | Prevalence | AUROC | AUPRC | Recall@0.5 | Precision | Balanced accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|
| critical checkpoint / physical criticality | 106 / 47 | 0.6928 | 0.5299 | 0.7988 | 0.4057 | 0.8431 | 0.6177 |
| full checkpoint / physical criticality | 106 / 47 | 0.6928 | 0.4147 | 0.7633 | 0.3679 | 1.0000 | 0.6840 |
| full checkpoint / K10 | 2003 / 60 | 0.9709 | 0.8228 | 0.9935 | 0.9795 | 0.9766 | 0.5981 |
| full checkpoint / safe release | 0 / 2063 | 0.0000 | n/a | n/a | n/a | 0.0000 | n/a |
| full checkpoint / instability | 3 / 2053 | 0.0015 | 0.6931 | 0.0032 | 0.0000 | n/a | 0.5000 |
| full checkpoint / gripper closing | 293 / 2058 | 0.1246 | 0.9982 | 0.9886 | 0.9693 | 0.8765 | 0.9749 |

The physical, safe-release, and instability rows remain proxy diagnostics. Safe-release has no positive known steps in this split; it is not evidence of a trained semantic head.

## Sealed R2 roots

Every root passed recursive payload verification and sidecar verification. The execution source commit is `f6f619b4fc6b1706aff1cf1967c73e8cc10b8c28`; source commit time was `2026-07-28T17:23:23+08:00`, before the recorded execution interval `2026-07-28T09:26:20Z`–`2026-07-28T09:26:54Z`.

| Bundle | Root | `SHA256SUMS.sha256` |
|---|---|---|
| critical checkpoint / critical-only shadow | `fresh40_v5_shadow_r2_critical_ckpt_criticalonly_f6f6_20260728T090000Z` | `0070eab7b5c7f996de240858006b49cf391b1e5af37855683a1834a9ed7628cf` |
| full checkpoint / critical-only equation | `fresh40_v5_shadow_r2_full_ckpt_criticalonly_f6f6_20260728T090000Z` | `5ef8f66320ebcadbef45df63b152cf92357508a028e36b581c7a00eb538abdd7` |
| full checkpoint / three-head equation | `fresh40_v5_shadow_r2_full_ckpt_threehead_f6f6_20260728T090000Z` | `f46f428870139451ddd05a1e3949c8f88efa96e6733adc64b6e1bee45488ba2b` |
| full checkpoint / full-five equation | `fresh40_v5_shadow_r2_full_ckpt_fullfive_f6f6_20260728T090000Z` | `5611a9e4e85c401b699a2ec2f891f468dcde3c01223b9374a4d1ef165f23e0c3` |
| corrected oracle ladder | `fresh40_v5_oracle_ladder_full5_f6f6_20260728T090000Z` | `9ac4c0bd64c52bedaa27c286477f6d64fd906f70233acf6f6aac51a315d143e8` |

## Final裁决

```text
R1-A variant/head isolation       = PASS
R1-B UNKNOWN step masking         = PASS
R1-C three-value event aggregation = PASS
R1-D event/candidate scientific denominator = HOLD_DIAGNOSTIC_ONLY
R1 provenance chronology          = PASS_FOR_NEW_R2_ROOTS
R2 oracle ladder                  = HOLD_INSUFFICIENT_SCIENTIFIC_COVERAGE
FRESH40 scientific canary         = NOT_EVALUATED_DUE_TO_CONFOUNDS
R3 formal retraining              = NOT AUTHORIZED
Fresh670                          = BLOCKED
Protected reads                   = 0
New Student training              = NOT RUN
Development Student shadow        = RUN ON SEALED DEV SPLIT ONLY
OpenVLA inference                 = NOT RUN
Rollout                           = NOT RUN
Attack                            = NOT RUN
```

The next meaningful scientific step remains a corrected contact-pair Teacher canary. These proxy results must not be used for formal model selection, threshold selection, Fresh670 promotion, or attack authorization.
