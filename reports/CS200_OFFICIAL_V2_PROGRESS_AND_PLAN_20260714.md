# Official-Aligned CS200 V2 — Progress and Execution Plan

**Snapshot:** 2026-07-14 22:33:52 CST  
**Evidence line:** `c2g_cs200_official_v2_fix2`  
**Global stage:** `CLEAN_2000`  
**Global status:** `ACTIVE` (handoff in progress; not final)

## Executive status

The current experiment is a fresh official-aligned V2 CLEAN census. It is not a completed CS200 matrix and must not be reported as final OpenVLA-LIBERO results.

The census contains 2,000 planned CLEAN cells:

| Suite | PASS | TASK_FAILURE | RUNNING | PENDING | Total |
|---|---:|---:|---:|---:|---:|
| `libero_object` | 22 | 6 | 1 | 471 | 500 |
| `libero_spatial` | 47 | 5 | 1 | 447 | 500 |
| `libero_goal` | 15 | 11 | 1 | 473 | 500 |
| `libero_10` | 11 | 4 | 1 | 484 | 500 |
| **Total** | **95** | **26** | **4** | **1,875** | **2,000** |

Current terminal count is 121/2,000. `PASS` here means a CLEAN episode reached environment success with a valid artifact; it is not a final paper SR, a verified 50-parent suite, or a completed attack condition.

## Runtime state

Evidence root:

```text
/mnt/sdc/dty_user/openvla_attack_evidence/c2g/c2g_cs200_official_v2_fix2
```

Authoritative files:

```text
CS200_OFFICIAL_RELAY_STATUS.json
manifests/OFFICIAL_GLOBAL_CELL_LEDGER_V1.csv
```

The four intended resident assignments are GPU2–GPU5, one suite worker per GPU:

| GPU | Suite | Worker |
|---:|---|---|
| 2 | `libero_object` | `official_fix2_object_gpu2` |
| 3 | `libero_spatial` | `official_fix2_spatial_gpu3` |
| 4 | `libero_goal` | `official_fix2_goal_gpu4` |
| 5 | `libero_10` | `official_fix2_l10_gpu5` |

Disk free space was approximately 84.2 GiB, above the 40 GiB soft stop and 30 GiB hard stop.

The acceleration handoff is not yet complete. The old supervisor was stopped so that workers could finish at cell boundaries. The spatial worker has already been stopped after a completed cell, while the remaining old workers are finishing their current cells. The replacement supervisor has not yet become authoritative. This is an operational HOLD for recovery, not an experiment-result PASS.

## Official protocol boundary

The V2 line is intended to use the official LIBERO configuration:

- 10 tasks per suite;
- 50 official initial states per task, state IDs 0–49;
- 10 wait steps before policy rollout;
- official horizons: Spatial 220, Object 280, Goal 300, L10 520;
- action execution through `model.predict_action()`;
- official action postprocessing and `env.step()`;
- fixed checkpoint, processor, normalization key, seed, and upstream provenance.

Each CLEAN artifact is compact evidence and includes the episode metadata, summary, runtime audit, step records, policy-intent records, privileged teacher sidecar, condition/config files, and recursive SHA-256 closure.

The intended detector-retraining inputs are recorded per step:

- canonical 25D state/kinematic feature vector;
- 9D clean policy-intent feature vector;
- action token IDs and compact token evidence;
- raw and postprocessed 7D actions;
- prompt and score-adapter parity fields;
- task, suite, state, initial-state hash, official horizon, and checkpoint provenance;
- privileged teacher sidecar kept separate from student-allowed inputs.

## Acceleration change under review

The current collector still has the old two-generation path in its completed artifacts:

```text
OfficialOpenVLAScoreAdapter.generate_same_inputs
generation_passes_per_step = not recorded / old path
```

The deployed but not-yet-active acceleration patch changes only the instrumentation path:

1. call `model.predict_action()` exactly as before;
2. intercept the same internal `generate()` call;
3. request generation scores and return `result.sequences` to `predict_action()`;
4. derive detector logits from that same generation;
5. remove the second model generation per step.

It does not change prompt, processor, horizon, wait steps, `do_sample=False`, action decoding, unnormalization, gripper inversion, or environment stepping. New artifacts must first pass an old/new same-input action parity canary. Any mismatch must fail closed and remain outside the official main table.

## Fixed phase plan

```text
UPSTREAM_PINNED
  -> OFFICIAL_ADAPTER_IMPLEMENTED
  -> PARITY_AUDIT
  -> CLEAN_2000
  -> CLEAN_2000_AUDIT
  -> DETECTOR_TRAIN
  -> DETECTOR_CAL
  -> DETECTOR_CHECK
  -> CS200_PARENT_FREEZE
  -> ATTACK_CANARY_48
  -> FULL_ATTACK_1000
  -> FINAL_AUDIT
```

No later phase is allowed to promote partial evidence:

1. Finish and audit all 2,000 official CLEAN cells.
2. Freeze identity split: states 0–23 FIT, 24–26 CAL, 27–29 CHECK, 30–49 final candidates.
3. Train, calibrate, and CHECK a new official-aligned detector. The old B2 detector remains baseline only.
4. Select exactly five verified CLEAN successes per task from states 30–49: 10 tasks × 5 parents × 4 suites = 200 parents.
5. Run a pre-registered 48-cell attack canary.
6. After canary PASS, run the 1,000 new R9Q/RAND_VALID/Direct-open/Shuffled/Gripper-only rollouts.
7. Produce the 1,200-cell matrix and final paired statistics only after checksum and identity closure.

## Safe parallel work

The following can proceed while CLEAN collection continues, without consuming formal attack identities:

- CPU-only checksum, schema, duplicate, task/state coverage, horizon, and teacher-sidecar audits;
- detector feature-loader and FIT/CAL/CHECK split dry-runs;
- protocol readiness tests for RAND_VALID_T10, Shuffled-gradient, and Gripper-only;
- non-executable canary/parent-manifest generation and deterministic hash verification;
- report generation and provenance reconciliation.

The following must remain blocked:

- formal detector CAL/CHECK promotion before the CLEAN census audit;
- parent freezing before all task quotas are met;
- R9Q, RAND_VALID, Direct-open, Shuffled, or Gripper-only rollout before detector CHECK and the 48-cell canary;
- a second project worker on GPU2–GPU5;
- any merge of legacy state-0–9 or manual-generate evidence into the official V2 line.

## Current claim boundary

The current evidence supports only:

- official-aligned CLEAN collection is active;
- 121 CLEAN cells have terminal artifacts at this snapshot;
- 95 terminal CLEAN episodes succeeded and 26 are task failures;
- the artifact schema is intended to support later detector retraining;
- the formal attack matrix has not started.

It does **not** support:

- a final 200-parent panel;
- detector CHECK success;
- any attack SR, trigger rate, paired risk difference, or mechanism conclusion;
- a completed 1,200-cell paper matrix.

## GPT review requests

Please review the following before the official line is allowed to advance:

1. Confirm that the single-generation score capture is semantically equivalent to the official `predict_action()` path under a same-input action-token parity test.
2. Confirm that the recorded 25D/9D fields and teacher sidecar are sufficient for detector FIT/CAL/CHECK without privileged-state leakage.
3. Confirm that the stale spatial lease created during the worker handoff is repaired as a runtime retry or resumed without duplicate identity.
4. Confirm that the current PASS/TASK_FAILURE counts are not being interpreted as final SR.
5. Confirm that the task-balanced states 30–49 rule is enforced before any attack manifest becomes executable.

## Review status

**Current review label: `PARTIAL / ACTIVE_WITH_HANDOFF_HOLD`**

This report is a progress and plan handoff only. It is not a final experimental result.
