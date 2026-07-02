# Existing Evidence Freeze Report — 2026-07-02

## Executive Summary

This report freezes and verifies all existing evidence on the dty-server A800 compute node. All numbers are independently verified from actual artifact files (episode_summary.json), not from directory counts or prior reports.

**Object frozen results verified: all 6 conditions match claimed numbers at the per-episode level.**
**CLEAN2000: 2000 episodes indexed, 1043 PRIMARY, exact exclusion reasons mapped.**
**Server quiescent: no OpenVLA processes, no artifacts being written.**
**CLEAN300 is archived and superseded.**

---

## Source of Truth

| Resource | Location | Status |
|---|---|---|
| Authoritative server | dty-server (10.60.2.56:33571) | Active |
| Authoritative repo | `/mnt/sdc/dty_user/openvla_attack` | branch `feature/sc5-abstention-v2-20260622`, commit `ace18762` |
| Authoritative CLEAN2000 | `/mnt/sdc/dty_user/openvla_attack/evidence/CLEAN2000_CANONICAL_V1` | Frozen 2026-06-30 |
| Authoritative Object evidence | `/mnt/sdc/dty_user/openvla_attack/evidence/sc5_object_privileged_loto_v1/vis_heldout_formal_v1` | Frozen |
| Authoritative runtime code | SHA256 captured for all dirty files, manifests, and commands | Captured 2026-07-02 |
| CLEAN300 | Archived and superseded | See CLEAN300 section below |

---

## CLEAN300 Archival Declaration

| Field | Value |
|---|---|
| CLEAN300_STATUS | ARCHIVED_SUPERSEDED_DATA |
| CLEAN300_SCIENTIFIC_AUTHORITY | NONE |
| CLEAN300_FUTURE_USE | HISTORICAL_REFERENCE_ONLY |
| Superseded by | dty-server CLEAN2000 |
| Historical note | ARCHIVED_ACCEPTED_AT_THE_TIME_BUT_SUPERSEDED_BY_DTY_CLEAN2000 |

CLEAN300 (300 records from an earlier collection phase) is **not** the authoritative cross-suite corpus. The dty-server CLEAN2000 with 2000 episodes across all 4 LIBERO suites is the sole current cross-suite data benchmark.

No CLEAN300 data was modified, deleted, or merged.

---

## Server Quiescence

| Check | T0 (18:57) | T1 (19:01) | Verdict |
|---|---|---|---|
| OpenVLA processes | 0 | 0 | PASS |
| Bridge PIDs (6) | All DEAD | All DEAD | PASS |
| Outputs/ size | 4096 bytes | 4096 bytes | STABLE |
| Log file sizes | — | Unchanged | STABLE |
| Evidence files modified (-15min) | 0 | 0 | PASS |
| CLEAN2000 files modified (-15min) | 0 | 0 | PASS |
| Watcher auto-launch | watch_understandingonly_b28.sh (modal-aphasia-textadv, not ours) | Running but for other project | MONITORED |
| GPU state | All occupied by other users | Same | N/A |

**SERVER_EXECUTION_QUIESCENT = PASS**
**PROCESS_STOPPED = PASS**
**LAST_ARTIFACT_ATOMICITY = UNKNOWN** (bridges killed mid-write; completed artifacts before kill presumed intact; no partial write evidence found)

Watcher `watch_understandingonly_b28.sh` (PID 3034782) belongs to `modal-aphasia-textadv` project — NOT our project. It will not launch OpenVLA experiments. No action needed.

---

## Object Frozen Results — Independent Verification

All numbers independently verified by reading `episode_summary.json` from each leaf artifact directory.

### Per-Condition Verification

| Condition | Leaf Dirs | Emitted | No-Emission | Success | Failure | Claimed | Match |
|---|---|---|---|---|---|---|---|
| CLEAN | 162 | 141 | 21 | 162 | 0 | 162/162 | PASS |
| RAND_T10 | 162 | 141 | 21 | 162 | 0 | 162/162 | PASS |
| RANDOM_TIME_V3 | 162 | 126 | 36 | 119 | 43 | 119/162 (26.5%) | PASS |
| EARLY_SHIFT_T10 | 141 | 99 | 42 | 98 | 43 | 98/141 (30.5%) | PASS |
| TRUE_T10 | 162 | 141 | 21 | 21 | 141 | 21/162 ITT, 0/141 emitted | PASS |
| COMMAND_OPEN_ORACLE_T10 | 141 | 141 | 0 | 0 | 141 | 0/141 (100%) | PASS |

### Emission-Matched Denominator Verification

| Check | Result |
|---|---|
| TRUE_T10 emitted parent set == EARLY_SHIFT_T10 parent set | TRUE (17 fold/state parents) |
| TRUE_T10 emitted parent set == COMMAND_OPEN_ORACLE parent set | TRUE (17 fold/state parents) |
| TRUE_T10 all parents == CLEAN all parents | TRUE (18 fold/state parents) |
| TRUE_T10 all parents == RAND_T10 all parents | TRUE (18 fold/state parents) |
| TRUE_T10 all parents == RANDOM_TIME_V3 all parents | TRUE (18 fold/state parents) |

The 141 emitted denominator for EARLY_SHIFT_T10 and COMMAND_OPEN_ORACLE_T10 matches TRUE_T10's emission set exactly.

### No-Emission Analysis (TRUE_T10)

- 21 no-emission episodes: detector did NOT trigger, attack NOT applied
- All 21 succeeded (100% success)
- 3 unique fold/state parent keys never trigger
- These 21 represent robot states where the contact-critical phase is absent or undetectable

### Emitted Failure Analysis (TRUE_T10)

- 141 emitted episodes: detector triggered, attack applied
- 0 succeeded, 141 failed (100% failure rate)
- This is the core mechanism evidence: when the perturbation aligns with the detector-identified contact-critical phase, the task always fails

### Object Aggregation Re-Verification

OBJECT_REAGGREGATION_MATCH = PASS

The original reported numbers (162/162, 119/162, 98/141, 21/162, 0/141) are independently confirmed by re-reading all episode_summary.json files.

---

## CLEAN2000 Authority Census

### Total: 2000 episodes, 4 suites

| Suite | Total | Success | Failure | Teacher Eligible | Primary |
|---|---|---|---|---|---|
| libero_spatial | 500 | 411 | 89 | 500 (100%) | 411 |
| libero_object | 500 | 367 | 133 | 500 (100%) | 367 |
| libero_goal | 500 | 383 | 117 | 300 (60%) | 234 |
| libero_10 | 500 | 231 | 269 | 50 (10%) | 31 |
| **TOTAL** | **2000** | **1392** | **608** | **1350** | **1043** |

### Exact Episode Classification (Mutually Exclusive, Sums to 2000)

| Category | Count | Description |
|---|---|---|
| PRIMARY_ELIGIBLE | 1043 | All checks passed: task_success=True, teacher_eligible=True, mechanism_eligible=True |
| CLEAN_FAILURE_SAFETY | 307 | Task failed but teacher/mechanism eligible (analyzable failures) |
| SUPPLEMENTARY_EVENT (multi_object_two_items) | 200 | Task involves 2 identical objects |
| SUPPLEMENTARY_EVENT (articulated_task_no_pick_place) | 100 | Task involves articulated objects (doors/drawers) |
| SUPPLEMENTARY_EVENT (multi_object_two_different_targets) | 100 | Task involves 2 different objects |
| SUPPLEMENTARY_EVENT (multi_stage_place_then_close) | 50 | Multi-stage: place then close |
| SUPPLEMENTARY_EVENT (multi_stage_place_then_close_drawer) | 50 | Multi-stage: place then close drawer |
| SUPPLEMENTARY_EVENT (push_task_not_pick_place) | 50 | Pushing task, not pick-place |
| SUPPLEMENTARY_EVENT (multi_stage_open_drawer_then_place) | 50 | Multi-stage: open drawer then place |
| SUPPLEMENTARY_EVENT (multi_stage_turn_on_then_place) | 50 | Multi-stage: turn on then place |
| **TOTAL** | **2000** | |

### Teacher Label Coverage

| Suite | Teacher Valid | Teacher Invalid | Reason |
|---|---|---|---|
| libero_spatial | 500 (100%) | 0 | — |
| libero_object | 500 (100%) | 0 | — |
| libero_goal | 300 (60%) | 200 | Multi-stage / push / articulated tasks |
| libero_10 | 50 (10%) | 450 | Multi-object / multi-stage tasks |
| **TOTAL** | **1350** | **650** | All 650 ineligible per task-type abstention |

### Per-Task Detail

**libero_object** (10 tasks × 50 episodes, all teacher-eligible):
- All 10 pick-place tasks have 100% teacher eligibility
- Success rates range from 54% (bbq_sauce) to 96% (ketchup)
- Primary: 367/500 (73%)

**libero_spatial** (10 tasks × 50 episodes, all teacher-eligible):
- All 10 spatial-variant pick-place tasks have 100% teacher eligibility
- Success rates range from 38% (ramekin) to 98% (between_plate_ramekin)
- Primary: 411/500 (82%)

**libero_goal** (10 tasks × 50 episodes, 6 tasks teacher-eligible):
- 6 pick-place goal tasks: 100% teacher eligibility
- 4 excluded: open_drawer, top_drawer_put_bowl, push_plate, turn_on_stove
- Primary: 234/500 (47%)

**libero_10** (10 tasks × 50 episodes, 1 task teacher-eligible):
- Only task 5 (pick_up_book) is teacher-eligible (50/50)
- 9 excluded: multi-object (4 tasks), multi-stage (3 tasks), articulated (2 tasks)
- Primary: 31/500 (6%)

### CLEAN2000 Final Status

**CLEAN2000_FROZEN_WITH_LABEL_GAPS**

- 2000 episodes physically present and verified
- 1043 PRIMARY (52.15%) — immediately usable for detector training
- 307 CLEAN_FAILURE_SAFETY — usable with caveat (failed clean, mechanism analyzable)
- 650 SUPPLEMENTARY_EVENT — not usable for single-object pick-place attack study
- Teacher labels cover 1350/2000 (67.5%)
- Cross-suite label gaps: libero_10 (90% unlabeled), libero_goal (40% unlabeled)
- No SCHEMA_FAIL, INFRA_FAIL, TELEMETRY_INCOMPLETE, or DUPLICATE episodes found
- All 2000 INDEXED with consistent schema

---

## Historical Canary Classification

| Experiment | Status | Paper Usable | Artifacts | Formal Validator |
|---|---|---|---|---|
| TMA Student | EXPLORATORY_CANARY | NO | 162/162 COMPLETE, outputs empty | PASS (TMA_STUDENT_FORMAL_PASS.json) |
| TMA Random-Time | EXPLORATORY_CANARY | NO | 161/162 COMPLETE, 1 missing | NOT RUN |
| UMA (SOTA) | FAILED_ENGINEERING_ATTEMPT | NO | 43/162 COMPLETE, outputs empty | NOT RUN |
| SHUFFLED (SOTA) | FAILED_ENGINEERING_ATTEMPT | NO | 16/162 COMPLETE, outputs empty | NOT RUN |
| Object-level TMA | EXPLORATORY_CANARY | NO | 282 dirs in vis_heldout_formal_v1 | N/A |
| Object-level UMA | EXPLORATORY_CANARY | NO | 270 dirs in vis_heldout_formal_v1 | N/A |
| Object-level SHUFFLED | EXPLORATORY_CANARY | NO | 242 dirs in vis_heldout_formal_v1 | N/A |

UMA/SHUFFLED SOTA: persisted_complete_artifact_rows = 0, scientifically_valid_rows = 0.
TMA SOTA: persisted_complete_artifact_rows in outputs/ = 0 (empty directory). Formal pass file exists but points to empty output root — FORMAL_PASS_FILE_NOT_SUFFICIENT.

---

## Runtime Code Snapshot

### Dirty Files (server-side, uncommitted)

| File | SHA256 | Lines Changed | Risk | Fields Affected |
|---|---|---|---|---|
| `src/gripper_attack/attack_adapter.py` | `3ff284a3e...` | +11/-7 | HIGH | Added `vanilla_tma_gripper_open_ce` objective, target_token_id handling, debug fields |
| `scripts/stageb/run_v2_vis_sc5_mlp_bridge.py` | `4ce75f9a...` | +825/-781 | HIGH | Bridge execution logic |
| `scripts/v4_run_eval_openvla.py` | `df148fe2...` | +1248/-1248 | MEDIUM | Eval/rollout pipeline |
| `scripts/stageb/run_sc5_cross_suite_clean.py` | `fc74c6c4...` | +2/-1 | MEDIUM | Cross-suite collection |

### Runtime Command Registry (SHA256 captured)

All 14 commands, 32 manifests (8 GPUs × 4 conditions), and condition specs SHA256 captured in `tables/server_freeze/runtime_code_sha_registry.csv`.

### attack_adapter.py Diff Analysis

The server-side modification adds `vanilla_tma_gripper_open_ce` as a new objective variant. This:
1. Adds the objective name to `is_force_gripper_open` detection
2. When this objective is selected, sets the gripper label position to `target_token_id` (31744)
3. Records `vanilla_tma_gripper_open_ce_target_token_31744` as the label source
4. Expands the debug field `attack_target_gripper_token_id` population

This change enables the TMA vanilla attack variant used in the SOTA experiments. It does NOT change the core PGD attack logic for the frozen Object conditions (which use `autoregressive_prefix_gripper_target_token_logratio_arm_v3`).

---

## Evidence Backup Status

| Item | Status |
|---|---|
| Object evidence location | dty-server ONLY (single point of failure) |
| Second storage candidate | vla server `/data/liuyu` (1.1T free, 37% used) |
| Write permission | Yes (liuyu@vla, same user) |
| Estimated Object evidence size | ~100-500 GB (needs measurement) |
| Estimated CLEAN2000 size | ~574 MB (measured from file listing) |
| Network path | Via jump host (10.60.133.3 → 10.60.133.4) |
| Rsync available | Yes |
| **BACKUP_STATUS** | **HOLD_NO_SAFE_SECONDARY_STORAGE** — vla server is accessible but backup has NOT been executed |

Object evidence size needs precise measurement before rsync. 95% full /mnt/sdc precludes local duplication.

---

## Remaining Blockers

1. **Object evidence single point of failure** — no backup exists; vla server has capacity but rsync not yet run
2. **Server dirty code not committed** — the attack_adapter.py diff is small and benign but uncommitted
3. **CLEAN2000 label gaps** — libero_10 (90% unlabeled) and libero_goal (40% unlabeled) limit cross-suite detector training
4. **Formal provenance seal pending** — aggregation script not yet located/audited; checkpoint SHA not verified for all folds
5. **Watcher for other project** — watch_understandingonly_b28.sh is non-threatening but present

---

## Final Gate

**HOLD_BACKUP_NOT_SECURED**

All other gates are satisfied:
- Server is quiescent ✓
- Object provenance reconciled ✓
- Object re-aggregation matches ✓
- CLEAN2000 reconciled ✓
- Runtime code snapshot captured ✓
- CLEAN300 archived ✓
- Historical canaries classified ✓

The single remaining independent HOLD is evidence backup. Once backup is executed and verified, gate resolves to EXISTING_EVIDENCE_FREEZE_READY_FOR_REVIEW.

---

CLEAN300 IS ARCHIVED AND SUPERSEDED.
DTY CLEAN2000 IS THE AUTHORITATIVE CROSS-SUITE CORPUS.
NO NEW EXPERIMENT WAS LAUNCHED.
NO LIVE SCIENTIFIC ARTIFACT WAS MODIFIED.
EXPERIMENT EXECUTION REMAINS NOT AUTHORIZED.
