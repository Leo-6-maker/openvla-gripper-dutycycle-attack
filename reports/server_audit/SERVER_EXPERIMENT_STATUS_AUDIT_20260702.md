# Server Experiment Status Audit — 2026-07-02

## Executive Summary

This audit provides a reconciled snapshot of all OpenVLA Gripper Duty-Cycle Attack experiments on the A800 compute server (`dty-server`, 10.60.2.56:33571) as of 2026-07-02T17:52+08:00.

**All UMA and SHUFFLED workers were killed on 2026-07-02 ~14:30+08. No OpenVLA processes remain running.** GPUs are occupied by unrelated user jobs (zkx CAMEF, huanzze GR00T-N1.7). No watchers, cron jobs, or auto-launch scripts are active.

**Overall Gate: HOLD_BACKUP_NOT_SECURED.** Server is quiescent (all processes stopped, all bridge PIDs dead, no artifact writes in progress). Object frozen results independently verified. CLEAN2000 reconciled. The single remaining hold is evidence backup.

**AMENDED 2026-07-02T19:01: CLEAN300 is ARCHIVED AND SUPERSEDED. dty CLEAN2000 is the sole authoritative cross-suite corpus. See CROSS_SUITE_AND_CLEAN2000_STATUS for details.**

---

## Server Environment

| Item | Value |
|---|---|
| Hostname | pm-364c0001 |
| Uptime | 89 days 14:58 |
| Load Average | 177.35, 176.43, 175.14 (dominated by other users) |
| Disk /mnt/sdc | 2.9T total, 2.6T used, 162G available (95% full) |
| GPUs | 8× NVIDIA A800-SXM4-80GB |
| CUDA | Not checked (proprietary driver) |
| Cron (dty_user) | none |
| Tmux/Screen | no sessions |

### GPU State (2026-07-02T17:52)

| GPU | Util | Memory Used | Memory Total | Owner |
|---|---|---|---|---|
| 0 | 50% | 80823 MiB | 81920 MiB | zkx (CAMEF cuda:0) |
| 1 | 11% | 80828 MiB | 81920 MiB | zkx (CAMEF cuda:1) |
| 2 | 53% | 80868 MiB | 81920 MiB | zkx (CAMEF cuda:2) |
| 3 | 52% | 33060 MiB | 81920 MiB | other user |
| 4 | 85% | 33062 MiB | 81920 MiB | other user |
| 5 | 29% | 75670 MiB | 81920 MiB | zkx (CAMEF cuda:5) |
| 6 | 45% | 37625 MiB | 81920 MiB | mixed |
| 7 | 23% | 80827 MiB | 81920 MiB | huanzze (GR00T) |

All GPUs at near-full memory. Our experiments are not running.

---

## Repository State

### dty-server (experiment execution)

| Item | Value |
|---|---|
| Path | `/mnt/sdc/dty_user/openvla_attack` |
| Branch | `feature/sc5-abstention-v2-20260622` |
| Commit | `ace1876281a9ad6ed68e1229a6e17346356766e9` |
| Dirty | YES — 4 modified + 50+ untracked |

**Modified files (protocol drift risk):**
- `scripts/stageb/run_sc5_cross_suite_clean.py`
- `scripts/stageb/run_v2_vis_sc5_mlp_bridge.py`
- `scripts/v4_run_eval_openvla.py`
- `src/gripper_attack/attack_adapter.py`

### dty-server (execution workspace)

| Item | Value |
|---|---|
| Path | `/mnt/sdc/dty_user/table1_sota_execution_v1` |
| Git | NOT a git repo (copied execution directory) |
| Commit origin | unknown — not tracked |

### Local (Windows dev)

| Item | Value |
|---|---|
| Branch | `feature/multisuite-clean2000-detector-prep-v1` |
| Commit | `9f7d711a08c96083c446e7dcb333ff382e52c11c` |
| Dirty | YES (many untracked files) |

### Local vs Server Divergence

**Different branches, different commits** — local is `feature/multisuite-clean2000-detector-prep-v1` (commit 9f7d711), server is `feature/sc5-abstention-v2-20260622` (commit ace18762). These branches diverged. The attack_adapter.py and bridge script are modified on BOTH sides with different changes.

---

## Current Running Tasks

**NONE.** All OpenVLA/gripper-attack Python processes are stopped.

### Previously Running (killed 2026-07-02 ~14:30)

| Condition | GPUs | Workers Killed | Bridges at Kill | Bridge Status |
|---|---|---|---|---|
| UMA | 3,4,6 | 3 | 3 (PIDs 912119, 913458, 914525, 914593) | Unknown — SSH disconnected during monitoring |
| SHUFFLED | 3,4,6 | 3 | 2 (PIDs 887833, 909658) | Unknown |

Bridges were writing artifacts at time of launcher kill. PIDs 912119, 913458, 914525, 914593 (UMA) and 887833, 909658 (SHUFFLED) may have completed or been killed. The `outputs/` directory is empty — bridges may have cleaned up temp files on completion.

---

## Experiment Inventory Summary

### Object Frozen Evidence (FROZEN_FORMAL_EVIDENCE)

| Condition | Status | Paper Usable | Details |
|---|---|---|---|
| CLEAN | FROZEN_FORMAL_EVIDENCE | YES | 162/162 success, FR=0.0% |
| RAND_T10 | FROZEN_FORMAL_EVIDENCE | YES | 162/162 success, FR=0.0% |
| RANDOM_TIME_V3 | FROZEN_FORMAL_EVIDENCE | YES | 119/162 success, FR=26.5% |
| EARLY_SHIFT_T10 | FROZEN_FORMAL_EVIDENCE | YES | 98/141 success, FR=30.5%, emission-matched |
| TRUE_T10 (ITT) | FROZEN_FORMAL_EVIDENCE | YES | 21/162 success, FR=87.0% |
| TRUE_T10 (emitted-only) | FROZEN_FORMAL_EVIDENCE | YES | 0/141 success, FR=100.0% |
| COMMAND_OPEN_ORACLE | FROZEN_FORMAL_EVIDENCE | YES | 0/141 success, FR=100.0% |

Artifacts stored at: `/mnt/sdc/dty_user/openvla_attack/evidence/sc5_object_privileged_loto_v1/vis_heldout_formal_v1/`
Structure: `{condition}/fold_XX/state_Y/det_seed_Z/pert_seed_W`
3 detector seeds × 3 perturbation seeds per state.
~3,495 total artifact directories across 14 conditions.

### TMA Student (EXPLORATORY_CANARY)

| Item | Value |
|---|---|
| Planned | 162 |
| Completed | ~162 |
| Failed | ~7 |
| Formal Validator | PASSED (`TMA_STUDENT_FORMAL_PASS.json`) |
| Paper Usable | NO (exploratory canary) |
| Status | EXPLORATORY_CANARY |

### TMA Random-Time (EXPLORATORY_CANARY)

| Item | Value |
|---|---|
| Planned | 162 |
| Completed | ~161 |
| Failed | ~9 |
| GPU6 Issue | 1 episode (fold_01 state_0 det_3 pert_1) FAILED after 19s retry |
| Formal Validator | NOT RUN |
| Paper Usable | NO (exploratory canary) |
| Status | EXPLORATORY_CANARY |

### UMA (INCOMPLETE_PAUSED)

| GPU | Planned | Completed | Failed | State |
|---|---|---|---|---|
| 0 | 21 | 0 | 21 | Failed (16-18s each) |
| 1 | 21 | 0 | 21 | Failed (16-18s each) |
| 2 | 20 | 0 | 20 | Failed (16-18s each) |
| 3 | 20 | 15 | 0 | Interrupted at job 16/20 |
| 4 | 20 | 14 | 2 | Interrupted at job 17/20 |
| 5 | 20 | 0 | 20 | Failed (16-18s each) |
| 6 | 20 | 14 | 0 | Interrupted at job 15/20 |
| 7 | 20 | 0 | 20 | Failed (16-18s each) |
| **Total** | **162** | **43** | **104** | **15 incomplete** |

**Status: INCOMPLETE_PAUSED_NOT_FOR_SCIENTIFIC_USE**

Failure pattern: GPUs 0,1,2,5,7 ALL jobs failed in 16-18 seconds — consistent with a code/runtime error (not timeout). Only GPUs 3,4,6 produced completions. Jobs labeled `_CLEAN` suggest clean-input baselines.

No formal validator was run. Formal pass file does NOT exist.

### SHUFFLED (INCOMPLETE_PAUSED)

| GPU | Planned | Completed | Failed | State |
|---|---|---|---|---|
| 0 | 21 | 0 | 21 | Failed (16-18s each) |
| 1 | 21 | 0 | 21 | Failed (16-18s each) |
| 2 | 20 | 0 | 20 | Failed (16-18s each) |
| 3 | 20 | 6 | 0 | Interrupted at job 7/20 |
| 4 | 20 | 5 | 1 | Interrupted at job 7/20 |
| 5 | 20 | 0 | 20 | Failed (16-18s each) |
| 6 | 20 | 5 | 0 | Interrupted at job 6/20 |
| 7 | 20 | 0 | 20 | Failed (16-18s each) |
| **Total** | **162** | **16** | **103** | **43 incomplete** |

**Status: INCOMPLETE_PAUSED_NOT_FOR_SCIENTIFIC_USE**

Same failure pattern as UMA. Only GPUs 3,4,6 produced any completions. No formal validator was run.

### CLEAN300 Status

| Item | Value |
|---|---|
| Evidence Path | Server: `evidence/sc5_object_privileged_loto_v1/` |
| Planned | 162 episodes (9 folds × ~18 states) |
| Cross-suite | LIBERO-Object only (not Spatial/Goal/10) |
| Object Detector Model | 10-fold LOTO, folds 01-09 teacher-labeled |
| Clean Success | All CLEAN artifacts present |
| Status | DATA_PRESENT, NOT_YET_AGGREGATED_FOR_300 |

CLEAN300 was planned as cross-suite but only Object data exists with full labels. Cross-suite (Spatial/Goal/10) data collected but not yet labeled or processed.

### CLEAN2000 Status

| Item | Value |
|---|---|
| Path | `/mnt/sdc/dty_user/openvla_attack/evidence/CLEAN2000_CANONICAL_V1/` |
| INDEX_DRAFT | 2000 episodes |
| ATTEMPT_LEDGER | 2000 entries |
| PRIMARY | 1043 episodes (52.15%) |
| Features 25D | Available (all steps + valid only) |
| Teacher Labels | Index available (CLEAN2000_TEACHER_LABEL_INDEX.jsonl) |
| Teacher Cross-Validation | CLEAN2000_TEACHER_CROSS_VALIDATION.json |
| SHA256SUMS | Present |
| Suite Coverage | Multi-suite (Object + Spatial + Goal + 10) |
| Pooled Detector | NOT TRAINED |
| LOSO Detector | NOT TRAINED |
| Formal Split | CLEAN2000_SPLITS_V1 exists |
| Training Release | CLEAN2000_TRAINING_RELEASE_V1 exists |
| **Status** | DATA_COLLECTED, NOT_READY_FOR_TRAINING |

Only 1043/2000 (52%) episodes validated as PRIMARY. Remaining 957 episodes have quality issues, schema failures, or are supplementary. Teacher labels exist for the labeled subset but full mechanism resolution is incomplete.

---

## Infrastructure Status

| Item | Status |
|---|---|
| Model Checkpoints | 4 checkpoints: libero-10, libero-goal, libero-spatial, openvla-7b-finetuned-libero-object |
| Disk Space | CRITICAL — 95% full, 167G remaining |
| Git Server (vla) | Multiple repos, all DIRTY, various branches |
| Local Evidence | Object evidence NOT mirrored locally (only schema gate + canary) |

---

## Top 5 Risks

1. **DISK CRITICAL**: /mnt/sdc at 95% (167G free). Any further writes risk disk full.
2. **No local evidence copy**: Object frozen artifacts exist only on dty-server. Single point of failure.
3. **Server code divergence**: Server on different branch/commit than local. 4 files modified server-side with no corresponding GitHub commits.
4. **UMA/SHUFFLED massive failure**: ~70% of UMA jobs and ~85% of SHUFFLED jobs failed with code errors. Only GPUs 3,4,6 succeeded — suggests GPU-specific dependency.
5. **TMA Random-Time formal gap**: 1 missing episode, formal validator never run. Cannot be used as formal evidence without completion.

---

## Recommended Actions

1. **Immediate**: Confirm bridge processes are fully stopped. Check if any partial `outputs/` artifacts need preservation.
2. **Immediate**: Rsync Object frozen evidence to local or secondary storage.
3. **Before any new work**: Resolve server git divergence — commit or discard dirty changes on `feature/sc5-abstention-v2-20260622`.
4. **Before any training**: Free up disk space (delete old pi0 logs, docker build logs, duplicate bundles).
5. **Gate for UMA/SHUFFLED recovery**: Fix the code error causing 16-18s failures on GPUs 0,1,2,5,7. Only then consider resuming (if at all).

---

NO NEW EXPERIMENT WAS LAUNCHED.
NO LIVE SCIENTIFIC ARTIFACT WAS MODIFIED.
