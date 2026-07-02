# Server Runtime and Process Ledger — 2026-07-02

## Current State: ALL PROCESSES STOPPED

No OpenVLA/gripper-attack processes are running on dty-server as of 2026-07-02T17:52+08:00.

All worker launchers were killed on 2026-07-02 ~14:30+08:00.

---

## UMA (Unconstrained Maximization Attack)

### Launcher Processes (KILLED)

| GPU | PID | Status |
|---|---|---|
| 0 | 503569 | KILLED |
| 1 | 503570 | KILLED |
| 2 | 503571 | KILLED |
| 3 | 503572 | KILLED |
| 4 | 503573 | KILLED |
| 5 | 503574 | KILLED |
| 6 | 503575 | KILLED |
| 7 | 503576 | KILLED |

### Bridge Processes at Kill Time

| PID | CPU% | Elapsed | Condition | Status |
|---|---|---|---|---|
| 912119 | 95.6% | 3:09 | UMA | Unknown — SSH disconnected |
| 913458 | 96.9% | 2:25 | UMA | Unknown — SSH disconnected |
| 914525 | 101% | 1:07 | UMA | Unknown — SSH disconnected |
| 914593 | 103% | 0:56 | UMA | Unknown — SSH disconnected |

### Per-GPU Completion

| GPU | Log File | Planned | Completed | Failed | Last Job |
|---|---|---|---|---|---|
| 0 | full_uma_gpu0.log | 21 | 0 | 21 | f09_s1_d3_p1_CLEAN FAILED exit=1 (16s) |
| 1 | full_uma_gpu1.log | 21 | 0 | 21 | f09_s1_d3_p2_CLEAN FAILED exit=1 (16s) |
| 2 | full_uma_gpu2.log | 20 | 0 | 20 | f09_s1_d1_p1_CLEAN FAILED exit=1 (18s) |
| 3 | full_uma_gpu3.log | 20 | 15 | 0 | f07_s1_d3_p0_CLEAN START (interrupted) |
| 4 | full_uma_gpu4.log | 20 | 14 | 2 | f08_s0_d3_p0_CLEAN START (interrupted) |
| 5 | full_uma_gpu5.log | 20 | 0 | 20 | f09_s1_d2_p1_CLEAN FAILED exit=1 (18s) |
| 6 | full_uma_gpu6.log | 20 | 14 | 0 | f07_s1_d1_p1_CLEAN START (interrupted) |
| 7 | full_uma_gpu7.log | 20 | 0 | 20 | f09_s1_d3_p0_CLEAN FAILED exit=1 (18s) |

### Failure Analysis

GPUs 0,1,2,5,7 show 100% failure rate with consistent 16-18 second runtime. This is NOT a timeout failure — it's a fast code error (likely import error, GPU availability check, or model loading failure). These GPUs may have been in use by other users at launch time.

GPUs 3,4,6 had reasonable completion rates (70-75%) before being interrupted.

### Key Observations

- ALL jobs ran with `_CLEAN` suffix — these are clean-input baselines, not attack runs
- Manifest: 21 lines per GPU (total 162 planned)
- No canary log for GPU0-2,5,7 (no completions to check)
- `full_uma_gpu3.log` shows 215s per successful job
- Output directory: EMPTY (no persisted artifacts from completed jobs)

---

## SHUFFLED (Shuffled Gradient Attack)

### Launcher Processes (KILLED)

Same PIDs 503569-503576 (co-launched with UMA on same GPUs).

### Bridge Processes at Kill Time

| PID | CPU% | Elapsed | Condition | Status |
|---|---|---|---|---|
| 887833 | 94.3% | 5:34 | SHUFFLED | Unknown — SSH disconnected |
| 909658 | 94.6% | 4:34 | SHUFFLED | Unknown — SSH disconnected |

### Per-GPU Completion

| GPU | Log File | Planned | Completed | Failed | Last Job |
|---|---|---|---|---|---|
| 0 | full_shuffled_gpu0.log | 21 | 0 | 21 | f09_s1_d3_p1_CLEAN FAILED exit=1 (16s) |
| 1 | full_shuffled_gpu1.log | 21 | 0 | 21 | f09_s1_d3_p2_CLEAN FAILED exit=1 (17s) |
| 2 | full_shuffled_gpu2.log | 20 | 0 | 20 | f09_s1_d1_p1_CLEAN FAILED exit=1 (17s) |
| 3 | full_shuffled_gpu3.log | 20 | 6 | 0 | f03_s3_d3_p0_CLEAN START (interrupted) |
| 4 | full_shuffled_gpu4.log | 20 | 5 | 1 | f03_s3_d3_p1_CLEAN START (interrupted) |
| 5 | full_shuffled_gpu5.log | 20 | 0 | 20 | f09_s1_d2_p1_CLEAN FAILED exit=1 (17s) |
| 6 | full_shuffled_gpu6.log | 20 | 5 | 0 | f03_s3_d1_p1_CLEAN START (interrupted) |
| 7 | full_shuffled_gpu7.log | 20 | 0 | 20 | f09_s1_d3_p0_CLEAN FAILED exit=1 (18s) |

### Failure Analysis

Identical pattern to UMA: GPUs 0,1,2,5,7 fail fast (16-18s), GPUs 3,4,6 succeed partially. GPUs 3,4,6 were working on fold_03 (low fold number) vs fold_09 for the failing GPUs.

Successful jobs take 477-570 seconds — much longer than failures.

---

## TMA Student (Timing-Matched Attack)

### All Launchers Completed Naturally

| GPU | Log File | Completed | Failed |
|---|---|---|---|
| 0 | full_tma_student_gpu0.log | 21 | 1 |
| 1 | full_tma_student_gpu1.log | 21 | 1 |
| 2 | full_tma_student_gpu2.log | 20 | 1 |
| 3 | full_tma_student_gpu3.log | 20 | 1 |
| 4 | full_tma_student_gpu4.log | 20 | 1 |
| 5 | full_tma_student_gpu5.log | 20 | 1 |
| 6 | full_tma_student_gpu6.log | 20 | 1 |
| 7 | full_tma_student_gpu7.log | 20 | 1 |
| **Total** | | **162** | **8** |

### Formal Validator

`TMA_STUDENT_FORMAL_PASS.json` — PASSED (162/162 expected)

---

## TMA Random-Time

### All Launchers Completed (GPU6 needed retry)

| GPU | Log File | Completed | Failed |
|---|---|---|---|
| 0 | full_tma_random_gpu0.log | 21 | 1 |
| 1 | full_tma_random_gpu1.log | 21 | 1 |
| 2 | full_tma_random_gpu2.log | 20 | 1 |
| 3 | full_tma_random_gpu3.log | 20 | 1 |
| 4 | full_tma_random_gpu4.log | 20 | 1 |
| 5 | full_tma_random_gpu5.log | 20 | 1 |
| 6 | full_tma_random_gpu6.log | 0 | 3 |
| 6r | full_tma_random_gpu6_r2.log | 19 | 1 |
| 7 | full_tma_random_gpu7.log | 20 | 1 |
| **Total** | | **161** | **11** |

### Missing Episode

fold_01 state_0 det_3 pert_1 — FAILED on both GPU6 initial run and retry (19s). Root cause not diagnosed.

### Formal Validator

**NOT RUN.** Only canary pass exists (`TMA_RANDOM_TIME_CANARY_PASS.json`).

---

## Watcher / Auto-Launch Status

| Check | Result |
|---|---|
| crontab (dty_user) | Empty |
| tmux sessions | None |
| screen sessions | None |
| Watcher scripts | None found running |
| Auto-launch scripts | None found running |
| Sleep loops | None found running |

**No automatic job restart mechanism is active.**

---

## GPU Lock / Reservation

No GPU locks found in `/mnt/sdc/dty_user/openvla_attack/gpu_locks/` or similar paths.

---

## Canary State

| Condition | Canary Pass |
|---|---|
| TMA_STUDENT | `TMA_STUDENT_CANARY_PASS.json` |
| TMA_RANDOM_TIME | `TMA_RANDOM_TIME_CANARY_PASS.json` |
| UMA_STUDENT | `UMA_STUDENT_CANARY_PASS.json` |
| SHUFFLED_STUDENT | `SHUFFLED_STUDENT_CANARY_PASS.json` |

Canary passes exist for all four conditions despite UMA/SHUFFLED main runs being incomplete.

---

## Supervisor / Chain Script Logs

| Log | Contents |
|---|---|
| full_chain.log | "Waiting for 16 TMA workers... FATAL" — early termination |
| full_chain_driver.log | Driver log (same content pattern) |
| auto_chain.log | Auto-chain attempt, also early termination |
| auto_chain_driver.log | Driver log |

The chain scripts were designed to auto-launch UMA+SHUFFLED after TMA pair completion, then run formal validators and aggregation. The chain detected TMA workers as incomplete and FATAL'd.

---

NO NEW EXPERIMENT WAS LAUNCHED.
NO LIVE SCIENTIFIC ARTIFACT WAS MODIFIED.
