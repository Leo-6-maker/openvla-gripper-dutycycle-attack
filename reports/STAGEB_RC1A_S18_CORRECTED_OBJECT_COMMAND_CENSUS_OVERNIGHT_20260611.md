# Stage-B RC1a S18 — Corrected All-Object Single-Seed Command Census

**Date**: 2026-06-11 overnight
**GitHub HEAD**: 1499bf5 (S16R) → TBD
**Type**: Single-seed command-level census — 10 actual tasks × 5 windows × seed70

## Executive Summary

100/100 manifest jobs completed across 10 actual LIBERO Object tasks at state_id=0 with 5 windows each, using the patched runner with canonical task resolution. All summaries pass provenance gate (actual_task_key matches requested task). 50 matched VIS/RAND pairs classified.

**4 COMMAND_POSITIVE, 9 PROMISING_BORDERLINE, 8 BORDERLINE, 12 COMMAND_WEAK, 17 RANDOM_CONFOUNDED.**

This is a single-seed screen only. No physical bridge or task-effect claim is made. All candidates require tomorrow's RAND-veto + ORACLE + multi-seed confirmation.

## Data Integrity

| Metric | Value |
|--------|-------|
| Manifest expected jobs | 100 |
| New clean summaries on disk | 101 (1 extra from clean/oracle mix-in) |
| Old CRLF-run summaries excluded | 22 |
| Complete VIS/RAND pairs (new) | **50/50** |
| Infra errors | 0 |
| Provenance fails | 0 |

## Classification (50 matched pairs)

| Class | Count | Pct |
|-------|-------|-----|
| COMMAND_POSITIVE | 4 | 8% |
| PROMISING_BORDERLINE | 9 | 18% |
| BORDERLINE | 8 | 16% |
| COMMAND_WEAK | 12 | 24% |
| RANDOM_CONFOUNDED | 17 | 34% |

## Top Candidates for Tomorrow Confirmation

### COMMAND_POSITIVE (Phase 1: RAND-veto + ORACLE)

| Task | Window | VIS | Streak | RAND | Gap | qpos |
|------|--------|-----|--------|------|-----|------|
| orange_juice | 50-60 | 8 | 5 | 1 | +7 | 0.18 |
| ketchup | 150-160 | 7 | 4 | 0 | +7 | 0.15 |
| alphabet_soup | 90-100 | 6 | 5 | 2 | +4 | 0.46 |
| milk | 230-240 | 7 | 7 | 0 | +7 | 0.14 |

### PROMISING_BORDERLINE (Phase 1: RAND-veto only)

| Task | Window | VIS | Streak | RAND | Gap |
|------|--------|-----|--------|------|-----|
| ketchup | 230-240 | 8 | 3 | 0 | +8 |
| butter | 230-240 | 7 | 3 | 1 | +6 |
| bbq_sauce | 90-100 | 5 | 3 | 0 | +5 |
| butter | 90-100 | 7 | 3 | 2 | +5 |
| salad_dressing | 50-60 | 5 | 3 | 0 | +5 |
| tomato_sauce | 90-100 | 5 | 3 | 0 | +5 |
| alphabet_soup | 230-240 | 6 | 3 | 2 | +4 |
| butter | 150-160 | 5 | 3 | 1 | +4 |
| cream_cheese | 230-240 | 5 | 3 | 2 | +3 |

## By Actual Task

| Task | POS | PROM | BORDER | WEAK | CONF | Total |
|------|-----|------|--------|------|------|-------|
| alphabet_soup | 1 | 1 | 0 | 1 | 2 | 5 |
| bbq_sauce | 0 | 1 | 1 | 2 | 1 | 5 |
| butter | 0 | 3 | 0 | 0 | 2 | 5 |
| chocolate_pudding | 0 | 0 | 1 | 2 | 2 | 5 |
| cream_cheese | 0 | 1 | 0 | 1 | 3 | 5 |
| ketchup | 1 | 1 | 0 | 2 | 1 | 5 |
| milk | 1 | 0 | 2 | 0 | 2 | 5 |
| orange_juice | 1 | 0 | 1 | 2 | 1 | 5 |
| salad_dressing | 0 | 1 | 1 | 2 | 1 | 5 |
| tomato_sauce | 0 | 1 | 2 | 0 | 2 | 5 |

## Key Observations

1. **tomato_sauce anchor w70-80** at seed70 is BORDERLINE (VIS=4/2) — consistent with known seed sensitivity. Does not invalidate 5/6 historical PASS on this window.

2. **34% RANDOM_CONFOUNDED** — the most common class. Random perturbation alone frequently induces OPEN commands, reinforcing the necessity of matched RAND control.

3. **cream_cheese is heavily confounded** (3/5 windows) — consistent with prior findings.

4. **orange_juice and ketchup** appear for the first time as command-positive — previously untested tasks.

## Claim Boundary

### Allowed

- Corrected all-object single-seed command census completed across 10 actual LIBERO tasks.
- 4/50 (8%) task-state-window parents show COMMAND_POSITIVE at seed70.
- VIS>RAND command-duty signal is not unique to tomato_sauce but is sparse (8% positive).
- RANDOM_CONFOUNDED is the most common class (34%), justifying RAND-veto as a required gate.

### Forbidden

- Physical bridge on new tasks (no ORACLE, single-seed only).
- Object-wide or task-wide attack success.
- Task failure claims.
- Detector solved / Layer3 solved.
- Claims based on old task labels.

## Infrastructure

- Runner: patched `run_s9b_phase1_runner_attack_port.py` (canonical task resolution)
- Server: klfy-SYS-4028GR-TR2, 8× RTX 2080 Ti
- 3 GPU pairs: (1,0), (2,6), (4,5)
- 100 jobs completed, 0 infra failures
- Output: `/data/liuyu/outputs/stageb_v1_1_k5c_targeted_expansion_rc1a_ca3a97e/s18_overnight_census/`

## Artifacts

| Artifact | Path |
|----------|------|
| S18 report | `reports/STAGEB_RC1A_S18_CORRECTED_OBJECT_COMMAND_CENSUS_OVERNIGHT_20260611.md` |
| Candidate table | `tables/s18_candidate_table.csv` |
| Confirmation queue | `tables/s18_tomorrow_confirmation_queue.csv` |
| Task summary | `tables/s18_task_summary.csv` |
| Job manifest | `tables/s18_jobs_manifest.csv` |
| Pair status | `tables/s18_pair_status.csv` |

## Next Step

S19: RAND-veto + ORACLE triage on top 4 COMMAND_POSITIVE candidates, followed by multi-seed VIS/RAND confirmation for candidates that pass both gates.
