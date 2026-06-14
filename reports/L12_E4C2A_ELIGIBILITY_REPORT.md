# L12 E4C.2a — Label-Audit Repair Report

**Date**: 2026-06-15
**Stage**: E4C.2a
**Branch**: `exp/l12-critical-close-window-selector-20260615`
**Remapper**: `rc1a_corrected_v2_e1_5`
**Host**: klfy-SYS-4028GR-TR2 (8× RTX 2080 Ti, CUDA available)
**Elapsed**: 6.4s (62.3 traces/s)
**TRAINING_STARTED: NO**

---

## Runtime Provenance Seal (P0-5)

| Artifact | SHA256 (first 16) |
|----------|-------------------|
| Input manifest (runtime) | `b11ac1e8df74f63f` |
| Input manifest (expected) | `B11AC1E8DF74F63F` |
| `remap_v4_trace_for_l12.py` | `5d9cf327b25da459` |
| `phase_detector.py` | `f9cc7e90f415ee31` |
| `critical_close_selector.py` | `81b510ec30716df1` |
| Runner (`run_l12_e4c2a_repair.py`) | `f1e1dd44ebb30999` |

Seal: PASS (manifest SHA matches, all source files present and recorded).

---

## Gate Results

| Gate | Pass | Fail |
|------|------|------|
| Provenance (SHA + row count + existence) | 402 | 0 |
| Field validity (all rows, 9 fields, domain checks) | 402 | 0 |
| Open convention | 402 | 0 |
| RC1a remap (rows > 0, invariants = 0) | 402 | 0 |

### Field validity detail (P1)
- Domain checks added: obj_x/obj_y/eef_x/eef_y ∈ [-10,10], obj_z/eef_z ∈ [-1,5],
  env ∈ [-10,10], decoded_open ∈ {0,1}, qpos ∈ [0,1]
- Zero domain violations across all 402 traces

### RC1a remap detail (P1)
- All 402 traces: rows > 0, invariant_violations = 0
- Per-trace gripper_valid_rows, neutral_rows, field_issue_count now recorded

---

## Eligibility Classification (P0-2: full 10-category taxonomy)

| Category | Count | % of 402 |
|----------|-------|----------|
| ELIGIBLE_MULTI_CANDIDATE | 131 | 32.6% |
| TEACHER_P_AMBIGUOUS | 131 | 32.6% |
| TEACHER_P_UNAVAILABLE | 79 | 19.7% |
| ELIGIBLE_SINGLE_CANDIDATE | 53 | 13.2% |
| NO_CLOSE_CANDIDATE | 8 | 2.0% |
| PROVENANCE_FAIL | 0 | 0.0% |
| FIELD_VALIDITY_FAIL | 0 | 0.0% |
| OPEN_CONVENTION_FAIL | 0 | 0.0% |
| RC1A_REMAP_FAIL | 0 | 0.0% |
| OTHER_ABSTAIN | 0 | 0.0% |

### TP-qualifying candidate distribution (per trace)

| TP-qualifying count | Traces |
|---------------------|--------|
| 0 | 87 |
| 1 | 184 |
| 2 | 79 |
| >=3 | 52 |

### Key finding: 131 traces have >1 Teacher-P-qualifying candidate

In E4C.2, these 131 traces were SILENTLY assigned the first qualifying
candidate as Teacher-P anchor. E4C.2a now correctly classifies them as
TEACHER_P_AMBIGUOUS. The per-candidate Teacher-P evidence (eef distance,
cumulative vertical dz, sustained frames, EEF attachment, abstain reason)
is exported for all 2957 candidates to support downstream disambiguation.

---

## Candidate Statistics (P0-1: all traces)

| Metric | Count |
|--------|-------|
| Total CLOSE candidates (all 402 traces) | 2,957 |
| TP-criteria-pass candidates | 547 |
| Selector-emittable candidates | 2,943 |
| Selector-not-emittable candidates | 14 (all `too_early`, t<3) |

### Candidate definition (P0-3)
- All 2,957 candidates defined by the 3-condition OR:
  1. raw_open_to_close_crossing OR 2. close_onset OR 3. close_streak == 1
- Selector abstain filter is recorded as `selector_abstain_reason` and
  `selector_emittable` but does NOT remove candidates from the label pool
- 14 candidates with `too_early` (t<3) are preserved for early-spurious
  hard-negative construction

### All 16 frozen features exported (P0-4)
Each candidate row includes:
- 6 score decomposition fields (total_score, raw_crossing_bonus, close_streak_bonus, close_onset_qpos_bonus, eef_deceleration_bonus, qpos_ready_bonus)
- 3 continuous dynamic (eef_speed_now, eef_speed_prev, eef_deceleration_delta)
- 3 candidate definition (close_streak, raw_crossing, close_onset)
- 1 continuous (qpos)
- 2 temporal context (time_since_prev_close, time_since_last_open)
- 1 index (candidate_index)

Plus 7 metadata/Teacher-P evidence fields.

---

## Per-Task Coverage

| Task | Total | TP1 | TP>1 | TP0 | Multi | Single | Una | Amb | NoCand |
|------|-------|-----|------|-----|-------|--------|-----|-----|--------|
| alphabet_soup | 38 | 23 | 12 | 3 | 18 | 5 | 3 | 12 | 0 |
| bbq_sauce | 43 | 16 | 3 | 24 | 13 | 3 | 20 | 3 | 4 |
| butter | 42 | 17 | 12 | 13 | 17 | 0 | 13 | 12 | 0 |
| chocolate_pudding | 41 | 7 | 25 | 9 | 7 | 0 | 9 | 25 | 0 |
| cream_cheese | 42 | 15 | 21 | 6 | 15 | 0 | 6 | 21 | 0 |
| ketchup | 42 | 27 | 12 | 3 | 21 | 6 | 3 | 12 | 0 |
| milk | 41 | 21 | 10 | 10 | 6 | 15 | 10 | 10 | 0 |
| orange_juice | 41 | 24 | 11 | 6 | 8 | 16 | 2 | 11 | 4 |
| salad_dressing | 37 | 18 | 13 | 6 | 10 | 8 | 6 | 13 | 0 |
| tomato_sauce | 35 | 16 | 12 | 7 | 16 | 0 | 7 | 12 | 0 |

TP1 = traces with exactly 1 TP-qualifying candidate
TP>1 = traces with >1 TP-qualifying candidate (ambiguous)
TP0 = traces with candidates but 0 TP-qualifying (unavailable)
Una = TEACHER_P_UNAVAILABLE (TP0 + grasp privilege valid but 0 TP-qualifying)
Amb = TEACHER_P_AMBIGUOUS (>1 TP-qualifying)
NoCand = NO_CLOSE_CANDIDATE (0 CLOSE candidates)

---

## Comparison: E4C.2 vs E4C.2a

| Metric | E4C.2 | E4C.2a |
|--------|-------|--------|
| ELIGIBLE_MULTI_CANDIDATE | 261 | 131 |
| TEACHER_P_AMBIGUOUS | 0 | 131 |
| TEACHER_P_UNAVAILABLE | 87 | 79 |
| ELIGIBLE_SINGLE_CANDIDATE | 54 | 53 |
| NO_CLOSE_CANDIDATE | 0 | 8 |
| Total candidates exported | 1,780 | 2,957 |
| Features exported | 12 | 16 + 7 evidence |
| Traces with candidates exported | 315 | 402 |

The 131-trace shift from ELIGIBLE_MULTI_CANDIDATE to TEACHER_P_AMBIGUOUS
is entirely due to per-candidate Teacher-P evaluation revealing multiple
qualifying closes that the original greedy-first-match algorithm hid.

---

## Training Eligibility Pool

**Eligible for within-trace ranking**: 131 traces
- Exactly 1 TP-qualifying candidate + >=1 other CLOSE candidate
- Non-TP candidates serve as within-trace negatives
- 10/10 tasks represented (min: milk with 6, max: ketchup with 21)

**Eligible for binary classification**: 53 traces
- Exactly 1 TP-qualifying candidate, 0 other candidates

**Ambiguous (needs disambiguation)**: 131 traces
- >1 candidate passes all 5 Teacher-P criteria
- Per-candidate evidence fields exported for downstream resolution
- Cannot be used for training without disambiguation rule

**Unavailable**: 79 traces
- 0 TP-qualifying candidates (grasp privilege valid but no sustained lift)

**No candidates**: 8 traces
- 0 CLOSE candidates from rule-based enumeration (4 bbq_sauce, 4 orange_juice)

## P0 Issues Resolved

| Issue | Status |
|-------|--------|
| P0-1: Candidates for all traces | FIXED — 2957 candidates across all 402 traces |
| P0-2: Full 10-category taxonomy | FIXED — all 10 categories reachable, per-candidate TP evidence |
| P0-3: Candidate definition matches prereg | FIXED — selector abstain as metadata, not filter |
| P0-4: All 16 features | FIXED — 16 features + 7 TP evidence + selector flags |
| P0-5: Runtime provenance seal | FIXED — manifest/config/source SHAs verified at startup |
| P1: Domain checks + remap detail | FIXED — domain validity, gripper_valid/neutral rows recorded |

## Next Step

Await reviewer audit of E4C.2a. After acceptance, D1b can freeze train/val/test
splits on the corrected eligibility pool (131 multi-candidate + 53 single-candidate
= 184 unambiguous traces) with the 131 ambiguous traces addressed by a
preregistered disambiguation rule.
