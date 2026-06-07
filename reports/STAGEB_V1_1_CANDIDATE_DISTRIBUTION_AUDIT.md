# Stage-B v1.1 Candidate Distribution Audit

**Date**: 2026-06-07
**Source**: 1198 candidates from 27 RC1a clean traces
**Status**: CPU-only — no VIS/random/PGD launched

## 1. Stratum Distribution

| Stratum | Count | open (mean) | qpos (mean) | eef_disp (mean) |
|---------|-------|-------------|-------------|-----------------|
| high_opportunity | 53 | 1.1 | 0.014 | 0.010 |
| medium_opportunity | 278 | 1.0 | 0.023 | 0.001 |
| hard_negative_or_idle | 867 | 9.3 | 0.074 | 0.004 |

### Key observation

**hard_negative_or_idle is dominated by ceiling-open windows** (93% have open>=8 and qpos>0.06). Only 10 true hard negatives exist (open<=3, qpos<0.04).

This is expected behavior: the stratum was defined as "everything else" — including high-open idle tails, saturated gripper states, and low-motion periods.

## 2. Per-Task Distribution

| Task | high | med | hard |
|------|------|-----|------|
| alphabet_soup | 8 | 32 | 128 |
| bbq_sauce | 3 | 66 | 34 |
| butter | 8 | 31 | 121 |
| cream_cheese | 8 | 49 | 81 |
| ketchup | **0** | **0** | **139** |
| milk | 4 | 7 | 103 |
| orange_juice | 10 | 31 | 109 |
| salad_dressing | 5 | 24 | 95 |
| tomato_sauce | 7 | 38 | 57 |

**ketchup has zero high/medium candidates** — all windows are hard_negative (always-open trajectories).

## 3. Butter s0 [55,65] Assessment

| Field | Value |
|-------|-------|
| open_count | 8/11 |
| qpos_mean | 0.079 |
| stratum | hard_negative_or_idle |

**Verdict**: This is a **ceiling_open_control** — gripper already fully open (qpos saturated at ~0.08). VIS cannot increase opening further. Should NOT be used as a hard negative control.

## 4. Smoke3 Proposals

### Smoke3-A: ceiling_open_control

| Window | Stratum | open | qpos |
|--------|---------|------|------|
| alphabet_soup s1 [45,55] | high_opportunity | 0/11 | 0.009 |
| bbq_sauce s0 [60,70] | medium_opportunity | 0/11 | 0.032 |
| butter s0 [20,30] | ceiling_open_control | 11/11 | 0.079 |

### Smoke3-B: true_hard_negative (RECOMMENDED)

| Window | Stratum | open | qpos |
|--------|---------|------|------|
| alphabet_soup s1 [45,55] | high_opportunity | 0/11 | 0.009 |
| bbq_sauce s0 [60,70] | medium_opportunity | 0/11 | 0.032 |
| cream_cheese s2 [45,55] | true_hard_negative | 0/11 | 0.034 |

## 5. Recommendation

**Execute Smoke3-B first.**

Rationale:
- Smoke3-A's ceiling_open_control (open=11/11, qpos=0.079) has zero room for VIS-induced opening — the gripper is mechanically saturated.
- Smoke3-B's true_hard_negative (open=0/11, qpos=0.034) has physical room for VIS to induce opening.
- Both proposals share the same high and medium candidates.
- Smoke3-B provides a cleaner scientific control: "window where model does NOT open, gripper is mid-range, VIS should be able to force opening."

## 6. Output Files

- `tables/stageb_v1_1_candidate_distribution_by_task.csv`
- `tables/stageb_v1_1_candidate_distribution_by_stratum.csv`
- `tables/stageb_v1_1_smoke3_alternatives.csv`
