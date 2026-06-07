# Stage-B v1.1 Reachable Window Candidate Generation

**Date**: 2026-06-07
**Source**: RC1a clean reachability scan (27 rollouts, 0 failures)
**Status**: CPU-only generation complete — awaiting Codex audit

## 1. Input

- 27 clean traces from `/data/liuyu/outputs/stageb_v1_1_clean_reachability_scan_rc1a_20260607/`
- All `trace_version == corrected_stageb_v1_1`
- All `source_snapshot_id == f9840cb1`
- All `prompt_style == official_in_out`
- All `image_preprocess_style == official_rot180_only`
- All `condition == clean` (no VIS, no random)

## 2. Generation Method

| Parameter | Value |
|-----------|-------|
| window_len | 11 steps |
| stride | 5 steps |
| min_window_steps | 8 |
| window_start >= | 5 |
| window_end <= | actual_max_step - 5 |
| done filtering | exclude windows starting at or after done_step |

## 3. Results

| Stratum | Count |
|---------|-------|
| high_opportunity | 53 |
| medium_opportunity | 278 |
| hard_negative_or_idle | 867 |
| **Total** | **1198** |

### Stratum criteria

- **high_opportunity**: open_frac <= 0.3, qpos not saturated, eef_displacement > 0.001
- **medium_opportunity**: open_frac <= 0.7, qpos not saturated
- **hard_negative_or_idle**: everything else (high open, saturated qpos, low motion)

## 4. Smoke3 Queue

| Window | Stratum | open | qpos_mean |
|--------|---------|------|-----------|
| alphabet_soup s1 seed=1 [45,55] | high_opportunity | 0/11 | 0.009 |
| bbq_sauce s0 seed=0 [60,70] | medium_opportunity | 0/11 | 0.032 |
| butter s0 seed=0 [55,65] | hard_negative | 8/11 | 0.079 |

3 different tasks. All ws >= 20, well within trace boundaries.

## 5. Pilot12 Queue

12 windows: 4 high + 4 medium + 4 hard_negative, across >= 4 tasks.

## 6. Output Files

| File | Rows |
|------|------|
| `tables/stageb_v1_1_clean_rollout_summary.csv` | 27 |
| `tables/stageb_v1_1_reachable_window_candidates.csv` | 1198 |
| `tables/stageb_v1_1_corrected_smoke3_queue.csv` | 3 |
| `tables/stageb_v1_1_corrected_pilot12_queue.csv` | 12 |

## 7. Provenance

- All candidates from RC1a clean traces
- No old windows, no old labels, no pre-v1.1 data
- No VIS/random/PGD evidence in any candidate
- source_snapshot_id = f9840cb1 throughout

## 8. Next Step

Codex CPU-only audit → PASS_FOR_3ROW_CORRECTED_SMOKE → corrected VIS smoke
