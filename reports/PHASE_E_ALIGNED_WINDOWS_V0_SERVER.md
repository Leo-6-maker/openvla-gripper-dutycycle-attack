# Phase E Aligned Windows V0

**Candidates source**: `tables/fast_vis_calibration_candidates_v0.csv`
**Labels source**: `tables/object_phase_response_labels_v2.csv`
**closed_threshold**: 0.015
**open_threshold**: 0.005
**Rows generated**: 120
**Recommended for Phase E**: 120
**Missing qpos rows**: 0
**Dry run**: False

This is a CPU-only candidate audit. It does not run rollout, VIS, watcher jobs, GPU work, or detector training.

## Notes

- trace qpos source for cream_cheese_s4: /data/liuyu/outputs/object_phase_response_batch3_VIS_20260604/cream_cheese_s4_near_w28_45/traces/cream_cheese_s4_vis_pgd_w28_45_trace.csv
- trace qpos source for milk_s4: /data/liuyu/outputs/object_phase_response_batch3_VIS_20260604/milk_s4_near_w19_36/traces/milk_s4_vis_pgd_w19_36_trace.csv
- trace qpos source for ketchup_s1: /data/liuyu/outputs/milestone_7_vis_controlled_rollout_micro_20260601/runs/vis_ketchup_state1_vis_pgd_full_d18_w28_45_seed0_185132_trace.csv
- trace qpos source for butter_s5: /data/liuyu/outputs/milestone_7_vis_controlled_rollout_micro_20260601/runs/vis_butter_state5_clean_full_d18_w25_42_seed0_225018_trace.csv
- trace qpos source for salad_dressing_s0: /data/liuyu/outputs/object_phase_response_batch3_VIS_20260604/salad_dressing_s0_far_w7_24/traces/salad_dressing_s0_vis_pgd_w7_24_trace.csv
- trace qpos source for bbq_sauce_s5: /data/liuyu/outputs/object_phase_response_batch3_VIS_20260604/bbq_sauce_s5_near_w27_44/traces/bbq_sauce_s5_vis_pgd_w27_44_trace.csv
- trace qpos source for ketchup_s5: /data/liuyu/outputs/milestone_7_vis_controlled_rollout_micro_20260601/runs/vis_ketchup_state5_clean_full_d18_w9_26_seed0_225735_trace.csv
- trace qpos source for milk_s5: /data/liuyu/outputs/milestone_7_vis_controlled_rollout_micro_20260601/runs/vis_milk_state5_random_linf_full_d18_w25_42_seed0_230805_trace.csv

## Qpos Phase Rule

- `qpos >= 0.015`: `true_closed`.
- `qpos <= 0.005`: `natural_open`.
- Otherwise: `transitional-pre-open`.
- `true_closed` may be recommended when denominator/provenance/mismatch gates pass.
- `transitional-pre-open` may be recommended when `true_closed_score >= 0.35` and gates pass.
- `natural_open` and missing-qpos rows are rejected.

## Qpos Phase Counts

- `true_closed`: 116
- `transitional-pre-open`: 4
- `natural_open`: 0
- `missing`: 0

## Selection Rule

- Do not assume centered L10 is valid.
- Recommend true_closed windows directly after denominator/provenance/mismatch gates.
- Recommend transitional-pre-open windows only when true_closed_score is at least 0.35.
- MuJoCo qpos is preferred; obs qpos is fallback; missing qpos is never auto-recommended.
- Polluted denominators, severe phase proxy mismatch, and infra-failed provenance block recommendation.

## Trace Root Guidance

- Broad `/data/liuyu/outputs` scans may miss traces because the script caps CSV scanning for safety.
- Prefer specific trace roots when available:
  - `/data/liuyu/outputs/nightly_object_batch3_20260604`
  - `/data/liuyu/outputs/object_phase_response_batch3_VIS_20260604`
  - `/data/liuyu/outputs/object_phase_response_batch4_...`
