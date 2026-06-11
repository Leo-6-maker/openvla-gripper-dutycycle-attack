# Object-100 Clean State Recovery

**Date**: 2026-06-11

## Inputs Audited

- `/data/liuyu/outputs/milestone_r1_official_eval_20260526/tables/object_official_script_100_manifest.csv`
- `/data/liuyu/outputs/milestone_r2_official_v4_object_alignment_20260526/tables/object_official_corrected_100_manifest_reconstructed.csv`
- `/data/liuyu/outputs/milestone_r2_official_v4_object_alignment_20260526/tables/object_v4_100_manifest_reconstructed.csv`
- `/data/liuyu/outputs/milestone_r2_official_v4_object_alignment_20260526/tables/object_official_vs_v4_per_episode_diff.csv`
- `/data/liuyu/outputs/milestone_1d_object_mujoco237_compat_20260526/object_full_10x10`

## Inventory Counts

- official_corrected_reconstructed: 100
- official_script_raw: 50
- v4_full10x10_raw: 100
- v4_runner_reconstructed: 100

## Ketchup State Provenance

Ketchup official-corrected R2 rows are low-confidence because the W45 CSV was overwritten and values were inferred from task-level SR. Raw per-state evidence is available from the v4 `object_full_10x10` run dirs.

- ketchup s0: success=True steps=155 runner=v4_full10x10_raw confidence=high source=/data/liuyu/outputs/milestone_1d_object_mujoco237_compat_20260526/object_full_10x10/obj_ketchup_s0
- ketchup s1: success=True steps=157 runner=v4_full10x10_raw confidence=high source=/data/liuyu/outputs/milestone_1d_object_mujoco237_compat_20260526/object_full_10x10/obj_ketchup_s1
- ketchup s2: success=True steps=160 runner=v4_full10x10_raw confidence=high source=/data/liuyu/outputs/milestone_1d_object_mujoco237_compat_20260526/object_full_10x10/obj_ketchup_s2
- ketchup s3: success=True steps=142 runner=v4_full10x10_raw confidence=high source=/data/liuyu/outputs/milestone_1d_object_mujoco237_compat_20260526/object_full_10x10/obj_ketchup_s3
- ketchup s4: success=True steps=155 runner=v4_full10x10_raw confidence=high source=/data/liuyu/outputs/milestone_1d_object_mujoco237_compat_20260526/object_full_10x10/obj_ketchup_s4
- ketchup s5: success=True steps=142 runner=v4_full10x10_raw confidence=high source=/data/liuyu/outputs/milestone_1d_object_mujoco237_compat_20260526/object_full_10x10/obj_ketchup_s5
- ketchup s6: success=True steps=148 runner=v4_full10x10_raw confidence=high source=/data/liuyu/outputs/milestone_1d_object_mujoco237_compat_20260526/object_full_10x10/obj_ketchup_s6
- ketchup s7: success=True steps=154 runner=v4_full10x10_raw confidence=high source=/data/liuyu/outputs/milestone_1d_object_mujoco237_compat_20260526/object_full_10x10/obj_ketchup_s7
- ketchup s8: success=True steps=125 runner=v4_full10x10_raw confidence=high source=/data/liuyu/outputs/milestone_1d_object_mujoco237_compat_20260526/object_full_10x10/obj_ketchup_s8
- ketchup s9: success=True steps=144 runner=v4_full10x10_raw confidence=high source=/data/liuyu/outputs/milestone_1d_object_mujoco237_compat_20260526/object_full_10x10/obj_ketchup_s9

## Tomato Sauce State Provenance

Tomato sauce has high-confidence official-corrected raw rows for states 0-9 in R1/R2 and high-confidence v4 full10x10 rows. Official state 7 failed, while v4 state 9 failed; this mismatch must be kept explicit.

- tomato_sauce s0: success=True steps=159 runner=v4_full10x10_raw confidence=high source=/data/liuyu/outputs/milestone_1d_object_mujoco237_compat_20260526/object_full_10x10/obj_tomato_sauce_s0
- tomato_sauce s1: success=True steps=153 runner=v4_full10x10_raw confidence=high source=/data/liuyu/outputs/milestone_1d_object_mujoco237_compat_20260526/object_full_10x10/obj_tomato_sauce_s1
- tomato_sauce s2: success=True steps=191 runner=v4_full10x10_raw confidence=high source=/data/liuyu/outputs/milestone_1d_object_mujoco237_compat_20260526/object_full_10x10/obj_tomato_sauce_s2
- tomato_sauce s3: success=True steps=135 runner=v4_full10x10_raw confidence=high source=/data/liuyu/outputs/milestone_1d_object_mujoco237_compat_20260526/object_full_10x10/obj_tomato_sauce_s3
- tomato_sauce s4: success=True steps=163 runner=v4_full10x10_raw confidence=high source=/data/liuyu/outputs/milestone_1d_object_mujoco237_compat_20260526/object_full_10x10/obj_tomato_sauce_s4
- tomato_sauce s5: success=True steps=171 runner=v4_full10x10_raw confidence=high source=/data/liuyu/outputs/milestone_1d_object_mujoco237_compat_20260526/object_full_10x10/obj_tomato_sauce_s5
- tomato_sauce s6: success=True steps=149 runner=v4_full10x10_raw confidence=high source=/data/liuyu/outputs/milestone_1d_object_mujoco237_compat_20260526/object_full_10x10/obj_tomato_sauce_s6
- tomato_sauce s7: success=True steps=169 runner=v4_full10x10_raw confidence=high source=/data/liuyu/outputs/milestone_1d_object_mujoco237_compat_20260526/object_full_10x10/obj_tomato_sauce_s7
- tomato_sauce s8: success=True steps=175 runner=v4_full10x10_raw confidence=high source=/data/liuyu/outputs/milestone_1d_object_mujoco237_compat_20260526/object_full_10x10/obj_tomato_sauce_s8
- tomato_sauce s9: success=False steps=280 runner=v4_full10x10_raw confidence=high source=/data/liuyu/outputs/milestone_1d_object_mujoco237_compat_20260526/object_full_10x10/obj_tomato_sauce_s9

## Level-3 Shortlist

| Rank | Task | State | Success | Steps | Confidence | Reason |
|---:|---|---:|---|---:|---|---|
| 1 | ketchup | 1 | True | 157 | high | clean_success=True; steps=157; runner=v4_full10x10_raw; confidence=high |
| 2 | tomato_sauce | 3 | True | 135 | high | clean_success=True; steps=135; runner=v4_full10x10_raw; confidence=high |
| 3 | ketchup | 3 | True | 142 | high | clean_success=True; steps=142; runner=v4_full10x10_raw; confidence=high |
| 4 | tomato_sauce | 5 | True | 171 | high | clean_success=True; steps=171; runner=v4_full10x10_raw; confidence=high |
| 5 | ketchup | 0 | True | 155 | high | clean_success=True; steps=155; runner=v4_full10x10_raw; confidence=high |
| 6 | ketchup | 2 | True | 160 | high | clean_success=True; steps=160; runner=v4_full10x10_raw; confidence=high |
| 7 | ketchup | 4 | True | 155 | high | clean_success=True; steps=155; runner=v4_full10x10_raw; confidence=high |
| 8 | ketchup | 5 | True | 142 | high | clean_success=True; steps=142; runner=v4_full10x10_raw; confidence=high |
| 9 | ketchup | 6 | True | 148 | high | clean_success=True; steps=148; runner=v4_full10x10_raw; confidence=high |
| 10 | ketchup | 7 | True | 154 | high | clean_success=True; steps=154; runner=v4_full10x10_raw; confidence=high |
| 11 | ketchup | 8 | True | 125 | high | clean_success=True; steps=125; runner=v4_full10x10_raw; confidence=high |
| 12 | ketchup | 9 | True | 144 | high | clean_success=True; steps=144; runner=v4_full10x10_raw; confidence=high |
