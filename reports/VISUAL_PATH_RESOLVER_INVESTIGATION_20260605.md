# Visual Path Resolver Investigation 2026-06-05

Scope: server read-only path investigation after server visual availability audit returned 0 trigger RGB rows. No GPU, rollout, VIS, watcher, detector v2 training, or model loading was run.

Server reviewed checkout:

```text
/data/liuyu/repos/openvla-gripper-dutycycle-attack-reviewed-20260605
HEAD: ceab89829d2f06ef3da1bf7a0fb07f34eb85b546
branch: exp/vis-prefix-margin-repair-20260603
```

## Directory Structure

Batch3 VIS directories contain logs, manifests, and trace CSVs:

```text
/data/liuyu/outputs/nightly_object_batch3_20260604/batch3_VIS/ketchup_s1_approach_near_closed_proxy_w21_38/phase_conditioned_attack_manifest.json
/data/liuyu/outputs/nightly_object_batch3_20260604/batch3_VIS/ketchup_s1_approach_near_closed_proxy_w21_38/traces/batch3_ketchup_s1_approach_near_closed_proxy_w21_38_vis_pgd_w21_38_trace.csv
/data/liuyu/outputs/object_phase_response_batch3_VIS_20260604/cream_cheese_s4_near_w28_45/phase_conditioned_attack_manifest.json
/data/liuyu/outputs/object_phase_response_batch3_VIS_20260604/cream_cheese_s4_near_w28_45/traces/cream_cheese_s4_vis_pgd_w28_45_trace.csv
```

Batch3b directories also contain logs, manifests, trace CSVs, and denominator audits:

```text
/data/liuyu/outputs/nightly_object_batch3b_20260604/tomato_sauce_s3/phase_conditioned_attack_manifest.json
/data/liuyu/outputs/nightly_object_batch3b_20260604/tomato_sauce_s3/traces/tomato_sauce_s3_clean_w17_34_trace.csv
/data/liuyu/outputs/nightly_object_batch3b_20260604/tomato_sauce_s3/audit/den_check_summary.csv
/data/liuyu/outputs/nightly_object_batch3b_20260604/tomato_sauce_s3/audit/den_check_prov.csv
```

## Image Search

Search over batch3-like paths found no image/array files:

```bash
find /data/liuyu/outputs -path "*batch3*" -type f \( \
  -iname "*.png" -o -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.webp" -o -iname "*.npy" -o -iname "*.npz" \
\) | head -300
```

Result: no files returned.

## Manifest / Trace Search

Manifest and trace files do exist:

```text
/data/liuyu/outputs/object_phase_response_batch3_VIS_20260604/milk_s4_near_w19_36/phase_conditioned_attack_manifest.json
/data/liuyu/outputs/object_phase_response_batch3_VIS_20260604/milk_s4_near_w19_36/traces/milk_s4_vis_pgd_w19_36_trace.csv
/data/liuyu/outputs/object_phase_response_batch3_precheck_20260604/ketchup_s1/traces/ketchup_s1_clean_w21_38_trace.csv
/data/liuyu/outputs/nightly_object_batch3b_20260604/tomato_sauce_s3/traces/tomato_sauce_s3_clean_w17_34_trace.csv
```

## Trace CSV Columns

Representative trace header for `cream_cheese_s4_vis_pgd_w28_45_trace.csv`:

```text
task,condition,seed,state_id,objective,eps_raw_pixels,eps_processor,window_start,window_end,semantics_version,trace_generated_by_repaired_runner,step,policy_step,in_window,attack_attempted,attack_invalid,attack_invalid_reason,attack_invalid_detail,pgd_applied,controller_active,controller_stopped,effective_attack_step_idx,raw_gripper,env_gripper,gripper_qpos,qpos_pre_step,qpos_post_step,clean_grip,adv_grip,clean_z,adv_z,nad_dof7,nad_z,nad_dof1_3,arm_l2,linf,token_flip,attack_dt,eef_x,eef_y,eef_z,done,reward,ctrl_mode,ctrl_stop_reason,ctrl_streak,ctrl_max_streak,ctrl_qpos_delta,ctrl_attacks,window_source,phase,selector_type,selector_checkpoint,detector_trigger_step,phase_label_3class,phase_label_6class,phase_label_validity,phase_window_selection_validity,phase_window_selection_reason,clean_natural_open_ratio,natural_release_confounded,phase_conditioned_wrapper_version
```

No `image`, `rgb`, `frame_path`, or camera path column is present.

## Manifest Content

Representative Batch3 manifest:

```json
{
  "condition": "vis_pgd",
  "episode_id": "cream_cheese_s4",
  "global_trace_path": "/data/liuyu/outputs/milestone_7_vis_controlled_rollout_micro_20260601/runs/vis_cream_cheese_state4_vis_pgd_full_d18_w28_45_seed0_002844_trace.csv",
  "localized_trace_path": "/data/liuyu/outputs/object_phase_response_batch3_VIS_20260604/cream_cheese_s4_near_w28_45/traces/cream_cheese_s4_vis_pgd_w28_45_trace.csv",
  "rc": 0,
  "selector_type": "fixed",
  "state_id": 4,
  "task": "cream_cheese",
  "trace_patch_failed": false,
  "window_end": 45,
  "window_source": "fixed",
  "window_start": 28
}
```

No image path is present in this manifest.

## Conclusion

RGB/image files appear to be absent from the checked Batch3/Batch3b output paths, not merely missed by the current path resolver. Trace CSVs and manifests contain rollout metadata and state/action/qpos summaries, but no image path columns.

## Required Resolver Changes

No path-rule patch is justified yet because the investigation did not find saved RGB/image files or manifest image fields. The next useful change would require DeepSeek to provide one of:

- a directory containing saved RGB frames,
- a manifest with image path fields,
- a trace CSV schema with image/camera path columns,
- or a replay-render/export procedure that creates trigger-centered RGB frames.

Until then, GPU6 frozen embedding smoke is blocked by missing visual inputs.
