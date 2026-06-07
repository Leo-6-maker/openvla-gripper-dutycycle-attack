# Stage-B Patched Rerun — Completion Checklist

**Expected**: 44 windows × 2 (VIS+random) = 88 jobs

## Pre-Postprocess Checks

- [ ] Total summaries >= 80
- [ ] Total traces >= 80
- [ ] Worker logs all show DONE
- [ ] No worker reported CUDA OOM
- [ ] GPU memory released on all pairs

## Manifest Quality

- [ ] No job9xxx (smoke test) traces in output
- [ ] No old unpatched traces (missing env_action_0..6)
- [ ] All traces have obs_gripper_qpos_0/1
- [ ] All summaries have condition, task_key, state_id, window_start/end

## Hotfix Postprocess

- [ ] Script: `run_patched_rerun_postprocess_hotfix.py`
- [ ] Open convention: `env_action_6 < -0.5`
- [ ] Qpos: `abs(q0) + abs(q1)` from trace
- [ ] Shifted qpos: `step_dict[s+1]`
- [ ] Condition/task read from summary JSON, not filename
- [ ] Pairing: (task, state, ws, we)
- [ ] Old overnight labels NOT reused

## Expected Outputs

| File | Min Rows |
|------|----------|
| stageb_selective_rerun_qpos_hotfix.csv | >=80 |
| stageb_selective_rerun_labels_hotfix.csv | >=30 paired |

## Post-Run

- [ ] Hotfix script ran without Python errors
- [ ] Label distribution makes sense (open convention correct)
- [ ] Qpos deltas show variation (not all zeros)
- [ ] SIGNED_MEAN_CANCELLATION diagnosis present for old summaries
- [ ] Codex audit PASS before Smoke-C
