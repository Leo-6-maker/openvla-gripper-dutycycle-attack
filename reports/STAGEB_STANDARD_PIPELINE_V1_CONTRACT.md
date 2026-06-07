# Stage-B Standard Pipeline v1 — Contract

**Date**: 2026-06-07
**Status**: v1 standard (replaces ad-hoc patches from overnight run)

## Pipeline Components

| Script | Purpose |
|--------|---------|
| `run_paired_vis_random_v1.py` | Paired VIS PGD20 + random_linf runner |
| `postprocess_patched_traces_v1.py` | Trace-level qpos recompute (abs_sum) |
| `build_pair_labels_v1.py` | Pair labels from matched traces |
| `run_stageb_watchdog_v1.py` | Auto-launch watcher with gate checks |
| `validate_stageb_outputs_v1.py` | Output schema validation |

## Per-Step Trace Schema (19 required columns)

```
step, in_window, attack_this_step, env_grip, arm_l2,
pgd_applied, attacks_applied, gripper_qpos, done,
env_action_0..6, obs_gripper_qpos_0, obs_gripper_qpos_1, qpos_source
```

## Qpos Convention

- Source: `obs["robot0_gripper_qpos"]` (NOT `env.sim.data.qpos[-2:]`)
- Storage: `obs_gripper_qpos_0`, `obs_gripper_qpos_1` (both finger joints)
- Aggregation: `abs_sum = abs(q0) + abs(q1)`, `abs_mean = abs_sum / 2`
- NEVER use signed mean — joints have opposite signs, cancels to zero

## GPU Isolation

- `CUDA_VISIBLE_DEVICES=<physical_pair>` with `--gpu_pair 0,1`
- GPU 1,0 → worker_10, GPU 2,6 → worker_26, GPU 4,5 → worker_45
- GPU 3,7 blacklisted (Xid31 MMU fault)

## Label Definitions

- `command_susceptible`: VIS open>=6 OR streak>=6, random NOT meeting threshold
- `random_confounded`: random open>=6 OR streak>=6
- `physical_response_sensitive`: VIS qpos_delta_shifted >= 0.01
- `physical_response_strict`: VIS qpos_delta_shifted >= 0.02

## Hard Fail Conditions

- qpos[-2:]/qpos[-1:] used for gripper
- obs["robot0_gripper_qpos"] missing without validated fallback
- env_action vector missing
- pair_id missing
- trace_version != "patched_stageb_v1"
