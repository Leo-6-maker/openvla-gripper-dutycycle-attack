# Phase E Qpos Cache Audit V0

**Status**: OK
**Cache root**: `/data/liuyu/outputs/phaseE_mujoco_qpos_cache_20260605`
**Candidates**: `tables/fast_vis_calibration_candidates_v0.csv`
**Rows audited**: 8
**Cache usable?**: no
**Usable MuJoCo task/states**: 7
**Obs-only task/states**: 0
**Missing task/states**: 1
**Phase E generator can safely rerun?**: yes

This is CPU-only. It does not run rollout, VIS, watcher jobs, GPU work, or detector training.

## Rules

- Obs-only qpos is not acceptable for Phase E recommendation.
- All-zero obs qpos is flagged as untrusted.
- Missing MuJoCo qpos is `missing_mujoco_qpos`.
- Cache must cover the parent calibration windows.
