# Pilot V2 State Selection After Success Fix

**Generated**: 2026-05-29 21:00 CST
**Server HEAD**: c62214f (blob-equivalent to remote freeze 0870443)

## Gate Assessment

| Criterion | Status |
|-----------|--------|
| ≥2 tasks with ≥3 eligible states | ✅ 4 tasks qualify |
| Pure clean success_official=True | ✅ |
| Detector clean success_official=True | ✅ |
| Detector clean attack_applied=0 | ✅ 9/9 verified |
| original==attacked all steps | ✅ 9/9 verified |
| No fresh Xid | ✅ |
| Manifest + step_records complete | ✅ |
| Working tree clean | ✅ |

## Selected Tasks

| Priority | Task | States | Avg Steps | Source |
|----------|------|--------|-----------|--------|
| 1 | tomato_sauce (task_id=5) | s0, s1, s2 | 197 | rerun |
| 2 | milk (task_id=7) | s0, s1, s2 | 172 | rerun |

**Total**: 2 tasks × 3 states × 4 conditions = 24 rollouts for pilot v2.

### Backup Pool
- salad_dressing (task_id=2): 3/3 valid
- ketchup (task_id=4): 3/3 valid
- cream_cheese (task_id=1): only 2/3 valid (s1 failed)

## Pilot v2 Design

- Output root: `/data/liuyu/outputs/milestone_2f_object_detector_matched_pilot_v2_successfix_20260529`
- Detector: `ProprioNoStep_baseline.pt` (SHA: 4b3f3d47...)
- Conditions: clean, oracle_open, random_control, gripper_inversion_proxy
- No `--force_detector_trigger`
- Primary GPU: 2,6 (clean gate first)

### Staged Launch
1. **Phase 3A**: clean condition only (6 rollouts) → gate check
2. **Phase 3B**: oracle_open, random_control, gripper_inversion_proxy (18 rollouts) — only if clean gate passes

## Decision

**PROCEED to Phase 3A (clean gate)**
