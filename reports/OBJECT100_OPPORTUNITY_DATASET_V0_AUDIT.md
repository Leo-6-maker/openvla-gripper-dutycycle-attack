# Object100 Opportunity Dataset v0 Audit

**Rows**: 222
**Episodes**: 74

## Stratum Distribution

| Stratum | Count |
|---|---|
| teacher_positive | 74 |
| late_noncritical_control | 74 |
| early_pregrasp_control | 74 |

## Label Balance

| Label | Count |
|---|---|
| 0 | 148 |
| 1 | 74 |

## Task Distribution

- **alphabet_soup**: 24 total (8 pos, 16 neg)
- **bbq_sauce**: 12 total (4 pos, 8 neg)
- **butter**: 24 total (8 pos, 16 neg)
- **cream_cheese**: 30 total (10 pos, 20 neg)
- **ketchup**: 30 total (10 pos, 20 neg)
- **milk**: 24 total (8 pos, 16 neg)
- **orange_juice**: 24 total (8 pos, 16 neg)
- **salad_dressing**: 27 total (9 pos, 18 neg)
- **tomato_sauce**: 27 total (9 pos, 18 neg)

## Feature Groups

| Group | Count | Example |
|---|---|---|
| action_gripper_command | 13 | gripper_qpos_mean |
| action_gripper_stats | 12 | gripper_qpos_mean |
| proprio_eef | 3 | eef_displacement |
| proprio_gripper_qpos | 13 | gripper_qpos_mean |
| proprio_gripper_width | 12 | gripper_qpos_mean |
| window_metadata | 5 | window_start |
| window_position | 5 | window_start |

## Hard Rules Check

- [x] No step_idx as input feature
- [x] No normalized_step as input feature
- [x] No teacher_phase as input feature
- [x] No teacher_hazard as input feature
- [x] No mechanism_eligible as input feature
- [x] Window position uses frac (not absolute step)
- [x] All features causal/online-legal
