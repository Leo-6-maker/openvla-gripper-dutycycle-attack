# Object100 Opportunity Dataset v0 Audit

**Rows**: 294
**Episodes**: 74

## Stratum Distribution

| Stratum | Count |
|---|---|
| teacher_positive | 74 |
| far_too_early_control | 74 |
| early_pregrasp_control | 74 |
| random_noncritical_control | 72 |

## Label Balance

| Label | Count |
|---|---|
| 0 | 220 |
| 1 | 74 |

## Task Distribution

- **alphabet_soup**: 32 total (8 pos, 24 neg)
- **bbq_sauce**: 16 total (4 pos, 12 neg)
- **butter**: 32 total (8 pos, 24 neg)
- **cream_cheese**: 40 total (10 pos, 30 neg)
- **ketchup**: 40 total (10 pos, 30 neg)
- **milk**: 32 total (8 pos, 24 neg)
- **orange_juice**: 31 total (8 pos, 23 neg)
- **salad_dressing**: 35 total (9 pos, 26 neg)
- **tomato_sauce**: 36 total (9 pos, 27 neg)

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
