# VIS Arm-Drift Diagnostic Run

Date: 2026-05-31

## Scope

No rollout and no training were run.

This diagnostic used the same one-frame Object sample as the bf16 budget smoke:

```text
/data/liuyu/outputs/milestone_2i_visual_fusion_online_detector_pilot_20260530/runs/libero_object/vis_ketchup_clean_ketchup_s0/frames/step_0000.png
```

GPU visibility:

```text
CUDA_VISIBLE_DEVICES=4,5
```

## Output

```text
tables/vis_arm_drift_sweep.csv
```

## Configuration

| Field | Value |
| --- | --- |
| objective | `target_action_ce` |
| eps | `4/255` |
| steps | `4` |
| step_size | `1/255` |
| random baseline | same processor-pixel Linf |

## Result

| Metric | Targeted VIS | Random same-Linf |
| --- | ---: | ---: |
| perturbation_linf | `0.015625` | `0.015625` |
| clean gripper token | `31872` | `31872` |
| adv gripper token | `31744` | `31872` |
| token flip | `true` | `false` |
| gripper delta | `0.996078` | `0.0` |
| arm L2 | `0.069530` | `2.14e-08` |
| gripper_to_arm_ratio | `14.3259` | `0.0` |

Targeted VIS changed the decoded gripper action strongly in the intended
direction. Random same-Linf did not flip the gripper token and did not change
the gripper action.

## Gate Decision

Limited one-frame arm-drift/random baseline gate: PASS.

This does not authorize rollout by itself. It only shows that, on this single
frame, the observed gripper effect is not reproduced by a random same-Linf
perturbation and arm drift is not dominant relative to gripper delta.

## Remaining Boundary

Do not run detector-triggered VIS or broad rollout from this single-frame gate.

Recommended next step is a small no-rollout confirmation across additional
Object contact frames before proposing any forced-window VIS micro rollout.
