# Visual V2 Full10 Offline Replay

**Date**: 2026-05-30 | **Phase**: C1 | **Episodes**: 50 (10 tasks × 5 states)

## Trigger Phase Distribution

| Phase | ProprioNoStep | VisualNoStep_v2 | VisualProprioNoStep_v2 |
|-------|-------------|-----------------|------------------------|
| pre_contact (0-20%) | 4 (8.5%) | **50 (100%)** | 0 |
| grasp (20-35%) | 16 (34.0%) | 0 | 1 (50%) |
| transport (35-55%) | 1 (2.1%) | 0 | 0 |
| placement (55-75%) | 3 (6.4%) | 0 | 0 |
| release (75-100%) | 23 (48.9%) | 0 | 1 (50%) |
| **Total triggered** | 47/50 | 50/50 | 2/50 |

## Per-Task First Trigger Step

| Task | Class | ProprioNoStep | VisualNoStep_v2 | VisualProprioNoStep_v2 |
|------|-------|-------------|-----------------|------------------------|
| cream_cheese | high | 136 | **4** | — |
| tomato_sauce | high | 152 | **4** | — |
| ketchup | robust | 132 | **4** | — |
| salad_dressing | robust | 102 | **4** | — |
| butter | medium | 111 | **4** | — |
| chocolate_pudding | medium | 135 | **4** | 164 |
| bbq_sauce | medium | 152 | **4** | — |
| alphabet_soup | medium | 259 | **4** | — |
| milk | low | 132 | **4** | 102 |
| orange_juice | low | 115 | **4** | — |

## Selectivity

| Model | High Coverage | Robust Coverage | Gap |
|-------|-------------|-----------------|-----|
| ProprioNoStep | 10/10 | 10/10 | 0.00 (timing differs) |
| VisualNoStep_v2 | 10/10 | 10/10 | 0.00 (identical behavior) |
| VisualProprioNoStep_v2 | 0/10 | 0/10 | N/A (too few triggers) |

## Key Findings

1. **VisualNoStep_v2 fires at step 4 on ALL 50 episodes** — universal pre-contact trigger. Despite being freshly trained on teacher labels with AUC 0.95, the model has not learned contact-phase timing.

2. **VisualProprioNoStep_v2 is effectively silent** at its calibrated threshold (0.55). Only 2/50 triggers. The model outputs low scores (~0.04 mean) well below threshold.

3. **ProprioNoStep fires at varied phases**: 34% grasp, 49% release, 8.5% pre-contact. The trigger timing varies by task (step 102-259), reflecting genuine task-dependent contact dynamics.

4. **The gap between teacher AUC (0.95) and online selectivity is fundamental**: Teacher labels mark per-step vulnerability in a gripper-centric window. Visual features encode what the scene looks like, not when the gripper is interacting. A model can predict "this object is difficult" from the first frame and score high on AUC, but that doesn't translate to contact-phase selective triggering.

## Conclusion

Visual v2, like V6, encodes scene-level difficulty rather than contact-phase timing. The visual signal is not useless — it genuinely contains information about task/object vulnerability — but it fires too early to be useful as a standalone online trigger.

**Recommendation**: Visual v2 is `visual_only_scene_prior`. Do not use as online detector. Future work should explore visual as a re-ranker on ProprioNoStep candidate windows only.
