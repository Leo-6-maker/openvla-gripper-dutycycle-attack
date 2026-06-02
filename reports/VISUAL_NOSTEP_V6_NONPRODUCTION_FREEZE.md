# VisualNoStep V6 Online Pilot — Non-Production Freeze

**Date**: 2026-05-30 | **Branch**: `exp/visual-detector-v2-training-20260530` | **Commit**: `e060d58`

## Result (threshold=0.05, detector_input_mode=visual)

| Task | Clean | sus30 |
|------|-------|-------|
| cream_cheese | 3/3 (trig=10,32,0) | 1/3 (trig=13,66,0; atk=92,131,0) |
| tomato_sauce | 3/3 (trig=8,5,10) | 1/3 (trig=13,10,7; atk=180,180,61) |
| ketchup | 3/3 (trig=96,35,43) | **0/3** (trig=199,131,183; atk=253,216,187) |
| salad_dressing | 3/3 (trig=0,29,42) | 2/3 (trig=0,29,206; atk=0,37,245) |

**Total**: 24 episodes, 1167 trigger steps, 10 episodes with attack, 16 success

## Selectivity vs ProprioNoStep

| Task | ProprioNoStep sus30 | VisualNoStep V6 sus30 |
|------|---------------------|-----------------------|
| cream_cheese (high) | 0/3 | 1/3 |
| tomato_sauce (high) | 0/3 | 1/3 |
| ketchup (robust) | **3/3** | **0/3** |
| salad_dressing (robust) | 3/3 | 2/3 |

VisualNoStep @ th=0.05 breaks the robust ketchup control that ProprioNoStep preserves.

## Clean Trigger Rate

VisualNoStep triggers on clean rollouts for ALL tasks at th=0.05:
- cream_cheese: 42 triggers across 3 clean episodes (14 avg)
- ketchup: 174 triggers across 3 clean episodes (58 avg) — highest
- salad_dressing: 71 triggers across 3 clean episodes (24 avg)
- tomato_sauce: 23 triggers across 3 clean episodes (8 avg)

Clean triggers do NOT produce attacks (attack_condition=clean prevents it), but they indicate the visual detector's hazard signal is not selective enough to distinguish clean vs. sus30.

## Mechanism: Why VisualNoStep Lacks Selectivity

ProprioNoStep's selectivity is not because its model is better trained. It's because **its input domain is naturally selective**:

| Dimension | ProprioNoStep | VisualNoStep V6 |
|-----------|--------------|-----------------|
| Input signal | 13-dim proprio/action (gripper, EEF, forces) | 2176-dim DINOv2+SigLIP (scene appearance) |
| What it encodes | Physical contact dynamics | Object/scene difficulty |
| Trigger phase | Contact / transport / placement (step 120-160) | Pre-contact / approach (step 14-63) |
| Attack effect | Selective contact-phase disruption | Non-selective grasp blocking |
| Selectivity | High 0/10, Robust 10/10 | Ketchup robust 0/3 broken |

Proprioceptive signal (gripper position, EEF velocity, action commands) directly encodes physical interaction — when the gripper touches, moves, or releases an object. ProprioNoStep naturally learns to fire at contact-relevant phases because that's when the signal changes.

Visual signal (DINOv2+SigLIP fused features) encodes what the scene looks like — which object, what pose, what background. The visual detector learns that "this scene/object looks difficult," which correlates with vulnerability but is NOT causally timed to contact. It fires before the gripper reaches the object, turning the attack into grasp prevention rather than contact disruption.

## Conclusions

### Valid
- VisualNoStep can trigger online and can drive sustained_command_open_proxy attacks at threshold 0.05.
- VisualNoStep @ th=0.05 is non-selective: it triggers pre-contact (step 14-63) rather than at contact phase.
- ProprioNoStep remains the production detector because proprio input is naturally selective for contact dynamics.
- Visual information is NOT useless — it correlates with task difficulty. But the current visual detector has not learned when contact is established.

### Forbidden
- "VIS attack failed" — The attack mechanism works; visual trigger timing is wrong.
- "Visual information is useless" — Visual signal encodes task/object difficulty; it's just not contact-phase calibrated.
- "VisualNoStep is production-ready" — It is not.

## Future Path

If Visual v2 is pursued:
1. Frame as **contact-phase re-ranker**: Proprio provides contact timing; Visual judges vulnerability at those moments.
2. Train on clean teacher labels at contact-phase windows only.
3. Evaluate whether visual adds information beyond proprio at contact-timed windows.
4. Do NOT use visual as standalone trigger — visual alone lacks contact-phase timing.

## Data

Output: `/data/liuyu/outputs/milestone_2j_visual_fusion_online_pilot_v6_20260530`
Checkpoint: `/data/liuyu/outputs/milestone_2e3_object100_visual_proprio_no_step_20260527/models/VisualNoStep_frozen.pt`
