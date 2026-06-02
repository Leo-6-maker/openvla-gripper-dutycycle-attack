# Sustained Command-Layer Proxy — Group Meeting Summary

**Date**: 2026-05-30 | **Status**: SELECTIVE SUSTAINED PROXY POSITIVE

## One-Line Result

ProprioNoStep + sustained_command_open_proxy_30 selectively causes failure on high oracle-sensitive Object tasks (0/10 sus30 success) while preserving robust controls (10/10 sus30 success).

## Key Numbers

| Class | Tasks | sus30 Success |
|-------|-------|--------------|
| High | cream_cheese, tomato_sauce | **0/10** |
| Robust | ketchup, salad_dressing | **10/10** |
| Medium | alphabet_soup, bbq_sauce, butter, chocolate_pudding | 6/20 |
| Low | milk, orange_juice | 7/10 |

**Selectivity**: High 0% vs Robust 100% — 100 percentage point separation.

## Production Pipeline

1. **Detector**: ProprioNoStep (CausalTCNDetector, 13-dim proprio/action input, 38,602 params)
2. **Attack**: sustained_command_open_proxy_30 (burst_steps=30, hold_mode=fixed)
3. **Mechanism**: Command-layer sustained proxy — gripper action overridden to fully open for 30+ steps when detector triggers consecutively

## Why It's Selective

### Detection Phase
- ProprioNoStep fires at **contact/grasp phase** (step 120-160)
- Proprioceptive signal naturally encodes gripper-object interaction dynamics
- Clean trigger rate is low on robust tasks

### Attack Phase  
- Sustained open grip interferes with **transport and placement**
- High-sensitive tasks (cream_cheese, tomato_sauce) require precise placement → open grip disrupts placement → max_steps reached
- Robust tasks (ketchup, salad_dressing) tolerate rough placement → task completes despite attack → episode ends early at step 140-200

### Task Dynamics
- cream_cheese: soft deformable object, needs careful transport → 0/5 survive
- tomato_sauce: round, rolls, needs precision basket placement → 0/5 survive
- ketchup: flat bottom, stable, basket has large tolerance → 5/5 survive
- salad_dressing: similar stability → 5/5 survive

## VisualNoStep V6 — Negative Result

VisualNoStep @ threshold=0.05 triggers online but is **non-selective**:

| Metric | ProprioNoStep | VisualNoStep V6 |
|--------|--------------|-----------------|
| ketchup sus30 | 5/5 success | 0/3 fail |
| First trigger (ketchup) | step 120-161 | step 14-63 |
| Trigger mechanism | Contact-phase proprio | Pre-contact visual appearance |

**Root cause — why proprio wins and visual loses**:

- **Proprioceptive signal** (13-dim gripper/EEF/action features) directly encodes physical contact dynamics. ProprioNoStep fires at contact/transport/placement phase (step 120-160). It isn't a more complex model — it just has an input domain that's naturally selective for when the gripper is actually interacting with the object.
- **Visual signal** (2176-dim DINOv2+SigLIP) encodes scene and object appearance. VisualNoStep V6 fires on "this looks like a difficult task" at pre-contact phase (step 14-63). The attack starts before grasp formation, turning selective contact-phase disruption into non-selective grasp blocking.
- The current attack needs **contact-phase timing**, not just object difficulty recognition. This is why ProprioNoStep is selective and VisualNoStep V6 is not.
- Visual information is not useless — it correlates with task difficulty. But to be useful for selective attack, a visual detector must learn **when contact is established**, not just whether the scene looks hard.

## Valid Claims

- "ProprioNoStep selects detector candidate windows at contact/grasp phase."
- "sustained_command_open_proxy_30 is a command-layer sustained proxy that selectively causes failures on high oracle-sensitive Object tasks while preserving robust controls."
- "Selectivity is 100 percentage points: High 0% vs Robust 100%."
- "VisualNoStep can trigger online but lacks selectivity due to early visual triggering at pre-grasp phase."

## Forbidden Claims

- "VIS attack successful" / "VIS attack failed"
- "Universal attack" — this is selective, not universal
- "Detector is oracle-optimal" — ProprioNoStep is best practical, not proven optimal
- "All Object tasks vulnerable" — only high-sensitive tasks fail
- "Visual information useless" — visual fires on appearance, just not calibrated for contact-phase selectivity
- "VisualNoStep production-ready"

## Next Steps

1. Visual v2/re-ranker training (requires approval)
2. Cross-suite generalization (Spatial, Goal)
3. Defense/mitigation study (contact-phase gripper hardening)
