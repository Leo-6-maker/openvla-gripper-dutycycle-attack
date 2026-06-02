# Next Phase — Visual Re-ranker Plan

**Date**: 2026-05-30 | **Branch**: `exp/visual-v2-reranker-training-20260530`

## Slide Storyline

### Slide 1: Production Result
- ProprioNoStep + sustained_command_open_proxy_30
- Full10 sus30: High 0/10, Robust 10/10
- 100 percentage point selectivity

### Slide 2: Why ProprioNoStep Wins
- Proprio signal encodes **contact dynamics** (gripper force, velocity, position)
- Fires at contact/transport/placement phase (step 120-160)
- Selectivity comes from contact-phase timing, not model complexity

### Slide 3: VisualNoStep V6 — What We Learned
- Visual signal encodes **scene/object appearance**
- Fires at pre-contact phase (step 14-63)
- Turns selective contact-phase disruption into non-selective grasp blocking
- Visual information is NOT useless — it correlates with task difficulty

### Slide 4: The Real Question
- NOT: "Can visual replace proprio?"
- BUT: "Can visual serve as a contact-phase vulnerability re-ranker?"
- Proprio provides timing; Visual judges vulnerability at contact moments

### Slide 5: Visual v2 Training (In Progress)
- Object-100 teacher labels (97 positive, 18,778 negative)
- CausalTCNDetector architecture (same as online runner)
- Task-holdout validation
- Models: VisualNoStep_v2, VisualProprioNoStep_v2, task-only baseline, label-shuffle baseline

### Slide 6: Next Steps
- Offline attack-relevance evaluation
- If visual v2 learns contact-phase timing: gated online pilot
- If visual v2 still fires pre-contact: visual as scene-level difficulty prior, not online trigger
- Production line unchanged: ProprioNoStep + sustained_command_open_proxy_30

## Key Messages

1. Selective sustained proxy is production-ready.
2. Selectivity mechanism is understood: proprio contact dynamics > visual scene appearance.
3. Visual v2 explores whether visual can be a re-ranker, not a replacement.
4. No change to production claims without gated validation.
