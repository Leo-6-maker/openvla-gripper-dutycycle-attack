# Final Detector Status — After VisualNoStep V6 Pilot

**Date**: 2026-05-30 | **Production Branch**: `exp/sustained-proxy-burst-control-20260530` | **Commit**: `703c172`

## Detector Status Matrix

| Detector | Online Validated | Selective | Production Ready | Notes |
|----------|-----------------|-----------|-----------------|-------|
| **ProprioNoStep** | YES (50 episodes) | YES (high 0/10, robust 10/10) | **YES** | Production detector |
| VisualNoStep | YES (24 episodes) | NO (breaks ketchup 0/3) | NO | Early visual triggering |
| VisualProprioNoStep | NO | N/A | NO | Not evaluated online |
| Flexible Detector | YES (infrastructure) | N/A | N/A | Code only |

## ProprioNoStep — Production Detector

- **Model**: `ProprioNoStep_baseline.pt`
- **Input**: 13-dim proprio/action (gripper_command, gripper_qpos, gripper_width, eef_xyz, eef_vxyz, action_dxyz, action_gripper)
- **Architecture**: CausalTCNDetector(in_dim=13, h_dim=64, n_ph=8, n_l=3, dropout=0.1)
- **Selectivity mechanism**: Proprioceptive signal (gripper position, EEF velocity, action commands) directly encodes physical contact dynamics. ProprioNoStep fires at contact/transport/placement phase (step 120-160) — when the gripper is actually interacting with the object. It doesn't win because it's more complex; it wins because its input domain is naturally selective for contact-phase timing.

## VisualNoStep V6 — Non-Production

- **Model**: `VisualNoStep_frozen.pt` (2176-dim DINOv2+SigLIP input)
- **Key failure**: Triggers at step 14-63 (~100 steps before ProprioNoStep)
- **Root cause**: Visual features encode scene/object appearance ("this looks difficult") rather than contact dynamics ("the gripper is now interacting with the object"). The current attack needs contact-phase timing for selective disruption; VisualNoStep V6 fires at pre-contact and turns it into non-selective grasp blocking.
- **What this teaches us**: Visual information correlates with task difficulty — it's not useless. But to be useful for selective attack, a visual detector must learn when contact is established, not just whether the scene looks hard. Future visual work should be framed as a contact-phase re-ranker on top of proprio timing, not a standalone trigger.
- **Selectivity comparison**:

| Metric | ProprioNoStep | VisualNoStep V6 |
|--------|--------------|-----------------|
| ketchup sus30 | 5/5 success | 0/3 fail |
| First trigger | step 120-161 | step 14-63 |
| Trigger domain | Contact-phase proprio | Pre-contact appearance |
| Clean trigger rate | Low | 35-96 per episode |

## Claim Boundaries

### Permitted
- "ProprioNoStep is the production online detector."
- "sustained_command_open_proxy_30 selectively causes failures on high oracle-sensitive Object tasks while preserving robust controls."
- "Selectivity is 100 percentage points: High 0% vs Robust 100%."
- "VisualNoStep triggers online but lacks contact-phase selectivity — fires ~100 steps earlier than ProprioNoStep."
- "Visual/Fusion retraining is future work."

### Forbidden
- "VIS attack successful/failed"
- "Visual information is useless"
- "VisualNoStep is production-ready"
- "All Object tasks are vulnerable"
- "Universal attack"
- "Detector is oracle-optimal"
