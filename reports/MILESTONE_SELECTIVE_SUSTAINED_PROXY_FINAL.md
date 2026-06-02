# Milestone — Selective Sustained Command-Layer Proxy

**Date**: 2026-05-30 | **Status**: COMPLETE | **Branch**: `exp/sustained-proxy-burst-control-20260530` | **Commit**: `703c172`

## 1. Detector Choice

**ProprioNoStep** is the production online detector (CausalTCNDetector, 13-dim proprio/action input, 38,602 params). Validated through Full10 sus30 (50 episodes): high-sensitive 0/10, robust 10/10.

## 2. Attack Mechanism

**sustained_command_open_proxy_30**: When ProprioNoStep triggers consecutively (duration=5), the gripper action is overridden to fully open (1.0) for 30 steps. This is a command-layer sustained proxy — not VIS or PGD.

## 3. Full10 sus30 Results

| Task | Class | sus30 Success | Attack Onset | Attack Steps | Final State |
|------|-------|--------------|-------------|-------------|-------------|
| cream_cheese | High | **0/5** | step 123-161 | 129-167 | max_steps (fail) |
| tomato_sauce | High | **0/5** | step 118-196 | 94-164 | max_steps (fail) |
| alphabet_soup | Medium | 4/5 | — | — | — |
| bbq_sauce | Medium | 1/5 | — | — | — |
| butter | Medium | 1/5 | — | — | — |
| chocolate_pudding | Medium | 0/5 | — | — | — |
| milk | Low | 4/5 | — | — | — |
| orange_juice | Low | 3/5 | — | — | — |
| ketchup | Robust | **5/5** | step 120-161 | 23-71 | early success (143-202) |
| salad_dressing | Robust | **5/5** | step 88-108 | 27-160 | early success (122-260) |

**Selectivity**: High 0% vs Robust 100% — 100 percentage point gap.

## 4. Mechanism of Selectivity

### High-sensitive tasks fail because:
1. ProprioNoStep fires at contact/grasp phase (step 120-160)
2. Sustained open grip forcibly opens gripper during transport
3. Object slips or placement precision is lost
4. cream_cheese (deformable) and tomato_sauce (round) are particularly vulnerable to grip disruption
5. Episode hits max_steps (290) without successful placement

### Robust tasks survive because:
1. Same trigger timing and same sustained open-grip attack
2. But task dynamics allow rough placement — ketchup (flat bottom) and salad_dressing (stable shape) can be placed with less precision
3. Task completes early (step 140-202) before sustained proxy accumulates enough disruption
4. Notable: salad_dressing s0 survived 160 attack steps out of 260 total

### Why not task identity:
- Both high and robust tasks trigger the detector at similar steps
- Both receive the same sustained open-grip attack
- Selectivity comes from **task dynamics** (object geometry, placement tolerance), not detector behavior
- Detector provides uniform trigger; task dynamics determine failure vs survival

## 5. VisualNoStep V6 — Ablation (Selectivity Mechanism Revealed)

VisualNoStep @ threshold=0.05 triggers at step 14-63 (~100 steps earlier than ProprioNoStep). Early trigger → attack before grasp → ketchup robust control destroyed (0/3).

### Selectivity Comparison

| Dimension | ProprioNoStep | VisualNoStep V6 |
|-----------|--------------|-----------------|
| Trigger phase | contact / transport / placement | pre-contact (approach) |
| Attack effect | selective contact-phase disruption | non-selective grasp blocking |
| High-sensitive | 0/10 — disrupted at contact phase | 2/6 — some survive, some fail |
| Robust controls | 10/10 — preserved | 2/6 — ketchup 0/3 (broken) |
| Clean trigger rate | Low | 35-96 per episode |
| Status | **production** | **non-production** |

### Why ProprioNoStep Wins

ProprioNoStep does not win because its model is more complex. It wins because **its input domain is naturally selective for contact dynamics**:

- **Proprioceptive signal** (13-dim: gripper position, EEF velocity, action commands) directly encodes physical interaction — force, motion, contact state.
- When ProprioNoStep fires (step 120-160), the gripper has already made contact with the object. The attack disrupts transport and placement — a **selective contact-phase disruption**.
- Whether a task fails or survives then depends on **task dynamics** (object geometry, placement precision), not detector identity.

**Visual signal** (2176-dim DINOv2+SigLIP) encodes scene and object appearance:
- It learns that "this object/scene looks difficult" — which correlates with vulnerability but is not causally timed to contact.
- Firing at step 14-63 means the attack starts **before grasp formation**. The robot cannot pick up the object at all.
- This turns a selective contact-phase disruption into a **non-selective grasp-prevention attack**.

### What Visual V6 Teaches Us

- Visual information is NOT useless — it correlates with task difficulty and vulnerability.
- But the current visual detector has not learned **when contact is established and the object is truly vulnerable**.
- Visual v2, if pursued, should be framed as a **contact-phase re-ranker**: Proprio provides the contact-timing signal, Visual judges whether that contact phase is truly exploitable.
- Visual should not try to replace proprio as the primary trigger.

## 6. Production Artifacts

| Component | Path |
|-----------|------|
| Detector | `milestone_2e3_object100_visual_proprio_no_step_20260527/models/ProprioNoStep_baseline.pt` |
| Runner | `scripts/run_official_eval_artifact_rich.py` (branch `exp/sustained-proxy-burst-control-20260530`) |
| Full10 Oracle | `milestone_2f_object_oracle_sensitivity_full10x5_20260529/` |
| Full10 sus30 | `milestone_2h_sustained_proxy_full10x5_sus30_20260530/` |
| Detector-clean | `milestone_2f_object_detector_clean_full10x5_prep_20260529/` |

## 7. Claim Boundaries

### Permitted
- ProprioNoStep is the production online detector.
- sustained_command_open_proxy_30 selectively causes failures on high oracle-sensitive tasks while preserving robust controls.
- Selectivity is 100 percentage points (High 0% vs Robust 100%).
- VisualNoStep triggers but is non-selective at current threshold/calibration.

### Forbidden
- VIS attack successful/failed
- Universal attack
- Detector is oracle-optimal
- All Object tasks vulnerable
- Visual information useless
- VisualNoStep production-ready
