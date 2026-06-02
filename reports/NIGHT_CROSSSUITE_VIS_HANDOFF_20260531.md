# Night Cross-Suite + VIS Handoff

**Date**: 2026-05-31 | **Branch**: `exp/crosssuite-proprio-vis-night-20260531`

## Track A — Cross-Suite ProprioNoStep Validation

### A1: Mechanism Inventory

| Suite | Tasks | Pick-Place Eligible | Transfer Status |
|-------|-------|--------------------|-----------------|
| Spatial | 10 | 10 (100%) | **Partial** (some triggers, low scores) |
| Object | 10 | 10 (100%) | **Production** (validated) |
| Goal | 10 | 6 (60%) | **No transfer** (detector silent) |
| L10 | 10 | 2 (20%) | Deferred |

### A2: Clean Shadow Results [IN PROGRESS — 8/32 collected]

**Goal**: ProprioNoStep is **silent** — 0/5 triggered so far.
- max_hazard_score = 0.000-0.011
- All episodes success but NO detector activity
- Object-trained detector does not recognize Goal contact dynamics

**Spatial**: ProprioNoStep **partially transfers** — 1/3 triggered so far.
- max_hazard_score = 0.071-0.449 (vs 0.9+ on Object)
- Only 1 episode triggered (at step 88)
- Hazard scores are significantly lower than Object baseline

### A4: Cross-suite sus30 — BLOCKED

sus30 cross-suite pilot cannot proceed because:
- Goal: detector silent → no trigger windows → sus30 would be clean
- Spatial: partial transfers with low scores → unreliable for attack
- ProprioNoStep is **Object-production only** — not cross-suite universal

## Track B — VIS

### B1: Code Audit

**VIS IS IMPLEMENTED** in `src/gripper_attack/attack_adapter.py`:
- `TokenPrefixPGDAttacker`: White-box PGD on OpenVLA action token loss
- `OpenVLAVisualAttacker`: Factory class
- Integration exists in `v4_run_eval_openvla.py`

**VIS is NOT integrated** into production runner `run_official_eval_artifact_rich.py`.

### B2: Gradient Smoke

**Partial success**:
1. Attacker instantiation confirmed on GPU4,5 (`ExistingDenseAttackAdapter`)
2. Model loads across GPU4,5 (~9GB each — backprop feasible)
3. `attack()` API requires `target_action` parameter — needs integration study

**Deferred** to dedicated VIS branch with runner integration.

## Production Line

- **Detector**: ProprioNoStep — Object-production only
- **Attack**: sustained_command_open_proxy_30
- **Selectivity**: High 0/10, Robust 10/10 (Object)
- **Cross-suite**: ProprioNoStep does NOT transfer to Goal/Spatial

## GPU Status (end of session)

All GPUs idle after rollouts complete.
GPU0: quarantined (lgzhou RoboTwin).

## Claims

### Valid
- ProprioNoStep is production for LIBERO-Object.
- ProprioNoStep does NOT transfer to LIBERO-Goal (detector silent).
- ProprioNoStep partially transfers to LIBERO-Spatial (low scores, infrequent triggers).
- Cross-suite sus30 is blocked without suite-specific detector training.
- VIS code is implemented (TokenPrefixPGDAttacker) but not validated.

### Forbidden
- ProprioNoStep universal across LIBERO
- VIS successful before gradient + rollout controls
- Command-layer sus30 equals VIS
- Universal attack
- Detector oracle-optimal

## Next Steps

1. Complete remaining cross-suite clean shadow rollouts
2. Suite-specific ProprioNoStep training for Spatial/Goal if cross-suite attack is desired
3. VIS runner integration in dedicated branch
4. Group meeting: Object selectivity is the headline result
