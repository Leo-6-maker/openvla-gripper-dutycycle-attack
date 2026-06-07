# ProprioNoStep Hard Gate Rejection

**Verdict**: PHASE_GATE_BLOCKS_VULNERABILITY_POSITIVES

## POC Details

| Metric | Value |
|--------|-------|
| Covered tasks | cream_cheese, salad_dressing, ketchup |
| Covered windows | 143 (from phase detector) |
| Joined vulnerability rows | 24 (7 pos, 17 neg) |
| Positive recall at tau=0 | 0/7 |
| Positive recall at tau=1e-12 | 0/7 |
| Negative rejection at tau=0 | 12/17 |

## Root Cause

ProprioNoStep is a learned proxy-phase / physical-hazard detector trained on CLEAN trajectories. It detects when the gripper enters critical physical phases (grasp, lift). Vulnerability detector labels are mostly 'no_action_bridge' — the VIS attack does not create a physical gripper bridge at these windows. Therefore the phase detector correctly identifies vulnerability windows as physically non-hazardous.

## Conclusion

**ProprioNoStep cannot be used as a hard cascade gate for vulnerability detection.** It blocks all vulnerability positives because the physical phase signal is orthogonal to VIS attack success.

## Allowed Roles

- Mechanism audit
- Physical bridge plausibility check
- Clean-control stratification
- False-alarm analysis by physical phase

## Disallowed Role

- Hard cascade gate for vulnerability detector
- Pre-filter before vulnerability screening
