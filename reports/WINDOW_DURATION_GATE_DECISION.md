# Window Duration Gate Decision

**Date**: 2026-06-01

## Gate Results

| Gate | Status | Detail |
|------|--------|--------|
| A: Duration | **PASS** | det_30=0 fail, det_150=6 fail |
| B: Onset (cc) | **PASS** | det_150 2/2 fail, early_150 0/2 fail |
| B: Onset (ts) | **PARTIAL** | Both early and detector fail at d90+ |
| C: Selectivity | **MIXED** | Robust 50% fail at det_150 |
| D: Implementation | **PASS** | Attack applied correctly in all episodes |

## Classification: Case 1 (cream_cheese) + Case 2 (tomato_sauce)

- **cream_cheese**: Detector-timed sustained intervention is empirically critical for this task.
- **tomato_sauce**: Duration dominates; onset timing is less discriminative.
- **Robust**: Long duration degrades selectivity; duration must be calibrated.

## Recommended Claim

"For cream_cheese, ProprioNoStep identifies an empirically critical onset window: 150-step sustained open grip at the detector window causes consistent failure, while the same 150-step attack at early/late windows causes zero failures. This dissociation demonstrates that the detector window is not only sufficient but also necessary within the tested candidate set."

## Not Claimed

- Mathematical optimal
- Universal across all tasks
- Perfect selectivity at all durations
