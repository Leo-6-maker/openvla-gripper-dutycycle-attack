# LOTO 10-Fold Claim Boundary V1

## Current Claims (Fold 0 Only)

- The SC5 ProprioNoStep detector, with train-only teacher calibration and Butter-only
  checkpoint selection, achieves **93.3% correct-window coverage** on held-out
  Chocolate Pudding task (σ=0 across 3 seeds).
- The detector emits with a **median +11 step delay** relative to the teacher anchor
  (IQR 1-2 steps, indicating tight timing precision).
- **No-corridor selectivity is poor**: 70-95% false positive rate on teacher-negative
  episodes (mean 85% across seeds).

## Claims NOT Yet Supported

- "LIBERO-Object task-level generalization" — requires all 10 folds.
- "Cross-suite generalization" — requires evaluation on unseen LIBERO suites.
- "Real-robot timing transfer" — requires real-robot evaluation.
- "Detector-driven attack effectiveness" — requires attack rollouts with frozen detector.
- "Selective attack triggering" — requires no-corridor FPR substantially below 50%.

## What 10-Fold LOTO Can Claim (If Successful)

- "The SC5 ProprioNoStep detector generalizes to unseen LIBERO-Object tasks."
- "The detector achieves consistent coverage across all 10 held-out tasks."
- "Timing precision (median signed/absolute error) is stable across tasks."

## What 10-Fold LOTO Cannot Claim

- Cross-suite generalization (different scenes, objects, task templates).
- Real-robot validity.
- Attack effectiveness (only timing transfer, not attack success).
- V3 model improvements (separate protocol required).

## Fold 0 Status

Fold 0 (test=8 Chocolate, val=6 Butter) is an **amendment-controlled corrected
evaluation**, not a pristine first-look. V1 was invalidated due to a corridor
label schema bug. V2 used the same architecture, seeds, thresholds, and protocol
with the only change being correct corridor label derivation.

Remaining folds 01-09 will be **prospective frozen evaluations** under the
two-phase test-open protocol.
