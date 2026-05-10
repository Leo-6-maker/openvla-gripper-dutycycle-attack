# Claim and Evidence

## Main Claim

The frozen claim is duty-cycle shaping via gripper-targeted temporal PGD: during contact-critical black-bowl manipulation windows, inference-time visual perturbations can bias the model toward repeated gripper-open commands, producing visible grasp/lift/carry failures.

## Template B Interpretation

The result is not framed as `prev_delta` uniqueness or margin-objective uniqueness. The supported interpretation is that gripper-targeted temporal perturbations can shape the command duty cycle in a vulnerable task phase.

## Frozen State7 Evidence

| Condition | Manual outcome | Interpretation |
|---|---:|---|
| VIS margin prev-delta | 4/5 manual gripper-opening failures | Primary positive evidence |
| Same-gate zero-margin | 3/5 comparable failures | Effect is not unique to `prev_delta`/margin |
| Random direction | 0/5 failures | Not arbitrary perturbation |
| Oracle continuous | 3/3 physical upper-bound failures | Task phase is physically sensitive to gripper opening |

## Claim Boundaries

- Current main evidence is LIBERO black-bowl state7 plus within-task state5 generalization.
- We do not claim broad cross-dataset generalization.
- We do not claim `prev_delta` is necessary.
- Simulator SR alone is insufficient; manual video review overrides SR for contact-rich slip/drop cases.
- Moka pots is Future Work / Dataset Diagnostics, not main attack evidence.
