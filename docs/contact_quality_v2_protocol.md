# Contact-Quality v2 Protocol

Official LIBERO success is insufficient for contact-rich manipulation because an object can be dropped or knocked into the target area while still receiving official success. CQ v2 detects these contact-quality failures from trajectory/log signals.

## Definitions

- `premature_release`: gripper opens during an active transport/release-sensitive interval.
- `drop_after_lift`: object returns near the pre-lift height after being lifted.
- `unstable_transport`: lifted object shows drop or failure-associated instability.
- `uncontrolled_final_drop`: object reaches or remains near the goal through an uncontrolled drop rather than a stable placement.
- `stable_controlled_place`: official success with lifted object and no uncontrolled final drop.
- `contact_quality_failure`: uncontrolled drop/release/transport failure.
- `contact_quality_success`: stable controlled placement without CQ failure.
- `sr_cq_mismatch`: official success coexists with CQ failure.

## Semantic Rule

Official success never overrides contact-quality failure. Near-target final position also does not override uncontrolled drop, slip, or unstable transport. If premature release plus drop/unstable transport occurs before stable controlled placement, CQ v2 marks the run as a contact-quality failure.

Manual labels are not production inputs. They are used only by `scripts/evaluate_cq_v2_manual_calibration.py` for evaluation and reporting.

## Calibration Status

CQ v2 is calibrated on the 20-video Black Bowl manual review set:

- manual positives detected: 9/9
- manual negatives preserved: 11/11
- precision/recall/F1: 1.0 / 1.0 / 1.0

Boundary: CQ v2 is calibrated on Black Bowl manual labels only. It is not yet LIBERO-wide validated.
