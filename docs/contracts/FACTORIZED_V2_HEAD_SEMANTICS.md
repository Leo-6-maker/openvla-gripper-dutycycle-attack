# Factorized V2 head semantics

Source: `src/gripper_attack/v5_factorized_teacher.py`,
`src/gripper_attack/v5_factorized_student_v2.py`, and
`scripts/detector_v5/predict_factorized_v2_inner_cv.py`.

| Head | Teacher target | Physical meaning | Runtime use | Status |
|---|---|---|---|---|
| `grasp_prob` | `grasp_established` | Physics-backed evidence that the object is stably held; known negatives are valid | Precondition, not a standalone attack trigger | CONDITIONALLY_COMPATIBLE |
| `manipulation_prob` | `manipulation_active` | Grasped object is being transported/lifted/placed | Positive phase evidence after grasp | CONDITIONALLY_COMPATIBLE |
| `release_prob` | `release_or_instability` | Release, drop, slip, or regrasp/instability | Conservative veto only; not equivalent to a pure release head | CONDITIONALLY_COMPATIBLE |

The three Student heads are direct predictions of these three targets. There is
no verified causal identity between `grasp_prob` and V5
`utility_probability`, or between `manipulation_prob` and V5
`regrasp_probability`. Those substitutions are forbidden.

`student_valid` is an input contract field, not `route_supported` alone. A
runtime row is valid only when the feature window is valid, action intent is
known, the route is supported, all required head inputs are present, and no
schema error occurred. Unknown head labels are not negatives; the scheduler
pauses or resets according to the frozen scheduler config.

The scheduler is detector-only. `attack_enabled=false` is required and no
action is produced or modified.
