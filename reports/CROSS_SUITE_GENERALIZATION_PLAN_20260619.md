# SC5 Cross-Suite Generalization Plan

## Research Question

Can the LIBERO-Object-trained SC5 causal detector transfer zero-shot to LIBERO-Spatial, LIBERO-Goal, and LIBERO-10 without retraining, target-suite normalization, online privileged state, manual anchors, or attack-outcome selection?

The claim is decomposed into:

1. feature/runtime compatibility;
2. detector timing transfer;
3. attack effect transfer.

These must be reported separately.

## Frozen Protocol

Protocol config: `configs/sc5_cross_suite_protocol_v1.yaml`.

Frozen detector:

- checkpoint SHA256: `66ec2d487ef4b4c673cb2c7c147c7f64c6e27c3e1eb6ced4470bf18466c11628`
- dataset SHA256: `f942f4b0856d3449fa4e98f6d6e74ac8d5e8e9af7082373f961f79b0a6930cd9`
- feature count: 25
- normalization: Object train-only checkpoint mean/std
- thresholds: `tau_corridor=0.3`, `tau_release=0.3`, `guard=5`

Frozen attack parameters:

- K: 10
- epsilon: 6/255
- PGD steps: 20
- objective: `autoregressive_prefix_gripper_target_token_logratio_arm_v3`
- target token: 31744
- target class: `CLIP_MEDIATED_OPEN`
- arm gate: 5/6

## Static Task Layering

Inventory: `tables/cross_suite_task_inventory.csv`.

Mechanism classes:

- `single_object_pick_place`: may enter zero-shot trigger validation.
- `multi_object_transfer`: event-level audit only in the first pass.
- `articulated_object`, `planar_or_push`, `rearrangement_non_grasp`, `unknown_or_low_signal`: abstain/negative compatibility checks, not gripper-duty attack denominators.

## Minimal Clean Smoke

Manifest: `tables/cross_suite_smoke_manifest.csv`.

Initial clean smoke is 18 rollouts:

- Spatial: tasks 0, 1, 2; states 0, 1.
- Goal: tasks 1, 6, 0; states 0, 1.
- LIBERO-10: tasks 5, 0, 2; states 0, 1.

Clean smoke measures only:

- clean success;
- 25D feature validity;
- detector trigger or abstain;
- emit timing;
- runtime crash;
- object/site parsing;
- artifact completeness.

## Minimal Attack/Control Smoke

Only after clean smoke gate:

- choose at most two eligible clean-success parents per suite;
- run detector-triggered VIS and same-trigger RAND_T10;
- no replacement to fill failed or ineligible slots.

Maximum additional runs: 12.

Total initial budget: 18 clean + up to 12 attack/control = 30 rollouts.

## Required Reporting

All reports must include denominators for:

- all scanned tasks;
- mechanism-eligible tasks;
- clean-success parents;
- valid-feature parents;
- triggered parents;
- attacked parents;
- manually audited parents.

Manual video audit is required before any gripper-induced physical failure claim.

## Claim Boundary

Allowed after Phase 1:

- protocol readiness;
- static task inventory;
- minimal smoke plan;
- no-target-leakage checks.

Forbidden until subsequent approved GPU phases:

- cross-suite generalization proven;
- paired ASR;
- timing superiority;
- universal VLA attack;
- gripper-induced physical task failure.
