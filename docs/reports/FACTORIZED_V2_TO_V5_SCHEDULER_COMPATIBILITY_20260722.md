# Factorized V2 -> V5 Scheduler Compatibility Audit

Status: `READ_ONLY / CONTRACT_AUDIT / NO_TRAINING`

Audit date: `2026-07-22`

Local source checkout: `ea10e1d81db845bceac54424bb2669e436f6365c`

This report audits the representation boundary between the Factorized V2
Student prediction stream and the V5 one-shot scheduler. It does not rerun
training, generate predictions, load an OpenVLA model, start a simulator, or
execute an attack. No existing evidence root is modified.

## Executive decision

The exact Factorized V2/W32 output is **not directly compatible** with the V5
scheduler contract. It is a plausible frozen input representation for a new,
causal decision head, but that bridge is not present or authorized yet.

```text
DIRECT_FACTORV2_TO_V5_STREAM       = HOLD
COMPATIBILITY_CLASS                = COMPATIBLE_WITH_CAUSAL_DECISION_HEAD
CAUSAL_DECISION_HEAD               = NOT IMPLEMENTED
L3_SCHEDULER_REPLAY                = BLOCKED
ENGINEERING_FULL_FIT               = HOLD
PASSIVE_SHADOW                     = HOLD
ACTIVE_ATTACK                      = HOLD
```

The classification is deliberately conditional. It does not mean that
`manipulation_prob` is already a V5 utility score, that `grasp_prob` is a
candidate-close gate, or that `release_prob` is a complete V5 release/regrasp
veto. Those mappings would be semantic substitutions without a frozen target
equivalence proof.

## Source contracts inspected

The conclusion is based on the checked-in implementation and protocols:

| Area | Source | Relevant contract |
| --- | --- | --- |
| Factorized model | `src/gripper_attack/v5_factorized_student_v2.py` | three route-specific heads: `grasp`, `manipulation`, `release`; 25D input; optional 9D fusion |
| Factorized loader | `src/gripper_attack/v5_factorized_dataset.py` | `valid_mask`, three Teacher targets/masks, event metadata, optional policy-intent tensor |
| Factorized predictor | `scripts/detector_v5/predict_factorized_v2_inner_cv.py` | sealed step stream contains `grasp_prob`, `manipulation_prob`, `release_prob` and Teacher evaluation fields |
| Factorized Teacher | `src/gripper_attack/v5_factorized_teacher.py` and `configs/DETECTOR_V5_FACTORIZED_TEACHER_PROTOCOL_V1.json` | `grasp_established`, `manipulation_active`, `release_or_instability`, `candidate_close`, prefix-causal Teacher rules |
| V5 scheduler | `src/gripper_attack/v5_scheduler.py` | `candidate_close`, `valid`, utility, release, regrasp, uncertainty; 3-of-5, dwell 10, one-shot |
| V5 shadow contract | `src/gripper_attack/v5_shadow.py` and `schemas/v5_student_scheduler_input_contract.schema.json` | exact Student-only stream fields; Teacher/future/action fields rejected |
| V5 development protocol | `configs/DETECTOR_V5_DEVELOPMENT_PROTOCOL_V2.json` | active heads are utility/release/regrasp; uncertainty disabled |

The local focused compatibility/scheduler tests pass (`22 passed`). This is an
engineering check of the contracts, not evidence that Factorized V2 has passed
the missing semantic bridge.

This is a source-contract audit. The exact remote W32 prediction root was not
rehashed or rewritten by this worktree; no raw prediction artifact was opened
or regenerated here. Numerical Factorized metrics therefore remain evidence
from their sealed Factorized evaluation line, not a result of this report.

## Field-by-field compatibility

| V5 scheduler input | Factorized V2 evidence | Classification | Required action |
| --- | --- | --- | --- |
| `candidate_close` | Present in Factorized Teacher rows, but not emitted as a Student prediction field. The runtime-authoritative value must come from the causal action contract, not from a Teacher label. | `EXTERNAL_INPUT_REQUIRED` | Supply an independently sealed runtime gate; do not derive it from `grasp_prob`. |
| `valid` / `student_valid` | Stored in the Factorized episode loader as `valid_mask`, but not included in the current Factorized prediction stream. | `EXTERNAL_INPUT_REQUIRED` | Carry the source Student-valid bit into a new scheduler stream and bind its schema. |
| `utility_probability` | No Factorized head has this target. `manipulation_active` means grasped object is being transported/lifted/placed; it is not defined as attack utility or criticality ranking. | `NOT_EQUIVALENT` | Train a causal decision head against the frozen V5 utility/causal-decision target, or keep L3 blocked. |
| `release_probability` | `release_prob` is present, but its Teacher target is `release_or_instability`: released, dropped, slipping, or regrasping. That is broader than a release-only veto. | `PARTIAL_ONLY` | It may be a frozen bridge feature; it is not a formally equivalent V5 release veto without an explicit semantic decision. |
| `regrasp_probability` | No separate regrasp output exists. Regrasp is folded into `release_or_instability`. | `MISSING` | Add a separately trained causal regrasp/instability head, with its own target and mask. |
| `uncertainty_probability` | No uncertainty head or calibration is emitted. V5 explicitly disables the uncertainty veto. | `DISABLED_BY_CONTRACT` | Pass an explicit zero only in a contract that keeps the veto disabled; never use an unsupervised random head. |
| `route_supported` / route | Factorized V2 has route-specific heads and deterministic unsupported-route abstention. | `COMPATIBLE_AS_METADATA` | Preserve the route binding, but do not treat route support as a utility score. |
| `features_25d` and feature order | Factorized V2 consumes the 25D source features. The current prediction JSONL does not itself emit the complete V5 feature contract/order hash. | `SOURCE_BINDING_REQUIRED` | Build a new Student-only stream with the frozen 25D schema and order SHA. |

The only safe direct numerical reuse is therefore as input to a new causal
decision layer, not as a positional rename of the three Factorized heads.

## Why the W32 output is not a V5 window stream

The Factorized V2 protocol permits a fixed causal context (`receptive_field_w`
including 32). In the Windowed GRU implementation, hidden state is reset every
32 steps. The predictor records `window_id = step // W` and
`position_in_window`, but those values describe the model's bounded encoder
context.

They do **not** describe a V5 candidate-close segment. In particular, they do
not prove:

- `candidate_close` is true for every member;
- the steps form one complete candidate segment;
- the segment is known and rankable;
- the segment has a V5 utility tier;
- the V5 decision anchor and scheduler dwell are satisfied.

Therefore the W32 identifier must not be passed to the V5 scheduler as a
candidate `window_id`, and W32 context boundaries must not be used as attack
window boundaries.

## Teacher target comparison

The Factorized Teacher is a causal, factorized state description:

```text
grasp_established      = object is stably held
manipulation_active    = a grasped object is transported, lifted, or placed
release_or_instability = released, dropped, slipping, or regrasping
```

The V5 development protocol is a different task:

```text
utility                = window-level criticality / attack-value proxy
release                = release veto
regrasp                = regrasp or instability veto
uncertainty            = calibrated uncertainty, currently disabled
```

The following tempting substitutions are rejected:

```text
manipulation_prob -> utility_probability       REJECTED
grasp_prob        -> candidate_close           REJECTED
release_prob      -> regrasp_probability       REJECTED
release_prob      -> release_probability       NOT PROVEN EQUIVALENT
```

`manipulation_prob` can be useful as a causal input feature for a new decision
head. It cannot silently become the scheduler's utility output. Likewise,
`grasp_prob` estimates a Teacher state and cannot replace the action-derived
candidate gate.

## Causal and leakage audit

The Factorized Teacher protocol marks its primary heads prefix-invariant and
future-forbidden. That supports use of its labels as clean-only supervision,
subject to the sealed Teacher contract. It does not make the labels valid
Student runtime inputs.

The current Factorized prediction stream also serializes targets, known masks,
event roles, durations, and other evaluation metadata. Those fields are useful
for offline metrics but are not an admissible V5 Student stream. The V5 shadow
contract rejects Teacher/future/action fields and requires an exact Student-only
field set. A scheduler adapter must therefore construct a new stream that:

1. reads only causal Student features and model outputs at the current step;
2. receives `candidate_close` and `student_valid` from runtime-bound external
   inputs;
3. excludes Factorized targets, known masks, event labels, durations, and
   Teacher metadata from the scheduler call;
4. sets uncertainty explicitly disabled;
5. preserves route support and exact feature-order provenance.

No existing Factorized prediction artifact has been promoted to that stream by
this audit.

## 9D / policy-intent status

The Factorized dataset can load a policy-intent tensor, but the current V2
inner-CV training and prediction entrypoints instantiate
`FactorizedStudentV2(..., use_9d=False)`. Thus the exact W32 Factorized output
does not establish a bound 9D policy-intent input for a V5 scheduler bridge.

If a future bridge uses 9D, it must bind the policy-intent root, feature-order
SHA, normalization, and causal availability separately. It must not infer a
policy-intent stream from Factorized event metadata.

## Minimal compatible bridge

The smallest defensible path is a new development-only causal decision head,
not a larger model and not a silent field rename.

### Frozen bridge inputs

At each step, the head may consume only:

- the frozen Factorized causal representation/logits or probabilities;
- the frozen 25D causal Student features and their order SHA;
- externally bound `candidate_close`;
- externally bound `student_valid`;
- route support metadata;
- causal history available before and at the current step.

### New outputs

The bridge must emit the exact V5 names and semantics:

- `utility_probability`;
- `release_probability`;
- `regrasp_probability`;
- `uncertainty_probability` only if a separately valid uncertainty method is
  added; otherwise the contract keeps it disabled and explicit.

The utility and regrasp heads require their own sealed causal target protocol.
The Factorized `manipulation` and `release_or_instability` labels may be used
as input features, but target equivalence must not be assumed.

### Required bridge evidence

Before L3 scheduler replay can start, the bridge needs:

1. a new protocol and model contract with exact input/output schemas;
2. causal target and mask definitions for utility, release, and regrasp;
3. train-only normalization and identity split binding;
4. strict checkpoint loading and checkpoint/input SHA binding;
5. Student-only prediction records with no Teacher fields;
6. candidate-close and student-valid source binding;
7. a leakage audit proving no window end, future label, target, or event
   metadata reaches the scheduler;
8. a new CPU shadow replay and negative tests for silent mappings.

This is a small causal head contract, but it is a new training/evaluation
stage. It is not a post-processing rename and must not be treated as an
already-compatible deployment.

## Legacy Stage-1 evaluation note

The older sealed Stage-1 evaluation artifact remains immutable. If its schema
needs to be compared with the corrected aggregator, the valid operation is a
derived reevaluation output root using the corrected evaluator and an explicit
new schema. It is not valid to edit the old evaluation JSONL or retroactively
rewrite its seal.

This migration is independent of the Factorized-to-V5 semantic bridge and does
not authorize scheduler replay, Full-FIT, passive shadow, or attack execution.

## Decision matrix

| Item | Status | Reason |
| --- | --- | --- |
| Factorized model integrity | `PASS_AS_CHECKED_IN` | three-head route model and causal W32/TCN code are internally defined |
| Factorized prediction seal compatibility | `PASS_AS_FACTOR_STREAM` | its own schema can be audited as a Factorized stream |
| Direct map to V5 scheduler | `HOLD` | utility, regrasp, candidate gate, valid bit, and feature stream are not all semantically bound |
| Causal decision-head route | `AUTHORIZED_AS_NEXT_DESIGN` | frozen Factorized outputs can be inputs to a new causal bridge after target/schema closure |
| L3 scheduler replay | `BLOCKED` | no exact V5 scheduler stream exists |
| Full-FIT / model selection | `HOLD` | no formal/engineering promotion from this audit |
| Passive shadow | `HOLD` | no scheduler-ready Student stream and no new deployment authorization |
| Active attack | `NOT AUTHORIZED` | no attack output or action path was created |

## Required next action

Do not patch the existing Factorized prediction JSONL or rename fields in
place. Keep the Factorized V2 evidence immutable. Create a separately sealed
causal decision-head protocol and Student-only bridge stream, then rerun the
shadow contract tests against that new schema.

Until that work is complete:

```text
leakage-free Platt/L1/L2 on Factorized heads = may continue within its own contract
Factorized output -> V5 L3 scheduler          = BLOCKED
Full-FIT / engineering deployment             = HOLD
Passive/active runtime                         = HOLD
Attack / CAL / CHECK                           = NOT STARTED
```

No GitHub push, PR state change, training run, prediction rerun, simulator run,
or attack was performed by this audit.
