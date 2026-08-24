# Official V3 Detector V5 Physics Teacher V2 audit

Date: 2026-07-19 CST  
Code HEAD: `ce74eb4cf6d55430e14297c45903ba6277e61f92`  
Scope: FIT states 0--19 only; clean-only derivation; no GPU, training, or attack.

## Inputs

The derivation consumed only sealed/read-only FIT inputs:

```text
FIT registry root SHA256SUMS = b42cc794bcf9e837106ecb54f99d70d85e2f47f8d44b1ce08862aebf9ef892f7
FIT registry CSV SHA          = 09f71b3a9b8250c80735382ba5deab6dbcadfa21b645e4a981eefb114b236af5
task decoder root SHA256SUMS = 168afe0b1e68ed18d8b066e4a30e750e6495eb199f2ac51c9e308e9a5da86fd9
physics source audit SHA256SUMS = abd25de6dcf18d5c6ca198f49d337e8598b46317107eb5940e1fd7322709bf08
protocol SHA                  = ccfac08fbccc92e9e8a7413c384d9e72d5195159f5305e755645afb7c73f4cb8
```

Every FIT artifact checksum list was independently rechecked before deriving
labels. The builder used official BDDL object order and the sealed task/object
state decoder; it did not use V4 Teacher labels, attack outcomes, or protected
splits.

## Explicit task-role boundary

38 of 40 tasks have a BDDL goal predicate whose first argument is a decoded
object-state object. Two `libero_goal` tasks are explicitly non-grasp goals
(cabinet opening and stove activation). They are recorded as:

```text
task_role_status = NO_MANIPULATION_TARGET
rankable Physics window = false
```

They are not silently assigned an object slice or synthetic physics label. An
actually ambiguous or malformed object role would be `ABSTAIN_DECODER_HOLD`;
none occurred in this run.

## Frozen proxy contract

The protocol freezes object--EEF relative-pose stability, object--EEF co-motion,
gripper/object contact, lift from episode-initial object height, support-contact
removal, target progress when a target object pose is decoded, and
future-only-for-Teacher release and regrasp/instability risks.

The utility tier is an ordinal clean-only criticality proxy. Tier 3 requires at
least 10 stable-grasp dwell steps plus fixed utility, lift, release-risk and
regrasp-risk bounds. Window length alone cannot produce Tier 3. T10 is only a
minimum stability condition. This is not a counterfactual attack label or a
measured vulnerability estimate.

## Materialization result

New non-overwrite Teacher root:

`OFFICIAL_V3_DETECTOR_V5_TEACHER_PHYSICS_V2_ce74eb4_20260719`

```text
status                         = PASS_WITH_EXPLICIT_NON_GRASP_TASKS
identities                     = 800
steps                          = 176336
known steps                    = 170107
candidate windows              = 5143
NO_MANIPULATION_TARGET tasks   = 2
ABSTAIN_DECODER_HOLD tasks     = 0
formal_training_authorized    = false
formal_attack_authorized      = false
```

Step-level tier counts:

```text
tier0 = 66686
tier1 = 55693
tier2 = 47059
tier3 = 669
```

The root seal was independently checked with `sha256sum -c`:

```text
Teacher root SHA256SUMS = 2d063fb42c379ad1a266a1a0e95751c711b4d23f307d9e70e63417712d3a9b83
Teacher root sidecar    = 4c8e7f5ac9c642c99a0572cfe6604e64cebf04d1d3ff3402583e8e8ef4bc4f73
```

## Independent audit

Separate auditor root:

`OFFICIAL_V3_DETECTOR_V5_TEACHER_PHYSICS_V2_AUDIT_ce74eb4_20260719`

```text
status                      = PASS
identities                  = 800
steps                       = 176336
known steps                 = 170107
windows                     = 5143
task-role rows              = 40
formal_training_authorized = false
formal_attack_authorized   = false
```

The auditor rechecked the Teacher root seal, all 800 label files,
identity/step closure, exact field whitelist, contiguous candidate-window
membership, protocol/input SHA bindings, role rows, finite component values,
manifest counts, and authorization boundaries.

Independent audit root seals:

```text
Audit root SHA256SUMS = 26beb3156ce74e59097bc23115ae6dca4df0b6f157848c69e607b0ba6ff18b22
Audit root sidecar    = f6836f3287756dc96286543336be5d42bd93d4c3be12182b63471d31460918ff
```

## Tests and execution boundary

The official A800 environment ran the targeted V5 CPU suite:

```text
22 passed in 3.07s
```

No GPU process was started. Existing processes were not stopped or shared. No
CLEAN/S1/old root was modified, and FIT-DEV/CAL/CHECK/states 30--49 semantics
and attack roots were not read.

## Decision

```text
PHYSICS_TASK_DECODER              = PASS_TASK_CONDITIONAL_DECODER
PHYSICS_TEACHER_V2_MATERIALIZED   = PASS_WITH_EXPLICIT_NON_GRASP_TASKS
PHYSICS_TEACHER_INDEPENDENT_AUDIT = PASS
V5_B/C/D                          = HOLD
GPU_SMOKE                          = NOT STARTED
FORMAL_TRAINING                    = NOT AUTHORIZED
ATTACK                             = NOT STARTED
```

This root is suitable as a sealed FIT-only development Teacher input. It is
not a formal training authorization and does not establish attack utility.
Before GPU smoke, the policy-intent runtime causality HOLD and C2F exact
trajectory-binding HOLD remain explicit; visual variants remain disabled until
C2F evidence is independently bound.
