# Provisional Layer2 Cross-Suite Engineering Preregistration

## Scope

This file starts the stacked provisional Layer2 branch. It depends on the H2
engineering-bypass record and does not constitute H2 approval.

```text
H2_HUMAN_REVIEW = DEFERRED_NONBLOCKING_FOR_ENGINEERING
APPROVE_GATE_H2 = NOT_GRANTED
PROVISIONAL_LAYER1_SNAPSHOT = FROZEN_FOR_ENGINEERING
LAYER2_PROVISIONAL_TRAIN_EVAL = GO
PAPER_CLAIMS = BLOCKED
```

## Output Roots

```text
provisional_layer1_root = /data/liuyu/layer1_outputs/provisional_layer1_6eb8863_20260621
provisional_layer2_root = /data/liuyu/layer2_outputs/provisional_cross_suite_20260621
sentinel = PROVISIONAL_ENGINEERING_ONLY_NOT_FOR_CLAIMS
```

## Frozen Provisional Layer1 Components

```text
resolver_commit = 6eb88630f20f99aac4d64356974b5a18345c3673
ontology_sha256 = 89bd296b15525c48a4fbd3be84eb4a8c0b269ca11170cfe901b449cc1bb77359
physics_config_sha256 = 1f5e0dbeb0e227d2c6708a310ef171863c216bf84a5cddda62af992766700059
teacher_schema_sha256 = ac2ffb8b064a502f8b5cab7b0c4183914701c7bf21443dd01189e023403ebca8
timing_contract_sha256 = 18b21f9e032fdba410e291f67edba60e649b2c6c4ae3d2f33df08a7c31f6ef60
```

## Dataset Roles

```text
train = Train300 states 10-17
validation = Train300 states 18-19
test = old CLEAN300 states 0-9
```

All clean failures remain in the funnel. No clean rollout recollection is
authorized.

## Supervision Policy

```text
ELIGIBLE_EVENT -> positive event/window supervision; out-of-window frames may be negative
CORRECT_SEMANTIC_ABSTAIN -> negative-only only for explicit non-grasp mechanisms
NO_RELEVANT_GRASP_EVENT -> negative-only
TARGET_BINDING_AMBIGUOUS/TARGET_BINDING_FAILED/OBJECT_BINDING_AMBIGUOUS/RESOLVER_NOT_IMPLEMENTED_FOR_MECHANISM -> ignore-mask for supervised loss
```

## Layer2 Matrix

Primary:

```text
causal 25D MLP
```

Diagnostics:

```text
TCN
GRU
```

Run order:

```text
M0 = existing frozen Object-only zero-shot baseline
M1 = per-suite in-domain
M2 = leave-one-suite-out
M3 = few-shot adaptation with budgets 0/1/2/4/8 states per task
```

No minimum accuracy threshold is required for provisional engineering
acceptance. Poor generalization is a valid result and must not trigger
result-driven label repair.

## Leakage Rules

Fail immediately on:

```text
state overlap
test-set normalization
test-tuned threshold
future-feature leakage
privileged-state leakage
success-only filtering
clean-failure deletion
duplicate canonical keys
```

## Forbidden Claims

```text
H2 is scientifically frozen
Teacher labels are final ground truth
cross-suite detector generalization is confirmed
VIS outperforms controls
attack effectiveness is established
```
