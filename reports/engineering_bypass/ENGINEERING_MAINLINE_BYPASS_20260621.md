# Engineering Mainline Bypass Record

## External Decision

```text
H2_HUMAN_REVIEW = DEFERRED_NONBLOCKING_FOR_ENGINEERING
APPROVE_GATE_H2 = NOT_GRANTED

PROVISIONAL_LAYER1_SNAPSHOT = FROZEN_FOR_ENGINEERING
FULL_TRAIN300_RESOLVER_PROVISIONAL = GO
FULL_CLEAN300_RESOLVER_PROVISIONAL = GO

LAYER2_PROVISIONAL_TRAIN_EVAL = GO
LAYER3_ENGINEERING_SMOKE = CONDITIONAL_GO_AFTER_LAYER2_SANITY_PASS

FINAL_BLIND = NOT_RUN
PAPER_CLAIMS = BLOCKED
```

## Reason

```text
prioritize an end-to-end Layer1->Layer2->Layer3 cross-suite engineering run
```

This record documents an engineering bypass only. It is not an H2 scientific
approval and it does not complete human review.

## Official H2 Status

```text
official_h2_status = NOT_GRANTED
human_review_status = DEFERRED_NONBLOCKING_FOR_ENGINEERING
```

## Provisional Layer 1 Snapshot

```text
provisional_layer1_resolver_commit = 6eb88630f20f99aac4d64356974b5a18345c3673
ontology_sha256 = 89bd296b15525c48a4fbd3be84eb4a8c0b269ca11170cfe901b449cc1bb77359
physics_config_sha256 = 1f5e0dbeb0e227d2c6708a310ef171863c216bf84a5cddda62af992766700059
teacher_schema_sha256 = ac2ffb8b064a502f8b5cab7b0c4183914701c7bf21443dd01189e023403ebca8
timing_contract_sha256 = 18b21f9e032fdba410e291f67edba60e649b2c6c4ae3d2f33df08a7c31f6ef60
```

## Required Disclaimer

```text
All downstream labels, models, metrics and attacks are provisional.
Any later Layer1 semantic change invalidates and supersedes them.
No provisional result may be used as final paper evidence.
```

## Preserved Human Review Package

The existing Reviewer A/B working copies must not be deleted or edited by this
bypass:

```text
human_reviews/in_progress/reviewer_A_h2_diagnostic_review_round_v2_1_WORKING_COPY.csv
human_reviews/in_progress/reviewer_B_initial_h2_diagnostic_review_round_v2_1_WORKING_COPY.csv
```

## Still Forbidden

```text
APPROVE_GATE_H2 = NOT_GRANTED
FINAL_BLIND = NOT_RUN
PAPER_CLAIMS = BLOCKED
```
