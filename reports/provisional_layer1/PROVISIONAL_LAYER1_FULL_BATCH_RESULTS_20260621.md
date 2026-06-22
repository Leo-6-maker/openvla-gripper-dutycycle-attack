# Provisional Layer1 Full-Batch Results 20260621

```text
STAGE:
  PROVISIONAL_LAYER1_FULL_BATCH

STATUS:
  ENGINEERING_PASS

OFFICIAL_H2_STATUS:
  NOT_GRANTED

HUMAN_REVIEW_STATUS:
  DEFERRED_NONBLOCKING_FOR_ENGINEERING

PAPER_CLAIMS:
  FORBIDDEN
```

This run is part of the engineering bypass recorded in
`reports/engineering_bypass/ENGINEERING_MAINLINE_BYPASS_20260621.md`.
It is not an H2 freeze and is not final paper evidence.

## Execution

```text
repo_commit:
  d1321845131e71e1a30315e48bbc67155f602230

server_checkout:
  /data/liuyu/repos/provisional_layer2_d132184_20260621

output_root:
  /data/liuyu/layer1_outputs/provisional_layer1_6eb8863_20260621_r2

sentinel:
  PROVISIONAL_ENGINEERING_ONLY_NOT_FOR_CLAIMS

python:
  /data/aviary/envs/openvla_official_libero_20260525/bin/python

gpu/libero/openvla:
  NOT_RUN
```

The first attempted output root
`/data/liuyu/layer1_outputs/provisional_layer1_6eb8863_20260621`
failed closed before resolver execution because the handoff component SHAs for
two text files were Windows working-tree byte hashes. The successful run records
both the handoff working-tree SHA and cross-platform Git blob SHA. The ontology
and Teacher schema were accepted by `git_blob_sha256`; the physics config and
timing contract matched the handoff working-tree SHA directly.

## Inputs

```text
Train300 ledger:
  tables/train300_20260620/final_primary_ledger.csv

CLEAN300 deep-integrity ledger:
  /data/liuyu/audit_outputs/cross_suite_clean_300_final_deep_integrity_20260619_202447/tables/cross_suite_clean_300_master_ledger.csv
```

All clean failures remain in the denominator. No clean rollout was recollected.

## Manifest Results

| split | expected | selected | duplicates | rejected | status |
|---|---:|---:|---:|---:|---|
| train300_train_s10_17 | 240 | 240 | 0 | 0 | PASS |
| train300_val_s18_19 | 60 | 60 | 0 | 0 | PASS |
| clean300_test_s0_9 | 300 | 300 | 0 | 0 | PASS |

## Resolver Results

| split | episodes | events | eligible events | failure count | validation errors |
|---|---:|---:|---:|---:|---:|
| train300_train_s10_17 | 240 | 105 | 105 | 0 | 0 |
| train300_val_s18_19 | 60 | 27 | 27 | 0 | 0 |
| clean300_test_s0_9 | 300 | 131 | 131 | 0 | 0 |

Teacher status counts:

| split | ELIGIBLE_EVENT | NO_RELEVANT_GRASP_EVENT | CORRECT_SEMANTIC_ABSTAIN | TARGET_BINDING_AMBIGUOUS | RESOLVER_NOT_IMPLEMENTED_FOR_MECHANISM |
|---|---:|---:|---:|---:|---:|
| train300_train_s10_17 | 105 | 23 | 24 | 8 | 80 |
| train300_val_s18_19 | 27 | 5 | 6 | 2 | 20 |
| clean300_test_s0_9 | 131 | 29 | 30 | 10 | 100 |

## Frozen Artifacts

```text
batch audit:
  reports/provisional_layer1/provisional_layer1_batch_audit_20260621.json

manifest summary:
  tables/provisional_layer1/provisional_layer1_manifest_summary_20260621.csv

recursive output SHA manifest:
  tables/provisional_layer1/provisional_layer1_recursive_sha256_manifest_20260621.csv

recursive manifest SHA256:
  c6efb23e5d25e07793798d6740427866e06a8ee06571a38876f63623182a4c8a
```

## Allowed Claim

The provisional Layer1 resolver can process the frozen Train300 and CLEAN300
ledgers end-to-end and produce complete provisional labels for engineering use.

## Forbidden Claims

- H2 is scientifically frozen.
- Teacher labels are final ground truth.
- Human review is complete.
- Layer2 generalization is established.
- VIS/RAND/shuffled or attack effectiveness is established.
- These provisional labels may be used as final paper evidence.
