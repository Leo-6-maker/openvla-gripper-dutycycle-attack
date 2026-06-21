# Provisional Layer3 Two-Suite Interface Smoke Results

## Status

```text
STAGE: PROVISIONAL_LAYER3_TWO_SUITE_INTERFACE_SMOKE
COMMIT: 592de8d3b65a4314b5104afb2c08c3b137e35271
RESULT_CLASS: INFRA_FAILED_NOT_ACCEPTED
TWO_SUITE_LAYER3_INTERFACE_SMOKE: NOT_ACCEPTED
THREE_SUITE_MAINLINE: NOT_RUN
VIS_GT_RANDOM: NOT_ESTABLISHED
PAPER_CLAIMS: BLOCKED
```

This was an engineering-only Layer2-to-Layer3 interface smoke for Spatial and Goal. It was not a three-suite run and not an attack-effectiveness validation.

## Output Root

```text
/data/liuyu/layer3_outputs/provisional_two_suite_interface_smoke_20260621
```

The output root contains the sentinel:

```text
PROVISIONAL_ENGINEERING_ONLY_NOT_FOR_CLAIMS
```

## Planned Versus Observed

| Item | Count |
| --- | ---: |
| Planned jobs | 16 |
| Completed jobs | 11 |
| Failed jobs | 1 |
| Not run after worker fail | 4 |
| Completed `episode_summary.json` files | 11 |

PAIR_B completed all Goal jobs:

```text
PAIR_B: planned=8 complete=8 failed=0
```

PAIR_A did not complete the Spatial shard:

```text
PAIR_A: planned=8 complete=3 failed=1 not_run=4
```

The failed job was:

```text
libero_spatial_3_4_0_CLEAN_shuffled
```

## Failure Evidence

PAIR_A failed during the first Spatial parent's SHUFFLED condition with return code 1. The worker ledger records:

```text
job_id = libero_spatial_3_4_0_CLEAN_shuffled
status = FAILED
returncode = 1
duration_sec = 988.8133533000946
log_path = /data/liuyu/layer3_outputs/provisional_two_suite_interface_smoke_20260621/_worker_PAIR_A/libero_spatial_3_4_0_CLEAN_shuffled.log
```

The server kernel log showed a matching Xid:

```text
timestamp = 2026-06-21 12:44:29 CST
pci = 0000:08:00
gpu_uuid = GPU-c1ee1619-6791-a734-bf4c-d0d2c47df580
pid = 39772
name = python
xid = 31
fault = MMU Fault, FAULT_PDE, ACCESS_TYPE_VIRT_READ
```

This is treated as an infrastructure failure. It is not evidence for or against VIS, RAND, SHUFFLED, detector timing, or attack effectiveness.

## Reconciliation Files

The full planned-job reconciliation is recorded in:

```text
tables/provisional_layer3/two_suite_interface_smoke_reconciliation_20260621.csv
```

The machine-readable summary is recorded in:

```text
reports/provisional_layer3/two_suite_interface_smoke_summary_20260621.json
```

## Allowed Claims

```text
Goal shard completed all 8 planned matched-condition rollouts.
Spatial shard was interrupted by an infrastructure failure.
The run exercised the provisional Layer2-to-Layer3 interface for completed jobs.
The overall two-suite interface smoke is not accepted because planned keys were not reconciled.
```

## Forbidden Claims

```text
Do not claim TWO_SUITE_LAYER3_INTERFACE_SMOKE = ENGINEERING_PASS.
Do not claim PROVISIONAL_LAYER123_MAINLINE pass.
Do not claim three-suite smoke.
Do not claim VIS > RAND or VIS > SHUFFLED.
Do not claim attack effectiveness.
Do not tune detector, objective, parent choice, or condition selection based on these partial results.
Do not treat the missing Spatial jobs as negative scientific evidence.
```

## Next Action

An external retry decision is required before any retry or Track C run. Retry planning should avoid using GPU3 as the C+G primary, per the current hardware constraint.
