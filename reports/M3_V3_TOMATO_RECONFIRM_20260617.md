# M3 V3 Tomato Reconfirmation

## Result

```text
RESULT_CLASS: L3-1_TOMATO_RECONFIRMED
SELECTED_LAMBDA: 2.0
SEEDS: 81, 82
GPU_PAIR: physical GPU 1,5
OUTPUT_ROOT: /data/liuyu/outputs/m3_gpu15_tomato_reconfirm_20260617_r1
```

Both seed81 and seed82 passed the fixed-frame Tomato confirmation gate.

## Primary Metrics

| seed | TRUE token | TRUE arm | TRUE margin | RAND21 margin | shuffled margin | TRUE-RAND | TRUE-shuffled |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 81 | 31744 | 6/6 | 30.48320198059082 | 0.5 | 0.0 | 29.98320198059082 | 30.48320198059082 |
| 82 | 31744 | 6/6 | 29.61040687561035 | 0.25 | 0.25 | 29.36040687561035 | 29.36040687561035 |

Both runs had:

- 21/21 candidate rows for TRUE, RAND21, and shuffled.
- Exactly one selected candidate per condition.
- Strict route/no fallback.
- Linf within 6/255.
- Preflight surrogate/official path match.

Committed table:

```text
tables/m3_v3_tomato_reconfirm_20260617.csv
```

## Stop Condition

No multi-parent fixed-frame panel was run because the required final Layer1/2
handoff v2 files were not present:

```text
tables/l12_to_l3_timing_handoff_v2.csv
tables/l12_timing_repeatability_v2.csv
```

Only an older `/data/liuyu/outputs/l12_to_l3_timing_handoff.csv` was found, and
the protocol explicitly forbids using that old handoff for this stage.

## Claim Boundary

Allowed:

- Tomato development frame is reconfirmed for lambda 2.0 on two seeds.

Forbidden:

- No multi-parent transfer claim.
- No closed-loop physical bridge claim.
- No task-level Layer3 success claim.
