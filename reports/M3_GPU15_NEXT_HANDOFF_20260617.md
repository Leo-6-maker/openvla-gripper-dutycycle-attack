# M3 GPU15 Next Handoff

## Current State

```text
L3-1_TOMATO_RECONFIRMED
MULTI_PARENT: NOT RUN
CLOSED_LOOP: NOT RUN
```

## Commits

```text
telemetry repair branch: exp/l3-v3-telemetry-repair-stage2-exec-20260617
head: d7e5a215c917c93d1fcb3c2f26498e1c01547a35
```

## Server Outputs

```text
r3 original watcher:
/data/liuyu/outputs/m3_gpu15_autonomous_20260617_r3

independent re-audit:
/data/liuyu/outputs/m3_gpu15_tomato_reaudit_20260617_r2

Tomato reconfirm:
/data/liuyu/outputs/m3_gpu15_tomato_reconfirm_20260617_r1
```

## Next Gate

Wait for final Layer1/2 handoff v2:

```text
tables/l12_to_l3_timing_handoff_v2.csv
tables/l12_timing_repeatability_v2.csv
```

Then run the executable Stage-2 watcher from the repaired branch. It must create
per-parent configs and run exactly:

```text
capture_input
preflight_zero_step
canary_v4 seed81
canary_v4 seed82
```

per parent. It must not use the old 7-row handoff and must not split
TRUE/RAND21/shuffled into fake separate runner conditions.

## Stop Rules

Stop before multi-parent if:

- handoff v2 is absent;
- selected parent is test/external/Teacher-P abstain;
- clean official token is not 31872;
- strict route or no-fallback fails;
- candidate count is not 21/21/21;
- any seed fails Tomato-style full selective gate.
