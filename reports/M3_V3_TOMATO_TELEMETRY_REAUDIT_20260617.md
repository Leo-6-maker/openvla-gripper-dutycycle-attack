# M3 V3 Tomato Telemetry Re-Audit

## Result

```text
S3_ORIGINAL_GATE: FAIL
S3_ORIGINAL_FAILURE_CLASS: TOMATO_NO_LAMBDA_PASS
S3_INDEPENDENT_REAUDIT: PASS
SELECTED_LAMBDA: 2.0
```

The original r3 gate was not modified. The original failure is retained as a
watcher aggregation result, but independent audit classifies it as telemetry
aggregation invalid rather than a scientific Tomato failure.

## Evidence

Source output root:

```text
/data/liuyu/outputs/m3_gpu15_autonomous_20260617_r3
```

Independent audit output:

```text
/data/liuyu/outputs/m3_gpu15_tomato_reaudit_20260617_r2
```

Committed table:

```text
tables/m3_v3_tomato_independent_reaudit_20260617.csv
```

All four lambda values passed independent audit. Each selected TRUE row emitted
token 31744, had arm match 6/6, passed strict route/no fallback via route audit,
stayed within Linf budget, and exceeded both RAND21 and shuffled controls.

The selected lambda under the preregistered ordering is `2.0`: all passing rows
have arm match 6/6, and lambda 2.0 has the largest TRUE-RAND margin.

## Claim Boundary

Allowed:

- The r3 watcher gate failed due to telemetry aggregation schema mismatch.
- The immutable r3 artifacts independently support Tomato fixed-frame pass for
  lambda 2.0.

Forbidden:

- Do not rewrite the original r3 `gate_result.json`.
- Do not claim multi-parent transfer from this audit.
- Do not claim closed-loop Layer3 success.
