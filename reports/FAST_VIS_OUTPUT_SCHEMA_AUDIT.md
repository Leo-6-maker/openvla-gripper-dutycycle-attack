# Fast VIS Output Schema Audit

**Overall status**: BLOCKED_MISSING_FAST_VIS_OUTPUTS

This is a CPU-only schema audit. It does not run rollout, VIS, watcher, or detector training.

## Dataset Status

| Dataset | Rows | Status | Issues |
|---|---:|---|---|
| policy_only | 0 | BLOCKED_MISSING_FAST_VIS_OUTPUTS | missing_csv |
| command_proxy | 0 | BLOCKED_MISSING_FAST_VIS_OUTPUTS | missing_csv |
| low_budget | 0 | BLOCKED_MISSING_FAST_VIS_OUTPUTS | missing_csv |

## Checks

- Required columns: task_key, state_id, window_start, window_end, label, label_source, label_confidence, gpu_pair, runtime_sec, provenance_status.
- denominator_status is required, including explicit not_applicable values for policy-only and command-proxy outputs.
- Proxy labels must not be marked gold.
- Rows with INFRA_FAILED/Xid/OOM/CUDA failures must not be treated as trainable labels.

## Claim Boundary

- Missing output CSVs are reported as BLOCKED_MISSING_FAST_VIS_OUTPUTS, not as failed experiments.
- Policy-only and command-open proxy results are screening/proxy evidence only, not gold VIS labels.
