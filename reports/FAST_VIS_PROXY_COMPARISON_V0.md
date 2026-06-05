# Fast VIS Proxy Comparison v0

**Status**: BLOCKED_MISSING_OR_INCOMPLETE_FAST_VIS_OUTPUTS

This report compares Fast cascade proxy outputs against full VIS reference labels. It is CPU-only and reads CSVs only.

## Metrics

| Dataset | Budget | n | TP | FP | FN | TN | Positive recall | Negative specificity | Agreement | Runtime reduction |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|

## Recommended Fast Budget

none: no complete comparable Fast VIS outputs

## Failure Modes / Issues

- command_proxy: missing_csv (1)
- low_budget: missing_csv (1)
- policy_only: missing_csv (1)

## Claim Boundary

- Policy-only outputs do not prove task-level success or failure.
- Command-open proxy outputs do not prove VIS.
- Silver/proxy labels are not gold labels.
- INFRA_FAILED, MEASUREMENT_FAILED, BLOCKED, and ERROR rows are excluded from comparison metrics.
- Agreement with full VIS is an acceleration-screening result, not detector validation.
