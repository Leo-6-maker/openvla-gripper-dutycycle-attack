# Fast VIS Proxy Comparison Audit

Date: 2026-06-05

Expected input: `reports/FAST_VIS_PROXY_COMPARISON_V0.md`

## Verdict

Status: BLOCKED_MISSING_COMPARISON_REPORT

`reports/FAST_VIS_PROXY_COMPARISON_V0.md` is not present in this checkout, so Codex did not audit the comparison claims.

## Checks Waiting On DeepSeek Output

When the comparison report exists, audit whether it correctly reports:

- positive recall
- negative specificity
- runtime reduction
- false positives on controls
- agreement with full VIS
- recommended fast budget
- failure modes

## Forbidden Claim Patterns

The comparison report must not claim:

- policy-only proves task success or task failure
- command-open proxy proves VIS
- silver/proxy labels are gold
- infra-failed rows are usable labels

## Claim Boundary

Fast cascade remains a candidate acceleration path only until the comparison report and output CSV schema audit pass.
