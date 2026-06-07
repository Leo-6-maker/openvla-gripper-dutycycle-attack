# VIS 1R vs 3R Calibration v1

**Status**: CALIBRATION_COUNT_PASS_BUT_AGREEMENT_LOW
**Config mismatch**: SUSPECTED (3R traces from batch1/3, 1R from calibration chain)
**Paired usable**: 10 (threshold: 6)
**Agreement**: 60% (6/10)
**4 disagreements**: all 3R success / 1R fail

## Verdict

- **1R recall is insufficient** — 1R fails to detect 4 candidates that 3R found.
- **1R failure must remain pending_negative_1r** — cannot be used as negative.
- **1R success cannot be promoted to silver yet.**
- **Config-matched calibration v2 required.**

## Promotion Rule

DO NOT promote adaptive 1R to silver labels until:
- Config-matched calibration v2 completed
- 1R positive precision >= 0.8
- No action-confounded positives

## Current adaptive 1R status

All overnight results: label_source=uncalibrated_1r_screening.
1R failure = pending_negative_1r (NOT train).
No detector v3 training until label gate passes.
