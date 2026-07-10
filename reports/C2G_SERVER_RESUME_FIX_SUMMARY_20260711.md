# C2g Detector-v2 — Server Resume Fix Summary

Date: 2026-07-11

Current branch: `codex/c2g-strict-server-smoke-20260710`

## Repairs added after the first Codex S1 return

```text
Goal v2 manifests are accepted only through the central byte-verifying validator.
Clean timing records the Goal validation result before any worker launch.
Matched-load model-map launch verifies Goal bytes before command execution.
Direct script execution now inserts the repository root before importing scripts.* modules.
Four-suite clean collection runs one suite per subprocess to release each 7B model before loading the next.
A combined collection report and immutable metadata/step manifest are rebuilt after all suite subprocesses pass.
Outcome-leakage scanning examines mapping keys only, so documentation strings such as
student_forbidden_modalities=["attack_outcome"] do not create false positives.
Regression tests cover v2 Goal acceptance and mutation rejection, suite partitioning,
command construction, and combined collection closure.
```

## Remaining live gates

```text
S0 server rerun
alias-aware 40-BDDL / 107-XML inventory
Goal frozen-byte restoration or explicit v2 current-byte rebase
strict four-suite model map
one clean parent per suite
Teacher-v2 dry audit
real tiny materialization/trainability
one-epoch checkpoint and clean calibration
four detector-only CLEAN timing parents
one-parent command dry run
one-parent four-attack runtime smoke
closed-world audit
```

## Known pre-scaling risk

The current collector is strongest for tasks with one active target entity. Before scaling
beyond the bounded cohort, report multi-target contact-positive/progress-positive coverage.
Systematic multi-target contact-positive but progress-negative rows are a HOLD requiring
per-step active-target/subgoal tracking; they must not be interpreted as known negatives.

The complete staged plan is:

`reports/C2G_DETECTOR_V2_LONG_RANGE_CODEX_PLAN_20260711.md`
