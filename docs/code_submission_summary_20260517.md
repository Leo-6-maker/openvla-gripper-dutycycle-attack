# Code Submission Summary 2026-05-17

Branch: `refactor/generic-autowindow-cqv2-20260517`

This branch consolidates source code, configs, tests, and documentation for the generic v4 clean-trajectory auto-window detector and CQ v2 contact-quality metric.

Commits:

- `9f51c83` refactor: add task-agnostic generic auto-window detector
- `b023949` test: isolate Black Bowl reference-window evaluation
- `8f3d2fd` fix: add contact-quality v2 metric for uncontrolled drops

Pushed branch:

- `origin/refactor/generic-autowindow-cqv2-20260517`
- Pull request URL: https://github.com/Leo-6-maker/openvla-gripper-dutycycle-attack/pull/new/refactor/generic-autowindow-cqv2-20260517

Validation:

- pytest: `76 passed in 1.96s` on old 2080Ti server Python
- generic detector hardcoding audit: `A. no_state_or_task_hardcoding_detected`
- CQ v2 calibration: 20/20 agreement with the Black Bowl manual review set

Not included:

- `outputs/`
- videos
- manual review packages
- raw `step_records.jsonl`
- raw `episode_records.jsonl`
- experiment locks

Known limitations:

- CQ v2 is calibrated on the Black Bowl manual set only.
- LIBERO clean denominator is insufficient with the current configured tasks.
- Moka remains helper/diagnostic only.
- No benchmark or Table 1 aggregation is included.
