# Stage X scripts

Canonical CODE-R2 audit-only command:

```text
python -m pytest -q tests/stage_x tests/test_stage_x_primary_matrix_runner.py
```

The tests are CPU/fixture/mock checks. Their PASS is engineering evidence only.

Lifecycle groups:

- Q3/Q3-AR/Q3R2/Q3R3 files are engineering qualification and branch-replay
  history. Their HOLDs are not negative attack results.
- E3/E4 files support sealed model-side structural realizability evidence.
  Parent units are primary; candidate slots are non-iid diagnostics; no
  physical efficacy is established.
- F1-A/B/C/C4 files are closed development and canary qualification history.
  F1T is terminal and sealed for PI. Do not tune, top up, recycle, reopen F1-D
  or BRIDGE, or rerun-to-pass.
- `build_stage_x1r2_f1t_synthesis.py` is an immutable historical paper-analysis
  producer. Read its sealed outputs; do not rerun or modify it.

No `run_*`, `freeze_*`, `repair_*`, `seal_*`, or artifact-producing `audit_*`
script is authorized by this README. Any scientific execution requires a new
prospective PI authorization and resource/protected-boundary checks.
