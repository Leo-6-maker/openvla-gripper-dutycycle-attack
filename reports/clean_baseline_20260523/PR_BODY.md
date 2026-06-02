## Summary

This PR establishes a clean protocol baseline for subsequent experiments.

No rollout, GPU job, Table2 run, or 142-eligible run was performed.

## Main changes

- Separates task identity from attack/condition protocols.
- Adds canonical condition protocol definitions.
- Restores the verified legacy Codex state5 protocol:
  - targeted attack: `force_gripper_open_token_ce`
  - eps=0.25, step_size=0.050, attack_steps=60
  - command_open uses rho=1.0 because rho=0 disables oracle override
- Marks old `MATCHED_CONDITIONS` / `TRIAGE_MATCHED_CONDITIONS` as fail-fast deprecated.
- Adds protocol validators for:
  - command_open rho/objective/env vars
  - same-seed matched conditions
  - clean autowindow source
  - Codex targeted protocol exactness
  - condition schema
- Documents attack mechanisms and claim boundaries.
- Adds no-GPU tests and smoke validation.

## Claim boundary

- State5 / bowl-on-plate spatial evidence is black-bowl-related.
- It is **not** a true non-Black-Bowl claim.
- Table1 prior windows are provenance only and must not be used as rollout input.
- True non-BB Object-suite work still requires gripper sign/window calibration.

## Validation

- Rebased onto `origin/main` with no conflicts.
- `compileall`: PASS
- smoke import + validators: PASS
- non-pytest regression: PASS
- pytest skipped because pytest is not installed.
