# Phase E Canary Root Cause — 2026-06-05

## Current Classification

Phase E is **not yet a silver-label generator**.

Known invalid/blocked phases:

- v0: `INVALID_ACTION_SPACE_CONFOUNDED`
- v1: `PHASE_MISALIGNED_COMPRESSED_WINDOW`
- v2: `PHASE_NOT_CAPTURED`

## v0: INVALID_ACTION_SPACE_CONFOUNDED

The first Phase E canary passed raw decoded OpenVLA actions directly to `env.step`. It skipped:

```python
normalize_gripper_action(raw_action, binarize=True)
invert_gripper_action(env_action)
```

As a result, policy-level OPEN token flips did not reliably become env-space OPEN commands. v0 cannot support any low-budget VIS mechanism claim.

## v1: PHASE_MISALIGNED_COMPRESSED_WINDOW

After the action transform was repaired, the centered L10 canary window landed in a natural-open phase. That window cannot test true closed-phase gripper disruption, so any failure/success observed there is phase-confounded.

## v2: PHASE_NOT_CAPTURED

Parent-start aligned L10 still measured qpos around open. The true closed/contact phase was not captured by the compressed window, so v2 cannot validate low-budget Phase E as a data accelerator.

## Current Fix

The repair adds:

- explicit phase-aligned window generation in `scripts/diagnostics/generate_phase_e_aligned_windows.py`;
- `--candidate-csv` support in `scripts/diagnostics/run_phase_e_canary.py`;
- mandatory explicit window use when a candidate CSV is provided;
- MuJoCo-qpos primary phase gate with obs qpos as fallback/audit;
- mechanism guard fields and `mechanism_status`;
- epsilon calibration through processor std when available, otherwise an explicit raw-div255 mismatch warning.

## Claim Boundary

Phase E can only become a silver-candidate generator after:

1. true_closed phase alignment;
2. mechanism_clean audit;
3. positive/negative canary separation;
4. 5-sample smoke pass.

Phase E rows must not enter labels_v2/v3 or detector training. Phase E remains a low-budget diagnostic scaffold until those gates pass.
