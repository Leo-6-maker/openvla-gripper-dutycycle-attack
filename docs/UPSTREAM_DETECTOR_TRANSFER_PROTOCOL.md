# Upstream Detector Transfer Protocol

## Gate

`UPSTREAM_DETECTOR_TRANSFER_AND_THRESHOLD_AUDIT`

## Status

**BLOCKED — CHECKPOINT_NOT_FOUND**

## ProprioNoStep Detector Contract

Per `src/gripper_attack/sc5_detector_runtime.py`:

- Model: SC5MLP (25 features → 64 → 64, 3 heads)
- Checkpoint: NOT FOUND in repo or on A800
- Training: NOT_STARTED (D1B doc confirms)
- Trigger: tau_corridor=0.3, tau_release=0.3, guard=5
- State: IDLE → ARMED → EMITTED (one-shot)

Per D1B checkpoint selection rule (reports/D1B_CHECKPOINT_SELECTION_RULE.md):
TRAINING_STARTED: NO

## EVAL19

19 common-success episodes locked in migration_audit/detector/eval19_lock_manifest.json.
All use restrictions documented.

## Blocking Issues

1. No trained ProprioNoStep checkpoint exists
2. Normalization mean/std unavailable (embedded in checkpoint)
3. Privileged teacher module not identified for offline re-labeling
4. Full D0.3-D0.6 replay cannot be executed without checkpoint

## Resolution Path

Must complete before attack pilot:

1. Train ProprioNoStep SC5MLP on upstream clean-only traces
2. Freeze checkpoint with dataset provenance
3. Extract normalization parameters
4. Run D0.4-D0.6 transfer evaluation on upstream clean30 traces
5. Gate classification based on transfer metrics
