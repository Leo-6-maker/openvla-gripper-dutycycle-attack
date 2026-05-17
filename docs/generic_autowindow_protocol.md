# Generic Auto-Window Protocol

This protocol selects candidate contact-critical windows from clean trajectories only. It is a phase detector, not a vulnerability-maximizing oracle.

## Inputs

- Clean `step_records.jsonl`
- Clean `episode_records.jsonl`
- `run_manifest.json`
- Frozen detector hyperparameters from `configs/generic_autowindow_detector.yaml`

Forbidden inputs:

- Attack outcomes
- Manual labels
- Fixed reference windows
- Black Bowl reference-window config

## Detector Semantics

The detector estimates task phase from trajectory signals:

- grasp / close phase
- lift
- carry
- pre-place, descent, or release-intent cue

Window selection prefers a fixed-length interval immediately before a release/pre-place cue. If that cue is unavailable but the lift-to-done interval is reliable, it may use a normalized late-carry fallback. If signals are weak, it abstains instead of forcing a window.

Required output fields include:

- `window_detected`
- `auto_window_start`
- `auto_window_end`
- `detector_mode`
- `confidence`
- `failure_reason`

## Guardrails

The production detector must not branch on task id, suite, state id, seed, run id, or Black Bowl names. It may output those values as metadata only.

Black Bowl fixed windows are evaluation-only and live outside the detector in `configs/reference_windows_blackbowl_eval_only.yaml` plus the reference evaluator.

## Current Validation Boundary

The generic v4 detector passed hardcoding audit and Black Bowl manual-audited sanity. It is not yet a complete LIBERO-wide protocol because the current clean denominator outside Black Bowl is insufficient.
