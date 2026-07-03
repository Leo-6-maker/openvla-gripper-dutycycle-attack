# Paper Claim Freeze V1

Date: 2026-07-03
Status: PLANNING_ONLY

This document freezes the next paper-planning target. It does not authorize
label building, detector training, rollout, attack execution, or server work.

## Claims

1. Clean-only localization

Clean proprio/action telemetry can localize contact-critical windows without
online privileged state.

2. Selective dimension vulnerability

For matched budget and timing, gripper-targeted perturbations change gripper
duty cycle more than arm trajectory.

3. Timing matters

Matched gripper payloads are more damaging in the detector window than at
random or early-shift times.

4. Cross-suite generalization

The detector and attack mechanism must be measured across Object, Spatial,
Goal, and LIBERO-10 before any cross-suite generalization claim.

## Current Claim Boundary

Historical Object evidence supports selective, phase-dependent gripper
vulnerability under `LEGACY_PROTOCOL_LIMITED`.

Current evidence does not support:

- clean-only detector generalization across all suites;
- formal cross-suite attack success;
- pooled or LOSO detector conclusions;
- final paper-level confirmatory Object benchmark.

## Gate

```text
AUDIT_FREEZE_STATUS = EXISTING_EVIDENCE_FREEZE_READY_FOR_REVIEW
SCIENTIFIC_CLAIM_STATUS = LEGACY_PROTOCOL_LIMITED
EXPERIMENT_AUTHORIZATION_STATUS = NOT_AUTHORIZED
```

