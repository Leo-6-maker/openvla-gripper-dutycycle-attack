# C3-T1D-R2 Teacher Quality Remediation Handoff

## Decision

```text
branch                         = codex/detector-completion-20260726
execution/source commit        = bf4857a21aabb5b930ec616f452a9421a5b14633
T1D-R2A semantic remediation   = PASS
T1D-R2B numerical fidelity    = HOLD_REFERENCE_MISSING
T1D-R2C pilot A/B             = NOT RUN
T1D-R2D quality report        = NOT RUN
T1D-R2E candidate packet      = NOT GENERATED
protected payload reads       = 0
OpenVLA inference             = NOT STARTED
Student training              = NOT STARTED
rollout / attack              = NOT STARTED
```

## R2A

The shared tri-state contract now uses explicit three-valued operators:

```text
AND: FALSE > UNKNOWN > TRUE
OR:  TRUE > UNKNOWN > FALSE
```

The runner now binds a versioned `C3_T0_TEACHER_SEMANTIC_CONTRACT_V1_1`
with frozen co-motion, placement-stability and slip thresholds.  It also:

- requires open qpos and released object-gripper contact for `released_state=TRUE`;
- keeps `release_event` as a diagnostic;
- computes placement stability from causal relative translation/quaternion deltas and consecutive support evidence;
- separates maintained-contact slip from contact-loss transitions;
- computes K10 only from protocol horizon and the current safe-release result;
- records observed-suffix censor information separately without changing K10.

Local and official-environment checks both pass:

```text
26 tests passed, 0 failed, 0 errors
Python compilation = PASS
git diff --check   = PASS
```

## R2B

The read-only source census consumed only the frozen 40-episode DEV pilot
metadata/payloads and the previously sealed geometry replay root.

```text
episodes                         = 40
steps                            = 9422
complete per-step simulator state = 0/40
partial object_state present     = 40/40
independent reference            = unavailable
```

The source sidecar contains `object_state`, gripper qpos, EEF fields and
contact pairs, but no sealed full simulator `qpos/qvel/sim_state` or complete
target-site state.  Therefore the existing action replay can only be reported
as a repeatability diagnostic, not as numerical accuracy:

```text
qpos sidecar max abs error       = 0.01582322290913525
EEF sidecar max abs error        = 0.028513823662753762
object/target/quaternion errors  = unavailable
predicate flip counts            = unavailable
```

The sealed R2B audit is:

```text
root = /mnt/sdc/dty_user/openvla_attack_outputs/n5/phase3_student/c3_t1d_r2b_reference_audit_bf4857a_20260727
SHA256SUMS SHA = 9650ae69398d446d15103866bc3ede4a7aa03fe412f4e326d86d1c385d8b4c8a
status          = HOLD_REFERENCE_MISSING
```

Because the independent reference is missing, the protocol requires stopping
before new A/B labels and before any quality PASS claim.  The previous A/B
roots remain immutable and are not reused as accuracy evidence.

## Scope boundary

No protected payload, T2R-D data, 670-episode data, OpenVLA inference,
Student training, rollout, or attack was started.  No threshold was relaxed
and no existing evidence root was overwritten.
