# G-REC data-gap fallback canary handoff

```text
FIT_ONLY_CLEAN_TELEMETRY_CANARY = PASS
GEOMETRY_CANARY_STRUCTURAL_AUDIT = PASS
GEOMETRY_ACCURACY_PROOF = NOT_ESTABLISHED
FALLBACK_CANARY_CONSUMABLE = false
G-REC = HOLD_DATA_GAP
T-PILOT = BLOCKED
```

The four canary roots are newly collected clean telemetry from the official
OpenVLA path, not replacements for historical episodes.  The official action
was passed unchanged to LIBERO; no detector, Teacher, Student, attack or
protected payload was used.

The two derived geometry roots contain 863 relation cases with identical
canonical case digest:

```text
881843e571987f51a820aa8b3fed83cdf6dd4833f572e0b97f1eed52792e6ab7
```

The independent CPU review verified recursive seals, exact input bindings,
finite recorded poses, relation/step joins, and the no-Teacher/no-attack
boundary.  It did not and cannot establish geometry accuracy because both
materialized sides intentionally use the recorded MuJoCo world pose.  This
avoids the rejected action-replay-as-truth path but remains a
`DERIVED_FIT_ONLY_CANARY_NONCONSUMABLE` evidence root.

The four-suite canary gate was followed by a 40-episode FIT-only batch
recovery.  Its 40 child seals and unified runtime audit passed; the batch
observed 8,332 steps and 44 relation rows, with two relation-empty telemetry
episodes retained outside geometry materialization.  The resulting derived
geometry A/B roots contain 10,317 cases and have identical canonical digest
`87e2ff5179cd733fdaa91970ae8b81ca5bf493d79663bd5ad941e5141fa3eea1`.
Because both sides still use direct recorded world pose, this remains
`NONCONSUMABLE`; G-REC geometry accuracy and T-PILOT remain blocked.
