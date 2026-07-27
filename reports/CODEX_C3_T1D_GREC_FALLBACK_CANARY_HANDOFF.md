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

The next permitted operation is the batch collection of the remaining
frozen FIT pilot identities, followed by per-episode seal/audit, recorded-pose
materialization, independent structural verification, and a new G-REC review.
No protected split is opened by that operation.
