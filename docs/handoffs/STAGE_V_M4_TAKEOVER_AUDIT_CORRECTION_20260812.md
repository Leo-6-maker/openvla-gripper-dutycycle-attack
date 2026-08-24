# Stage V M4 takeover audit correction

Status: `HOLD_RESERVE_ROOT_PROVENANCE_MISMATCH`

The read-only takeover audit on 2026-08-12 found that the formal A/B parent
receipts reconcile to the recorded 29 stable identities, but both reserve
roots contain the same malformed `PREFLIGHT_INPUT.json` (`7ec418...`). The
file is not valid JSON and neither reserve root has a `SHA256SUMS` root seal.

The fourteen reserve parent receipts were still read and hashed. Their A3/B3
canonical fields and key set match, with three stable Object identities, and
all observed protected counters are zero. That result is now descriptive only:
the reserve roots are not consumable provenance and must not be used to justify
formal M4 or a label claim.

The formal corridor remains `HOLD_FORMAL_M4_CORRIDOR_INSUFFICIENT` at 29/40
(`libero_10=9`, `libero_goal=4`, `libero_object=10`, `libero_spatial=6`). No
intervention, label generation, protected read, Teacher, Student, Scheduler,
Timing, or VIS action was started.

Exact machine-readable evidence is in
`reports/STAGE_V_M4_TAKEOVER_AUDIT_CORRECTION_20260812.json`. Verify the PR
tip from Git metadata; this document does not self-reference a future tip.

Next legal action: add the append-only stale-M4 supersession/HOLD gate and
make the future runner/authorization issuer require a new independent
`PASS_FORMAL_M4_CORRIDOR` receipt. Do not repair or rerun the sealed reserve
roots in place.
