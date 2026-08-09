# C3-G-DEV Stage 3 independent review hold

## Decision

`C3-G-DEV-STAGE3 MAIN = PASS` but `INDEPENDENT REVIEW = HOLD`.
Therefore:

```text
C3-G-DEV FINAL = HOLD
C3-T             = NOT STARTED
Clean2000        = NOT READ
Teacher/Student  = NOT STARTED
ROLLOUT          = NOT STARTED
ATTACK           = NOT STARTED
```

The gate rule requires an independent PASS; no later stage was started.

## Main evidence

- code commit: `6ea950af3316d077e42a683a0304c78317394b1d`
- code tree: `703cf4d31a4aac22596d70d4d774133a6e93bcbb`
- frozen predicate contract: `C3_G_PREDICATE_CONTRACT_V1`
- Stage 2 official tests: `51 passed`, `0 failed`, `0 errors`
- Stage 3 main: `44 relations × 6 cases = 264`, A/B/comparison PASS
- Stage 3 canonical digest: `3880fbc5dd23be5d465943140e6aeb41ed9f3672a4de1a942bd3f20b5c06a3dc`
- run_A `SHA256SUMS`: `23271c253c0d757af6b276f30d5230eea4af321f6ed3b3af9d734ae619b476a7`
- run_B `SHA256SUMS`: `481c9d1db7c1d6fefaee7c320a1196077ab48e44c5401eec936799159220077e`
- comparison `SHA256SUMS`: `fb7629cac2ec5bea954e5ae5097c9c8c7d42370d3003fb2c5af5c759431b9ab4`

## Independent blocking findings

The sealed `predicate_records.jsonl` contains only result-level fields:

`case_kind`, `expected_value`, `observed_value`, `predicate`, `reason`,
`relation_id`, `episode_id`, `step`, and `pass`.

It does not contain the case-level object/target roles, identities, poses,
quaternions, or half-extents. An independent reviewer therefore cannot directly
recompute coordinate transforms, q/-q equivalence, NaN/Inf behavior, role/identity
semantics, or the tri-state result from the sealed output.

The independent mutation check consequently rejected `0/1`; the reviewer could
not demonstrate that a mutated case is rejected from the sealed case geometry.

The comparison summary also does not expose a complete independently consumable
case-level comparison closure. Canonical equality and the comparison seal alone
are insufficient for this gate.

## Required remediation before rerun

Add immutable case-level geometry to each Stage 3 record (or a separately sealed
case-input bundle bound to the record stream): object/target IDs and roles,
episode/step identity, world poses, target-local geometry, extents, and the
frozen contract SHA. Extend the comparison receipt with exact case counts,
per-relation counts, case-kind counts, and input/output binding. Then rerun fresh
Stage 3 A/B/comparison roots and an independent source/record mutation review.

No existing Stage 3 root is modified or reused as a new PASS.
