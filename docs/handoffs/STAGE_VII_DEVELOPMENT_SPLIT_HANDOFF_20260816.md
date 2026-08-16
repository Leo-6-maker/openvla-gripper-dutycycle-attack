# Stage VII development split handoff

Status: `PASS_STAGE_VII_DEVELOPMENT_SPLIT`

The split was frozen after the domain-shift forensic and before any Stage VII candidate training.

- Source commit/tree: `ab09b7fa4b4f7dbcd71530a3d1c1ad6bbe4f6da7` / `221a45ec67cd184afa7ef8548e4eed38b4e0f86c`.
- Sealed output: `/mnt/sdc/dty_user/openvla_attack_outputs/n5/phase3_student/STAGE_VII_CONTEXT_CONDITIONED_VULNERABILITY_DETECTOR/STAGE_VII_DEVELOPMENT_SPLIT_20260816T075626Z`.
- Population: 40 Stage V development parents + 16 Stage VI-B2 development parents = 56 unique parents.
- Selection inputs: canonical parent identity and suite only; labels, Student/Teacher scores, V_phys, attack outcomes, horizon behavior, and protected identities were not used.
- Split rule: SHA-256(`STAGE_VII_SPLIT_V1_20260816|canonical_parent_key`) order within each suite, then deterministic 60/20/20 TRAIN/VAL/DEVTEST assignment.
- Counts: TRAIN 32, VAL 11, DEVTEST 13.
- Suite counts:
  - `libero_10`: 10 / 3 / 4
  - `libero_goal`: 7 / 3 / 3
  - `libero_object`: 7 / 2 / 3
  - `libero_spatial`: 8 / 3 / 3
- Four frozen leave-one-suite-out partitions are included.
- Root `SHA256SUMS` binding matches; `candidate_training_performed=false`; protected counters are zero; Eval160/protected evaluation remain unread.

The next legal scientific action is S7-A only: the multidose 16x25D control candidate on this frozen split. S7-B/S7-C remain unavailable until a complete clean language/visual embedding contract is independently materialized; no new M4 is authorized by this split alone.
