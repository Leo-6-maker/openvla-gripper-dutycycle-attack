# Stage X X1R T1-D1M0R handoff — 2026-08-18

## Final state

`STAGE_X_X1R_T1D1M0R_HUMAN_SHEET_PASS`

`next_gate = OWNER_MANUAL_CONTACT_LABELS_REQUIRED`

D1M0R repaired only the delivery layer. It did not recompute candidates,
change anonymous order, change the rubric, rerender clips, modify raw videos,
infer labels, or run any model/simulator/attack operation.

## Safe human-facing sheet

The sealed sheet contains exactly these fields:

```text
review_id
task_instruction
review_clip_path
review_clip_sha256
review_frame_strip_path
review_frame_strip_sha256
contact_label
reason_code
reviewer
review_timestamp
optional_short_note
```

There are exactly 14 rows in frozen order `M001` through `M014`. The task
instruction is copied from the frozen private mapping, while clip/strip paths
and hashes are copied from the frozen render manifest. All owner-label fields
are blank.

The sheet does not expose suite, task/state identity, ordinal, parent key,
rank, seed, numeric emit step, policy horizon, Student/physical scores, raw
receipt paths, telemetry paths, or attack/V_phys data. The private mapping is
not copied into the human-facing packet.

## Review material

Durable review root:

```text
/llm_jzm/dty_user/openvla_attack_d1_screening_clean_20260818/D1R_CONTINUATION/T1D1M0_REVIEW_PACKET
```

Use only the safe sheet, the fixed `M001`–`M014` clips/frame strips, and the
frozen D1M0 rubric. Do not open the private mapping, D1R receipts, telemetry,
Student scores, physical scores, or attack/V_phys artifacts while labeling.

## Immutable D1M0 bindings

- candidate ledger SHA256:
  `5f1f036b47b1c9a8c1bafe7a400b6be9269cd3e67587691018005c824dc8d89e`
- mapping SHA256:
  `3d7f59a736cc2c7bcb5ecdc49e9e57a7e8b547c9e7554251e88158017366f0fe`
- review-order digest:
  `30a73b0e4ab13e149d8c991906fc9067844797e39113201e9e76a10a8be40d67`
- safe CSV SHA256:
  `9c42b3d6486f2082c414ff799d9efe2f2633e797d2b92e1f45fb23426470a7b2`
- safe JSON SHA256:
  `9742206be7e948a9e5df7903edb1c5a071819654e983fe950817dc8bed163eff`
- human-sheet audit SHA256:
  `b1d39852d0687f9aa90215acf933290e70f69e9488a21827e2aa04f5f41f0080`
- D1M0R SHA256SUMS SHA256:
  `a0d3f1fc98803bf61a2b6bccea7c3e5ee128361e7280b7081281418017ab27fc`
- D1M0R root-seal SHA256:
  `7b32a23ae6a7808c42c94a6c3caa082cf096a102ecd9d10835f3a2869a0a724d`

## Owner label return

After reviewing `M001` → `M014`, copy the safe sheet to a new file named
`STAGE_X_X1R_T1D1M1_OWNER_LABEL_SUBMISSION_V1.csv`, fill only the frozen label
fields, and report the filled file's raw SHA256. Leave the sealed D1M0R sheet
unchanged. Do not map labels to parent identity or compute a final eligible
count.

## Mandatory stop

D1M0R does not authorize D1M1 ingestion, CLEAN_EVAL, TRUE_PGD, RAND,
SHUFFLED, physical intervention, V_phys, Eval160, or protected evaluation.
`Eval160=UNREAD` and `protected_evaluation=UNREAD` remain in force.
