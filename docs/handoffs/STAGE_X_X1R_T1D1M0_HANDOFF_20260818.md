# Stage X X1R T1-D1M0 handoff — 2026-08-18

## Final state

`STAGE_X_X1R_T1D1M0_REVIEW_PACKET_PASS`

`next_gate = OWNER_MANUAL_CONTACT_LABELS_REQUIRED`

This is a pre-registered manual contact-validity packet only. No model
inference, Student re-inference, simulator step, attack, V_phys read, Eval160
read, or protected read occurred in D1M0.

## Frozen candidate and blinded order

The 14 candidates were recomputed directly from the sealed D1R parent receipts
before any video pixels were opened:

- candidate ledger SHA256:
  `5f1f036b47b1c9a8c1bafe7a400b6be9269cd3e67587691018005c824dc8d89e`
- mapping SHA256:
  `3d7f59a736cc2c7bcb5ecdc49e9e57a7e8b547c9e7554251e88158017366f0fe`
- blinded order digest:
  `30a73b0e4ab13e149d8c991906fc9067844797e39113201e9e76a10a8be40d67`

The frozen rule is:

```text
SHA256("STAGE_X_X1R_T1D1M0_MANUAL_REVIEW_ORDER_V1_20260818"
       + "|" + canonical_parent_key)
```

The human packet exposes only `M001`–`M014`, task instruction, fixed review
copies, and blank label fields. The private mapping retains identity and hash
bindings. Student probabilities, physical scores, suite/task/state identity,
and attack/V_phys results are not in the human form.

## Fixed review material

Durable review root:

```text
/llm_jzm/dty_user/openvla_attack_d1_screening_clean_20260818/D1R_CONTINUATION/T1D1M0_REVIEW_PACKET
```

Each anonymous ID has:

```text
Mxxx/review_clip.mp4
Mxxx/review_frame_strip.png
```

The source videos were frame-count checked against recorded policy steps. Each
derived clip uses the sealed window `[max(0,t_emit-10),
min(policy_steps_executed-1,t_emit+14)]`, marks `T_EMIT`, and preserves the raw
clean video unchanged.

## Review rubric

`CONTACT_VALID=PASS` requires task-relevant gripper-dependent coupling at
`t_emit`, persistence through `[t_emit,t_emit+4]`, no already-safe release or
independent support, no pre-existing slip/lost contact, and sufficient visual
evidence. Handles/doors/fixtures use sustained gripper-dependent manipulation;
free-space lifting is not required.

`CONTACT_INVALID=FAIL` uses one frozen reason code. `CONTACT_AMBIGUOUS=ABSTAIN`
is used when the fixed material cannot support a defensible judgment. ABSTAIN
is not attack-eligible.

## Blank owner form

- CSV: `reports/STAGE_X_X1R_T1D1M0_MANUAL_REVIEW_FORM_V1.csv`
- JSON: `reports/STAGE_X_X1R_T1D1M0_MANUAL_REVIEW_FORM_V1.json`

Both forms contain 14 rows and no labels, reason codes, reviewer, timestamp, or
notes. Codex has not labeled any candidate.

## Audit and seal

- packet audit SHA256:
  `a444d60eefd2c75ff871e8e5c1577465ec60e28918e13c5df89968bbe2df5c89`
- SHA256SUMS SHA256:
  `26026321baa8d9dd3e54af29b472855ad56d287c63bc54f9fc9977e49d1371b9`
- root seal SHA256:
  `0722fa2dd1fc66db2b6744346aa1a407953d0f40f095fe1dbabf9213a5b31f40`

The packet audit confirms 14/14 candidate closure, all clean success, all
Student traces PASS, all first emits legal, D1 ordinals 1/11/20/30 absent,
fixed-copy/raw-video SHA matches, and blank forms. `Eval160=UNREAD` and
`protected_evaluation=UNREAD`; all D1R/D1M0 attack counters remain zero.

## Mandatory stop

Stop at `OWNER_MANUAL_CONTACT_LABELS_REQUIRED`. Do not ingest labels, form a
final attack-eligible set, or run CLEAN_EVAL/TRUE_PGD/RAND/SHUFFLED until the
owner returns the 14 manual labels and a new review authorizes label ingestion.
