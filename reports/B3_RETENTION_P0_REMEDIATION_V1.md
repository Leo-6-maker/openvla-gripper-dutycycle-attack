# B3-Retention P0 remediation record

Status: `CPU_REVIEW_READY / FORMAL_TRAINING_HOLD / ATTACK_HOLD`

This record addresses the attached review findings for PR #75. It does not
change the active Official CLEAN collector, the CLEAN_2000 evidence root, or
any attack queue.

## Closed findings

1. The scheduler now implements a literal 2-of-3 gate. It retains the last
   three valid gate decisions for one event and emits when at least two are
   true. Event changes, invalid evidence, and release clear the candidate
   history. The persistence window and threshold are fixed by the protocol.
2. The scheduler exposes `trigger_started`, `attack_active`, `attack_index`,
   and `attacked_frames_emitted`. The trigger call is frame index 0; indices
   0 through 9 are active, and the next call transitions to `DONE`. This is
   an exact ten-frame contract rather than an inclusive end-step shortcut.
3. A release step returns `event_id = -1` and `event_active = false`. The
   released event is retained separately in `released_event_id`; the online
   tracker and offline rebuilder exclude the release step from the event span.
4. Raw OpenVLA gripper values and applied environment gripper values have
   separate extraction paths and conventions. If both are present, a
   disagreement is a hard error; an ambiguous fallback is not silently used.
   Official qpos and opening calculations are explicit and parity-checked.
5. `release_imminent` now masks unknown future evidence instead of treating
   missing/invalid evidence as a negative label.

## Formal offline materialization

`materialize_b3_retention_episode.py` performs a strict, sealed-episode join
of step records, policy-intent records, and privileged sidecar records. It
checks official identity, contiguous unique steps, finite 25D/9D student
inputs, source `artifact_sha256.json`, and source/config/rebuilder/materializer
hashes. It emits separate:

- `student_input_records.jsonl`: only `features_25d` and
  `clean_policy_intent_9d` plus immutable identity;
- `teacher_retention_records.jsonl`: event fields, four head targets, masks,
  and derived provenance;
- `retention_events.json` and `materialization_manifest.json`.

`audit_b3_retention_materialization.py` verifies output checksum closure,
identity/step alignment, field isolation, head presence, mask types, and the
explicit `formal_training_ready = false` / `formal_attack_ready = false`
holds.

## Test and CI coverage

The CPU suite covers:

- multi-event regrasp and release semantics;
- non-contiguous-step rejection;
- true 2-of-3 persistence;
- exact ten active frames;
- raw/env semantic mismatch rejection;
- official qpos/opening parity;
- four-head and mask materialization;
- strict materializer join, student/teacher separation, and checksum audit.

The existing Stage-B CPU workflow now compiles the B3 runtime/materializer and
runs the B3 tests together with the existing Stage-B tests.

## Remaining holds

These fixes make the preparation line auditable; they do not establish
detector effectiveness. The following remain fail-closed:

- old-label agreement and trajectory audit;
- label-distribution smoke, including per-suite and later-event L10 coverage;
- offline/online stateful parity;
- CHECK acceptance;
- the pre-registered 48-cell downstream canary;
- formal B3 training and all attack execution.

No result from this branch should be reported as an improved detector or as a
formal attack result before those gates pass.
