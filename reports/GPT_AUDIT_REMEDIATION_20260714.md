# GPT audit remediation — Official-aligned CLEAN worker

Date: 2026-07-14
Repository: `Leo-6-maker/openvla-gripper-dutycycle-attack`
PR: #74 (`agent/official-openvla-clean-schema-review`)

## Scope

This patch addresses the code-level findings in the GPT review. It updates the
GitHub review branch only. The active server-side V2 workers and their evidence
root were not stopped or modified by this patch.

## Follow-up audit disposition — 2026-07-15

The follow-up review correctly found that the first remediation still had four
gaps. They are addressed by the follow-up code change on this branch:

- Runtime-invalid metadata now contains explicit `runtime_valid=false`,
  `success=false`, identity, schema, and protocol fields, so restart recovery
  reaches `RUNTIME_HOLD` without granting another formal attempt.
- The parity harness now compares an observational wrapper that preserves the
  original `model.predict_action()` generation kwargs against the
  score-capturing single-generation path. It also resets LIBERO before state
  injection and checks raw plus postprocessed action equality.
- A checksum-valid artifact that is incompatible with the current schema is
  now `PROTOCOL_HOLD` and cannot be overwritten in place.
- The worker now verifies the declared OpenVLA and LIBERO checkout paths and
  their actual Git HEADs, in addition to checkpoint and collector hashes.

The remaining audit boundary is intentional: the schema script is a
per-episode audit, not the global 2,000-cell census/ledger closure. The global
census remains a separate formal gate and the new evidence line remains HOLD
until the CPU checks and GPU canary are run in the pinned environment.

## Fixed findings

| Finding | Remediation | Gate |
|---|---|---|
| `set_init_state` without a preceding reset | CLEAN now calls `env.reset()` after seeding and before injecting the frozen initial state; the event is recorded in metadata and runtime audit | `env_reset_called=true` |
| Recovery could turn a failed artifact into PASS | Recovery validates the sealed artifact, reads the persisted `success` value, preserves `TASK_FAILURE`, and preserves `RUNTIME_INVALID` as `RUNTIME_HOLD` | no unconditional recovery PASS |
| Checkpoint binding was declaration-only | Worker verifies resolved checkpoint path, full tree SHA, file count, byte count, `config.json`, `dataset_statistics.json`, processor config/tokenizer SHAs, and collector source SHAs before model load | `checkpoint_binding_pass=true` |
| CLEAN used two generation passes | `predict_action_with_scores()` instruments the official `model.predict_action()` call, returns the captured sequence to that call, and decodes scores/tokens from the same generation | `generation_passes_per_step=1` and action max error `<=1e-6` |
| Gripper zero-boundary mismatch | Removed the non-official zero-to-`+1` rewrite; postprocessing now matches upstream `normalize_gripper_action` followed by `invert_gripper_action` | exact upstream boundary behavior |
| Schema audit was permissive | Audit now requires artifact files and checksum closure, runtime/success semantics, identity and official horizon, reset/provenance/single-generation gates, exact seven action tokens, parity flags, and absence of privileged fields from student records | `OPENVLA_OFFICIAL_CLEAN_SCHEMA_AUDIT_V2` |

## Deliberate holds

- The task-balanced `5 Clean-Success states per task` rule remains fail-closed;
  this patch does not add cross-task substitution or reuse FIT/CAL/CHECK
  identities.
- RAND_VALID, detector training/check, attack canary, and formal attacks remain
  separate protocol gates. No attack condition is enabled by this CLEAN patch.
- Existing artifacts produced by the pre-fix worker are not silently promoted
  to the corrected schema. They must pass the new audit or remain legacy/hold.
- A new formal run must regenerate `UPSTREAM_PROVENANCE.json` with the updated
  collector source SHAs before the worker can start; a stale provenance file is
  intentionally rejected.

## Verification performed

Passed locally:

- Python bytecode compilation for all four modified Python files.
- Python AST parsing for all four modified Python files.
- `git diff --check`.

Not run in this Windows checkout:

- GPU/MuJoCo rollout.
- Official action parity rollout.
- Full repository pytest suite; the local checkout does not have `pytest` or
  the project numerical runtime installed.

The PR therefore requests GPT review of the implementation and protocol gates;
it does not claim that the active server evidence has already been regenerated
under this branch.
