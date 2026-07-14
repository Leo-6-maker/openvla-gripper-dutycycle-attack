# Official CLEAN detector-data contract

This contract belongs to the official OpenVLA/LIBERO V1 collector. It is
separate from the legacy state-0--9 and manual-`generate()` runners.

## What one CLEAN episode writes

`episode_metadata.json` binds the identity and provenance:

- `protocol_id`, `suite`, `task_idx`, `task_name`, `task_language`, `state_id`
- `canonical_parent_key`, `split`, `initial_state_sha256`
- `official_horizon`, `max_steps`, `num_steps_wait`
- `runtime_valid`, `env_success`, `success`
- model/checkpoint/processor/unnormalization provenance and adapter hashes
- `feature_names_25d` and `policy_intent_feature_names_9d` in frozen order
- `student_allowed_modalities` and `student_forbidden_modalities`
- `teacher_labels_materialized` and `teacher_label_source`

`step_records.jsonl` contains, per causal step:

- identity: `step`, `suite`, `task_idx`, `state_id`, `condition`
- detector inputs: finite `features_25d` and finite `clean_policy_intent_9d`
- action evidence: `clean_action_raw_7d`, `applied_action_7d`,
  `action_token_ids`, top-token ids/logits, and score-adapter parity
- official execution evidence: raw action, postprocessed environment action,
  prompt, and the max continuous action discrepancy

`policy_intent_records.jsonl` repeats the 9D policy vector and compact token
evidence in a detector-focused stream. `privileged_teacher_sidecar.jsonl`
contains simulator-only evidence such as end-effector pose, gripper qpos,
object state, and MuJoCo contact pairs. It is audit/label input only and is
forbidden as a deployed student input.

The bundle also contains `episode_summary.json`, `runtime_audit.json`,
`condition_config.json`, `attack_config.json`, `step_records.jsonl`, and
`artifact_sha256.json`.

## Detector retraining boundary

The V2 schema is sufficient to materialize the detector's student inputs:

- 25D causal proprio/action stream, in the existing frozen SC5 order;
- 9D clean gripper policy-intent stream, derived from the official score
  adapter's clean gripper logits;
- task identity/language and parent-level provenance.

It is not, by itself, a completed detector-training result. Teacher labels must
still be generated in a separate offline step from the privileged sidecar,
then split by the pre-registered FIT/CAL/CHECK identities. The collector writes
`teacher_labels_materialized=false` until that step exists. No attack outcome
may enter either the student inputs or teacher-label generation.

The collector intentionally stores compact top-token evidence, not the full
vocabulary logits or RGB frames. That is enough for the frozen 9D feature
definition, but a future detector that needs arbitrary logit reprocessing or
visual inputs must add a separately versioned capture contract before running.

Use `scripts/audit_official_clean_schema.py` before detector materialization;
it fails closed on missing vectors, changed feature order, or incomplete step
records.
