# STAGE X X1R T1-D0R2 — Clean Runtime Authority Closure

Status: `STAGE_X_X1R_T1D0R2_HOLD_STUDENT_REPLAY_PARITY`

Runtime source before evidence: `836b6ff65edee32dae2688567de83790083da6d6` / `13425a44d1c914b83dacfaad78852a0193cafbc1`.

This is a static/CPU historical-replay authority audit only. It did not load OpenVLA, use a GPU, reset a simulator, call an environment step, materialize a fresh parent, run PGD, read V_phys, read Eval160, or read protected evaluation.

The historical Student training source remains `NOT_IDENTIFIABLE`: T1 handoff, T1 receipt runtime identity, and current server file are distinct provenance statements. The prospective implementation is the tracked PR127 source bound by raw bytes and Git blob.

Student replay parity: `HOLD_STUDENT_REPLAY_PARITY`; sealed per-step reference available: `False`. The T1-C receipt is summary-only, so deterministic repeat/prefix checks are diagnostic and cannot be promoted to historical per-step parity.

Success/horizon authority: `PASS_SUCCESS_AND_HORIZON_BINDING`. The canonical evaluator is the immutable upstream OpenVLA LIBERO evaluator; `done` is consumed after `env.step`, and LIBERO's domain step derives it from `_check_success()`. Policy horizons are 520/300/280/220 for L10/goal/object/spatial; the ten dummy wait steps are outside the policy-decision horizon.

D0R1 population/seed invariance: `PASS_D0R1_INVARIANTS`; 1200 G10 -> 990 exclusion union -> 210 fresh -> 40 nominal cells -> 39 executable parents; missing cell `libero_goal/task_01`; replacement false.

Authorization remains closed: `openvla_model_inference_authorized=false`, `clean_parent_materialization_authorized=false`, `env_step_authorized=false`, `pgd_authorized=false`, `physical_intervention_authorized=false`, `attack_outcome_authorized=false`, `protected_authorized=false`. Next gate remains `CLEAN_PARENT_MATERIALIZATION_REVIEW_REQUIRED`.

Frozen scientific claim: `STUDENT_HELDOUT_GENERALIZATION_NOT_ESTABLISHED`.
