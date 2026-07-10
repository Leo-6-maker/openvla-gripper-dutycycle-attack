# C2g P0 Static Patch — 2026-07-10

## Boundary

This branch contains repository-side static corrections only.

- Base: `e5ce916f0b7ddbfbedeac6aa5e99b60bedc771ad`
- Branch: `assistant/c2g-p0-static-patch-20260710`
- No GPU episode was launched.
- No LIBERO environment was created.
- No OpenVLA inference was run.
- No counterfactual replay, dataset materialization, detector training, Goal/Object/Spatial expansion, or D7 parity was run.
- D7 Table1 remains frozen.

## P0 corrections

### Contiguous persistence and triggerability

- `_persistent_score` no longer joins non-contiguous eligible points through a global top-k fallback.
- Positive episodes are explicitly split into triggerable and untriggerable under the frozen 2-of-3 policy.
- Dataset fold coverage now records positive intervals, persistent-positive windows, triggerable attackable episodes, and untriggerable positive episodes.
- Sequence-level losses require sequence outputs explicitly; patch attention now accepts a validity mask.

### Causal-label provenance

Teacher-v2 schema was bumped to `c2g.teacher_v2.2026-07-10.v2`.

- `GROUNDING_ONLY` rows may provide grounding/contact auxiliary supervision but cannot create known vulnerability labels.
- Known causal labels require a counterfactual manifest SHA, comparison tier, replay-valid flag, and frozen attack protocol.
- Known causal negatives require all harm outcomes to be explicitly false and use `NO_MATERIAL_HARM_AFTER_CMDOPEN` or `RELEASE_SAFE_COUNTERFACTUAL`.
- Teacher-only feature checks now include prefix-based denial.

### Counterfactual manifest

Manifest schema was bumped to `c2g.counterfactual_manifest.2026-07-10.v2`.

- Required snapshot components include simulator, controller, wrapper, termination, environment RNG, and policy RNG state.
- Per-component snapshot/restore hashes are required.
- Restore parity is derived from frozen metrics and thresholds.
- Causal replay is bound to exact T10, raw open `+1.0`, environment open `-1.0`, and `C2G_CMDOPEN_CAUSAL_REPLAY/2026-07-10.v1`.
- Effect-threshold keys are mandatory.

### Target and contact semantics

- Target resolution now uses operator-specific argument roles, including reversed `contains(receptacle, object)` semantics.
- Non-placement operators such as open/close/toggle/press/push/pull/slide/rotate/grasp/hold/lift/move/pour are represented.
- Direct targets are validated against declarations and output ordering is deterministic.
- Contact identity is role-aware: static receptacles remain excluded, while explicitly manipulable receptacles/fixtures can be attack targets.
- Additional finger-name patterns and optional model-derived aliases are supported.

### Closed-world Track A audit

- Duplicate expected jobs and duplicate parent entries are fatal.
- Unexpected metadata or step-record artifacts are fatal.
- Completion requires exact equality between expected and actual job-key sets.

## Local static validation

Executed in an isolated local source tree containing the patched modules:

```bash
python -m unittest \
  tests.test_c2g_p0_patch \
  tests.test_c2g_teacher_v2_target_resolution \
  tests.test_c2g_teacher_v2_contact_identity \
  tests.test_c2g_teacher_v2_schema \
  tests.test_c2g_counterfactual_manifest
```

Result:

```text
Ran 41 tests
OK
```

`py_compile` passed for the seven patched modules and five corresponding test modules.

This was not the server-mounted full test suite. The branch still requires Codex/server validation of all existing tests and a remote diff review before any replay or GPU authorization.

## Static gate status

```text
PERSISTENCE_CONTIGUITY                 = PASS_LOCAL_STATIC
UNTRIGGERABLE_POSITIVE_ACCOUNTING      = PASS_LOCAL_STATIC
PATCH_TOKEN_MASK                       = PASS_LOCAL_STATIC
SEQUENCE_LOSS_CONTRACT                 = PASS_LOCAL_STATIC
CAUSAL_LABEL_REPLAY_BINDING            = PASS_LOCAL_STATIC
KNOWN_NEGATIVE_COMPLETENESS            = PASS_LOCAL_STATIC
FULL_SNAPSHOT_CONTRACT                 = PASS_LOCAL_STATIC
T10_PROTOCOL_BINDING                   = PASS_LOCAL_STATIC
NONPLACEMENT_TARGET_RESOLUTION         = PASS_STATIC_SUBSET
ROLE_AWARE_CONTACT_IDENTITY            = PASS_SYNTHETIC
CLOSED_WORLD_RUN_AUDIT                 = PASS_LOCAL_STATIC
TRIGGERABLE_FOLD_VIABILITY             = PASS_LOCAL_STATIC
SERVER_FULL_TEST_SUITE                 = NOT_RUN
LIVE_BDDL_OPERATOR_CENSUS              = NOT_RUN
LIVE_MUJOCO_GEOM_CENSUS                = NOT_RUN
DETERMINISTIC_RESTORE_SMOKE            = NOT_RUN
COUNTERFACTUAL_REPLAY                   = NOT_RUN
C2G_TRAINING                            = NOT_RUN
GPU_EPISODES_LAUNCHED                  = 0
```

## Required next review

Codex should fetch this branch and run the full CPU test suite, `py_compile`, and Bash syntax checks on the server. It must additionally audit real LIBERO BDDL operator names and robot geom names without launching rollouts. No replay, materialization, training, or GPU job is authorized by this static patch.
