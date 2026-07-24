# C2F Counterfactual Teacher v2 Specification

Date: 2026-07-10

Status: `PASS_SPEC_ONLY`; replay and materialization are not authorized.

## TeacherLabelerV2 grounding

TeacherLabelerV2 is privileged and offline-only. None of its simulator fields may enter student inputs.

1. Resolve task targets from parsed BDDL/task metadata: task object identities, target receptacle/site, and ordered subgoals. Language matching is fallback-only and must record that fallback.
2. Identify held/contacted objects from MuJoCo contact geom pairs involving left/right gripper fingers. Canonicalize geom/body names to structured task object identities. Nearest-body distance alone is insufficient.
3. Track grasp persistence and object-relative lift from the contacted object's pose at grasp onset. Absolute EEF-z may be used only when relative object evidence is unavailable, with reduced confidence and an explicit reason code.
4. Define release safety from object-to-target/receptacle geometry and task progress. A closed-to-open gripper transition alone is not release-safe.

Required stable reason codes:

```text
PRIMARY_TARGET_CARRY
AUXILIARY_GRASP
DISTRACTOR_CARRY
NO_CONFIDENT_CONTACT_OBJECT
TARGET_ID_UNRESOLVED
PRE_LIFT_GRASP
RELEASE_NEAR_TARGET
RELEASE_AWAY_FROM_TARGET
APPROACH_OR_SETUP
```

Every row records `teacher_confidence` in `[0,1]`, `teacher_reason_code`, grounding source, contacted object, resolved target object/receptacle, relative-lift evidence, and release-distance evidence. Missing evidence produces abstention.

Candidate replay strata are not limited to stable carry: `CLOSE_ONSET`, `STABLE_GRASP`, `PERSISTENT_CONTACT`, `RELATIVE_OBJECT_MOTION`, `STABLE_CARRY`, `PRE_RELEASE`, plus a deterministic `RANDOM_NONCANDIDATE_AUDIT` sample for candidate-recall estimation.

## Counterfactual state protocol

For candidate clean steps `t`, freeze a simulator snapshot containing all state needed for deterministic restoration: qpos, qvel, actuator state, mocap/userdata, simulation time, environment task state, and relevant RNG state. Record a state hash and restoration check.

From the same snapshot compare two continuations for horizon `H`:

- clean continuation: the recorded clean raw actions;
- command-open continuation: identical non-gripper clean raw actions, with the gripper command forced open for exactly T10 delivered steps.

This is an offline privileged teacher comparison, not D7 image-space PGD. If exact restoration or action alignment fails, the window is `unknown`.

## Label definitions

```text
y_cmdopen_vulnerable
y_contact_loss
y_object_drop
y_progress_regression
y_success_flip
y_release_safe
teacher_confidence
teacher_reason_code
label_known_mask
```

- `y_contact_loss=1`: clean continuation preserves target-object contact/grasp while force-open loses it within `H`.
- `y_object_drop=1`: force-open causes a frozen object-height/contact degradation beyond the clean continuation tolerance.
- `y_progress_regression=1`: target-relative task progress worsens beyond a frozen margin versus clean continuation.
- `y_success_flip=1`: full clean continuation succeeds and the matched force-open continuation fails.
- `y_release_safe=1`: target-relative placement evidence says opening is benign/required; this vetoes vulnerability.
- `y_cmdopen_vulnerable=1`: at least one high-confidence causal harm label is positive, clean progress/contact is preserved, and release-safe is false.
- `y_cmdopen_vulnerable=0`: the window was replayed validly, evidence remained within frozen benign tolerances, and release-safe/ambiguity rules are satisfied.
- unknown/unreplayed/ambiguous windows use `label_known_mask=0`; they are never converted to negatives.

Suggested counterfactual reason codes include `CONTACT_LOSS_AFTER_CMDOPEN`, `OBJECT_DROP_AFTER_CMDOPEN`, `PROGRESS_REGRESSION_AFTER_CMDOPEN`, `SUCCESS_FLIP_AFTER_CMDOPEN`, `RELEASE_SAFE_COUNTERFACTUAL`, `RESTORE_MISMATCH`, `ACTION_ALIGNMENT_FAILED`, `AMBIGUOUS_EFFECT`, and `NOT_REPLAYED`.

## Confidence and acceptance

Confidence combines restoration parity, target grounding confidence, contact identity confidence, clean/counterfactual action alignment, and margin from effect thresholds. All thresholds, horizon, T10 semantics, simulator/code commit, and candidate-filter reason are frozen in the replay manifest.

Before any large replay, a separately authorized smoke must verify deterministic restore, clean-continuation parity, exact T10 delivery, and unknown masking. Current status:

```text
COUNTERFACTUAL_TEACHER_SPEC = PASS
COUNTERFACTUAL_MANIFEST_SCHEMA = PASS_STATIC
COUNTERFACTUAL_REPLAY_SMOKE = NOT_STARTED
GPU_REPLAY_JOBS = 0
```
