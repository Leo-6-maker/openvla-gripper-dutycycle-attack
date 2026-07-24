# C2g Teacher-v2 Static Readiness

Date: 2026-07-10

Status: `PASS_STATIC`; no replay or label materialization was performed.

## Structured target resolution

`c2g_teacher_v2_target_resolution.py` accepts already-parsed task/BDDL metadata. Resolution order is frozen:

1. structured goal/task metadata;
2. structured BDDL goal predicates;
3. explicit task target metadata;
4. task-language fallback.

The result preserves object instance IDs, multiple targets, receptacles, sites, and ordered subgoals. Structured/language conflict is recorded rather than hidden. Language fallback lowers confidence. Missing or ambiguous targets return an explicit reason and never select the first declaration by list order.

## Contact identity

`c2g_teacher_v2_contact_identity.py` canonicalizes MuJoCo body/geom names while retaining instance identity. It strips link, visual, collision, geom, mesh, and shape suffixes; maps composite components to declared objects; distinguishes left/right fingers; excludes robot/static/receptacle contacts; and reports unilateral, multi-object, or canonical-mapping ambiguity. Raw and canonical contact pairs remain in provenance.

## Frozen Teacher-v2 row schema

Schema version: `c2g.teacher_v2.2026-07-10.v1`.

The validator enforces finite confidence, explicit known masks, null labels for unknown rows, causal-harm evidence for vulnerability positives, release-safe veto, reason/field consistency, and teacher-only field exclusion from student features. Unresolved grounding, restore mismatch, snapshot failure, failed action alignment, and unreplayed rows cannot become known negatives.

Primary grounding reason codes:

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

Counterfactual reason codes include contact loss, object drop, progress regression, success flip, release-safe, restore/action failure, ambiguity, not replayed, target-grounding failure, and incomplete snapshots.

## Candidate strata

The frozen candidate strata are:

```text
CLOSE_ONSET
STABLE_GRASP
PERSISTENT_CONTACT
RELATIVE_OBJECT_MOTION
STABLE_CARRY
PRE_RELEASE
RANDOM_NONCANDIDATE_AUDIT
```

Every candidate records the selection reason, stratum, sampling probability, deterministic seed, privileged-selection flag, and random noncandidate recall-audit flag. Stable-carry-only replay is prohibited.

## Static gates

```text
STRUCTURED_TARGET_RESOLUTION = PASS_STATIC
CONTACT_CANONICALIZATION = PASS_STATIC
TEACHER_V2_ROW_SCHEMA = PASS_STATIC
CANDIDATE_STRATA_SCHEMA = PASS_STATIC
TEACHER_V2_LABEL_MATERIALIZATION = NOT_RUN
COUNTERFACTUAL_REPLAY = NOT_RUN
```
