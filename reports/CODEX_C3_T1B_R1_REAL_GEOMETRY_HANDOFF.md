# C3-T1B-R1 Real Geometry and V23 Runner Handoff

Date: 2026-07-27
Branch: codex/detector-completion-20260726
Code snapshot: a10b9b155e8c2ef9c6135736a636e9a2feabc072

## Scope

This handoff covers the authorized FIT-development pilot only. The geometry
source was deterministic MuJoCo replay from the sealed 40-episode pilot input;
no OpenVLA, policy inference, Student training, Teacher relabeling, rollout, or
attack was run.

## Geometry evidence

| Evidence | Result |
|---|---|
| Run A | /mnt/sdc/dty_user/openvla_attack_outputs/n5/phase3_student/c3_t1_real_geometry_run_A_a10b9b1_20260727 |
| Run B | /mnt/sdc/dty_user/openvla_attack_outputs/n5/phase3_student/c3_t1_real_geometry_run_B_r1_a10b9b1_20260727 |
| A/B SHA256SUMS SHA | 2f54462be1df71e022be7d60cdd3a9dac0af3d74410d6ee8d9e3a4c976b1f354 |
| A/B comparison | PASS; 40 episodes, 9422 steps, canonical differences 0 |
| Comparison root SHA256SUMS SHA | c955a2b6d3b31c33fa7030919f929f866dc3e34222047593b79bb73f14bbc348 |
| Relation-count values | 0, 1, 2 |
| Articulated unsupported rows | 2 episodes retained as empty-relation UNKNOWN geometry |
| Geometry source commit | a10b9b155e8c2ef9c6135736a636e9a2feabc072 |
| Model inference / Teacher labeling / attack | 0 / 0 / 0 |

Run A and Run B were independent processes with disjoint output roots. The
comparison was performed by n5/phase2_labels/compare_v23_geometry_runs.py,
which independently verifies recursive file closure before canonical row
comparison.

The replay audit also records nonzero source-side diagnostic maxima:

    qpos_sidecar_max_abs_error = 0.01582322290913525
    eef_feature_sidecar_max_abs_error = 0.028513823662753762

These fields are reported, not hidden. They are not the frozen C3-S3 object
pose replay thresholds and no claim of zero numerical parity is made here.

## Runner evidence

| Evidence | Result |
|---|---|
| Four-suite smoke | PASS; one deterministic episode per suite |
| Smoke root SHA256SUMS SHA | 416f1f7475cf2ee75c533f6e28a2f71280114a9c53075719fb281847bb5f4e26 |
| Full V23 pilot root | /mnt/sdc/dty_user/openvla_attack_outputs/n5/phase3_student/c3_t1_v23_dev_pilot_a10b9b1_20260727 |
| Full pilot SHA256SUMS SHA | 1dbf459e0767c277def8ceeb8f40bd8193dfabdf653adcbc79678bc605bb329a |
| Full output manifest SHA | f71d2deeed246c2a752070e4d140d5a3cdb77733cc0eee327d4b3a5c3382e1c7 |
| Episodes / unique identities | 40 / 40 |
| Suite counts | libero_10=10, libero_goal=10, libero_object=10, libero_spatial=10 |
| Steps | 9422 |
| Unknown-to-FALSE | 0 |
| Forbidden runtime keys in emitted JSONL | 0 |
| Output recursive seal and file closure | PASS |

The runner uses the C3-G V1.1 predicate contract, preserves multi-relation
geometry, treats the two no-relation articulated cases as UNKNOWN, applies the
per-head persistence contract, and uses observed future length only for
right-censoring. K10 uses the frozen suite protocol horizon.

## Current gate interpretation

    C3-T1A-R2 CONTRACT                 = PASS (previously sealed)
    C3-T1B-R1 GEOMETRY REPLAY          = PASS AS FIT-ONLY EVIDENCE
    C3-T1C FOUR-CASE RUNNER SMOKE      = PASS
    C3-T1D 40-EPISODE DEV PILOT        = PASS AS DEVELOPMENT PILOT
    NUMERICAL ZERO-PARITY CLAIM        = NOT MADE
    PROTECTED READS                    = 0
    OPENVLA / STUDENT / TRAINING       = NOT STARTED
    ROLLOUT / ATTACK                   = NOT STARTED

The nonzero qpos/EEF diagnostic maxima remain an explicit follow-up for any
future stricter numerical-parity claim. They do not alter the clean-only
scope of this pilot and must not be silently converted to zero.

## Code changes

    n5/phase2_labels/replay_v23_real_geometry.py
    n5/phase2_labels/run_v23_dev_pilot.py
    n5/phase2_labels/compare_v23_geometry_runs.py
    n5/phase2_labels/tests/test_v23_dev_pilot_runner.py
    configs/C3_T0_TEACHER_SEMANTIC_CONTRACT_V1.json

The latest branch commit is a10b9b155e8c2ef9c6135736a636e9a2feabc072.
The runner output was executed from the preceding runner snapshot
61b183839102ee14c92c06b3e2afb3932f3784c6; the intervening a10b9b1 change
is replay-bound geometry caching only and does not change runner semantics.
Local contract tests: 19 passed, 0 failed, 0 errors.

## Boundary

This is not authorization for Clean2000 relabeling, Student training,
protected-split access, OpenVLA rollout, or attack. All subsequent gates
remain separately controlled.
