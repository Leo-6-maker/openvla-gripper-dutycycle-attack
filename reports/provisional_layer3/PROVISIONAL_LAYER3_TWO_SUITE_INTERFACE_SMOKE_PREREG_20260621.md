# Provisional Layer3 Two-Suite Interface Smoke Preregistration

Status: `PROVISIONAL_LAYER3_TWO_SUITE_INTERFACE_SMOKE`

This is an engineering-only wiring smoke for Spatial and Goal. It is not a three-suite smoke, not a Layer123 mainline pass, and not attack-effectiveness evidence.

## Upstream Status

- `LAYER2_PROVISIONAL_MATRIX_V3 = PARTIAL_ENGINEERING_PASS`
- `PROVISIONAL_LAYER2_ENGINEERING_PASS = NOT_GRANTED`
- `SPATIAL_GOAL_TRANSFER_PATH = ENGINEERING_PASS`
- `LIBERO10_SUPERVISED_DENOMINATOR = STRUCTURALLY_MISSING`
- `APPROVE_GATE_H2 = NOT_GRANTED`
- `PAPER_CLAIMS = BLOCKED`

## Parent Selection

Source:

- `tables/layer1_h2_20260620/h2_diagnostic_review_round_v2_1_form_template.csv`
- Source SHA256: `0340c6aa37bcbe9a92239f3c40c8cd7dbcb4a31e8dbf0552284295a3fbeac3df`

Allowed strata:

- `DEV_CANARY`
- `DIAGNOSTIC_HOLDOUT`

Allowed suites:

- `libero_spatial`
- `libero_goal`

Selection rule:

```text
sort by SHA256(review_round_id | canonical_episode_key)
take first two ELIGIBLE_EVENT parents per suite
```

Forbidden selection inputs include Student emit/probability, attack outputs, visual ease, task success, and human-review judgments.

Frozen parent manifest:

- `reports/provisional_layer3/two_suite_parent_manifest_20260621/provisional_layer3_two_suite_parent_manifest.csv`
- `reports/provisional_layer3/two_suite_parent_manifest_20260621/provisional_layer3_two_suite_parent_manifest_audit.json`

Selected parents:

| Suite | Review ID | Episode Key | Anchor | Detector |
| --- | --- | --- | ---: | --- |
| `libero_spatial` | `v2_dev_002_event_00` | `libero_spatial|3|4|0|CLEAN` | 50 | `M2_leave_one_suite_out_test_libero_spatial` |
| `libero_spatial` | `review_023_event_00` | `libero_spatial|3|1|0|CLEAN` | 49 | `M2_leave_one_suite_out_test_libero_spatial` |
| `libero_goal` | `review_016_event_00` | `libero_goal|2|8|0|CLEAN` | 290 | `M2_leave_one_suite_out_test_libero_goal` |
| `libero_goal` | `v2_dev_004_event_00` | `libero_goal|1|4|0|CLEAN` | 42 | `M2_leave_one_suite_out_test_libero_goal` |

`review_016` is retained because metadata-only selection is binding.

## Detectors

Spatial parent detector:

- Run: `M2_leave_one_suite_out_test_libero_spatial`
- Checkpoint SHA256: `f0ff9acdc77d1ca000214dae5d2758ba6474d3748248078cf99d1bdc79195da0`
- Supervised source: Goal only

Goal parent detector:

- Run: `M2_leave_one_suite_out_test_libero_goal`
- Checkpoint SHA256: `d98256ea6c29f5aed4d96b58d0f5a9497358de54de6633aa52fe944828067994`
- Supervised source: Spatial only

LIBERO-10 contributed zero supervised rows to both detectors.

## Planned Conditions

For each frozen parent:

```text
CLEAN
VIS
RAND
SHUFFLED
```

Maximum planned rollouts:

```text
4 parents x 4 conditions = 16
```

No oracle is authorized.

## Frozen Attack Configuration

```text
K = 10
epsilon = 6/255
PGD steps = 20
objective = autoregressive_prefix_gripper_target_token_logratio_arm_v3
target token = 31744
mode = CLIP_MEDIATED_OPEN
arm gate = 5/6
```

Attack remains detector-triggered only, one-shot, gripper override only, with arm action preserved. No manual trigger fallback and no parent replacement are allowed. No-emit cases remain in the denominator.

## Acceptance Boundary

`TWO_SUITE_LAYER3_INTERFACE_SMOKE = ENGINEERING_PASS` requires planned-key reconciliation, no duplicates, no output collision, matched conditions, Student-only trigger source, arm preservation, retained no-emit cases, decodable videos, and matching telemetry lengths.

No minimum attack success is required. Poor or zero effect is valid and must not trigger detector tuning.

## Forbidden Claims

- This is not a three-suite smoke.
- This is not `PROVISIONAL_LAYER123_MAINLINE`.
- This does not grant H2.
- This does not prove cross-suite detector generalization.
- This does not prove VIS superiority over controls.
- This does not establish attack effectiveness.

