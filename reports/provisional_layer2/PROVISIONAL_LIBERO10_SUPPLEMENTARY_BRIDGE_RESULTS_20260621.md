# Provisional LIBERO-10 Supplementary Event Bridge Results

## Status

```text
TRACK_B_LIBERO10_SUPPLEMENTARY_EVENT_BRIDGE: ENGINEERING_PASS
PROVISIONAL_LAYER2_ENGINEERING_PASS: PASS
H2_PRIMARY_TEACHER_FREEZE: NOT_GRANTED
PAPER_CLAIMS: BLOCKED
VIS_RAND_SHUFFLED_ORACLE_ATTACK: NOT_RUN
```

This is an engineering bridge for provisional Layer 1 -> Layer 2 -> Layer 3 wiring. It does not modify the primary Layer 1 denominator:

```text
primary_positive_denominator = single_object_pick_place
supplementary_label_role = supplementary_multievent_grasp_carry_bridge
```

## Outputs

```text
LIBERO-10 bridge root:
/data/liuyu/layer1_outputs/provisional_libero10_event_bridge_20260621_r1

Combined Layer1 root:
/data/liuyu/layer1_outputs/provisional_layer1_plus_libero10_event_bridge_20260621_r1

Combined dataset v2 root:
/data/liuyu/layer2_outputs/provisional_cross_suite_20260621/dataset_v2_libero10_bridge_combined_r1

Layer2 v4 matrix root:
/data/liuyu/layer2_outputs/provisional_cross_suite_20260621/mlp_matrix_v4_libero10_bridge_cpu_r1
```

## Bridge Coverage

| split | episodes | supplementary eligible | no relevant event | target ambiguous | event count | multi-event episodes | positive task coverage |
|---|---:|---:|---:|---:|---:|---:|---:|
| train300_train_s10_17 | 80 | 33 | 39 | 8 | 47 | 14 | 6 |
| train300_val_s18_19 | 20 | 10 | 8 | 2 | 14 | 4 | 6 |
| clean300_test_s0_9 | 100 | 41 | 49 | 10 | 56 | 15 | 6 |

Coverage gate:

```text
planned = processed
missing = 0
extra = 0
resolver crashes = 0
schema errors = 0
train supervised episodes > 0
validation supervised episodes > 0
test supervised episodes > 0
train positive task coverage >= 2 task IDs
test positive task coverage >= 2 task IDs
```

## Dataset v2

```text
dataset_sha256 = b7a6d4bc4dd9106dba4f36e39c6e3058c7a43524f8f6fa84454594780eedecaf
frame_count = 139757
supervised_frame_count = 129446
model_input_columns_exactly_sc5_features = true
```

Frame counts by label role:

```text
primary_single_object_pick_place = 34536
supplementary_multievent_grasp_carry_bridge = 26099
negative_only = 68811
ignore = 10311
```

`task_success` is not present in the model-facing frame dataset. It is stored only in the evaluator sidecar.

## Layer2 v4 Matrix

All six runs completed; no v3 checkpoints were reused as v4 results.

| run | status | checkpoint SHA | test F1 | precision | recall | AUROC |
|---|---|---|---:|---:|---:|---:|
| M1_in_domain_libero_spatial | COMPLETED | deb730bc2c4a94a4763087a14afee234c6448596680c132329670c7fe4f88c07 | 0.7947 | 0.8824 | 0.7229 | 0.9754 |
| M1_in_domain_libero_goal | COMPLETED | 46be5d69f2e24184b3fa89c1c490716266f1fbea34fee6a141cbf1d5e96fc802 | 0.7865 | 0.8537 | 0.7292 | 0.9789 |
| M1_in_domain_libero_10 | COMPLETED | 687ef93d7d66b46aede654968708d91260b68bd56a4734dc0d1eaaff9db7a7c4 | 0.8571 | 0.7800 | 0.9512 | 0.9669 |
| M2_leave_one_suite_out_test_libero_spatial | COMPLETED | fd2f45f11f7ceca9c8744a9c9f74cddc7e68f01da2eb4838f05d6a802d7656ef | 0.5139 | 0.6066 | 0.4458 | 0.8756 |
| M2_leave_one_suite_out_test_libero_goal | COMPLETED | 2dd37f6eb68d697e6b0aecccf9f09135742917284c8ef176b65a84a0cb70da7b | 0.7234 | 0.7391 | 0.7083 | 0.9458 |
| M2_leave_one_suite_out_test_libero_10 | COMPLETED | e2778bf499a83dfbe0d16826a72a5fad4141fe391d9db3b80b8507795e109f8e | 0.4000 | 0.5417 | 0.3171 | 0.7566 |

## Allowed Claims

- The provisional LIBERO-10 supplementary bridge produced nonzero supervised train, validation, and test rows.
- Dataset v2 contains explicit primary/supplementary label-role fields and uses exactly `SC5_FEATURES` as model inputs.
- The provisional Layer2 v4 matrix reached terminal artifacts for all six M1/M2 runs and produced all three M2 checkpoints.
- The result is sufficient to unblock a three-suite Layer3 engineering smoke, subject to the separate Track C execution boundary.

## Forbidden Claims

- H2 primary Teacher labels are scientifically frozen.
- LIBERO-10 has been reclassified into the primary single-object denominator.
- Cross-suite detector generalization is scientifically confirmed.
- VIS is superior to RAND or shuffled controls.
- Any Layer3 attack effectiveness or physical task degradation has been established.
