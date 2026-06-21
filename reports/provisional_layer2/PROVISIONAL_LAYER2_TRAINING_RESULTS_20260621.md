# Provisional Layer2 Training Results 2026-06-21

Status: `PROVISIONAL_ENGINEERING_ONLY_NOT_FOR_CLAIMS`

This run uses provisional Layer1 labels and is not a replacement for H2 human-reviewed Teacher freeze. It is intended only to test the engineering path from provisional Layer1 labels to Layer2 MLP training/evaluation.

## Inputs

- Dataset root: `/data/liuyu/layer2_outputs/provisional_cross_suite_20260621/dataset_v1`
- Dataset SHA256: `104a26f026782be1cb7c5e1ef94fc19f12f519b3a201ba1a6aaa69b6d8dcaca0`
- Training output root: `/data/liuyu/layer2_outputs/provisional_cross_suite_20260621/mlp_matrix_v3`
- Source commit: `ee7b8e7f5900f48542265bd164b4856a0b65b5a3`
- Recursive manifest SHA256: `2bc572f4cb9ed4cd11a94cadfc6ed8b5726b2b40e9ec8f70b89e096e2458a646`

## Run Status

- Completed runs: 4
- Skipped runs: 2
- Skip class: `SKIPPED_NO_SUPERVISED_ROWS`

Skipped runs:

- `M1_in_domain_libero_10`: no supervised train/val/test rows for LIBERO-10.
- `M2_leave_one_suite_out_test_libero_10`: train/val rows exist from Spatial+Goal, but LIBERO-10 has no supervised test rows.

The skip is an explicit denominator result from provisional Layer1 coverage, not a hidden training failure.

## Primary Metrics

| Run | Status | Val F1 | Test F1 | Test Precision | Test Recall | Test AUROC |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `M1_in_domain_libero_spatial` | completed | 0.7879 | 0.8129 | 0.8750 | 0.7590 | 0.9752 |
| `M1_in_domain_libero_goal` | completed | 0.8000 | 0.7368 | 0.7447 | 0.7292 | 0.9779 |
| `M1_in_domain_libero_10` | skipped | - | - | - | - | - |
| `M2_leave_one_suite_out_test_libero_spatial` | completed | 0.8000 | 0.7333 | 0.8209 | 0.6627 | 0.9499 |
| `M2_leave_one_suite_out_test_libero_goal` | completed | 0.7879 | 0.7273 | 0.8000 | 0.6667 | 0.9438 |
| `M2_leave_one_suite_out_test_libero_10` | skipped | - | - | - | - | - |

## Allowed Claims

- The provisional Layer1 to Layer2 training pipeline runs end to end for trainable Spatial/Goal splits.
- The run records LIBERO-10 supervised-label absence as explicit skip artifacts.
- Thresholds are selected from validation predictions; test metrics are computed afterward.

## Forbidden Claims

- This does not grant H2.
- This does not validate final Teacher labels.
- This does not prove detector generalization to LIBERO-10.
- This does not authorize GPU/LIBERO/VIS/RAND/shuffled/attack execution.

