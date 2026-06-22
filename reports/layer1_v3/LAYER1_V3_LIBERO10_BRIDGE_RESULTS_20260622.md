# Full Layer1 Dataset v3 Gate v2 - 20260622

Status: PHASE_D_PASS_OWNER_AI_PROTOCOL

Root: `/data/liuyu/layer1_outputs/frozen_owner_ai_v2_libero10_bridge_20260622/full_layer1_dataset_v3_e2e21c7`
Dataset root: `/data/liuyu/layer2_outputs/frozen_owner_ai_v2_libero10_bridge_20260622/cross_suite_layer2_dataset_v3_owner_ai_frozen_e2e21c7`
Repo commit: `e2e21c79450b34bb779f6e9f3cfb7389c9597ccb`

## Gate Results

- train/val/held_out processed: PASS
- resolver failures: 0
- validation/schema errors: 0
- duplicate/missing/extra: not detected by copied frozen manifests; manifest row counts match 240/60/300
- Spatial/Goal semantic drift vs a7a7188: 0 rows
- model_input_columns_exactly_sc5_features: true
- task_success_not_in_model_rows: true (present only in evaluator sidecar)
- dataset leakage audit: PASS

## Counts

See `tables/full_layer1_dataset_v3_gate_v2_summary.csv` and `tables/full_layer1_status_by_suite_task_split.csv`.

## Claim Boundary

Allowed: owner-AI protocol Layer1 dataset v3 has nonzero supervised Spatial, Goal and LIBERO-10 supplementary rows, suitable for provisional Layer2 engineering.

Forbidden: H2 scientifically frozen, final Teacher ground truth, detector generalization proof, or VIS/RAND attack effectiveness.
