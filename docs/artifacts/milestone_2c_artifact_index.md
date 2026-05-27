# Milestone 2C Artifact Index

This page indexes the external Milestone 2C artifact package. The package is stored outside git and should not be committed to the repository.

## External Artifact

Output root:

```bash
/data/liuyu/outputs/milestone_2c_proprio_causal_student_20260526
```

Export package:

```bash
/data/liuyu/outputs/milestone_2c_proprio_causal_student_20260526/export/milestone_2c_proprio_causal_student_20260526.tar.gz
```

SHA256:

```text
2ebc109ebc384e4f59fc5ace2a6fc0e63414d93ecdcc2c8700e5d6c6381d5a68
```

## Key Files in the Artifact Package

- `checkpoints/best_model.pt`
- `checkpoints/last_model.pt`
- `configs/train_config.json`
- `tables/train_epoch_metrics.csv`
- `tables/test_predictions.csv`
- `tables/test_metrics_overall.csv`
- `tables/test_metrics_by_suite.csv`
- `tables/test_metrics_by_mechanism.csv`
- `tables/threshold_sweep.csv`
- `reports/TRAINING_SUMMARY.md`
- `reports/EVAL_SUMMARY.md`
- `reports/NEXT_ACTION_STATUS.md`

## SHA256 Verification

Use:

```bash
echo "2ebc109ebc384e4f59fc5ace2a6fc0e63414d93ecdcc2c8700e5d6c6381d5a68  /data/liuyu/outputs/milestone_2c_proprio_causal_student_20260526/export/milestone_2c_proprio_causal_student_20260526.tar.gz" | sha256sum -c -
```

If a checksum file is available beside the export package, use:

```bash
sha256sum -c /data/liuyu/outputs/milestone_2c_proprio_causal_student_20260526/export/milestone_2c_proprio_causal_student_20260526.tar.gz.sha256
```

## Git Boundary

Do not commit the export package, checkpoints, rollout directories, videos, or other generated output artifacts to git. Only source, tests, and documentation belong in the repository.
