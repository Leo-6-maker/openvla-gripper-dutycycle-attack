# Reproducing Milestone 2C

Milestone 2C trains a small proprio-only PyTorch model. It does not require MuJoCo, EGL, OpenVLA rollout, visual models, internet access, or attack outcomes.

## Training Command

CPU is the default and sufficient for this baseline:

```bash
python scripts/train_proprio_causal_student.py \
  --data_csv /data/liuyu/outputs/milestone_2b_parser_visual_linkage_20260526/tables/student_train_dataset.csv \
  --output_root /data/liuyu/outputs/milestone_2c_proprio_causal_student_20260526 \
  --split_mode task_id \
  --model mlp \
  --device cpu \
  --seed 1 \
  --epochs 50 \
  --batch_size 1024 \
  --lr 1e-3 \
  --weight_decay 1e-4 \
  --early_stop_patience 8 \
  --hazard_threshold 0.5 \
  --save_best
```

This training command should not create MuJoCo or EGL contexts and should not run OpenVLA rollouts.

## Validation Commands

```bash
python -m compileall scripts src tests
pytest tests/v4/test_clean_protocol.py \
       tests/v4/test_protocol_validation.py \
       tests/v4/test_contact_quality_v2.py \
       tests/v4/test_language_grounded_student_scaffold.py \
       tests/v4/test_student_replay_leakage.py \
       tests/v4/test_task_language_parser.py \
       tests/v4/test_episode_identity_export.py \
       tests/v4/test_visual_feature_linkage.py \
       tests/v4/test_proprio_student_dataset.py \
       tests/v4/test_proprio_student_training.py \
       tests/v4/test_proprio_student_leakage.py \
       tests/v4/test_proprio_student_splits.py
```

Previous freeze validation passed with 69 tests.

## Protocol Constraints

- No visual model is used.
- No detector-triggered attack rollout is run.
- No VIS, oracle, random, or manual outcomes are used.
- Teacher window fields are not model inputs.
- Identity fields such as `task_id`, `state_id`, `run_id`, and `episode_key` are used only for grouping and reporting.
- Object suite remains a clean reproducibility gap and must not be used as strong attack evidence.

## Boundary Statement

Milestone 2C trains and evaluates a proprio-only causal student baseline for detector development.
It does not use visual features.
It does not run detector-triggered attack rollouts.
It does not use attack/oracle/random/manual outcomes.
The trained model is not final attack evidence; it is a candidate online phase detector for offline validation.
