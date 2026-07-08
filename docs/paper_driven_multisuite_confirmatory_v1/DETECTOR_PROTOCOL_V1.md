# Detector Protocol V1

Status: PLANNING_ONLY

This phase runs detector-only experiments. It does not run attacks.

## Main Model Freeze

The main detector is fixed to the current repository production path:

| Field | Value |
|---|---|
| main_model_name | `SC5MLPV1` |
| source_git_sha | `2525779bdb4d5f4f48b96f3d784550df3ad1bf27` |
| source_path | `src/gripper_attack/sc5mlp_v1.py` |
| source_sha256 | `0cb0ffd08290a058cbba289d87128662cb7c90568e0e056c48340d69a2d5de3a` |
| train_path | `tools/multisuite_detector/train_detector.py` |
| train_sha256 | `12019f05aabcabfc372e4e99864041c84c575c0b9cd5326c01a78a1827fc2539` |
| eval_path | `tools/multisuite_detector/evaluate_detector.py` |
| eval_sha256 | `f63d9d3240f445e7128f1bee13814b4479fc6dff3288a7408d8e54233ba628b5` |
| feature_order | `SC5_FEATURES`, 25 canonical features |
| history construction | causal features from current and past telemetry only |
| hidden dimensions | 64, 64 |
| heads | phase(9), corridor(1), release(1) |
| loss functions | phase CE + 0.5 corridor BCE(pos_weight=5.0) + 0.3 release BCE |
| class weighting | corridor positive weight 5.0; release unweighted |
| normalization | train-only mean/std |
| threshold | `tau_corridor=0.3`, `tau_release=0.3` |
| guard duration | 5 steps |
| checkpoint-selection rule | validation loss or validation suite-macro event F1, preselected before training |
| FSM | `legacy_v1` |

TCN or revocable FSM variants are ablations only; they cannot replace the main
model after results are inspected.

## Baselines

- fixed normalized-time;
- close-onset heuristic;
- rule-based proprio;
- logistic regression or shallow MLP;
- one other lightweight temporal model;
- privileged teacher upper bound for offline evaluation only.

## Training Regimes

```text
Object-only x 3 seeds
Pooled x 3 seeds
LOSO x 4 held-out suites x 3 seeds
Object leave-one-task-out x 10 folds x 3 seeds
```

Total: 48 lightweight detector runs, no OpenVLA rollout.

## Threshold Rule

Thresholds are selected on validation only, for example: maximize event F1
subject to false-trigger rate <= 0.20. Object-only zero-shot uses no
suite-specific threshold.

## Gate C

```text
Object held-out event recall >= 0.70
Object held-out +/-10 recall >= 0.65
Object held-out false-trigger rate <= 0.20
Object held-out median timing error <= 10 steps
Cross-suite macro event recall >= 0.60
Each eligible suite recall >= 0.50
Cross-suite false-trigger rate <= 0.30
```

A suite that fails Gate C must not start formal detector-triggered attack. It
may only run separately authorized teacher-timing mechanism smoke.
