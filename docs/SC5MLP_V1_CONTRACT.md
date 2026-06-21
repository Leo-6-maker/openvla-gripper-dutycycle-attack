# SC5MLP-v1 Contract

## Architecture

```
input: 25D canonical features
backbone: Linear(25,64) → ReLU → Linear(64,64) → ReLU
heads:
  phase_logits: Linear(64,9)    # 9 phase classes
  corridor_logit: Linear(64,1)  # binary corridor classifier
  release_logit: Linear(64,1)   # binary release classifier
```

The confidence head (`confidence_logit`) is **LEGACY** from SC5MLP base definition.
It must NOT appear in new checkpoints — training produces 3 heads; the 4th head in
`SC5MLP` base class is retained for backward compatibility but its output is unused.

## Features (25D, frozen order)

1-13: Proprio/action (direct)
14-16: Gripper history (causally derived)
17-25: Advanced causal features

See `sc5_streaming_features_v2.py` for canonical order and unit definitions.

## Phase Classes (9)

approach, grasp_close, stable_grasp, first_lift, stable_carry,
pre_place_unsupported, release_safe, recovery_or_regrasp, abstain_unsupported

## Training Objective

```
loss = phase_CE + 0.5 * corridor_BCE + 0.3 * release_BCE
```

Corridor BCE uses pos_weight=5.0 for class imbalance.
Phase CE uses class-balanced weights from training set.

## Normalization

- mean/std computed from TRAIN set only
- shape: (25,)
- std epsilon: 1e-8
- no future rows, no leaked episode statistics

## Trigger State Machine

```
IDLE → (pred_phase==stable_carry AND corridor_p > tau_corridor) → ARMED
ARMED → (step >= arm_step + guard AND corridor_p > tau_corridor AND release_p < tau_release) → EMITTED
EMITTED → one-shot latch (no further transitions)
```

Frozen defaults: tau_corridor=0.3, tau_release=0.3, guard=5

## Checkpoint Format

```python
{
    "model_state": OrderedDict,          # SC5MLP state dict
    "mean": np.ndarray(25,),             # training set mean
    "std": np.ndarray(25,),              # training set std
    "feature_names": List[str],          # 25 canonical names
    "phase_classes": List[str],          # 9 phase class names
    "dataset_sha256": str,               # SHA of training CSV
    "dataset_path": str,                 # relative path
    "split_mode": "frozen",              # must be "frozen"
    "normalization_source": "train_only",
    "n_train": int, "n_val": int,
    "train_episode_ids": List[str],
    "val_episode_ids": List[str],
    "seed": int,
    "tau_corridor": float,
    "tau_release": float,
    "guard": int,
}
```

## Train/Val/Cal/Test Split

| Set | Source | Count | Purpose |
|---|---|---|---|
| TRAIN | 10 tasks × init3-8 | 60 | Model training + normalization |
| VAL | 10 tasks × init9 | 10 | Checkpoint selection (seed/model) |
| CAL | 10 tasks × init10 | 10 | Threshold/policy calibration |
| XFER-TEST | 10 tasks × init11-12 | 20×2 | Zero-shot transfer test |
| EVAL19 | init0-2 common success | 19×2 | Final one-time verification |
| HELD-OUT | init0-2 (existing clean30) | 30×3 | NEVER for training/calibration |

## Forbidden

- Using EVAL19 or HELD-OUT for any model/training decision
- Selecting threshold on XFER-TEST
- Profile-specific thresholds (FP32 and Flash2 share same tau)
- Using PIL-preprocessed traces
- Future-row feature peeking
- Training on success episodes only
