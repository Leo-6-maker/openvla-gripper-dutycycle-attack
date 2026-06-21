# Full Layer1 Dataset V3 Gate 20260622

```text
PHASE_D_FULL_LAYER1_GENERATION = STOPPED
STOP_REASON = EMPTY_SUPERVISED_SUITE
resolver_commit = a7a7188012ac63d9e650e663add21f474752bd51
server_output_root = /data/liuyu/layer1_outputs/frozen_owner_ai_v1_20260622/full_layer1_dataset_v3_a7a7188
train_episodes = 240
train_events = 105
train_failures = 0
train_validation_errors = 0
train_duplicate_keys = 0
train_phase_order_violations = 0
val_episodes = 60
val_events = 27
val_failures = 0
val_validation_errors = 0
val_duplicate_keys = 0
val_phase_order_violations = 0
held_out_episodes = 300
held_out_events = 131
held_out_failures = 0
held_out_validation_errors = 0
held_out_duplicate_keys = 0
held_out_phase_order_violations = 0
total_phase_order_violations = 0
all_suites_nonzero_supervised_train_val_test = false
```

## Supervised Counts

| suite | train | val | held_out |
| --- | ---: | ---: | ---: |
| libero_spatial | 68 | 17 | 83 |
| libero_goal | 37 | 10 | 48 |
| libero_10 | 0 | 0 | 0 |

LIBERO-10 has zero supervised rows in all splits under the repaired primary resolver. Per the sprint hard gate, Layer2 training/evaluation and Layer3 execution were not run.

## Claim Boundary

- Full resolver execution completed for 240 train, 60 validation, and 300 held-out episodes.
- This is not a frozen dataset v3 because the nonzero-supervision gate failed for LIBERO-10.
- No Layer2 training, GPU, LIBERO rollout, VIS, RAND, shuffled, oracle, or attack execution was run after this gate failure.
