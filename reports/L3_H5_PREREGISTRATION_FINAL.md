# H5: Oracle Closed-Loop Preregistration

## Parent

- butter_s11 step60 (D5 emit = anchor = 60)

## Conditions (per seed)

1. CLEAN — baseline rollout
2. TRUE — V4 selected hard-feasible candidate
3. RAND — best-arm control (arm=6/6, token=31872 CLOSE)
4. SHUFFLED — best-arm control (arm=6/6, token=31872 CLOSE)

## Frozen Bindings

| Seed | Condition | Candidate ID | Arm | Token | Margin |
|------|-----------|-------------|-----|-------|--------|
| 81 | TRUE | 12 | 5/6 | 31744 | 17.8 |
| 81 | RAND | 0 | 6/6 | 31872 | -3.6 |
| 81 | SHUFFLED | 19 | 6/6 | 31872 | -3.5 |
| 82 | TRUE | 9 | 5/6 | 31744 | 16.8 |
| 82 | RAND | 11 | 6/6 | 31872 | -3.9 |
| 82 | SHUFFLED | 16 | 6/6 | 31872 | -3.6 |

## Control Selection Rule

1. Maximize arm_prefix_match_count
2. Maximize official_target31744_margin
3. Minimize processor_linf
4. Lowest candidate_id

## H5 Execution Gates

- Seed81 first: CLEAN→TRUE→RAND→SHUFFLED
- All 4 conditions must complete
- TRUE uses frozen adversarial delta — no online re-optimization
- Physical bridge: token→command→qpos/width→grasp→task
- Token change alone ≠ closed-loop PASS
