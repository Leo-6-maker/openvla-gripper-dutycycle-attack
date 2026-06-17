# H5 Dual-Seed Final Report

**Classification:** L3-3_DUAL_SEED_PHYSICAL_BRIDGE_PASS_TASK_EFFECT_NOT_PROVEN

## Episode Results

| Seed | Condition | Steps | Token | Arm | Open Frac | Mean Q | Lift H | Dist Inc |
|------|-----------|-------|-------|-----|-----------|--------|--------|----------|
| seed81 | CLEAN | 161 |  |  | 0.68 | -0.00381 | 0.226 | -0.006 |
| seed81 | RAND | 161 | 31872 | 6 | 0.68 | -0.00381 | 0.226 | -0.006 |
| seed81 | SHUFFLED | 161 | 31872 | 6 | 0.68 | -0.00381 | 0.226 | -0.006 |
| seed81 | TRUE | 157 | 31744 | 5 | 1.16 | -0.00949 | 0.173 | -0.006 |
| seed82 | CLEAN | 161 |  |  | 0.68 | -0.00381 | 0.226 | -0.006 |
| seed82 | RAND | 161 | 31872 | 6 | 0.68 | -0.00381 | 0.226 | -0.006 |
| seed82 | SHUFFLED | 161 | 31872 | 6 | 0.68 | -0.00381 | 0.226 | -0.006 |
| seed82 | TRUE | 157 | 31744 | 5 | 1.16 | -0.00949 | 0.173 | -0.006 |

## Bridge Gates

- B1 Token: PASS (TRUE=31744 dual-seed)
- B2 Command: PASS (TRUE env=-1 OPEN)
- B3 Physical: PASS (TRUE excess=0.0057 > 0.003)
- B4 Grasp: NOT_PROVEN
- B5 Task: NOT_PROVEN (CLEAN+TRUE both success=True)
- B6 Selectivity: PHYSICAL_PASS
