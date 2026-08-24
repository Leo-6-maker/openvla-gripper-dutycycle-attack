# Stage V M4 current-source final-40 corridor — terminal HOLD

Date: 2026-08-13 (Asia/Shanghai)

## Verdict

The current-source final-40 clean-only corridor completed all 80 A/B
receipts, but the formal corridor gate did not pass:

- `PASS/PASS`: 32/40 stable parents
- `PASS/CLEAN_FAILURE`: 3/40 A/B mismatches
- `CLEAN_FAILURE/CLEAN_FAILURE`: 4/40
- `INELIGIBLE/INELIGIBLE`: 1/40
- required stable parents: 40/40

The first non-passing scientific gate is
`CURRENT_SOURCE_FINAL40_CORRIDOR_AB_STABILITY`. Formal M4 was not launched,
no `V_phys`/`V_t` labels were generated, and this line is terminal HOLD under
the frozen failure action. Failed or replicate-mismatched parents must not be
rerun to obtain a pass.

## Frozen bindings

- Source: commit `3b2ffdb5809710e3c7f2e0a529600c8d7a79b9b2`, tree
  `2492a075e782a112d1e857248956b2647e751039`
- Official Python:
  `/mnt/sdc/dty_user/openvla_attack/envs/openvla-official-a800/bin/python`
- Governed worktree:
  `/mnt/sdc/dty_user/openvla_attack_worktrees/stage-v-m4-governed-20260812`
- Corridor root:
  `/mnt/sdc/dty_user/openvla_attack_outputs/n5/phase3_student/STAGE_V_M4_ARCHITECTURE_FREEZE_REBUILD_V1_20260813T000000Z/STAGE_V_M4_CORRIDOR_PREFLIGHT_CURRENT_SOURCE_FINAL40_AB_20260813T000000Z`
- Launch manifest SHA256:
  `7fcd777bf6b912d1ccbc3b292dd434c0be6100dc55d8bb1750856328f9c80a60`
- Sealed independent reconciliation:
  `/mnt/sdc/dty_user/openvla_attack_outputs/n5/phase3_student/STAGE_V_M4_ARCHITECTURE_FREEZE_REBUILD_V1_20260813T000000Z/STAGE_V_M4_CORRIDOR_CURRENT_SOURCE_AB_RECONCILIATION_HOLD_V1.json`
- Reconciliation SHA256:
  `866ce90f73cd542584c4db3fca4b590ebc014e7e7e9dbd2a91adcdee210c7fd9`

The frozen population remains 40 unique parents, TRAIN/VAL/TEST `24/8/8`,
and `6/2/2` per suite. The observed stable counts are
`libero_10=9`, `libero_goal=7`, `libero_object=10`, and `libero_spatial=6`.

## Non-passing identities

- `libero_10/task_02/state_35` — `CLEAN_FAILURE/CLEAN_FAILURE`
- `libero_goal/task_03/state_45` — `CLEAN_FAILURE/CLEAN_FAILURE`
- `libero_goal/task_07/state_28` — `CLEAN_FAILURE/CLEAN_FAILURE`
- `libero_goal/task_07/state_39` — `CLEAN_FAILURE/CLEAN_FAILURE`
- `libero_spatial/task_03/state_20` — `INELIGIBLE/INELIGIBLE`
- `libero_spatial/task_02/state_45` — `PASS/CLEAN_FAILURE`
- `libero_spatial/task_09/state_22` — `PASS/CLEAN_FAILURE`
- `libero_spatial/task_09/state_32` — `PASS/CLEAN_FAILURE`

## Boundary audit

All 80 receipts were parsed directly and independently. Receipt-field audit
errors were zero; A/B key sets exactly matched both the frozen split and the
formal parent manifest. Protected counters were all zero:

```text
protected_reads=0
eval160_reads=0
attack_rollouts=0
vis_pgd_attack_rollouts=0
outcomes_read=false
formal_m4_authorization_issued=false
v_phys_map_generated=false
```

Do not launch formal M4, Teacher/Student, VIS, attack evaluation, or protected
Eval160 from this HOLD. Any future attempt requires a separately authorized
protocol/version and must not mutate or reuse this sealed result.
