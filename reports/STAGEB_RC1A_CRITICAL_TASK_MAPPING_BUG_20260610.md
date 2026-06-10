# CRITICAL: Task Mapping Bug — All S6-S16 Object Labels Inverted

**Date**: 2026-06-10 ~23:40 CST
**GitHub HEAD**: 08191c3 (S16b freeze, now suspended)
**Severity**: CRITICAL — halts all experiments

## Bug Description

`run_s9b_phase1_runner_attack_port.py` hardcodes a `TASK_CFG` dict that maps requested task names to LIBERO suite indices. This mapping is **off-by-shift**: the indices do not correspond to the canonical LIBERO Object task order.

```python
# BUGGY code (line 92)
TASK_CFG = {'ketchup':0,'butter':1,'cream_cheese':2,'salad_dressing':3,
            'bbq_sauce':4,'milk':5,'alphabet_soup':6,'tomato_sauce':7,'orange_juice':8}
cfg = TASK_CFG.get(args.task)          # gets wrong index
task_obj = task_suite.get_task(cfg)    # loads wrong task
```

The actual LIBERO Object task order (verified on server):
```
0: alphabet_soup    1: cream_cheese    2: salad_dressing
3: bbq_sauce        4: ketchup         5: tomato_sauce
6: butter           7: milk            8: chocolate_pudding
9: orange_juice
```

## Impact: Every `--task` label in S6-S16 is wrong

| Requested `--task` | TASK_CFG→idx | **Actual LIBERO task** | Used in |
|-------------------|-------------|----------------------|---------|
| milk | 5 | **tomato_sauce** | S9b/S9c/S13a/S15b/S16b (PRIMARY) |
| tomato_sauce | 7 | **milk** | S11b/S12a/S13a/S15b/S16b |
| cream_cheese | 2 | **salad_dressing** | S12b/S15b/S16b |
| butter | 1 | **cream_cheese** | S10a/S12a/S16b |
| salad_dressing | 3 | **bbq_sauce** | S16b |
| bbq_sauce | 4 | **ketchup** | never used directly |
| alphabet_soup | 6 | **butter** | never used directly |
| orange_juice | 8 | **chocolate_pudding** | never used directly |
| (none) | — | **orange_juice** | never used at all |
| (none) | — | **chocolate_pudding** | used as "orange_juice" |

## What remains valid

1. **Physical rollout data**: env, BDDL, init_state, prompt language all follow the actual LIBERO task. The model receives the correct instruction for the task it's actually executing.
2. **Qpos measurements**: gripper qpos[7] is correctly sampled from `env.sim.data.qpos`.
3. **VIS/RAND attack mechanism**: adversarial pixel generation, token decode, env action binarization are all correct.
4. **OPEN convention**: `env_action[-1] = -1.0` for open, detected via `env_action_6 < -0.5`.
5. **Trace-level data**: all per-step metrics are valid for the actual task being run.

## What is invalid

1. **All object labels** in summaries, pair_ids, filenames, reports, tables.
2. **All object-level claims** from S6 through S16.
3. **All task-level taxonomy** (milk bridge POC, tomato RAND-confounded, etc.).
4. **All cross-object comparisons** and failure-mode assignments.

## Corrected Label Mapping

| Old Label (reports) | **Corrected Label** | Old Claim | New Status |
|---------------------|-------------------|-----------|------------|
| milk (PASS 4/5) | **tomato_sauce** | physical bridge POC | physical bridge POC on tomato_sauce |
| tomato (RAND-confounded) | **milk** | random-contaminated | milk is RAND-confounded in tested windows |
| cream (cmd-weak 0/3) | **salad_dressing** | command-weak | salad_dressing cmd-weak |
| butter (manual) | **cream_cheese** | manual/cmd-weak | cream_cheese manual |
| salad (cmd-weak/borderline) | **bbq_sauce** | cmd-weak/borderline | bbq_sauce first test |
| bbq_sauce | **ketchup** | never tested | never tested |
| alphabet_soup | **butter** | never tested | never tested |
| orange_juice | **chocolate_pudding** | never tested | never tested |

## S16b Relabeled Results

| Old Parent Label | Corrected Parent | VIS OPEN | Streak | RAND OPEN | Class |
|-----------------|-----------------|----------|--------|-----------|-------|
| milk_s0_w240-250 | **tomato_sauce_s0_w240-250** | 7 | 4 | 2 | COMMAND_ATTACK_POSITIVE |
| milk_s0_w70-80 | **tomato_sauce_s0_w70-80** | 8 | 4 | 0 | COMMAND_ATTACK_POSITIVE |
| tomato_s2_w95-105 | **milk_s2_w95-105** | 10 | 10 | 2 | COMMAND_ATTACK_POSITIVE |
| salad_s1_w50-60 | **bbq_sauce_s1_w50-60** | 5 | 2 | 2 | BORDERLINE |
| tomato_s0_w50-60 | **milk_s0_w50-60** | 3 | 2 | 1 | COMMAND_WEAK |
| salad_s0_w55-65 | **bbq_sauce_s0_w55-65** | 0 | 0 | 1 | COMMAND_WEAK |

## S16c Partial Results (KILLED mid-run, do not claim)

| Old Label | Corrected Label | Seed | VIS OPEN | Streak | RAND OPEN | Notes |
|-----------|----------------|------|----------|--------|-----------|-------|
| milk_s0_w240-250 | tomato_sauce_s0_w240-250 | 51 | 4 | 4 | 0 | BORDERLINE |
| milk_s0_w240-250 | tomato_sauce_s0_w240-250 | 52 | 7 | 5 | — | POSITIVE (partial) |
| tomato_s2_w95-105 | milk_s2_w95-105 | 51 | — | — | **3** | RAND-veto REJECT |
| tomato_s2_w95-105 | milk_s2_w95-105 | 52 | — | — | 1 | CLEAN |
| tomato_s2_w95-105 | milk_s2_w95-105 | 53 | — | — | **4** | RAND-veto REJECT |

ORACLE refs: tomato_sauce_s0_w240-250=0.2755, milk_s2_w95-105=0.6474

## Immediate Actions

1. ✅ **All GPU jobs killed** — S16c tmux sessions terminated.
2. ✅ **Task order audit completed** — `tables/libero_object_actual_task_order.csv`.
3. 🔲 **Patch runner** — remove TASK_CFG, use canonical task resolution from task metadata.
4. 🔲 **Relabel all summaries** — generate corrected aggregate tables.
5. 🔲 **Rewrite claim boundary** — under corrected task labels.

## Current Allowed Claim (during relabel)

A task-mapping bug invalidated all requested object labels in S6-S16.
Physical trace data remain valid under actual LIBERO task identities.
No object-level conclusion is valid until relabel audit is complete.
The corrected mapping is confirmed via server-side task order enumeration.

## Next Steps After Relabel

1. Fix runner and verify with canonical task key assertion.
2. Regenerate all tables with corrected labels.
3. Reassess the scientific narrative under corrected object identities.
4. Only then decide whether to resume S16c confirmation or redesign experiments.
