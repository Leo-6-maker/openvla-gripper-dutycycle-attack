# Stage-B RC1a S16R — Relabel Recovery After Task Mapping Bug

**Date**: 2026-06-11 (spanning 2026-06-10 bug discovery)
**GitHub HEAD**: e9578b0 (runner fix) → TBD (this freeze)
**Branch**: exp/vis-prefix-margin-repair-20260603
**Type**: Pure analysis — no GPU, relabel of all S6-S16 artifacts

## Executive Summary

A hardcoded TASK_CFG index mismatch in `run_s9b_phase1_runner_attack_port.py` caused all `--task` labels from S6 through S16 to map to wrong LIBERO Object tasks. **All 199 summary JSONs and all reports/tables used wrong object names.** Physical rollout data (env, BDDL, prompt, qpos) remain valid under actual LIBERO task identities. This report provides the corrected mapping, relabeled tables, and updated claim boundary.

**Corrected conclusion**: The only confirmed clean repeated physical bridge POC is on **tomato_sauce_s0_w70-80** (6 seeds across S9b/S9c/S13a/S16b calib, with PHYSICAL_BRIDGE_PASS classification for all seeds except seed12 which is PHYSICAL_BORDERLINE). The actual milk task shows RAND contamination in most tested windows but has one strong single-seed command-positive candidate (s2_w95-105, VIS 10/10) that failed 3-seed RAND-veto. Actual salad_dressing is command-weak. Actual cream_cheese was a manual candidate.

## Root Cause (Fixed)

```python
# OLD (buggy):
TASK_CFG = {'ketchup':0,'butter':1,'cream_cheese':2,...,'milk':5,...}
cfg = TASK_CFG.get(args.task)       # wrong index
task_obj = task_suite.get_task(cfg) # wrong task

# NEW (e9578b0):
_actual_by_key = {}
for _i in range(len(task_suite.tasks)):
    _key = _canonical_task_key(suite.get_task(_i))
    _actual_by_key[_key] = _i
cfg = _actual_by_key[args.task]
actual_task_key = _canonical_task_key(task_obj)
assert actual_task_key == args.task  # FATAL if mismatch
```

Runner now writes `actual_task_key`, `actual_task_idx`, `actual_language`, `actual_problem_folder`, `actual_bddl_file` to both summary JSON and trace CSV. Duplicate key detection and coverage assertion added.

## Task Mapping

| Old `--task` | Old TASK_CFG→idx | **Actual LIBERO** | Old Role | Corrected Role |
|-------------|-----------------|-------------------|----------|----------------|
| milk | 5 | **tomato_sauce** | physical bridge POC | POC confirmed on tomato_sauce |
| tomato_sauce | 7 | **milk** | RAND-confounded | milk: mostly RAND-contaminated, one cmd-positive candidate |
| cream_cheese | 2 | **salad_dressing** | command-weak | salad_dressing: cmd-weak 0/3 bridge |
| butter | 1 | **cream_cheese** | manual candidate | cream_cheese: manual, cmd-weak/RAND |
| salad_dressing | 3 | **bbq_sauce** | cmd-weak/borderline | bbq_sauce: first test, weak |
| bbq_sauce | 4 | **ketchup** | never used | ketchup: untested |
| alphabet_soup | 6 | **butter** | never used | butter: untested |
| orange_juice | 8 | **chocolate_pudding** | never used | chocolate_pudding: untested |
| (none) | — | **orange_juice** | never used | orange_juice: completely untested |

## Corrected Physical Bridge Results

See [tables/s16r_corrected_layer3_physical_bridge_status_all.csv](../tables/s16r_corrected_layer3_physical_bridge_status_all.csv).

### tomato_sauce_s0_w70-80 (old "milk" anchor) — 6 seeds

| Seed | VIS OPEN | Streak | VIS Norm | RAND OPEN | RAND Norm | Classification |
|------|----------|--------|----------|-----------|-----------|----------------|
| 9 | 7 | 4 | 0.423 | 0 | 0.000 | **PHYSICAL_BRIDGE_PASS** |
| 10 | 6 | 5 | 0.265 | 0 | 0.055 | **PHYSICAL_BRIDGE_PASS** |
| 11 | 6 | 4 | 0.378 | 1 | 0.000 | **PHYSICAL_BRIDGE_PASS** |
| 24 | 6 | 4 | 1.297 | 0 | 0.000 | **PHYSICAL_BRIDGE_PASS** |
| 50 | 8 | 4 | 0.621 | 0 | 0.000 | **PHYSICAL_BRIDGE_PASS** |
| 12 | 6 | 4 | 0.129 | 0 | 0.030 | PHYSICAL_BORDERLINE |

**6/6 seeds have VIS OPEN ≥ 6, streak ≥ 4, RAND OPEN ≤ 1. 5/6 exceed norm 0.20.**

### tomato_sauce_s0_w240-250 (old "milk" late, S16b fresh) — partial S16c

| Seed | VIS OPEN | Streak | VIS Norm | RAND OPEN | Classification |
|------|----------|--------|----------|-----------|----------------|
| 50 | 7 | 4 | 0.279 | 2 | MIXED (cmd-positive, RAND=2 borderline) |
| 51 | 4 | 4 | 0.071 | 0 | PHYSICAL_TRANSFER_WEAK |
| 52 | 7 | 5 | 0.391 | 0 | **PHYSICAL_BRIDGE_PASS** |
| 53 | — | — | — | — | NOT RUN (S16c killed) |

**1/2 confirmed PASS (seed52), seed51 weak. Incomplete — seed53 not run.**

### milk_s2_w95-105 (old "tomato" S16b fresh candidate)

| Seed | VIS OPEN | Streak | VIS Norm | RAND OPEN | RAND Norm | Classification |
|------|----------|--------|----------|-----------|-----------|----------------|
| 50 | 10 | 10 | 1.018 | 2 | 0.019 | MIXED (cmd-strong, RAND=2 borderline) |
| 51 | — | — | — | **3** | — | **RAND-veto REJECT** |
| 52 | — | — | — | 1 | — | RAND-veto CLEAN |
| 53 | — | — | — | **4** | — | **RAND-veto REJECT** |

**RAND-veto: 2/3 seeds REJECT. → milk_s2_w95-105 = random-confounded / abstain.**

### salad_dressing_s2_w50-60 (old "cream") — 4 seeds

| Seed | VIS OPEN | VIS Norm | RAND OPEN | Classification |
|------|----------|----------|-----------|----------------|
| 21 | 2 | 0.007 | 0 | COMMAND_WEAK |
| 22 | 3 | 0.044 | 0 | COMMAND_WEAK |
| 23 | 2 | 0.002 | 0 | COMMAND_WEAK |
| 50 | 2 | 0.008 | 0 | COMMAND_WEAK |

**All 4 seeds COMMAND_WEAK. VIS cannot produce ≥ 4 OPEN on this task/window.**

### cream_cheese_s0_w90-100 (old "butter" manual) — 2 seeds

| Seed | VIS OPEN | VIS Norm | RAND OPEN | RAND Norm | Classification |
|------|----------|----------|-----------|-----------|----------------|
| 13 | 2 | 0.014 | 1 | 0.158 | COMMAND_WEAK |
| 14 | 2 | 0.025 | 3 | 0.261 | RAND_CONFOUNDED_ABSTAIN |

**Manual candidate. Not Layer1-selected. Both command-weak and RAND-contaminated.**

## Corrected Command-Level Screen (S16b Relabeled)

| Corrected Task | State | Window | VIS OPEN | Streak | RAND OPEN | Gap | Class |
|---------------|-------|--------|----------|--------|-----------|-----|-------|
| **tomato_sauce** | 0 | 240-250 | **7** | **4** | 2 | +5 | **CMD_POSITIVE** |
| **tomato_sauce** | 0 | 70-80 | 8 | 4 | 0 | +8 | CMD_POSITIVE (calib) |
| **milk** | 2 | 95-105 | **10** | **10** | 2 | +8 | CMD_POSITIVE (RAND-veto failed) |
| **bbq_sauce** | 1 | 50-60 | 5 | 2 | 2 | +3 | BORDERLINE |
| tomato_sauce | 0 | 235-245 | 5 | 3 | 2 | +3 | BORDERLINE (calib) |
| tomato_sauce | 0 | 230-240 | 4 | 2 | 2 | +2 | BORDERLINE (calib) |
| milk | 0 | 50-60 | 3 | 2 | 1 | +2 | COMMAND_WEAK |
| **bbq_sauce** | 0 | 55-65 | **0** | **0** | 1 | -1 | COMMAND_WEAK |
| salad_dressing | 2 | 50-60 | 2 | 1 | 0 | +2 | COMMAND_WEAK (calib) |
| cream_cheese | 0 | 80-90 | 1 | 1 | 1 | 0 | COMMAND_WEAK (calib) |
| milk | 2 | 155-165 | 4 | 3 | **8** | -4 | RANDOM_CONFOUNDED (calib) |
| salad_dressing | 0 | 85-95 | 4 | 1 | **3** | +1 | RANDOM_CONFOUNDED (calib) |

## Allowed Claims (post-relabel)

### Allowed

1. **tomato_sauce_s0_w70-80 is the only confirmed clean repeated physical bridge POC**, with 5/6 seeds PHYSICAL_BRIDGE_PASS and 1 PHYSICAL_BORDERLINE (seed12).
2. **tomato_sauce_s0_w240-250 is a command-positive candidate** with partial confirmation (seed52 PHYSICAL_BRIDGE_PASS, seed51 weak, seed53 not run).
3. **VIS attack produces stronger OPEN duty-cycle than matched random** on selected task/state/window parents.
4. **Layer1 command selectivity alone is insufficient** for Layer3 physical bridge specificity — multiple Layer1-selected windows fail at command or physical level.
5. **milk family shows RAND contamination** in most tested windows, with s2_w95-105 being a very strong single-seed command-positive that failed 3-seed RAND-veto.
6. **salad_dressing is command-weak** on tested windows — VIS cannot produce sufficient OPEN commands.
7. The task-mapping bug invalidated all S6-S16 object labels; physical traces remain valid; this relabel is the authoritative correction.

### Forbidden

1. "milk" as any positive claim — the old "milk" was actually tomato_sauce.
2. "tomato RAND-confounded" — the old "tomato" was actually milk.
3. "cream command-weak" — the old "cream" was actually salad_dressing.
4. Any object-level claim using old task names.
5. Any claim about object-wide success, Layer3 solved, or detector solved.
6. milk_s2_w95-105 as a positive candidate (RAND-veto REJECTED).
7. Claims about LIBERO Object task coverage (only 4/10 tasks tested: tomato_sauce, milk, salad_dressing, cream_cheese; bbq_sauce touched in S16b only).

## Artifact Inventory

| Artifact | Path |
|----------|------|
| S16R report | `reports/STAGEB_RC1A_S16R_RELABEL_RECOVERY_20260610.md` |
| Relabel manifest | `tables/s16r_relabel_all_s6_s16_summaries.csv` (199 rows) |
| Corrected physical bridge | `tables/s16r_corrected_layer3_physical_bridge_status_all.csv` |
| Task mapping | `tables/task_mapping_bug_requested_to_actual.csv` |
| LIBERO task order | `tables/libero_object_actual_task_order.csv` |
| Bug report | `reports/STAGEB_RC1A_CRITICAL_TASK_MAPPING_BUG_20260610.md` |
| Patched runner | `scripts/stageb/run_s9b_phase1_runner_attack_port.py` (e9578b0) |

## Runner Patch Summary (e9578b0 + postfix)

- TASK_CFG removed
- `_canonical_task_key()` from task metadata (language + problem_folder + bddl_file)
- Runtime enumeration of task_suite.tasks → `_actual_by_key`
- `assert actual_task_key == args.task`
- Duplicate key detection + coverage assertion
- Summary: `actual_task_key`, `actual_task_idx`, `actual_language`, `actual_problem_folder`, `actual_bddl_file`
- Trace CSV: `requested_task`, `actual_task_key`, `actual_task_idx`, `actual_language`, `actual_bddl_file`

## Next Steps

S16R is complete. Next justified step: **S17a patched-runner smoke test** — run a minimal verification that the patched runner correctly executes `--task tomato_sauce` on tomato_sauce and reproduces the known bridge on w70-80. Only after smoke passes should any new experiments proceed.
