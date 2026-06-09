# Stage-B Patched Rerun — Final Experiment Summary

**Date**: 2026-06-07
**Runner**: `run_stageb_vis_labeling.py` (P0-patched)
**Postprocess**: `run_patched_rerun_postprocess_hotfix.py` (hotfix v1)
**Server**: klfy-SYS-4028GR-TR2

## 1. Execution Summary

| Metric | Value |
|--------|-------|
| Windows queued | 44 |
| Jobs (VIS + random Linf) | 88 |
| Completed | 88 (100%) |
| Smoke contaminants excluded | 4 (job9xxx, qpos_delta=0.0) |
| Workers used | 3 pairs × 28-30 jobs each |
| GPU pairs | 1,0 / 2,6 / 4,5 |
| GPU3,7 blacklist | Respected (0 MiB throughout) |
| CUDA OOM | 0 |
| Xid errors | 0 (fresh) |
| Duration | ~3h 10m (11:39–14:48) |
| Waiter triggered | Yes, at 14:48 |
| Hotfix postprocess | Fixed REPO path bug, re-run at 14:50 |

## 2. Hotfix Postprocess Outputs

| File | Rows |
|------|------|
| `tables/stageb_selective_rerun_qpos_hotfix.csv` | 88 (all traces) |
| `tables/stageb_selective_rerun_labels_hotfix.csv` | 44 (paired windows) |

**Postprocess methodology (verified)**:
- Open convention: `env_action_6 < -0.5` → OPEN
- Qpos: `abs(q0) + abs(q1)` from `obs_robot0_gripper_qpos`
- Shifted qpos: `step_dict[s+1]` (action at t → qpos at t+1)
- Metadata: read from summary JSON, not filename
- Pairing: (task_key, state_id, window_start, window_end)
- Old overnight labels: NOT reused

## 3. Label Distribution

| Label | Count | % |
|-------|-------|---|
| **cmd_susceptible** | **3** | **6.8%** |
| random_confounded | 35 | 79.5% |
| Neither (non-perturbable) | 6 | 13.6% |
| physical_response_sensitive (>=0.01) | 9 | 20.5% |
| physical_response_strict (>=0.02) | 4 | 9.1% |
| vis_specific_physical_response | 6 | 13.6% |

### Comparison with Original Queue (OLD wrong convention)

| Label | Original Queue | Patched Rerun | Change |
|-------|---------------|---------------|--------|
| cmd_susceptible | 27 (61.4%) | 3 (6.8%) | **-24** |
| random_confounded | 6 (13.6%) | 35 (79.5%) | **+29** |
| hard_negative | 8 (18.2%) | 6 (13.6%) | -2 |

**The original queue was 61% wrong on cmd_susceptible.** The old wrong open convention (`g > 0` = OPEN) systematically overcounted VIS OPEN actions.

## 4. The 3 cmd_susceptible Windows

| Window | VIS open | VIS streak | VIS qpos | RAND open | RAND streak | RAND qpos | phys_sens | vis_spec |
|--------|----------|------------|----------|-----------|-------------|-----------|-----------|----------|
| butter s6 [182,192] | 10 | 7 | **0.064** | 1 | 1 | -0.012 | ✅ | ✅ |
| butter s7 [143,153] | 6 | 3 | -0.0004 | 5 | 5 | 0.005 | ❌ | ❌ |
| ketchup s8 [154,164] | 10 | 8 | **0.034** | 5 | 4 | -0.001 | ✅ | ✅ |

- **2 out of 3 cmd_susceptible show genuine physical bridging** (qpos >= 0.01, VIS-specific)
- butter s7 has command change but NO physical response → command ≠ physical bridge

## 5. Per-Task Breakdown

| Task | n | cmd | rand_conf | phys_sens | vis_spec | VIS qpos mean | RAND qpos mean |
|------|---|-----|-----------|-----------|----------|--------------|----------------|
| alphabet_soup | 6 | 0 | 4 | 3 | 3 | 0.007 | 0.0003 |
| bbq_sauce | 3 | 0 | 3 | 1 | 0 | 0.017 | 0.011 |
| **butter** | **6** | **2** | 4 | 1 | 1 | 0.012 | 0.010 |
| cream_cheese | 8 | 0 | 7 | 1 | 1 | -0.001 | 0.002 |
| ketchup | 5 | 1 | 1 | 2 | 1 | 0.010 | 0.013 |
| milk | 2 | 0 | 2 | 0 | 0 | 0.002 | -0.002 |
| orange_juice | 5 | 0 | 5 | 0 | 0 | -0.0004 | 0.0001 |
| salad_dressing | 6 | 0 | 6 | 1 | 0 | 0.003 | 0.006 |
| tomato_sauce | 3 | 0 | 3 | 0 | 0 | 0.0003 | 0.0005 |

### Task-Level Observations

- **butter**: Only task with cmd_susceptible windows. Both come from different states.
- **cream_cheese**: VIS-immune. VIS opens 0-1 per window, RAND opens 10-11/11. **INVERTED signal** — VIS actually suppresses opening relative to random.
- **orange_juice**: Same inverted pattern. VIS 0-1, RAND 8-11.
- **ketchup**: Extreme bimodality. 2 windows have VIS+RAND both 0, 1 window has VIS 10 vs RAND 5.
- **alphabet_soup**: Highest vis_specific rate (3/6) despite 0 cmd_susceptible. VIS has specific physical effects without command selectivity.

## 6. Qpos Analysis

| Statistic | VIS qpos_delta_shifted | RAND qpos_delta_shifted |
|-----------|----------------------|--------------------------|
| mean | 0.005160 | 0.004677 |
| median | 0.000428 | 0.000297 |
| count > 0.01 | 9 | 6 |
| count > 0.02 | 4 | 4 |

- VIS and RAND qpos distributions are SIMILAR — both can induce physical gripper changes
- VIS has slightly higher mean but the difference is marginal
- **qpos response is rare overall**: only 9/44 VIS windows cross the 0.01 threshold

## 7. Smoke-C Interpretation

### Verdict: CASE C (with B elements)

**Primary signal (Case C):**
- random_confounded rate = 79.5% (>>30% threshold)
- VIS and RAND produce similar qpos distributions
- Windows are generally perturbation-sensitive, not VIS-specific

**Secondary signal (Case B):**
- Command OPEN exists but qpos response is rare (only 9/44 >= 0.01)
- Most windows that VIS opens show near-zero physical response
- "Command ≠ physical bridge" confirmed for majority of windows

**Case A signal (rare but present):**
- 2 windows (butter s6, ketchup s8) show genuine cmd_susceptible + physical_response
- These are the only candidates for "command → physical bridge chain"

### Implication

The original premise — "VIS PGD20 produces selective command changes that cause measurable physical gripper response" — holds for only **2/44 (4.5%)** of teacher-identified vulnerable windows. The remaining 95.5% of windows show either:
- No selective command effect (random does same or better) — 79.5%
- Command effect without physical response — most of the 20.5% with some qpos

## 8. P0 Bug Impact Assessment

| Bug | Impact on Labels | Mitigated? |
|-----|-----------------|------------|
| Open convention inverted | Caused 24 false cmd_susceptible predictions | ✅ Fixed in runner + postprocess |
| Qpos wrong indices | All old qpos=0.5 (constant) | ✅ `obs_robot0_gripper_qpos` |
| Qpos timing (before env.step) | Old summary qpos_delta always ~0 | ✅ Shifted step_dict[s+1] |
| Signed mean cancellation | q0≈-q1 caused ~0 mean | ✅ abs_sum = abs(q0)+abs(q1) |
| Condition from filename | `vis_pgd` → split to `vis`+`pgd` | ✅ Summary JSON |
| Task from filename | `tomato_sauce` → `tomato` | ✅ Summary JSON |
| Shifted qpos indexing | enumerate() local index | ✅ step_dict lookup |
| REPO path hardcoded | Hotfix crashed on first run | ✅ Fixed + re-ran |

All 8 P0 bugs verified as fixed in final output.

## 9. Files Produced

### Server
- `/data/liuyu/outputs/stageb_selective_rerun_patched_20260607/` — 92 traces + 92 summaries
- `/data/liuyu/repos/.../tables/stageb_selective_rerun_qpos_hotfix.csv` — 88 rows
- `/data/liuyu/repos/.../tables/stageb_selective_rerun_labels_hotfix.csv` — 44 rows

### Local
- `reports/STAGEB_PATCHED_RERUN_MIDRUN_HEALTH_CHECK.md`
- `reports/STAGEB_PATCHED_RERUN_COMPLETION_CHECKLIST.md`
- `reports/STAGEB_SMOKE_C_INTERPRETATION_GUIDE.md`
- `tables/stageb_patched_rerun_midrun_trace_audit.csv`
- `tables/stageb_patched_rerun_midrun_pairing_progress.csv`
- `tables/stageb_patched_rerun_midrun_qpos_preview.csv`

## 10. Next Steps

1. **Copy label/qpos CSVs locally** for audit
2. **Mechanism readout**: Deep-dive into the 2 genuine cmd+phys windows vs inverted cream_cheese/orange_juice
3. **Smoke-C**: CPU-only physical_response model (LR/RF) — only if audit PASS
4. **Decision gate**: With only 2 genuine positives, is the window detector approach viable? Or pivot strategy?
