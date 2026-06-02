# CODEX HANDOFF — VIS Adaptive Controller Freeze

**Date**: 2026-06-02 | **Branch**: `exp/vis-payload-upgrade-validation-20260601` | **Commit**: `2b1489f`

## 0. Executive Summary

This is the freeze commit for the VIS (visual-input perturbation) adaptive controller exploration phase. The branch validates corrected OpenVLA decode for VIS PGD diagnostics and implements an online adaptive controller for exploratory calibration.

**This is NOT a production VIS controller. It is an engineering/mechanism exploration artifact.**

Key conclusions at freeze:

1. ProprioNoStep handles phase/timing detection.
2. VIS PGD can change decoded gripper action, but there is a **physical-transfer gap**: decoded OPEN commands do not always translate to gripper qpos/width opening.
3. **qpos_delta is the strongest failure mediator** (LogReg weight +4.51), not OPEN count or fixed duration.
4. **No universal selective fixed-duration D\***: cream_cheese fails at d16-d20, but salad_dressing (originally classified as "robust" under command-layer proxy) is VIS-vulnerable, and ketchup is only partially tolerant.
5. **Pure streak-stop adaptive controller (open_streak_stop) is the wrong policy**: K=2 stops too early, K=3 insufficient, K=5 over-threshold.
6. **min_hold_qpos_cap shows exploratory promise** but requires qpos timing fixes before assessment.
7. **GPU layout discovery**: GPUs with Xid history (0,3,7) survive as pure-compute secondary cards when healthy GPUs handle C+G rendering.

Next mainline: **VisualTransferHead** — a lightweight model predicting physical response / failure susceptibility from task features + VIS action trace, replacing rule-based adaptive controllers.

## 1. Repository and Commit State

```
Branch:   exp/vis-payload-upgrade-validation-20260601
HEAD:     2b1489f — Finalize VIS adaptive controller and GPU layout findings
Remote:   https://github.com/Leo-6-maker/openvla-gripper-dutycycle-attack
Status:   Clean (all changes committed and pushed)
Parent:   main (diverged, exp branch contains VIS-specific additions)
```

### Files changed in `2b1489f`:
```
M scripts/vis_rollout_adaptive_v3.py    (updated from 856eba5)
A reports/GPU_HEALTH_AND_LAYOUT_FINDINGS.md
M reports/VIS_FAILURE_MECHANISM_AUDIT.md
```

### Full diff from main (cumulative across all VIS commits on this branch):
- 60+ files changed
- Core: `src/gripper_attack/attack_adapter.py` (dtype fix), `src/gripper_attack/openvla_redecode.py` (new)
- Scripts: `scripts/vis_rollout_adaptive_v3.py`, `scripts/diagnostics/` (7 files)
- Reports: 20+ VIS report files
- Tables: 30+ CSV tables
- Tests: 8 test files in tests/v4/

### Not committed (by design):
- Rollout trace CSVs (~90 files in `/data/liuyu/outputs/...`)
- Log files
- Checkpoint/tensor dumps
- Backup patches

## 2. Files Changed in Commit 2b1489f

### scripts/vis_rollout_adaptive_v3.py

Standalone adaptive controller rollout script. Not integrated into production runner.

**Supported controllers:**

| Controller | Stop Condition | Parameters |
|------------|---------------|------------|
| `fixed` | Window ends or episode done | — |
| `open_streak_stop` | Consecutive OPEN streak >= K | `--K` |
| `open_count_stop` | Total OPEN count >= K | `--K` |
| `qpos_safety_stop` | qpos_delta_online >= Q | `--Q` |
| `min_hold_qpos_cap` | After min_attacks, stop if qpos >= Q | `--min_attacks`, `--Q` |
| `streak_with_qpos_cap` | Streak >= K, but stop early if qpos >= Q | `--K`, `--Q` |

**New CLI arguments:**
- `--controller`: controller mode (default: fixed)
- `--K`: OPEN streak/count threshold (default: 0 = disabled)
- `--Q`: qpos_delta threshold (default: 0 = disabled)
- `--max_duration`: max PGD attacks before forced stop (default: 0 = use window size)
- `--min_attacks`: min PGD attacks before controller can stop (default: 0)

**Run ID naming:**
```
vis_{task}_s0_{condition}_{strategy}_d{duration}_w{start}_{end}_seed{seed}_{controller}_K{K}_Q{Q}_md{max_dur}_{timestamp}_trace.csv
```

**Trace CSV fields** (per-step): task, condition, seed, step, policy_step, in_window, raw_gripper, env_gripper, gripper_qpos, clean_grip, adv_grip, arm_l2, linf, token_flip, attack_dt, eef_x/y/z, done, reward, ctrl_mode, ctrl_stop_reason, ctrl_streak, ctrl_max_streak, ctrl_qpos_delta, ctrl_attacks

**Summary JSON fields** (per-episode): task, condition, seed, success, total_steps, window_start, window_end, window_steps, window_token_flips, avg_arm_l2, total_dt_s, controller, K, Q, max_dur, min_att, stop_reason, attacks_applied, max_open_streak, qpos_delta_online

**Known limitations:**
- Fixed perturb_start/perturb_end per task (hardcoded in TASK_CONFIGS)
- State always 0 (state_id hardcoded)
- No CQ/CQFR integration
- qpos is pre-step (lagged one step behind physical response)
- attack_dt is 0 (not measured per step)

### reports/GPU_HEALTH_AND_LAYOUT_FINDINGS.md

Documents the discovery that GPUs 0,3,7 survive PGD when used as pure-compute secondary cards with healthy GPUs handling C+G rendering.

**Stable layout:**
```
g10: CUDA_VISIBLE_DEVICES=1,0  (GPU1 primary C+G, GPU0 secondary compute)
g23: CUDA_VISIBLE_DEVICES=2,3  (GPU2 primary C+G, GPU3 secondary compute)
g45: CUDA_VISIBLE_DEVICES=4,5  (GPU4 primary C+G, GPU5 secondary compute)
g67: CUDA_VISIBLE_DEVICES=6,7  (GPU6 primary C+G, GPU7 secondary compute)
```

**Recovery methods:**
1. Server reboot (~2-3 min)
2. Nvidia driver reload: `sudo rmmod nvidia_uvm nvidia_drm nvidia_modeset nvidia && sudo modprobe nvidia && sudo modprobe nvidia_modeset && sudo modprobe nvidia_uvm` (~10 sec)

**CAVEAT**: Driver reload kills ALL CUDA contexts. Only run when no other GPU processes exist.

### reports/VIS_FAILURE_MECHANISM_AUDIT.md

Mechanism summary showing qpos_delta as strongest failure predictor.

**Key numbers** (from LogReg on 31-35 VIS rollout rows):
- qpos_delta weight: +4.51
- longest_open_streak weight: +2.85
- arm_l2 weight: +0.29
- duration weight: -0.33
- open_count weight: +0.08 (very weak)

**Best threshold baselines:**
- streak>=5: acc=0.857, f1=0.839
- qpos>=0.004: acc=0.800

**Per-task vulnerability:**
- salad_dressing: 71% fail rate, avg qpos_delta=0.014 (highest transfer)
- cream_cheese: 57% fail rate, avg qpos_delta=0.003 (moderate transfer)
- ketchup: 33% fail rate, avg qpos_delta=0.005 (low physical transfer)
- tomato_sauce: 17% fail rate (very resilient, d60 required)

**IMPORTANT**: These fail rates come from log-based success/failure summaries. Source traces and logs are in `/data/liuyu/outputs/milestone_7_vis_controlled_rollout_micro_20260601/`. Some older traces may have been overwritten by later runs with the same task/condition naming (pre-timestamp-fix era).

## 3. Raw Output / Trace Inventory

### Primary output root:
```
/data/liuyu/outputs/milestone_7_vis_controlled_rollout_micro_20260601/
├── runs/     (~90 trace CSV files)
├── logs/     (~60+ log files)
└── tables/   (summary CSVs)
```

### Key trace subsets:
- Fixed-duration traces: `vis_{task}_s0_{cond}_full_d{len}_w{start}_{end}_seed{seed}_{ts}_trace.csv`
- Adaptive controller traces: `vis_{task}_s0_vis_pgd_full_d{len}_w{start}_{end}_seed{seed}_{controller}_K{K}_Q{Q}_md{max}_{ts}_trace.csv`
- 16 adaptive controller traces total (open_streak_stop + min_hold_qpos_cap)
- 3 adaptive jobs crashed (GPU3 Xid31 on g23/g36 pairs)

### Other outputs:
```
/data/liuyu/outputs/overnight_vis_adaptive_8h_20260601/logs/watcher.log
/data/liuyu/outputs/code_backups/vis_*_20260601/
/data/liuyu/outputs/patches_vis_20260602/
```

### Validity notes:
- Pre-timestamp-fix traces (before commit 856eba5) may have been overwritten
- Salad_dressing traces from g23 (GPU3) are suspect due to Xid31 crashes
- All adaptive controller traces include controller config in filename
- Fixed-duration traces from the main sweep (33 rollouts) are in the runs/ directory but some may be pre-fix overwrites

## 4. Adaptive Controller Result Summary

From log-based extraction (source: `/data/liuyu/outputs/milestone_7_vis_controlled_rollout_micro_20260601/logs/`):

| Task | Controller | K | Q | Min | Max | Attacks | Stop Reason | Success | Flips | ArmL2 | GPU |
|------|-----------|---|---|-----|-----|---------|-------------|---------|-------|-------|-----|
| cc | open_streak | 2 | 0 | 0 | 16 | 2 | streak_threshold | True | 2/16 | 0.11 | 1,2 |
| cc | open_streak | 3 | 0 | 0 | 16 | 13 | streak_threshold | True | 4/16 | 0.70 | 0,1 |
| cc | open_streak | 3 | 0 | 0 | 16 | 16 | max_duration | True | 7/16 | 0.88 | 1,0 |
| cc | open_streak | 5 | 0 | 0 | 16 | 11 | force_window_end | True | 7/11 | 0.91 | 1,2 |
| cc | min_hold_qpos | 0 | 0.008 | 14 | 20 | 20 | max_duration | **False** | 11/20 | 0.87 | 1,2 |
| salad | open_streak | 2 | 0 | 0 | 16 | 16 | max_duration | **False** | 4/21 | 0.70 | 4,5 |
| salad | open_streak | 2 | 0 | 0 | 16 | 8 | streak_threshold | True | 4/21 | 0.46 | 6,7 |
| salad | open_streak | 3 | 0 | 0 | 16 | 15 | streak_threshold | **False** | 6/21 | 0.83 | 2,3 |
| salad | open_streak | 5 | 0 | 0 | 16 | 16 | max_duration | True | 5/21 | 0.80 | 4,5 |
| salad | open_streak | 5 | 0 | 0 | 16 | 16 | max_duration | True | 7/21 | 0.67 | 4,5 |
| salad | min_hold_qpos | 0 | 0.008 | 14 | 20 | 14 | qpos_threshold | **False** | 1/21 | 0.29 | 4,5 |
| ket | open_streak | 3 | 0 | 0 | 16 | 16 | max_duration | True | 3/16 | 0.86 | 6,7 |
| ket | open_streak | 3 | 0 | 0 | 16 | 14 | streak_threshold | **False** | 8/16 | 0.43 | 4,5 |
| ket | open_streak | 5 | 0 | 0 | 16 | 5 | streak_threshold | **False** | 5/11 | 0.42 | 4,5 |
| ket | open_streak | 5 | 0 | 0 | 16 | 11 | force_window_end | True | 1/11 | 0.89 | 6,3 |
| ket | min_hold_qpos | 0 | 0.012 | 16 | 20 | 20 | max_duration | True | 8/20 | 0.86 | 4,5 |

(crashed jobs excluded; see known bugs)

**Key observations:**
- K=2 streak-stop is too conservative (cc stops at 2 attacks)
- K=3 streak-stop is insufficient (cc survives with 13 attacks)
- min_hold_qpos_cap at min14_q008: cream fails (20 attacks, never hit qpos), salad fails at qpos_threshold (14 attacks, qpos hit 0.01)
- ket min_hold16_q012 survives (20 attacks, qpos stayed at 0)
- Salad failures often have very few OPEN flips — may be denominator/environment noise, not VIS-specific

## 5. Known Bugs / Scientific Caveats

See `tables/codex_handoff_known_bugs_and_fixes.csv` for structured list.

Critical items:

**BUG-001 qpos timing bug (severity: HIGH)**
- Current controller uses qpos observed BEFORE env.step()
- This is lagged signal; physical gripper response happens AFTER env.step()
- Fix: log both qpos_pre_step and qpos_post_step
- Controller can continue using online qpos_pre, but audit must use qpos_post

**BUG-002 denominator mixing (severity: HIGH)**
- After controller stops, in_window still contains clean decode rows
- window_token_flips counts across ALL window steps, not just attacked steps
- Fix: add pgd_applied, attack_attempted, controller_active per-step flags
- Add attacked_step_denominator column to summary

**BUG-003 official_success insufficient (severity: MEDIUM)**
- success=True/False from done flag is not contact-quality conclusion
- Salad_K2 failure: 4/21 flips, armL2=0.70, but success=False — could be environmental
- Fix: add CQFR/CQSR placeholders, manual audit flag

**BUG-004 attack_dt not computed (severity: LOW)**
- attack_dt is always 0 or meaningless
- Fix: measure PGD elapsed time per step

**BUG-005 GPU driver reload wording (severity: LOW)**
- GPU_HEALTH report says "without affecting other GPUs" — incorrect
- rmmod nvidia kills ALL CUDA contexts across all GPUs
- Fix: update report language

**BUG-006 FAILURE_MECHANISM_AUDIT provenance (severity: MEDIUM)**
- Fail rates/averages in report are from log extraction, not trace recomputation
- Need source trace table linking each number to a file path
- Fix: recompute from trace CSVs, add source column

**BUG-007 adaptive controller overclaim risk (severity: MEDIUM)**
- min_hold_qpos_cap shows promise but is exploratory
- Current 1-sample-per-config is insufficient for strong claim
- Fix: downgrade language, mark as negative calibration

## 6. Current Scientific Conclusions

### Allowed Claims

1. VIS PGD can change decoded gripper action in corrected decode pipeline (confirmed by prompt() wrapper + action prefix token fix).
2. cream_cheese d16/d18/d20 is the strongest controlled VIS positive (4/5 seeds fail at d16, random baseline mostly clean).
3. qpos_delta and longest OPEN streak mediate failure better than OPEN count or fixed duration.
4. salad_dressing is VIS-vulnerable, not a robust VIS control (fails at d12/d14/d16/d20/d40).
5. ketchup is more tolerant than salad but not perfectly robust (1/3 seeds fail at d16).
6. tomato_sauce is highly resilient (requires d60, 3x the selective budget).
7. No universal fixed-duration selective D* exists across tested tasks.
8. ProprioNoStep handles phase/timing better than VIS transfer.
9. GPU layout: healthy-primary + flaky-secondary enables all 8 GPUs.

### Forbidden Claims

1. Object-wide selective VIS attack solved.
2. Adaptive controller solved or validated.
3. Detector-triggered VIS validated.
4. Simulator success rate alone proves VIS failure mechanism.
5. d20 is universally selective.
6. salad_dressing is a robust control.
7. Production-grade learned controller from 35-row dataset.

### Pending (Requires Further Work)

1. VisualTransferHead may bridge VIS physical-transfer gap.
2. min_hold_qpos_cap may reduce over-budget if qpos timing is fixed.
3. All-8-GPU layout needs watcher-level Xid monitoring.
4. Post-step qpos audit may reveal stronger physical response than pre-step.
5. CQ/CQFR integration needed for failure attribution.

## 7. GPU Operational Handoff

### Stable layout:
```bash
# g10: GPU1 (primary, C+G) + GPU0 (secondary, compute only)
CUDA_VISIBLE_DEVICES=1,0

# g23: GPU2 (primary, C+G) + GPU3 (secondary, compute only)
CUDA_VISIBLE_DEVICES=2,3

# g45: GPU4 (primary, C+G) + GPU5 (secondary, compute only)
CUDA_VISIBLE_DEVICES=4,5

# g67: GPU6 (primary, C+G) + GPU7 (secondary, compute only)
CUDA_VISIBLE_DEVICES=6,7
```

### Rules:
- Primary GPU must be a known-healthy card (1,2,4,5,6)
- Secondary can be historical-Xid card (0,3,7)
- NEVER use GPU0/3/7 as primary in CUDA_VISIBLE_DEVICES
- Log nvidia-smi before and after every batch
- Monitor Xid every 90-120 seconds
- If Xid occurs on a pair, isolate that pair and continue with remaining pairs
- Do NOT run rmmod/modprobe while other GPU jobs exist on ANY GPU
- After driver reload, re-verify all GPUs with basic stress test

### Monitoring:
```bash
nvidia-smi --query-gpu=index,pci.bus_id,memory.used,temperature.gpu --format=csv
dmesg -T | grep -i "NVRM\|Xid" | tail -50
```

## 8. Immediate Codex Tasks

### P0 — Static code audit and minimal fixes

1. Inspect `scripts/vis_rollout_adaptive_v3.py` for qpos_pre/post logging gap
2. Add `qpos_post_step` field to trace CSV
3. Add `pgd_applied`, `attack_attempted`, `controller_active` per-step flags
4. Fix `attack_dt` to measure actual PGD elapsed time
5. Add CQ/CQFR placeholder (None if unavailable)
6. Run `py_compile` to verify syntax

### P1 — Rebuild adaptive result table

1. Parse all 16 adaptive controller trace CSVs
2. Recompute: attacks_applied, attacked OPEN count, attacked streak, posthoc qpos_delta
3. Flag missing/overwritten/crashed traces
4. Output to `tables/codex_handoff_adaptive_result_summary.csv`

### P2 — Downgrade adaptive controller language

1. Update `VIS_FAILURE_MECHANISM_AUDIT.md`: "exploratory negative calibration"
2. Add "NOT production controller" disclaimer
3. Add source trace table linking each number to file path

### P3 — Prepare VisualTransferHead mainline

1. Create `reports/VIS_TRANSFER_HEAD_DATASET_SCHEMA.md`
2. Create `tables/vis_transfer_feature_schema_audit_template.csv`
3. No model training unless explicitly approved

### P4 — Optional smoke validation (only after P0/P1)

Run 2-3 minimal smoke rollouts only to verify logging fixes:
```bash
python scripts/vis_rollout_adaptive_v3.py \
  --task cream_cheese --condition vis_pgd \
  --gpu_pair 1,0 --seed 0 \
  --controller min_hold_qpos_cap --min_attacks 14 --max_duration 20 --Q 0.008
```
Do NOT launch real rollouts unless Leon explicitly approves.

## 9. Exact Commands for Codex

```bash
# Verify syntax
PYTHONPATH=src python -m py_compile scripts/vis_rollout_adaptive_v3.py

# Run tests
PYTHONPATH=src python -m pytest tests/v4/test_token_prefix_pgd_interface.py \
  tests/v4/test_openvla_redecode.py tests/v4/test_vis_arm_drift_sweep.py \
  tests/v4/test_success_predicate_regression.py tests/v4/test_sustained_proxy_burst.py

# List all adaptive traces
ls /data/liuyu/outputs/milestone_7_vis_controlled_rollout_micro_20260601/runs/vis_*_open_streak* \
   /data/liuyu/outputs/milestone_7_vis_controlled_rollout_micro_20260601/runs/vis_*_min_hold*

# Parse one adaptive trace
python -c "
import csv
rows = list(csv.DictReader(open('/data/liuyu/outputs/milestone_7_vis_controlled_rollout_micro_20260601/runs/vis_cream_cheese_s0_vis_pgd_full_d20_w65_84_seed0_min_hold_qpos_cap_K0_Q0.008_md20_114917_trace.csv')))
attacked = [r for r in rows if float(r.get('ctrl_attacks',0)) > 0 or r.get('ctrl_active','') == 'True']
print(f'Total window: {len([r for r in rows if r[\"in_window\"]==\"True\"])}, attacked: {len(attacked)}')
"

# Dry-run (no actual rollout)
python scripts/vis_rollout_adaptive_v3.py --task cream_cheese --condition vis_pgd \
  --gpu_pair 1,0 --seed 0 --controller min_hold_qpos_cap \
  --min_attacks 14 --max_duration 20 --Q 0.008 --dry_run
```

## 10. Final Handoff Summary

**Frozen:**
- VIS corrected-decode pipeline (prompt wrapper, action prefix, openvla_redecode)
- Fixed-duration sweep results (~50 rollouts)
- Adaptive controller exploratory calibration (streak-stop = negative, min_hold_qpos = exploratory)
- GPU layout discovery (healthy-primary + flaky-secondary = all 8 GPUs usable)
- VISTransfer dataset v2 (61 rows, qpos_delta dominant feature)

**Unresolved:**
- qpos timing (pre-step vs post-step)
- Denominator mixing (clean rows in window after controller stop)
- CQ/CQFR integration
- Salad failure attribution (VIS-specific vs environmental)
- Cream leave-one-task-out weakness (acc=0.40)
- VisualTransferHead not yet implemented

**Codex should do first:**
1. P0: Fix qpos timing and denominator flags in adaptive script
2. P1: Rebuild adaptive result table from traces
3. P2: Downgrade adaptive controller language
4. P3: Prepare VisualTransferHead schema

**Codex must NOT claim:**
- Adaptive controller solved
- Object-wide selective VIS
- Detector-triggered VIS validated
- Salad is robust control
- Production-grade learned controller

**Data locations:**
- Rollout traces: `/data/liuyu/outputs/milestone_7_vis_controlled_rollout_micro_20260601/runs/`
- Rollout logs: `/data/liuyu/outputs/milestone_7_vis_controlled_rollout_micro_20260601/logs/`
- Transfer dataset: `/data/liuyu/outputs/vis_transfer_controller_autorun_20260601/tables/`
- Code backups: `/data/liuyu/outputs/code_backups/vis_*_20260601/`
- Server repo: `/data/liuyu/repos/openvla-gripper-dutycycle-attack-clean-main-20260524/`
- GitHub: `https://github.com/Leo-6-maker/openvla-gripper-dutycycle-attack` (branch `exp/vis-payload-upgrade-validation-20260601`)
