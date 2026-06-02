# Handoff — OpenVLA Gripper Duty-Cycle Attack / Detector Project

**Generated**: 2026-05-29 ~19:00 CST
**For**: New ChatGPT / Claude session handoff
**Branch**: `eval/official-libero-clean-20260525`
**Remote commit**: `087044390498f271fddfe009d95e6701fc9450fd`

---

## 0. TL;DR

We are building a detector-triggered attack pipeline for OpenVLA gripper duty-cycle vulnerability. Current focus is **Object-only ProprioNoStep TCN detector → matched command-layer control pilot**.

**Most important facts right now**:
1. A success predicate regression caused a false 0/30 clean collapse. LIBERO uses `done=True` to signal task success; `info["success"]` is NOT populated. The bug was fixed in commit `0870443` and GitHub is frozen. Do NOT change code without explicit approval.
2. Pilot v1 (40-rollout) was aborted due to this bug. It is **debug-only, not formal evidence**.
3. Candidate re-scan with fixed success predicate is running/complete on GPU2,6. All results so far show clean success (cream_cheese, ketchup, salad_dressing all True).
4. Next: complete pure-clean re-scan → detector-clean scan on GPU4,5 → state selection → pilot v2 (24 rollouts) only if clean-stability gate passes.
5. `gripper_inversion_proxy` is a command-layer gripper inversion proxy, **NOT visual attack (VIS)**. Never call it VIS or visual PGD.
6. GPU0 is quarantined (Xid13). GPU7 cannot run OpenVLA 7B (OOM). GPU2,6 is primary pair for pilot work.

---

## 1. Server Access

```bash
# SSH jump host connection
ssh -J scene@10.60.133.3 liuyu@10.60.133.4

# Primary conda environment
/data/aviary/envs/openvla_official_libero_20260525

# Python binary
/data/aviary/envs/openvla_official_libero_20260525/bin/python

# Code repository
cd /data/liuyu/repos/openvla-gripper-dutycycle-attack-clean-main-20260524

# GitHub
https://github.com/Leo-6-maker/openvla-gripper-dutycycle-attack
Branch: eval/official-libero-clean-20260525
Frozen commit: 087044390498f271fddfe009d95e6701fc9450fd
```

---

## 2. Safety / Hard Rules

### GPU Rules
| GPU | Status | Permitted Use |
|-----|--------|---------------|
| 0 | ⛔ QUARANTINED | Never use. Xid13 / CUDA illegal memory access history |
| 1,3 | L10-B done, now likely idle | Check before use |
| 2,6 | Primary pair for Object pilot v2 | Use after confirming no fresh Xid |
| 4,5 | Secondary pair | Detector-clean scan, short tasks |
| 7 | Idle | Lightweight tasks only (visual extraction, small model training). NO OpenVLA 7B rollout (OOM at 10.66/10.75 GB). NO VIS PGD |

### Code Rules
- **Do NOT modify code without explicit approval**
- If code change needed: explain reason → proposed fix → test → commit → push → report new SHA
- Current frozen commit: `087044390498f271fddfe009d95e6701fc9450fd`
- Before any experiment: confirm `local == remote` and `working tree clean`

### Experiment Rules
- Never call `gripper_inversion_proxy` as VIS or visual attack
- Never use pilot v1 as formal evidence
- Never launch attack pilot unless clean-stability gate passes
- Never use `info["success"]` as primary success predicate
- If fresh Xid appears on any active GPU: stop job, quarantine outputs, report

### Success Predicate Rule (CRITICAL)
- **LIBERO signals task success through `done=True`**
- `info["success"]` is absent in LIBERO's info dict
- Use `success_official = done_any` as the primary success predicate
- `success_source = "done_any_LIBERO_official"`

---

## 3. Current Git State

```bash
cd /data/liuyu/repos/openvla-gripper-dutycycle-attack-clean-main-20260524
git branch --show-current || git rev-parse --abbrev-ref HEAD
# Expected: eval/official-libero-clean-20260525

git rev-parse HEAD
# Expected: c62214f... (local)

git status --short
# Expected: clean (0 modified files, only untracked backups/configs)

git ls-remote origin eval/official-libero-clean-20260525
# Expected: 087044390498f271fddfe009d95e6701fc9450fd

# Verify local == remote
```

**Recent commits (server)**:
```
c62214f Fix git branch command compatibility in launch script
c078626 Fix LIBERO success predicate — use done=True not info[success]
ab6cd5c Hotfix 3: fix run_dir collision + manifest provenance
2fc98b5 Hotfix 2: fix P0/P1 bugs + add smoke flag
7032d61 Hotfix P0/P1 bugs in detector-triggered attack runner
f1beaa5 Freeze Object detector pilot readiness and GPU provenance audits
```

---

## 4. Core Method

**Pipeline**: artifact-rich clean rollout → privileged teacher labels → no-timestep causal student detector → detector-triggered matched attack/control pilot

**Current detector**: ProprioNoStep TCN (Object-only, 38,602 params, 16-step history, 13 proprio/action features)

**Detector checkpoint**:
```
/data/liuyu/outputs/milestone_2e3_object100_visual_proprio_no_step_20260527/models/ProprioNoStep_baseline.pt
SHA256: 4b3f3d479d6bbb92b2bd15cffec0be587bf221dc81663aaff93e44afdd9c7b1f
```

**Detector inputs** (13 features):
```
gripper_command, gripper_qpos, gripper_width,
eef_x, eef_y, eef_z, eef_vx, eef_vy, eef_vz,
action_dx, action_dy, action_dz, action_gripper
```

**Student MUST NOT use**: normalized_step, object_pose, target_pose, object_to_target_distance, teacher_window fields, future timesteps, attack/manual/oracle/random outcomes, success/done as predictive input.

**Attack conditions**:
1. `clean` — detector logging, no attack
2. `oracle_open` — force gripper fully open on trigger
3. `random_control` — random gripper sign on trigger
4. `gripper_inversion_proxy` — action-level gripper inversion + noise (NOT visual PGD)

---

## 5. Critical Bug History

### The 0/30 False Collapse (May 29)

**Symptom**: All 30 candidate scan episodes (5 tasks × 3 states × 2 conditions) recorded as `success=False`. Object-100 had cream_cheese 10/10.

**Root cause**: A hotfix changed the success predicate from:
```python
if done:
    success = True  # CORRECT
```
to:
```python
if done:
    success = bool(info.get("success", False))  # BROKEN
```

**Why**: LIBERO's `env.step()` returns `done=True` when the task succeeds, but the `info` dict does NOT contain a `"success"` key. So `info.get("success", False)` always returns `False`.

**Fix** (commit `c078626`):
- Reverted to `success = True` when `done=True`
- Added per-episode tracking: `ep_done_any`, `ep_reward_max`, `ep_info_success`
- Added manifest fields: `success_official`, `success_done`, `success_info_present`, `success_info_value`, `success_reward`, `reward_max`, `done_any`, `timeout`, `success_source`
- Added `done` and `reward` to step_records
- Added regression test: `tests/v4/test_success_predicate_regression.py` (6/6 pass)

**IMPORTANT**: The earlier conclusion of "LIBERO physics non-determinism" has been **withdrawn**. The collapse was a code bug, not physics.

### Other Fixed Bugs
1. Detector loaded then reset to None (double `detector=None` line)
2. Step records missing detector/attack fields
3. Online feature schema wrong (gripper_command=0, action_gripper=raw[3], eef_vel=0)
4. `VIS_targeted` was fake action-level inversion — renamed to `gripper_inversion_proxy`
5. `run_dir` collision across conditions — fixed to use `run_id` (includes condition prefix)
6. Clean condition incorrectly setting `attack_applied=True`
7. Attack burst duration not implemented — added `attack_remaining`
8. Launch script git compatibility (older git version)
9. Aggregate/CQ scripts still referencing `VIS_targeted` — changed to `gripper_inversion_proxy`

---

## 6. Completed Milestones

### Object-100
- 100 episodes, SR 81/100
- Privileged coverage: 100%
- Teacher detector: 81/81 clean-success windows, 0/19 failed FP
- Labeled dataset: 18,875 rows, 0 leakage
- Visual features: 18,875 images, 2176-dim (DINOv2+SigLIP fused)
- ProprioNoStep TCN: coverage 99.1%, AUROC 0.969, 0 miss, 0 FP
- Output: `/data/liuyu/outputs/milestone_2e2_object100_privileged_artifact_rich_20260527`
- Models: `/data/liuyu/outputs/milestone_2e3_object100_visual_proprio_no_step_20260527/models/`

### Goal-100
- 100 episodes, SR 71/100
- Aggregation: 0 missing, 0 duplicate
- Privileged audit: pick_place 60 eps 100% priv, articulated 30 eps correctly gated, planar 10 eps correctly gated
- Teacher labels: 47/60 eligible windows, 0 FP on 29 failures
- Visual features: 17,003 images, dim=2176, 0 NaN/Inf, 100/100 eps joined
- Labeled export: 17,003 rows, 0 model leakage
- Output: `/data/liuyu/outputs/milestone_2e5_goal100_parser_v2_privileged_rerun_20260527`
- Features: `/data/liuyu/outputs/milestone_2g_goal100_frozen_visual_features_gpu7_20260529`

### L10
- 100/100 episodes (L10-A + L10-B), SR 52%
- Aggregation complete, audit complete
- 3 mechanisms: articulated (10), multi_object_transfer (60), pick_place_transfer (30)
- Teacher labels frozen (episode-level)
- Output: `/data/liuyu/outputs/milestone_2e5_l10100_parser_v2_privileged_rerun_20260527`

### Forced Micro Smoke (GPU2,6)
- Verified detector/attack injection works
- Clean: triggers>0, attacks=0 ✓
- Oracle: triggers>0, attacks>0, gripper=1.0 ✓
- Random: triggers>0, attacks>0 ✓
- Proxy: triggers>0, attacks>0 ✓
- 4 unique run_dirs, all detector fields logged
- Output: `/data/liuyu/outputs/milestone_2f_object_detector_forced_micro_gpu26_20260529`

### Spatial
- 100 episodes exist, SR 74%
- Parser v2: 10/10 pick_place_transfer
- **Must rerun**: missing `object_pose_json`, `target_pose_json`, `object_to_target_distance`, `object_eef_distance` from step_records
- Audit report: `/data/liuyu/outputs/milestone_2g_spatial_reuse_audit_20260529/tables/spatial_rerun_decision.csv`
- Output: `/data/liuyu/outputs/milestone_2e4_cross_suite300_privileged_artifact_rich_20260527`

---

## 7. Current Running / Pending Experiments

### Pure Re-scan (GPU2,6)
- **Status**: Running. 8+/15 done, ALL True with `done_any=True`
- Tasks: cream_cheese, ketchup, salad_dressing, tomato_sauce, milk
- 3 states each (0, 1, 2)
- Output: `/data/liuyu/outputs/milestone_2f_object_pilot_v2_clean_candidate_scan_20260529`

### Detector-Clean Scan (GPU4,5 — PENDING)
- Same tasks/states as pure re-scan
- `--detector_path` + `--attack_condition clean` + NO `--force_detector_trigger`
- Must verify: attack_applied=0, original_env_action==attacked_env_action, success_official uses done_any

### Pilot v1 (GPU2,6 — ABORTED)
- 10 clean cream_cheese + milk: all recorded as False due to success predicate bug
- **DEBUG-ONLY, NOT FORMAL EVIDENCE**
- Output: `/data/liuyu/outputs/milestone_2f_object_detector_matched_attack_pilot_20260529`

### GPU Status
```
Check with: nvidia-smi
GPU0: QUARANTINED — do not use
GPU2,6: Running pure re-scan
GPU4,5: Idle — ready for detector-clean scan
GPU1,3: Likely idle (L10 done)
GPU7: Idle — lightweight tasks only
```

---

## 8. Pending Decisions

1. **Pilot v2 launch**: Only if clean-stability gate passes after both scans
2. **State selection**: Choose 2 tasks × 3 states from candidates with highest clean stability
3. **Spatial rerun**: Deferred until Object pilot v2 is done or explicit approval
4. **Goal smoke training**: Dataset format needs adaptation (not a blocker)
5. **L10 step-level export**: Deferred
6. **Universal detector**: Blocked until Goal/L10 labels fully audited

---

## 9. Exact Next Commands

### 9.1 First Checks (run immediately on new session)
```bash
# Connect
ssh -J scene@10.60.133.3 liuyu@10.60.133.4

# Git verification
cd /data/liuyu/repos/openvla-gripper-dutycycle-attack-clean-main-20260524
git rev-parse --abbrev-ref HEAD
git rev-parse HEAD
git status --short
git ls-remote origin eval/official-libero-clean-20260525

# GPU verification
nvidia-smi
dmesg -T | grep -i "NVRM\|Xid" | tail -30
df -h /data/liuyu
```

### 9.2 Check Pure Re-scan Status
```bash
echo "Pure re-scan manifests:" $(find /data/liuyu/outputs/milestone_2f_object_pilot_v2_clean_candidate_scan_20260529/runs/libero_object/scan2_pure_* -name run_manifest.json 2>/dev/null | wc -l) "/15"
for f in /data/liuyu/outputs/milestone_2f_object_pilot_v2_clean_candidate_scan_20260529/runs/libero_object/scan2_pure_*/run_manifest.json; do
  python3 -c "import json; d=json.load(open('$f')); print(d['task_name'][:50], 's'+str(d['state_id']), d['success'])"
done
```

### 9.3 Launch Detector-Clean Scan (if pure scan done and GPU4,5 idle)
```bash
cd /data/liuyu/repos/openvla-gripper-dutycycle-attack-clean-main-20260524
for TID in 1 2 4 5 7; do
  CUDA_VISIBLE_DEVICES=4,5 MUJOCO_GL=egl PYTHONUNBUFFERED=1 \
  /data/aviary/envs/openvla_official_libero_20260525/bin/python -u \
  scripts/run_official_eval_artifact_rich.py \
    --model_path /data/aviary/models/openvla/openvla-7b-finetuned-libero-object \
    --task_suite_name libero_object \
    --task_start $TID --task_count 1 \
    --num_trials_per_task 3 \
    --worker_id "scan2_det_t${TID}" \
    --save_step_records \
    --cuda_visible_devices 4,5 --render_gpu_device_id 4 \
    --output_root /data/liuyu/outputs/milestone_2f_object_pilot_v2_clean_candidate_scan_20260529 \
    --run_id_prefix scan2_det \
    --detector_path /data/liuyu/outputs/milestone_2e3_object100_visual_proprio_no_step_20260527/models/ProprioNoStep_baseline.pt \
    --detector_hazard_threshold 0.1 --detector_trigger_duration 5 \
    --attack_condition clean
done
```

### 9.4 Select States and Launch Pilot v2
After both scans complete, select states where:
- pure_clean success_official=True AND detector_clean success_official=True
- detector_clean attack_applied=0
- No CQ failure, no fresh Xid

If gate passes (≥2 tasks × ≥3 states), launch:
```bash
cd /data/liuyu/repos/openvla-gripper-dutycycle-attack-clean-main-20260524
# Use scripts/launch_object_matched_attack_pilot_gpu26.sh
# BUT modify: use selected tasks/states only
# Output root: /data/liuyu/outputs/milestone_2f_object_detector_matched_pilot_v2_successfix_20260529
```

### 9.5 Post-Pilot Aggregation and CQ
```bash
cd /data/liuyu/repos/openvla-gripper-dutycycle-attack-clean-main-20260524
bash scripts/aggregate_object_attack_pilot.sh
bash scripts/evaluate_cq_object_attack_pilot.sh
```

---

## 10. Output Roots

| Name | Path |
|------|------|
| Object-100 | `/data/liuyu/outputs/milestone_2e2_object100_privileged_artifact_rich_20260527` |
| Object visual features | `/data/liuyu/outputs/milestone_2e3_object100_visual_features_openvla_20260527` |
| Object detector models | `/data/liuyu/outputs/milestone_2e3_object100_visual_proprio_no_step_20260527` |
| Goal-100 | `/data/liuyu/outputs/milestone_2e5_goal100_parser_v2_privileged_rerun_20260527` |
| Goal visual features | `/data/liuyu/outputs/milestone_2g_goal100_frozen_visual_features_gpu7_20260529` |
| Goal labeled datasets | `/data/liuyu/outputs/milestone_2g_goal100_labeled_datasets_20260529` |
| L10 | `/data/liuyu/outputs/milestone_2e5_l10100_parser_v2_privileged_rerun_20260527` |
| Cross-Suite-300 (old) | `/data/liuyu/outputs/milestone_2e4_cross_suite300_privileged_artifact_rich_20260527` |
| Pilot v1 (ABORTED) | `/data/liuyu/outputs/milestone_2f_object_detector_matched_attack_pilot_20260529` |
| Forced micro | `/data/liuyu/outputs/milestone_2f_object_detector_forced_micro_gpu26_20260529` |
| Candidate scan | `/data/liuyu/outputs/milestone_2f_object_pilot_v2_clean_candidate_scan_20260529` |
| Pilot v2 (PLANNED) | `/data/liuyu/outputs/milestone_2f_object_detector_matched_pilot_v2_successfix_20260529` |
| Parity audit | `/data/liuyu/outputs/milestone_2f_pilot_clean_parity_audit_gpu45_20260529` |
| GPU0,7 smoke | `/data/liuyu/outputs/gpu07_engineering_smoke_object_detector_20260529` |
| Spatial audit | `/data/liuyu/outputs/milestone_2g_spatial_reuse_audit_20260529` |
| Goal smoke | `/data/liuyu/outputs/milestone_2g_goal100_detector_smoke_20260529` |

---

## 11. Metrics / Interpretation Rules

- **official SR**: use `success_official = done_any` (not `info["success"]`)
- **CQFR** (Contact Quality Failure Rate) and **CQSR** (Contact Quality Success Rate): use alongside official SR to detect failures missed by LIBERO's loose success criteria
- **SR-CQ mismatch**: `official_SR - CQSR` identifies episodes LIBERO says succeeded but contact quality says failed
- **gripper_inversion_proxy**: command-layer gripper inversion + noise, NOT visual attack, NOT VIS PGD
- **Clean-stability gate**: only states where clean succeeds (both pure and detector-clean) enter matched causal analysis
- **Failed clean states**: diagnostic only, excluded from primary matched analysis
- **Oracle gate**: oracle_open CQFR >> clean CQFR proves detector-selected windows are physically gripper-sensitive
- **VIS-vs-random gate**: proxy CQFR > random CQFR, proxy gripper-open metrics > random
- **CQ gate**: CQ metrics detect failures that official SR may miss

---

## 12. Known P1 / Technical Debt

1. **step_records reward/done timing**: `reward` and `done` are logged BEFORE `env.step()` for that action, so they may be one-step-shifted relative to the action that caused them. Episode-level `success_official` via `done_any` is correct. If using per-step done/reward for analysis, account for this offset.
2. **Goal smoke dataset format**: The training script (`tmp_train_obj100.py`) expects Object-100 data format. Goal-100 labeled export has different column structure. Needs adaptation script.
3. **L10 step-level export**: Teacher labels exist at episode level. Step-level labeled export is deferred.
4. **Spatial rerun**: Existing artifacts lack `object_pose_json`, `target_pose_json`, `object_to_target_distance`, `object_eef_distance`. Must rerun with parser-v2.
5. **pytest environment**: `pytest` is broken in the conda env (missing `pygments` dependency). Use `python -m unittest` for specific test modules, `py_compile` for syntax, and `bash -n` for shell script validation.
6. **Watcher**: The autonomous 15-min watcher was killed during session compaction. May need restart for long-running experiments.

---

## 13. Recommended Next 2–4 Hours Plan

1. **Complete pure re-scan** on GPU2,6 (if not already done)
2. **Launch detector-clean scan** on GPU4,5 (parallel to any remaining pure-clean)
3. **Generate state selection table**: identify eligible task/state pairs
4. **If gate passes**: launch 24-rollout pilot v2 on GPU2,6
5. **After pilot v2**: aggregation → CQ evaluation → clean-stability gate → trigger/attack audit
6. **Lower priority**: Goal smoke adaptation, L10 export, Spatial planning

---

## 14. Final Warning

- **Do NOT call `gripper_inversion_proxy` VIS or visual attack.** It is a command-layer proxy.
- **Do NOT use pilot v1 as formal evidence.** It was aborted due to success predicate bug.
- **Do NOT launch attack pilot unless clean-stability gate passes.**
- **Do NOT use `info["success"]` as primary success predicate.** Use `done_any`.
- **Do NOT change code** without explaining reason, fix, test, commit, push, and reporting new SHA.
- **Do NOT use GPU0.** It has permanent Xid13 and CUDA illegal memory access.
- **Do NOT use GPU7 for OpenVLA 7B rollout or VIS PGD.** It OOMs at inference.
- **Confirm `local == remote` and working tree clean** before starting any new experiment.
- **If fresh Xid appears**: stop affected job, quarantine outputs, report immediately.
