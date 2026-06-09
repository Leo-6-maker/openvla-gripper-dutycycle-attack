# S7 Handoff / Context Compression

**Date**: 2026-06-10
**Branch**: exp/vis-prefix-margin-repair-20260603
**S6 frozen**: c0ca025

---

## 1. Scientific State

### S6 Closed (c0ca025)

**Allowed main claim:**
> Layer-1 CleanRand abstain-first pipeline validated at command level.
> A-group robustness over attack_seed 5–8: cmd_hit=0.94, cmd_rand=0.06, yield_cmd=+0.94.

**Supporting evidence:**
- Fixed-env K-repeat protocol gives stable probability labels.
- Stable pool v2: 40 parents across 9 tasks.
- Pipeline v0.3 fresh confirmation: 5/5 gates PASS, yield=+1.00 on seeds 5-6.
- Robustness seeds 7-8: yield=+0.88, combined 5-8 yield=+0.94.

**Action-logit conclusion (6bb53db):**
> Mechanism evidence only. Explains FP/FN structure but does not improve selector.
> Minimal PASS — not detector improvement.

**Forbidden claims:**
- Detector solved.
- Layer-2 ranking solved.
- Strict physical bridge solved.
- Action-logit improves detector.
- Action-hidden result exists (S7 not yet complete).
- Cross-task/suite generalization demonstrated.

### S7 Current Target

Test whether action-hidden features (last-layer hidden state at gripper token)
improve Layer-2 ranking beyond action-logit/action-dynamics.

**No VIS/RAND attack confirmation until hidden readout strong PASS.**

---

## 2. Current Running Jobs (2026-06-10)

| GPU | Task | Progress | ETA | Status |
|-----|------|----------|-----|--------|
| 1,0 | hidden shard 0–19 | 3/20 | ~34 min | running |
| 4,5 | hidden shard 20–39 | 4/20 | ~32 min | running |
| 2,6 | idle | — | — | healthy, available for auxiliary |
| 3,7 | blacklisted | — | — | do not use |

---

## 3. Server Runbook

### Connection

```bash
ssh vla  # jump: scene@10.60.133.3 → liuyu@10.60.133.4
```

Server: klfy-SYS-4028GR-TR2, user: liuyu

### Repo

```
Path: /data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607
Branch: exp/vis-prefix-margin-repair-20260603
Remote: origin → /data/liuyu/repos/openvla-gripper-dutycycle-attack-reviewed-20260605 (local bare)
GitHub: https://github.com/Leo-6-maker/openvla-gripper-dutycycle-attack
```

### Python Environment

```
Conda env: openvla_official_libero_20260525
Python: 3.10.13
PYTHONPATH: src
Model: /data/aviary/models/openvla/openvla-7b-finetuned-libero-object
```

### Headless Rendering

```bash
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
unset DISPLAY
```

Set BEFORE importing torch/mujoco/OpenGL.

### tmux

```bash
tmux new -s s7_hidden_shard10
tmux new -s s7_hidden_shard45
tmux new -s watcher
tmux ls
```

---

## 4. GPU Pairing Convention

### Physical mapping (8× RTX 2080 Ti, 11 GiB each)

| Pair | Physical GPUs | CUDA_VISIBLE_DEVICES | Role |
|------|---------------|---------------------|------|
| worker_10 | 1,0 | 1,0 | primary extraction |
| worker_45 | 4,5 | 4,5 | trusted sidecar / second shard |
| worker_26 | 2,6 | 2,6 | auxiliary/duplicate only |
| blacklist | 3,7 | — | permanently blacklisted (Xid31 history) |

### Critical EGL Rule

CUDA_VISIBLE_DEVICES remaps device IDs inside PyTorch, but EGL/MuJoCo refers to PHYSICAL GPU IDs. Scripts must map CUDA index to physical GPU for `render_gpu_device_id`.

```python
visible = [int(x) for x in os.environ.get('CUDA_VISIBLE_DEVICES','').split(',')]
render_physical_gpu = visible[gpu_ids[1]]  # NOT gpu_ids[1]!
```

### Launch Template

```bash
export CUDA_VISIBLE_DEVICES=1,0
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
unset DISPLAY
cd /data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607
PYTHONPATH=src nohup python -u scripts/<name>.py --gpu_pair 0,1 ... > log 2>&1 &
```

---

## 5. S7 Hidden Extraction Gates

### Extraction gates
- coverage >= 36/40 stable pool parents
- feature_source = pre_window_only
- online_safe = True
- prompt present, task-language aligned
- hidden_dim = 4096 (consistent with smoke)
- no VIS/RAND/yield/qpos_delta/success/failure as features
- no window/post steps in readout features

### Readout models
```
TaskOnly
CleanProprio
ActionLogit
ActionHiddenOnly
ActionLogit + Hidden
CleanRand + ActionHiddenRank
CleanRand + RandomRank
Oracle upper bound
```

### Decision rules
- **Strong PASS**: CleanRand + ActionHiddenRank > CleanRand + RandomRank → generate Layer-2 confirmation queue ONLY, do not launch attack
- **Medium PASS**: ActionHiddenOnly > ActionLogit/CleanProprio but not best → mechanism report, no attack
- **Minimal PASS**: hidden separates FP/FN only → mechanism evidence, no attack
- **FAIL**: stop Layer-2 hidden route, preserve S6 Layer-1 result

---

## 6. Forbidden Actions

- Do NOT launch new VIS/RAND attack jobs without explicit approval.
- Do NOT run Layer-2 confirmation automatically.
- Do NOT modify S6 frozen reports (c0ca025).
- Do NOT use GPU 3,7.
- Do NOT use old 72-pair single-shot labels, pre-v1.1 traces, or Bronze labels as final labels.
- Do NOT treat random_sensitive as negative.
- Do NOT claim detector solved, Layer-2 solved, Layer-3 solved.
- Do NOT call action-logit a detector improvement.
- Do NOT start two workers writing to the same CSV.
- Do NOT merge hidden shards before both complete and gates pass.

---

## 7. Recovery Commands

```bash
# Check state
git status
git rev-parse --abbrev-ref HEAD
nvidia-smi
tmux ls
ps -ef | grep -E "hidden|action|watcher|python"

# Check hidden shard progress
grep -c 'dim=' action_hidden_full/shard10.log
grep -c 'dim=' action_hidden_full/shard45.log
tail -5 action_hidden_full/shard10.log
tail -5 action_hidden_full/shard45.log

# Merge shards after both complete
python scripts/diagnostics/merge_hidden_shards.py \
  --input action_hidden_full/action_hidden_full_features_w0.csv \
         action_hidden_full/action_hidden_full_features_w20.csv \
  --out tables/action_hidden_full_features.csv

# Run hidden readout
python scripts/diagnostics/run_action_hidden_readout.py \
  --features tables/action_hidden_full_features.csv
```

---

## 8. Key Commit Chain

```
c0ca025  S6 final freeze
6bb53db  action-logit readout / mechanism evidence
d9ceab0  P0 hard-fail watcher gates
288b072  tighten watcher gates
836f1fa  action-logit full extraction + watcher
8f2ea07  P1 online-safe action-logit smoke
ee50a9e  P0-fix action-logit smoke
3c47f93  true action-logit smoke
3f3eb8f  robustness seeds 7,8
19aaf87  pipeline v0.3 fresh confirmation
```
