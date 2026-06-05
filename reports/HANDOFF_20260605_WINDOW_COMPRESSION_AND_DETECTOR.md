# HANDOFF — Window Compression & Detector Design

**Date**: 2025-06-05  
**Author**: Leon + Claude (DeepSeek executor)  
**Next**: GPT / Codex review

---

## 0. HANDOFF Target Audience

| Role | Responsibility |
|------|---------------|
| **DeepSeek** | Server experiment execution, watcher, audit, labels, training, compression rollout |
| **Codex** | Design docs, schema audit, training script audit, candidate generators, feasibility — NO GPU |
| **GPT / Human** | Quick restore of scientific mainline, claim boundary, experiment state |

---

## 1. Server Access

SSH (see `~/.ssh/config` for exact alias):

```bash
ssh vla
```

Repo:

```bash
cd /data/liuyu/repos/openvla-gripper-dutycycle-attack-clean-main-20260524
git checkout exp/vis-prefix-margin-repair-20260603   # our branch
git pull
```

**⚠️ Server may be on a different branch at connection time. Always verify with `git branch` and `git log --oneline -3` before running commands.**

Environment:

```bash
conda activate openvla_official_libero_20260525
PY=/home/liuyu/.conda/envs/openvla_official_libero_20260525/bin/python
$PY --version   # Python 3.10
```

GPU monitor:

```bash
watch -n 20 nvidia-smi
dmesg | tail -n 200 | grep -i "xid\\|nvrm" || true
```

**Key paths:**

| Purpose | Path |
|---------|------|
| Repo | `/data/liuyu/repos/openvla-gripper-dutycycle-attack-clean-main-20260524` |
| Python | `/home/liuyu/.conda/envs/openvla_official_libero_20260525/bin/python` |
| Batch3 output | `/data/liuyu/outputs/nightly_object_batch3_20260604` |
| Batch3b output | `/data/liuyu/outputs/nightly_object_batch3b_20260604` |
| Batch3 Wave1 VIS | `/data/liuyu/outputs/object_phase_response_batch3_VIS_20260604` |
| Disk | `/data` 612G used / 1.8T total (36%) |

---

## 2. Git & Environment State

```text
Server: klfy-SYS-4028GR-TR2
User: liuyu
Repo branch: exp/vis-prefix-margin-repair-20260603
Latest commit: ba6f860
Remote: git@github.com:Leo-6-maker/openvla-gripper-dutycycle-attack.git
Python: 3.10 (conda env openvla_official_libero_20260525)
```

**⚠️ GPU health:**

| GPU | Status | Notes |
|-----|--------|-------|
| 0 | Healthy | Xid13 history |
| 1 | Healthy | — |
| 2 | Healthy | — |
| 3 | **Probation** | Xid31 ×2 overnight. GPU23 passed one probation task (alphabet_s4) |
| 4 | Healthy | — |
| 5 | Healthy | — |
| 6 | Healthy | — |
| 7 | Healthy | Xid31 history (old Xid from previous boot) |

**Rules:**
- Xid/OOM/missing trace = infrastructure failure, NEVER scientific negative
- GPU23 probation budget=1 used, pair stays probation (not healthy)
- Do NOT use GPU3 for critical controls or unique pre_lock tasks

---

## 3. Scientific Mainline Summary

```
clean rollout / denominator
→ phase-response candidate windows
→ VIS/random matched rollout
→ denominator/provenance audit
→ vulnerability_ready labels
→ detector diagnostic
→ window compression
→ detector design v2 / VisualTransferHead
```

### Claim Boundary

**Allowed:**
- Batch3 expanded Object far/near closed vulnerability to cream_cheese, milk
- Detector v1 training pipeline works (CSV-reading mode)
- v1 exposed prevalence confound (no model beats always_positive baseline)
- Batch3b/c are designed to add negatives/controls
- Phase bins are learnable from runtime features (CPU smoke, F1=0.55 at task-level)

**Forbidden:**
- Detector v1 works / deployable / online
- Object-wide vulnerability_ready learned
- Cross-suite generalization
- Simulator SR alone proves success/failure
- Polluted/random-failed rows are negatives
- GPU Xid/OOM = scientific negative
- pre_lock is universally safe
- far_closed universally vulnerable
- Student target = fixed lead=-50

---

## 4. Experiment Progress

### Batch1 (ketchup seed0 phase-offset)

- 12 windows, task-critical band T+5 to T+25
- Right boundary between T+25 and T+30
- Physical bridge persists through T+40
- Provenance: `reports/VIS_PHASE_OFFSET_BOUNDARY_KETCHUP_SEED0.md`

### Batch2b (Object teacher-oracle delay=-50 probe)

- 9 VIS completed, 4 claim_usable, 4 task-negative, 1 weak
- pre_lock 0/2 positive (ruled out as vulnerable phase)
- far_closed 3/5 (60%), near_closed 1/2 (50%)
- CSV: `tables/object_phase_response_batch2b_vis_summary.csv`

### Batch3 (Object phase-response expansion)

- **11 VIS completed**: 7 claim_usable, 4 task-negative
- New positive tasks: cream_cheese, milk
- First coverage: salad_dressing (task-negative)
- CSV: `tables/object_phase_response_batch3_vis_summary.csv`

### Batch1 + Batch2b + Batch3 combined labels

- `tables/object_phase_response_labels_v1.csv`
- 20 total: 11 pos, 8 neg, 1 ignore = 19 train rows
- Tasks: alphabet_soup, bbq_sauce, butter, cream_cheese, ketchup, milk, salad_dressing (7 tasks)

### Detector v1

- 19 labels, LOTO diagnostic
- always_positive baseline: F1_pos=0.7333, F1_neg=0.0, balAcc=0.5
- LR task-key: F1_pos=0.7333 — exact match with prevalence baseline
- **No model beats prevalence**
- RF phase+causal: balAcc=0.528 — barely above chance
- v1 underpowered, cannot claim vulnerability_ready learned

### ba6f860 update

- `scripts/train_vulnerability_ready_detector_v1.py` now supports `--labels-csv`
- `--freeze-v1-hardcoded` only for v1 reproduction
- Ready for Batch3b/c label expansion → v2

### Batch3b (14 candidates, 9 tasks)

- far_closed=5, near_closed=8, pre_lock=1
- Precheck launched on 4 GPU pairs (1,0 / 4,5 / 6,7 / 2,3)
- Next: audit → VIS-ready → selected VIS

### Batch3c (11 controls)

- stable_post_lock × 6, far_too_early × 4, pre_lock × 1
- Role-specific gates implemented: `scripts/diagnostics/role_specific_gates.py`
- stable_post_lock: late_open_control taxonomy, skips clean_OPEN≤0.1
- far_too_early/pre_lock: standard closed-window gates
- Next: precheck → role-aware audit → VIS if controls needed

---

## 5. DeepSeek Execution Plan (Server)

### Priority A — Finish Batch3b

1. Wait for precheck completion
2. Run denominator audit on localized traces only
3. Generate VIS-ready list
4. Run VIS on denominator-clean rows
5. Prioritize: underrepresented tasks, near_closed, far_closed

### Priority B — Batch3c Controls

1. Run Batch3c precheck with role-specific gates
2. Apply late_open_control taxonomy for stable_post_lock
3. Run Batch3c VIS only if negatives insufficient
4. Never treat stable_post_lock done=False as automatic positive

### Priority C — Merge Labels v2

After Batch3b/c VIS summaries:
```bash
$PY scripts/train_vulnerability_ready_detector_v1.py \
  --labels-csv tables/object_phase_response_labels_v2.csv
```

Training gates: valid≥24, pos≥8, neg/control≥8, tasks≥6

### Priority D — Train Detector v2

```bash
$PY scripts/train_vulnerability_ready_detector_v1.py \
  --labels-csv tables/object_phase_response_labels_v2.csv \
  --min-rows 24
```

Pass condition: phase+causal beats prevalence on balanced_accuracy, negative_recall>0, control FP reduced

---

## 6. Window Compression Plan

**Goal**: Compress 18-step windows while preserving VIS effect.

**Smoke test candidates** (5 from Batch3):

Positives: cream_cheese s4 [28,45], milk s4 [19,36], ketchup s1 [21,38]
Negatives: salad_dressing s0 [7,24], bbq_sauce s5 [27,44]

**Compression**: L12, L10, L8 centered. Run VIS + random for each.

**Success**: ≥2/3 positives claim_usable at L10 or L8. Freeze as default.

---

## 7. Codex Parallel Plan (NO GPU)

| Task | Output | Description |
|------|--------|-------------|
| 1 | — | Handoff consistency audit |
| 2 | `reports/DETECTOR_V2_AND_VISUAL_TRANSFER_DESIGN.md` | Detector design doc |
| 3 | `scripts/diagnostics/audit_label_schema.py` | Label schema audit |
| 4 | — | Review train_vulnerability_ready_detector_v1.py |
| 5 | `scripts/diagnostics/generate_window_compression_candidates.py` | Compression candidate gen |
| 6 | `reports/VISUAL_TRANSFER_HEAD_FEASIBILITY_NOTE.md` | Visual module feasibility |

---

## 8. Local Sync

```bash
cd <local_repo_path>
git fetch origin
git pull
git log --oneline -5
```

Optional server copy:
```bash
scp vla:/data/liuyu/outputs/nightly_object_batch3_20260604/HANDOFF_20260605_WINDOW_COMPRESSION_AND_DETECTOR.md .
```

---

## 9. Checklist

- [x] Branch: `exp/vis-prefix-margin-repair-20260603`, commit `ba6f860`
- [x] Python 3.10, conda env openvla_official_libero_20260525
- [x] Batch3 final results: 11 VIS, 7 claim_usable
- [x] Batch3b precheck launched
- [x] Batch3c controls ready
- [x] Detector v1 CSV-reading mode, prevalence baselines
- [x] GPU blacklist: GPU3 probation, GPU23 probation PASS
- [x] DeepSeek tasks listed (A/B/C/D)
- [x] Codex tasks listed (1-6)
- [x] Window compression plan
- [x] Forbidden claims listed
- [ ] Handoff copied to output dirs → pending server copy
- [ ] Commit pushed
