# S5 Repeat-Stability-First Protocol — 6-Hour Autonomous Runbook

**Target session**: Next DeepSeek
**Start anchor**: `26da793`
**Branch**: `exp/vis-prefix-margin-repair-20260603`
**Goal**: Complete K=5 repeat stability round. Do NOT train detector. Do NOT expand new parents.

---

## Context (for new session)

Today we discovered that **8/8 Silver confirmation parents failed repeat-stability**.
Single-shot VIS/RAND labels are highly seed-dependent. The 72-pair pool is exploratory-only.

### Retained claims
1. RC1a gripper semantics correction is correct
2. Corrected VIS produces command OPEN on some windows
3. Physical transfer exists but unstable
4. Random-sensitive/confounded behavior is real
5. Single-shot labels are NOT ground truth

### Downgraded claims
- 72-pair pool CANNOT train detector
- abstain AUROC=0.889 is single-shot readout only
- Visual sidecar tested on unstable labels; may change with stable labels

### Forbidden (entire 6h window)
- Train detector on 72-pair labels
- Use single-shot labels as ground truth
- Run global visual sidecar or cross-suite
- Blind expansion of new parents
- Use GPU 3/7
- Overwrite old outputs
- Report "detector success"

---

## Server Access

```bash
ssh vla   # jump: scene@10.60.133.3 → liuyu@10.60.133.4
```

GPU pairs: `1,0` / `2,6` / `4,5` (3,7 BLACKLISTED)
Conda: `/home/liuyu/.conda/envs/openvla_official_libero_20260525/bin/python`
Execution copy: `/data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607/`

Key outputs:
- `/data/liuyu/outputs/stageb_v1_1_detector_v0_rc1a_20260608/all_labels_rc1a_14cfabe_72pairs.csv`
- `/data/liuyu/outputs/stageb_v1_1_silver_confirmation_rc1a_e33b5e4/` (32 traces, 32/32 PASS)
- `/data/liuyu/outputs/visual_sidecar_14cfabe_72pairs/`

---

## 6-Hour Timeline

### Phase A (0:00–0:30): Freeze + Downgrade Declarations
Gate: downgrade claims written, forbidden claims explicit.

### Phase B (0:30–1:15): Prefix Determinism Audit
Gate: confirm same-parent repeats have identical clean prefix before attack window.
If NOT: stop, write nondeterminism report, fix runner.

### Phase C (1:15–1:45): Build K=5 Repeat Queue
**8 parents × 5 attack seeds × 2 conditions = 80 jobs.**
6 fixed parents + 2 from 72-pair pool (non-edge strict-phys + non-edge clean negative).
Job IDs: 500000–500079. Must be exactly 80.

Fixed parents:
1. milk s0 [70,80] — cmd stable candidate
2. milk s0 [230,240] — confounded/cmd swing
3. tomato_sauce s2 [150,160] — rand_cmd/confounded/no_cmd
4. tomato_sauce s2 [90,100] — rand_phys → positive flip
5. salad_dressing s2 [120,130] — negative → rand_cmd drift
6. bbq_sauce s2 [100,110] — HN surprise → disappeared

Gate: 80 jobs exactly, env_seed fixed, attack_seed varied, no edge parents.

### Phase D (1:45–2:15): K=2 Mini-Smoke
2 parents × 2 seeds × 2 conditions = 8 jobs.
milk [70,80] + salad [120,130].
Gate: validator PASS, prefix hash consistent.

### Phase E (2:15–4:45): K=5 Full Run
80 jobs on 3 GPU pairs. Hard stops: pair mismatch, provenance fail, 2 consecutive infra fail, GPU 3/7 usage.

### Phase F (4:45–5:30): Postprocess + Probability Labels
Compute per-parent: pV_cmd, pR_cmd, pV_phys, pR_phys, yield_cmd, risk_rand, stability label.

Stability rules (K=5):
- stable_cmd: pV_cmd ≥ 0.6, pR_cmd ≤ 0.2, yield ≥ 0.4
- stable_phys: pV_phys ≥ 0.6, pR_phys ≤ 0.2, yield ≥ 0.4
- stable_rand: pR_cmd ≥ 0.4 or pR_phys ≥ 0.4
- stable_negative: ALL ≤ 0.2
- unstable_or_unknown: otherwise

### Phase G (5:30–6:00): Self-Check + Commit
Answer: do stable labels exist? If stable < 3 parents, do NOT train detector.

---

## Final Deliverable

1-page summary with gate results, parent probability table, main finding, allowed/forbidden claims, next action.

## Core Instruction

**Do not try to prove the detector works. First prove that labels can be stable.**
