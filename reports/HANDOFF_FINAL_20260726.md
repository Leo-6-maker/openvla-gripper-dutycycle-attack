# N5 Final Handoff — 2026-07-26

**Branch**: `deepseek/integration-final-detector-20260724`
**Commit**: `f9ba872`
**Server**: dty (10.60.2.56:33571, user dty_user)
**Env**: `/mnt/sdc/dty_user/openvla_attack/envs/openvla-official-a800/bin/python`

---

## 1. V4 Formal Matrix (Complete)

```text
20/20 parents, 19 DONE_VALID + 1 DONE_CLASSIFIED_TC (spatial s0 K10 truncation)
Queue: /mnt/sdc/dty_user/openvla_attack_outputs/fec_formal_v2/queue.sqlite

Per-suite emission:
  libero_10:     5/5 emit (K10=10 all attack arms)
  libero_object: 5/5 emit (K10=10 all attack arms)
  libero_goal:   0/5 emit (detector never fires — root cause confirmed)
  libero_spatial: 2/5 emit (s0=TC, s4=K10=10; s1,s3 no emit)

Action semantics audit:
  V4 TRUE_VIS = RELEASE-BLOCKING (CLOSE), not PREMATURE_OPEN
  clean_env_grip = +1.0 (CLOSE) at emission for ALL 42 attack samples
  target_action = clean_raw_action (BUG — release-blocking)
  TARGET_TOKEN_ID = 31744 (BUG — out of bounds, disc=255)
  target_execution_class = CLIP_MEDIATED_OPEN (BUG — empty class, 0 tokens)
```

## 2. VIS-PGD Attack Fixes (Committed)

Three bugs fixed in `run_gpu_smoke_v5_open.py` + `fec_attack_v5_open.yaml`:

| Bug | Fix |
|-----|-----|
| target_action = clean_raw_action (CLOSE) | target_action[6] = 1.0 (CANONICAL_RAW_OPEN) |
| TARGET_TOKEN_ID = 31744 (invalid) | 31745 (NATIVE_OPEN, disc=254) |
| execution_class = CLIP_MEDIATED_OPEN (empty) | NATIVE_OPEN |

8-GPU budget sweep results (libero_10 s1, emit=83):
```
ε=0.03 PGD-5:  0/10 OPEN
ε=0.03 PGD-10: 0/10 OPEN
ε=0.03 PGD-20: 4/10 OPEN  ← WORKS
ε=0.06 PGD-5:  3/10 OPEN
ε=0.06 PGD-10: 4/10 OPEN
ε=0.06 PGD-20: 5/10 OPEN
ε=0.10 PGD-5:  5/10 OPEN
ε=0.10 PGD-10: 5/10 OPEN
```
**Conclusion**: ε=0.03/PGD-5 insufficient. PGD-20 at ε=0.03 flips gripper. Budget radius is fine, optimization steps were insufficient.

ORACLE at same emit: task_success=False, POLICY_HORIZON — confirms physical vulnerability.

Region-mass objective (`prefix_locked_gripper_open_region_ce`) config created but not yet tested.

## 3. Label Production Pipeline

| Gate | Status | Key Metric |
|------|--------|------------|
| G2 Canary | PASS | 32/32, 0 unk2neg |
| G3 Cohort | PASS | 800 training IDs, CS200-verified |
| G4 Labels | PASS | 800/800 episodes, 176,336 steps |
| G5 Seal | PASS | See distribution below |

### Label Seal Distribution

```
Suite          Critical%  SafeRel%  Instab%  K10%   GripperClose%  Unknown%
libero_10      71.3%      0.01%     0.8%     66.3%  33.3%          0%
libero_goal    43.7%      0.003%    0.3%     37.0%  25.5%          20.7%
libero_object  40.6%      0%        0.2%     33.7%  20.4%          0%
libero_spatial 58.8%      0.03%     0.4%     47.2%  49.9%          0%
```

Key findings:
- Goal: 43.7% critical (V4 was 0%). 20.7% unknown = drawer/stove tasks correctly abstain.
- Safe release: sparse but real (16/800 episodes, 16 total positive steps). Placement-gated.
- critical && gripper_not_closing = 58,904 steps — cc decoupling confirmed working.
- 0 unknown→negative throughout.

### Label Data Locations

```
Training identities: /mnt/sdc/dty_user/openvla_attack_evidence/c2g/c2g_cs200_official_v3_20260716/ops/FACTORIZED_PHASE_B2_DETERMINISTIC_ALLOCATION_V3_804113EE_20260723/checkpoint_training_identity_manifest.json
Label output:        /mnt/sdc/dty_user/openvla_attack_outputs/n5/phase2_labels/g4_label_production/
Label Seal:          /mnt/sdc/dty_user/openvla_attack_outputs/n5/phase2_labels/g4_label_production/LABEL_SEAL.json
Canary output:       /mnt/sdc/dty_user/openvla_attack_outputs/n5/phase2_labels/canary_32_output/
Pilot V3 output:     /mnt/sdc/dty_user/openvla_attack_outputs/n5/phase2_labels/pilot_12_v3_output/
```

## 4. Key Source Files

| File | Purpose | Tests |
|------|---------|-------|
| `scripts/fec/label_contract_v2.py` | Label Contract V2 (5-head tri-state evidence lattice) | 15/15 |
| `n5/phase2_labels/v22_production_v2.py` | V22 Physics Teacher (19/19) | 19/19 |
| `n5/phase2_labels/physics_teacher_v22.py` | V22 Schema (9/9) | 9/9 |
| `n5/phase3_student/n5_student_model.py` | N5 Dual TCN Student | 18/18 |
| `scripts/fec/run_gpu_smoke_v5_open.py` | Fixed VIS-OPEN smoke runner | server |
| `scripts/fec/run_gpu_smoke_v6_open.py` | V6 exact-prefix snapshot runner | not tested |
| `configs/fec_attack_v5_open.yaml` | V5 attack config (single-token 31745) | repo |
| `configs/fec_attack_v5_open_region.yaml` | V5 attack config (OPEN-region CE) | not tested |
| `configs/sweep_v5_e*.yaml` | Budget sweep configs | 5 files |
| `configs/V5_ACTION_SEMANTICS_SEAL_V1.json` | Action space mapping | frozen |
| `configs/V5_SCIENTIFIC_CONTRACT_V1.json` | POST_GRASP_PREMATURE_OPEN contract | frozen |

## 5. V4 vs V5 Architecture

```
V4:  cc funnel → label=0 for cc=False → Goal 0% coverage
     EEF Z proxy, arbitrary contact, comotion bug
     VIS = CLOSE (release-blocking)
     TARGET_TOKEN_ID = 31744 (invalid)

V5:  cc is auxiliary head, not physical gate
     Target-finger contact, target object Z, correct history
     VIS = OPEN (CANONICAL_RAW_OPEN=1.0, token 31745)
     Evidence lattice: any known+ → 1; all known- → 0; else unknown
     Goal: 43.7% critical (pick-place tasks), 20.7% unknown (articulated, correct abstain)
     Safe release: placement-gated (not just gripper opening)
```

## 6. Critical Constants

```
CANONICAL_RAW_OPEN  = 1.0    CANONICAL_RAW_CLOSE = 0.0
CANONICAL_ENV_OPEN  = -1.0   CANONICAL_ENV_CLOSE  = +1.0
CORRECT_OPEN_TOKEN  = 31745  (NATIVE_OPEN, disc=254)
V22 Schema SHA: 48472ce8ec95...  V22 Config SHA: 936aab0292bf...
```

## 7. GPU Artifacts (/tmp)

```
Gate B baseline (single-token 31745, PGD-5): /tmp/gate_0/  → 0/10 OPEN
Gate B region (OPEN-region CE, PGD-5):       /tmp/gate_1/  → not tested (GPU stuck)
Gate C PGD-20 restarts:                      /tmp/gate_2..6/ → all >=1 OPEN
Gate D three-arm (PGD-20):                   /tmp/gate_7/  → 4/10 OPEN, ORACLE=10
```

## 8. Next Steps (By Priority)

### Immediate (can start without new code)

1. **G6 Training Data Seal**: Freeze feature order, train/val/cal split, pos_weight, sampler. Input: 800 labels at `g4_label_production/`. Output: frozen config JSON.

2. **GPU sweep region objective**: Fix GPU 1 idle issue (likely loss computation slow with 127-token logsumexp). Investigate attacker adapter's `prefix_locked_gripper_open_region_ce` code path.

### Short-term (needs training infra)

3. **G7 Baselines**: Train Prior (V4 checkpoint→V5 heads), Last-frame MLP, RF32 TCN, RF128 TCN on 800 labels. GPU: 1× A800.

4. **G8 N5 Training**: Dual TCN × 3 seeds. Same 800 labels. GPU: 1× A800.

### Medium-term

5. G9: Calibration + Scheduler on calibration split (from held-out 1200)
6. G10: Detector replay (V4 vs V5 on held-out)
7. G11-G13: V5 Smoke + Formal Wave-0 + Formal 20

## 9. Known Limitations

- **Articulated tasks** (drawer, stove): 20.7% of goal steps unknown. No joint qpos in CS200 sidecar. Cabinet geoms contact gripper fingers → can detect handle engagement but not joint progress. N5-v2 needs Articulated Teacher.
- **Safe release**: Very sparse (16/800 episodes). Placement detection strict. K10 veto may rarely trigger. Consider relaxing placement requirement for release detection.
- **Instability**: 74-99% unknown. Target-relative checks make it conservative. May need threshold calibration.
- **VIS region objective**: Not yet tested. Single-token 31745 confirmed working with PGD-20.
- **V6 runner**: exact-prefix snapshot branching implemented but not tested (N4Bridge lacks get_state/set_state).

## 10. Server Quick Reference

```bash
# Label data
ls /mnt/sdc/dty_user/openvla_attack_outputs/n5/phase2_labels/g4_label_production/

# Formal queue
python3 -c "import sqlite3; conn=sqlite3.connect('/mnt/sdc/dty_user/openvla_attack_outputs/fec_formal_v2/queue.sqlite'); conn.row_factory=sqlite3.Row; rows=conn.execute('SELECT state,COUNT(*) FROM tasks GROUP BY state').fetchall(); [print(r[0],r[1]) for r in rows]"

# GPU status
nvidia-smi --query-gpu=index,memory.free,utilization.gpu --format=csv,noheader

# Run smoke test (single GPU)
CUDA_VISIBLE_DEVICES=0 python /mnt/sdc/dty_user/openvla_attack_outputs/n5/phase3_student/run_gpu_smoke_v5_open.py \
  --gpu-id 0 --suite libero_10 --task-index 0 --state-index 1 \
  --output-root /tmp/test --model-path /mnt/sdc/dty_user/openvla_attack/models/libero-10/openvla-7b-finetuned-libero-10 \
  --config /mnt/sdc/dty_user/openvla_attack/configs/sweep_v5_e0.03_s20.yaml \
  --n4-module /mnt/sdc/dty_user/openvla_attack/scripts/fec/n4_detector_adapter_v4.py \
  --n4-norm-data /mnt/sdc/dty_user/openvla_attack_evidence/fec_implementation_v1/n4_norms_o0i0.pt \
  --expected-attacker-sha256 26cfb9f5d8a5a29e7ac2729f5c9cdd58dadfd75e45eebe935ee66214cc9402be \
  --seed 19903 --rand-direction-seed 20903 --random-time-seed 21903 --random-time-start 29
```
