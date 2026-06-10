# Stage-B RC1a S12B Running Handoff — Layer3 RAND-Clean Physical Bridge Search

**Date**: 2026-06-10 ~20:10 CST
**Server HEAD**: `ee8bf3e`
**GitHub HEAD**: `77d0eee` (merge)
**Branch**: `exp/vis-prefix-margin-repair-20260603`

## 0. First-Read Instructions

下一轮接手前必须先读：
1. `reports/STAGEB_RC1A_S9B_VISRAND_SMOKE_PASS_20260610.md`
2. `tables/layer3_physical_bridge_status_all.csv`
3. `tables/s12a_rand_veto_results.csv`

**S12b 正在运行中，不要基于猜测写结论。不要扩实验。**

禁止：
- 不启动 full queue / 第三第四个 object
- 不把 S12b 说成 PASS（需等 summary+trace audit）
- 不把 tomato 1/3 当 positive
- 不把 random_sensitive 当 negative
- 不说 detector solved / Layer3 solved / Object-wide success
- 不使用旧 OPEN convention / old traces

---

## 1. Current Scientific Status

### Layer Definitions
- **Layer1** = detector v0.3 / CleanRand abstain-first command-level selector。过滤 random-sensitive，保留 VIS-specific OPEN candidate。不是 physical bridge proof。
- **Layer2** = post-Layer1 ranker，未验证，不作为主 claim。
- **Layer3** = physical bridge: VIS command OPEN → gripper qpos physical opening。不是 detector。

### Current Pipeline (after S12a)
```
Layer1 detector v0.3
→ ORACLE physical reachability
→ Phase1-port RAND-clean veto
→ VIS/RAND physical bridge test
```
Tomato (S11b) 证明：Layer1-selected + ORACLE-reachable 仍可能 RAND contaminated。

---

## 2. Completed Results

### Milk — Clean Repeated Physical Bridge POC
- Object: milk, state_id=0, window=[70,80), open_duration=10
- Layer1✅ ORACLE✅ RAND-clean✅
- S9b seed9: VIS norm 0.423 (PASS), seed10: 0.265 (PASS)
- S9c seed11: 0.378 (PASS), seed12: 0.129 (FAIL)
- **3/4 matched pairs PASS.** Attack-seed sensitive but reproducible.

### Butter w90-100 — Manual ORACLE-Referenced Negative
- Layer1: NO (manual). ORACLE✅. VIS/RAND 0/2 FAIL.
- VIS open=2/10, RAND pos > VIS pos. Command-weak + random-confounded.
- Does NOT evaluate detector v0.3.

### Tomato w155-165 — Layer1 + ORACLE, RAND Contaminated
- Layer1✅ ORACLE✅. VIS/RAND: control FAILED.
- RAND seed16: open=7/10, norm=0.950. VIS seed17 only: norm=0.253.
- **Abstain/reject for Layer3.**

### S11a ORACLE Overlap
- 6/6 STRONG non-milk Layer1-selected parents ORACLE-reachable.
- Gap is not ORACLE; gap is Layer3 RAND-cleanliness.

### S12a RAND-First Veto (frozen `ee8bf3e`)
| Parent | Seeds | RAND OPEN | RAND pos | Verdict |
|--------|-------|-----------|----------|---------|
| cream_s2_w50-60 | 18,19,20 | 0,0,0 | ~0.000 | **STRICT** |
| tomato_s2_w150-160 | 18,19,20 | 2,5,3 | 0.01-0.04 | REJECT |
| butter_s0_w80-90 | 18,19,20 | 3,2,3 | 0.02-0.15 | REJECT |

Cream is first non-milk to clear Layer1✅ + ORACLE✅ + RAND-clean✅.

---

## 3. S12b Currently Running — DO NOT CLAIM RESULT

**cream_s2_w50_60**, ORACLE ref = 0.2838, RAND-clean (S12a strict).

| tmux | GPU | Seed | Jobs | Status |
|------|-----|------|------|--------|
| `s12b_s21` | 1,0 | 21 | VIS+RAND | **RUNNING** |
| `s12b_s22` | 2,6 | 22 | VIS+RAND | **RUNNING** |
| `s12b_s23` | 4,5 | 23 | VIS+RAND | **RUNNING** |

Output: `/data/liuyu/outputs/stageb_v1_1_k5c_targeted_expansion_rc1a_ca3a97e/s12b_cream_visrand/`
Runner: `scripts/stageb/run_s9b_phase1_runner_attack_port.py`

**PASS gate (after completion):**
- VIS open > RAND open; VIS pos > 0; VIS pos > RAND pos
- VIS/0.284 >= 0.2 for ≥2/3 seeds
- RAND still clean (open≤2, pos low)

---

## 4. Server Connection & Environment

- **SSH**: `ssh vla` (jump: scene@10.60.133.3 → liuyu@10.60.133.4)
- **Server**: klfy-SYS-4028GR-TR2, user: liuyu
- **Repo**: `/data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607`
- **Branch**: `exp/vis-prefix-margin-repair-20260603`
- **Conda**: `openvla_official_libero_20260525`
- **Python**: `/home/liuyu/.conda/envs/openvla_official_libero_20260525/bin/python`
- **Model**: `/data/aviary/models/openvla/openvla-7b-finetuned-libero-object`
- **Output root**: `/data/liuyu/outputs/stageb_v1_1_k5c_targeted_expansion_rc1a_ca3a97e/`

### GPU Protocol
| Group | CUDA_VISIBLE_DEVICES | Runner --gpu_pair |
|-------|---------------------|-------------------|
| GPU10 | 1,0 | 0,1 |
| GPU26 | 2,6 | 0,1 |
| GPU45 | 4,5 | 0,1 |

Launch: `CUDA_VISIBLE_DEVICES=1,0 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl unset DISPLAY`

---

## 5. Immediate Next Steps

### A. Check S12b completion
```bash
ssh vla
tmux ls | grep s12b
ls /data/liuyu/.../s12b_cream_visrand/summary_*.json | wc -l  # expect 6
```

### B. Postprocess
```python
OR=0.2838
for each seed 21,22,23:
  VIS: open, streak, pos_area, norm=pos/OR
  RAND: open, streak, pos_area, norm
```

### C. Classify
- 2-3/3 PASS → first non-milk clean physical bridge POC
- 1/3 PASS → mixed, not established
- 0/3 PASS → cream is clean but VIS bridge fails

### D. Freeze
- `reports/STAGEB_RC1A_S12B_CREAM_VISRAND_BRIDGE_20260610.md`
- `tables/s12b_cream_visrand_results.csv`
- Update `tables/layer3_physical_bridge_status_all.csv`

---

## 6. Claim Boundary

**Allowed (S12a):** Milk 3/4 repeated POC. Cream clears all pre-VIS gates. S12b running.

**Forbidden until S12b audit:** Cream PASS. Non-milk bridge established. Layer3 solved.

**If S12b 2-3/3 PASS:** "Cream provides first clean non-milk physical bridge POC under Layer1+ORACLE+RAND-clean gating." Still not Object-wide.

---

## 7. Next-Session Checklist

1. What is GitHub HEAD?
2. Did S12b finish? 6/6 summaries + traces?
3. Any FAIL / infra error?
4. VIS/RAND open_count + pos_area for seeds 21/22/23?
5. Is RAND still clean on cream?
6. VIS ≥2/3 seeds with norm ≥0.2?
7. What claim is now allowed?
8. What next experiment is justified?
