# R9Q Online Detector — 实验汇报

**Date:** 2026-07-13
**PR:** [#73](https://github.com/Leo-6-maker/openvla-gripper-dutycycle-attack/pull/73)
**Base:** PR #72 (`codex/c2g-r9q-final-detector-attack-20260713`)

---

## 1. 实验架构

```
Detector: C2gGripperCriticalWindowDetector (GRU, h=128)
Model:    B2_seed456_epoch_011 (Codex训练)
Config:   τ_crit=0.7 τ_rel=0.4 τ_grnd=0.3, 2-of-3 persistence, burst=10
Gate:     susceptibility_gate_enabled = false (R9Q tri-head gate only)
Attack:   VIS-PGD, target=token 31744 (CLIP_MEDIATED_OPEN), ε=6/255, 20 steps
```

## 2. 主表：20-Parent Matched Attack (预注册)

| Suite | n | Clean | R9Q | RAND | Oracle | R9Q ΔSR | Induced Fail | Trigger | Arm ΔL2 |
|-------|---|-------|-----|------|--------|---------|-------------|---------|---------|
| Object | 5 | 3/5 (60%) | 1/5 (20%) | 3/5 (60%) | 2/5 (40%) | **−40pp** | 3/3 (100%) | 5/5 | 0.218 |
| Spatial | 5 | 5/5 (100%) | 2/5 (40%) | 5/5 (100%) | 3/5 (60%) | **−60pp** | 3/5 (60%) | 5/5 | 0.261 |
| Goal | 5 | 5/5 (100%) | 1/5 (20%) | 4/5 (80%) | 2/5 (40%) | **−80pp** | 4/5 (80%) | 5/5 | 0.252 |
| L10 | 5 | 1/5 (20%) | 1/5 (20%) | 0/5 (0%) | 1/5 (20%) | **+0pp** | 0/1 (N/A) | 3/5 | 0.115 |
| **MACRO** | **20** | **14/20 (70%)** | **5/20 (25%)** | **12/20 (60%)** | **8/20 (40%)** | **−45pp** | **10/14 (71%)** | **18/20 (90%)** | **0.233** |

**核心指标:**
- R9Q induced failure: **10/14 = 71.4%** (Clean-success前提下被攻击转为失败)
- R9Q trigger rate: **18/20 = 90%** (2个L10 parent未触发)
- Trigger step range: **51-165** (median 72, 无退化)
- susceptibility_gate_enabled=true: **0**
- runtime_invalid: **0**, multi-trigger: **0**

## 3. 攻击机制诊断 (Canary 8-parent)

| Condition | EnvOpen | Flip% | GripΔ | Arm ΔL2 |
|-----------|---------|-------|-------|---------|
| R9Q (VIS-PGD) | **100%** | 45-80% | −0.9~−1.6 | 0.12-0.35 |
| RAND (同VIS-PGD, 随机时机) | 90-100% | 23-65% | −0.4~−1.3 | 0.09-0.21 |
| Oracle (direct gripper open) | **100%** | 50-85% | −1.0~−1.7 | **0.000** |

**关键发现:**
1. VIS-PGD成功打开夹爪 (100% EnvOpen)，不是"攻击失败"
2. Arm偏移伴随夹爪攻击（R9Q Arm ΔL2=0.12-0.35 vs Oracle=0）
3. RAND的Flip%低于R9Q（随机时机常在夹爪已open时触发）
4. R9Q-specific induced: 1 parent (goal/task_01/024) — VIS-PGD击败direct command

## 4. R9Q vs RAND vs Oracle 效能对比

```
                 Clean=14/20    Induced Fail    
R9Q (detector):   5/20 (25%)    10/14 (71%)    ← 最强
RAND (random):   12/20 (60%)     2/14 (14%)    ← 弱，时机不对
Oracle (direct):  8/20 (40%)     6/14 (43%)    ← 纯夹爪上界

Δ(R9Q-RAND) = −35pp  → Detector timing价值
Δ(R9Q-Oracle) = −15pp → VIS-PGD策略级攻击 vs 纯夹爪干预
```

**Detector时机 vs 随机：相同的VIS-PGD，不同时机，效果差 −35pp。** 证明Detector选择了任务关键窗口。

## 5. Clean-Success 20 构建中

```
目标: 20个Clean-Success parents (Object=5, Spatial=6, Goal=6, L10=3)

已完成: OGS=17 ✅
  Object:  3 existing + 2 enrichment = 5
  Spatial: 5 existing + 1 enrichment = 6  
  Goal:    5 existing + 1 enrichment = 6

进行中: L10=1 + 2 screening 🔄
  GPU4正在筛选L10 CLEAN候选 (成功率~20%)
```

完成后将对20个Clean-Success parents运行核心攻击矩阵 (R9Q+RAND+Oracle)，在Clean-Success条件下重新评估induced failure rate。

## 6. Canary 32-Cell 审计结果

```
32/32 cells PASS
Bundle SHA closure: ✅ (checkpoint/config/normalization/SHA256SUMS)
R9Q: 8/8 triggered, 100% EnvOpen, exact T10
sg_enabled=True: 0
RAND exact-T10: 5/8 (3 truncated by task success)
provenance_mismatch: 0
```

## 7. 结论

1. **R9Q在线Detector有效运行**: 90%触发率, 100% EnvOpen, exact T10
2. **R9Q诱导71% Clean-Success任务失败**: VIS-PGD在Detector时机下显著破坏任务
3. **Detector时机价值−35pp**: 相同VIS-PGD, Detector时机比随机强35个百分点
4. **VIS-PGD是夹爪靶向攻击，且产生arm副作用**: Arm ΔL2=0.12-0.35
5. **susceptibility_gate修复确认**: 0残留旧gate，在线telemetry完整
