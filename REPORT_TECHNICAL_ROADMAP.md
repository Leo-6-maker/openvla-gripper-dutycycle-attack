# R9Q Online Detector — 技术路线与实验进展

**PR:** [#73](https://github.com/Leo-6-maker/openvla-gripper-dutycycle-attack/pull/73)
**HEAD:** `24af426`

---

## 1. Detector 技术架构

```
C2gGripperCriticalWindowDetector (147K params)
├── Input:  25D proprio (SC5 streaming features) + 9D policy intent (OpenVLA token semantics)
├── Model:  GRU(hidden=128) → Fusion MLP(256→128→128) → 6-head outputs
├── Visual: use_visual=False (1152-dim预留，未启用)
├── Lang:   use_language_conditioning=False
└── Output: 6 per-timestep logits →
              sigmoid(critical) × (1−sigmoid(release)) × sigmoid(grounding)
              → FixedBurstTriggerScheduler (2-of-3, one-shot, burst=10)

Runtime Gate:
  effective_valid = detector_ready (susceptibility_gate_enabled=false)
  τ_crit=0.7, τ_rel=0.4, τ_grnd=0.3, persistence=2-of-3

Attack: VIS-PGD, target token 31744 (CLIP_MEDIATED_OPEN), ε=6/255, 20 steps
        objective = autoregressive_prefix_gripper_target_token_logratio_arm_v3
```

## 2. 关键设计决策

| 决策 | 选择 | 理由 |
|------|------|------|
| 模型版本 | Codex B2_seed456_epoch_011 | 94.7% CAL feasible-hit, τ参数更保守(2-of-3 vs 3-of-5) |
| 输入特征 | 25D+9D, 无visual | 94.7% feasible-hit证明proprio+policy_intent已足够 |
| 参数量 | 147K | 在线推理零延迟，不与OpenVLA竞争显存 |
| 旧gate处理 | susceptibility_gate_enabled=false | PR #72修复，消除旧clean-policy gate阻断 |
| 4-worker架构 | GPU6:object+spatial, GPU7:goal+l10 | suite模型隔离, model-load lock串行化 |

## 3. 实验结果

### 3.1 20-Parent 预注册主表

| Suite | n | Clean | R9Q | RAND | Oracle | R9Q ΔSR | IndFail | Trigger | Arm ΔL2 |
|-------|---|-------|-----|------|--------|---------|---------|---------|----------|
| Object | 5 | 3/5 | 1/5 | 3/5 | 2/5 | −40pp | 3/3 | 100% | 0.218 |
| Spatial | 5 | 5/5 | 2/5 | 5/5 | 3/5 | −60pp | 3/5 | 100% | 0.261 |
| Goal | 5 | 5/5 | 1/5 | 4/5 | 2/5 | −80pp | 4/5 | 100% | 0.252 |
| L10 | 5 | 1/5 | 1/5 | 0/5 | 1/5 | +0pp | 0/1 | 60% | 0.115 |
| MACRO | 20 | 14/20 | 5/20 | 12/20 | 8/20 | −45pp | 10/14 | 90% | 0.233 |

**核心指标:**
- R9Q induced failure: 10/14 = 71.4%
- R9Q trigger rate: 18/20 = 90%
- R9Q vs RAND: −35pp (Detector timing价值)
- R9Q vs Oracle: −15pp (VIS-PGD策略级 vs 纯夹爪)
- EnvOpen (R9Q): 100% — VIS-PGD确实打开了夹爪

### 3.2 Canary 32-Cell Audit

```
32/32 PASS, susceptibility_gate_enabled=True: 0
Bundle SHA全匹配 (checkpoint/config/normalization/SHA256SUMS)
R9Q: 8/8 triggered, 100% EnvOpen, exact T10 burst
Trigger diversity: steps 55-165, median 81
```

### 3.3 L10 数据缺口分析

```
R7 L10 registry:        500 identities
R8W+R8Y union:          478 (470 + 8 unique)
Residual gap:           22 (全部 ATTACK_EVAL + DETECTOR_TEST)
DETECTOR_TRAIN (300):   完整覆盖 ✅
→ 补全L10-500对detector训练改善有限, 残缺口仅影响攻击评估完整性
```

## 4. 进行中: Clean-Success 20 核心攻击矩阵

```
20 Clean-Success parents × 3 conditions (R9Q+RAND+Oracle)
├── 复用已有cell: 39/60
└── 新运行: 21 cells (GPU6 object + GPU4 l10, 并行)
当前: 11/21 done, 0 failures
```

完成后生成 Clean-Success 条件化对比表（所有parent Clean SR=100%）。

## 5. 消融实验计划 (排队中)

| 条件 | 时机 | 梯度 | 回答 |
|------|------|------|------|
| R9Q_DETECTOR_T10 | Detector | Targeted | 完整方法 |
| RAND_T10 | Random | Targeted | Detector timing价值 |
| DET_TIME_SHUFFLED_GRAD_T10 | Frozen R9Q trigger | Permuted | Targeted gradient价值 |
| R9Q_VIS_GRIPPER_ONLY_T10 | Frozen R9Q trigger | Targeted, arm clamped | Arm副作用贡献 |
| COMMAND_OPEN_ORACLE | Detector | Direct open | 纯夹爪上界 |

## 6. 未回答问题

1. **Visual embedding贡献**: detector设计支持visual_dim=1152但未启用, 消融实验可回答visual是否降低false-trigger或改善L10
2. **L10为何弱**: 仅Clean=1/5, Trigger=3/5, 可能因520步长时程使proprio-only detector信号衰减
3. **Shuffled-grad消融**: 等待core20完成后在Clean-Success条件下运行
