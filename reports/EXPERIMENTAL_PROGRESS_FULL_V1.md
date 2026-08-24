## V4 Formal 矩阵最终结果

### 完成状态：20/20 全部执行完毕

| Suite | 完成 | States | 平均重试 |
|-------|------|--------|---------|
| libero_10 | **5/5** | 0, 1, 3, 5, 7 | 1.6 |
| libero_goal | **5/5** | 0, 2, 8, 9, 10 | 1.0 |
| libero_object | **5/5** | 1, 2, 3, 4, 6 | 16.8 |
| libero_spatial | **5/5** | 0, 1, 2, 3, 4 | 87.0 |

- 18/20 已提交队列 (DONE_VALID)
- 2/20 直接运行完成，待入队 (spatial state_0, state_2)

### 事故记录

| 事故 | 影响 | 状态 |
|------|------|------|
| Phase 1 部署覆盖 `/tmp/n4_detector_adapter.py` | 5 个 spatial + object 任务 400+ 次 retry | 已修复 (SHA 匹配 Seal) |
| Persistent worker subprocess bug | 最后 2 任务无法通过 worker 完成 | 直接 nohup 绕过 |
| Spatial state_0 K10 8/10 截断 | TRUE_T10 attack window 被 episode 结束截断 | CLASS_C，科学有效 |

### 冻结参数

| 参数 | 值 |
|------|-----|
| Detector | N4 V4 (Dual CausalTCN RF32+RF128) |
| Provider SHA | `6a7ab61d8dba8cb3` |
| Checkpoint SHA | `685ddadf90ad2ac4` |
| Platt A/B | 0.519 / 0.813 |
| τ / D_PERSIST | 0.855 / 6 |
| candidate_close | raw_gripper ≤ 0.5 |
| PGD ε / steps | 0.03 / 5 |
| K10 | 10 |
| Seal SHA | `22a97d57...` |

---

## Phase 1：Goal 零覆盖根因审计

### 结论：H2 CONFIRMED — K10 labeler 以 candidate_close 为第一漏斗过滤器

| 分析 | 结果 |
|------|------|
| Counterfactual replay | cc gate 移除无效 (score-only ≡ V4 hard gate) |
| Joint distributions | raw_close ≡ env_close (XOR=0)，语义正确 |
| Full vs sparse pipeline | cal_prob max 0.005 on goal (score 本身太低) |
| K10 labeler 审计 | **candidate_close 是第一漏斗过滤器** |
| Goal K10 funnel | 8/10 goal 任务有 critical 窗口 (44-510 steps)，但被 cc gate 截断 |
| Student raw_logit | ≈ -11 on goal (强负判断，不是 gate 问题) |

### 最终裁决

> **V4 实现正确，但训练目标定义错误。Student 精确学会了一个被 `candidate_close` 截断的标签，而非物理 criticality。**

```
根因链：
cc 是第一过滤器 → cc=False → label=0 (70% goal 步)
→ cc=True 但 K10 corridor < 10 → label=0
→ Student 学到 "goal → 负分"
→ raw_logit ≈ -11, cal_prob ≈ 0.005
→ 无论用什么 scheduler 都不触发
```

---

## N5 开发线

### Commit 历史

| Commit | 内容 | 测试 |
|--------|------|------|
| `162d667` | N5 初始提交 (Phase 1 + Label V2 + Student) | — |
| `eaada44` | P0 方向修正 | Label 7/7, N5 11/11 |
| `ebff71e` | P0 执行修复 | Label 11/11, N5 15/15 |
| `de8fc64` | P0 证据格 | Label 11/11, N5 17/18 |
| `1c807de` | P0 instability + atomic + CUDA | Label 11/11, N5 **18/18** |
| `7ba225a` | **Label V2 P0 闭合** (production_mode 接线) | Label **15/15** |
| `1c1ea12` | Physics Teacher V22 schema | V22 **9/9** |
| `23783d5` | CS200 只读清单 | — |
| `2f23df2` | CS200 元数据盘点 (2000 集, 3.4GB) | — |

### 当前分支测试状态

| 模块 | 测试 | 状态 |
|------|------|------|
| Label Contract V2 | **15/15** | PASS |
| N5 Student | **18/18** | PASS (含 CUDA A800) |
| V22 Schema | **9/9** | PASS |
| V22 Production | **10/10** | PASS |
| 12-episode Pilot | **PILOT PASS** | 12/12 identity, 0 unknown→negative |

### Pilot 关键结果

| 指标 | 值 |
|------|-----|
| Identity join | 12/12 |
| Unknown→negative | 0 |
| Goal blanket NO_TARGET | 0 |
| Total steps | 2259 |
| Critical steps | 2185 (96.7%, 需校准阈值) |
| Attack opportunity | 2063 (91.3%) |

> 注：critical 率偏高因 grasp 检测对 contact_count + gripper_qpos 阈值宽松。链路已贯通但需校准。

### CS200 元数据盘点

| 发现 | 值 |
|------|-----|
| Clean 集数 | **2000** (500/suite, 非 1400) |
| 总大小 | 3.4 GB |
| 每集文件 | 6 JSON + 3 JSONL |
| 物理数据 | `privileged_teacher_sidecar.jsonl` (EEF pos, gripper qpos, object_state, mujoco_contact_pairs) |
| Factorized labels | 600 文件 (states 35-49) |

---

## N5 代码模块清单

| 模块 | 路径 | 用途 | 状态 |
|------|------|------|------|
| Label Contract V2 | `scripts/fec/label_contract_v2.py` | 五头三态标签合同，证据格，atomic writer | **P0 闭合** |
| N5 Student | `n5/phase3_student/n5_student_model.py` | 双 TCN 多头模型 (5 heads) | **18/18 PASS** |
| V22 Schema | `n5/phase2_labels/physics_teacher_v22.py` | 10 物理因子定义，独立 known_mask | **9/9 PASS** |
| V22 Production | `n5/phase2_labels/v22_production.py` | Typed schema, sidecar parser, physics factors, V22→Label V2 adapter | **10/10 PASS** |
| Pilot Pipeline | `n5/phase2_labels/run_pilot_12.py` | CS200 → V22 → Label V2 → K10 → atomic receipt | **PILOT PASS** |
| Pilot Manifest | `n5/phase2_labels/pilot_12_manifest.json` | 12 集精确身份 | 已冻结 |
| CS200 Inventory | `n5/phase2_labels/cs200_inventory.py` | 元数据盘点脚本 | 完成 |
| CS200 Manifest | `n5/reports/CS200_READ_ONLY_MANIFEST_V1.json` | 只读授权清单 | 待审批 |

---

## 当前裁决矩阵

| 模块 | 状态 |
|------|------|
| V4 Formal 计算 | **20/20 完成** |
| V4 Formal 队列提交 | **18/20, 2 待入队** |
| Label V2 代码 P0 | **CLOSED** |
| V22 Schema + 合成测试 | **PASS** |
| V22 Production | **10/10 PASS** |
| 12-episode Pilot | **PILOT PASS** |
| CS200 内容读取 | **HOLD** (待精确 manifest 审批) |
| 1400 全量制标 | **HOLD** (待 CS200 pilot 授权) |
| N5 训练 | **HOLD** (待 Label Seal) |
| N5 攻击矩阵 | **HOLD** |
