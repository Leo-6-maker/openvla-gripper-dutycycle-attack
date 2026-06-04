# VIS Phase-Offset & Online Trigger — 项目交接文档

**日期**: 2026-06-04（覆盖 2026-06-03 夜间实验）
**分支**: `exp/vis-prefix-margin-repair-20260603`
**最新 commit**: `7cf041c`（及后续修复）
**服务器**: `klfy-SYS-4028GR-TR2` (8× RTX 2080 Ti, 11GB)
**Conda 环境**: `openvla_official_libero_20260525`
**Python 路径**: `/home/liuyu/.conda/envs/openvla_official_libero_20260525/bin/python`
**REPO**: `/data/liuyu/repos/openvla-gripper-dutycycle-attack-clean-main-20260524`

---

## 0. 项目当前核心方向

OpenVLA gripper duty-cycle / VIS prefix_margin 推理时攻击。

当前最重要的科学发现已经从"固定窗口攻击"升级为：

> VIS prefix_margin 可以稳定诱导 OpenVLA autoregressive generated OPEN；
> 但 **generated OPEN 本身不是充分条件**。
> OPEN 是否转化为 physical qpos opening 和 task failure，**强依赖任务 phase / offset**。

**正确故事线**:
```
clean early-grasp trigger T_gform
→ apply positive delay Δ
→ attack [T_gform+Δ, T_gform+Δ+17]
→ 在 Δ≈5–20 时出现 strong physical qpos opening 和任务失败
```

**禁止 claim**:
- online detector 已经完成
- trained detector works
- LIBERO-wide generalization
- window independence
- ProprioNoStep-guided VIS established

**可以 claim**:
- heuristic/oracle phase-conditioned smoke
- phase-offset boundary ablation
- early-grasp physical coupling band
- online-feasible trigger+delay policy candidate
- generated OPEN is necessary but not sufficient

---

## 1. 已冻结的 Fixed Primary 主结果

**Task**: ketchup | **Window**: [10,27] | **eps_raw_pixels**: 6
**Objective**: `prefix_locked_gripper_open_margin` | **Code status**: post-repair

### Prefix 4/4 — all perfect

| Seed | OPEN | qpos_opening | armL2 | done | failure_phase |
|------|------|-------------|-------|------|---------------|
| 0 | 18/18 | 0.03763 | 0.000000 | False | early_grasp_disruption |
| 1 | 18/18 | 0.03756 | 0.000000 | False | early_grasp_disruption |
| 2 | 18/18 | 0.03755 | 0.000000 | False | early_grasp_disruption |
| 3 | 18/18 | 0.03756 | 0.000000 | False | early_grasp_disruption |

### Random 6/6 — all clean

| Seed | OPEN | done |
|------|------|------|
| 0-5 | 0/18 | True |

### Provenance notes

- generated OPEN 来自 actual autoregressive generation，不是 teacher-forced probability mass
- `raw_gripper_is_open` 语义已修复：`raw_gripper < 0.5` ⇔ `env_gripper > 0` ⇔ qpos 下降
- prefix armL2=0 表示 arm action 被完全保持
- random controls 使用相同 raw-pixel 预算

---

## 2. ProprioNoStep Bridge Smoke

**Frozen ProprioNoStep 触发步**: T≈93（late-phase selector）

| Window | VIS OPEN | qpos_opening | done | Taxonomy |
|--------|----------|-------------|------|----------|
| W-20 [73,90] | 18/18 | 0.00014 | True | action-positive, physical-negative |
| W-10 [83,100] | 18/18 | ~0.00000 | True | action-positive, physical-negative |
| W0 [93,110] | 18/18 | ~0.00000 | True | action-positive, physical-negative |

**结论**: Frozen ProprioNoStep 是 **late-phase selector**（触发在 natural release 之后），不是 early-grasp vulnerability selector。这是 action bridge positive 但 physical bridge negative 的证据，不是"VIS 失败"。

---

## 3. Phase-Offset Ketchup Seed0 曲线

### 已完成 VIS（全有 clean/random denominator）

| Policy | Window | VIS OPEN | qpos_opening | armL2 | done | Taxonomy |
|--------|--------|----------|-------------|-------|------|----------|
| T+0 | [0,17] | 18/18 | **0.01925** | 0 | False | weak physical |
| T+5 | [5,22] | 18/18 | **0.03458** | 0 | False | **strong physical** |
| T+10 | [10,27] | 17-18/18 | **0.03756** | 0 | False | **strong physical** |
| T+15 | [15,32] | 18/18 | **0.03804** | 0 | False | **strong physical** |
| T+20 | [20,37] | 18/18 | **0.03813** | 0 | False | **strong physical** |

**所有已完成窗口**: random OPEN=0, done=True. clean OPEN=0/18. denominator clean.

### 物理阈值

| 类别 | 条件 |
|------|------|
| strong physical | qpos_opening_delta >= 0.03 |
| weak physical | 0.01 <= qpos_opening_delta < 0.03 |
| physical negative | qpos_opening_delta < 0.01 |

### 物理耦合曲线

```
T+0  [0-17]:   ██░░░░  (weak, 0.019)
T+5  [5-22]:   ██████  (strong, 0.035)  ← 左拐点
T+10 [10-27]:  ██████  (strong, 0.038)
T+15 [15-32]:  ██████  (strong, 0.038)
T+20 [20-37]:  ██████  (strong, 0.038)
T+25 [25-42]:  ??????  (running)
T+30 [30-47]:  ??????  (running)
T+35 [35-52]:  ??????  (running)
T+40 [40-57]:  ??????  (running)
Late [73-110]: ░░░░░░  (none, ~0)
```

---

## 4. 当前 GPU 运行状态

**4 组 GPU 全部在跑右边界 VIS（~60 min ETA）**:

| GPU Pair | Window | Policy | Log |
|----------|--------|--------|-----|
| 0,1 | [25,42] | T+25 | `/data/liuyu/outputs/right_boundary_vis_20260604/vis_Tplus25_to_Tplus42.log` |
| 2,3 | [30,47] | T+30 | `/data/liuyu/outputs/right_boundary_vis_20260604/vis_Tplus30_to_Tplus47.log` |
| 4,5 | [35,52] | T+35 | `/data/liuyu/outputs/right_boundary_vis_20260604/vis_Tplus35_to_Tplus52.log` |
| 6,7 | [40,57] | T+40 | `/data/liuyu/outputs/right_boundary_vis_20260604/vis_Tplus40_to_Tplus57.log` |

**⚠️ 不要杀当前运行任务。不要在当前 4 个 VIS 完成前再开新 VIS/PGD。**

---

## 5. GPU 组使用策略

| GPU Pair | 主卡 | 状态 | 策略 |
|----------|------|------|------|
| 0,1 | GPU1 | GPU0 Xid13 历史 | **默认备用**；仅手动确认健康后使用；不用于 overnight PGD |
| 2,3 | GPU2 | 正常 | 可用于 VIS/random/clean |
| 4,5 | GPU4 | 正常 | 可用于 VIS/random/clean |
| 6,7 | GPU6 | 正常 | 可用于 VIS/random/clean |

### 使用规则

1. 不打断已运行任务
2. long VIS/PGD 优先给 2,3 / 4,5 / 6,7
3. GPU 0,1 仅健康检查通过后使用
4. 同一 pair 不同时启动多个 OpenVLA 模型
5. clean/random ~2-3 min；VIS ~60-70 min
6. 每个 condition 之间 sleep 10-15 秒
7. 所有输出必须写 output_dir + log + manifest
8. 重启后需 `sudo modprobe -r nvidia_uvm && sudo modprobe nvidia_uvm`

### GPU 健康检查

```bash
nvidia-smi
dmesg | tail -n 100 | grep -i "xid\|nvrm" || true
```

---

## 6. Watcher / Queue 策略

### 队列格式 (TSV)

```
id | priority | status | allowed_pairs | gate_file | task_type | task | seed | policy | condition | proposal_csv | output_dir
```

### task_type 说明

| Type | 行为 |
|------|------|
| VIS | 直接运行 vis_pgd |
| RANDOM | 直接运行 random_linf |
| CHAIN | clean → random → audit denominator → VIS only if denominator clean |

### 队列规则

- T+25/30/35/40 当前已直接 VIS running（denominator 已干净）
- 新任务默认 CHAIN
- cream_cheese / salad_dressing **仅允许 random precheck**，不自动 VIS
- tomato_sauce 低优先级候选
- 不训练 detector，不 broad LIBERO sweep

### Overnight 优先级

```
P0: ketchup right-boundary T+25/T+30/T+35/T+40 VIS [当前 running]
P1: ketchup seed1 T+10 CHAIN + seed2 T+10 CHAIN
P2: ketchup seed1/2 T+15 CHAIN
P3: tomato_sauce seed0/1/2 T+10 CHAIN
P4: cream/salad T+10 RANDOM-only
P5: extra clean rollouts seeds 3-5
```

---

## 7. 多任务状态

### 4-task Phase Dataset

- 26 traces，全部 `label_validity=heuristic`，qpos_missing=0
- Oracle proposals 全部 valid，T_gform≈0/1（heuristic 等价于 close-onset detector）
- CSV: `tables/phase_alignment_clean_rollouts_4tasks_seed012.csv`

### Random 预检

| Task | [0,17] denominator | T+10 delayed |
|------|-------------------|-------------|
| ketchup | clean | clean |
| cream_cheese | **polluted** (seeds 0,1 fail) | pending |
| salad_dressing | **polluted** (seeds 0,1 fail) | pending |
| tomato_sauce | clean | pending |

---

## 8. Online Detector 准备

**当前结论: 不训练 learned detector。**

先完成: offset-boundary 曲线 → seed robustness → detector-readiness audit

### 推荐 Online 策略

```
causal early-grasp / close-onset trigger T
→ wait positive delay Δ
→ attack [T+Δ, T+Δ+17]
```

Empirically: Δ=0 weak, Δ=5-20 strong, late ProprioNoStep negative.
**这是 online-feasible positive-delay policy，不需要回溯。**

### Detector-readiness audit（待生成）

需要回答:
- T_gform 是否几乎总是 0/1？
- 是否 rule-based causal trigger 就足够？
- 是否需要 learned TCN？

**决策规则**:
- 如果 T_gform 几乎全是 0/1 → 不要训练 TCN，用 rule-based trigger baseline
- 如果 T_gform 有 task/seed 差异 → 训练 clean-only 3-class phase selector

### 如果训练（未来）

- 输入: 仅 runtime features（gripper command, qpos, width, EEF pose/velocity, action history）
- 禁止输入: object pose, target pose, distance, attacked outcome, VIS success/failure
- 标签: pre_grasp / grasp_formation / post_grasp
- Split: by seed 或 by task

---

## 9. 关键文件路径

### 代码

| 文件 | 用途 |
|------|------|
| `src/gripper_attack/gripper_semantics.py` | 标准 OPEN/CLOSE 语义，物理 qpos 常量 |
| `src/gripper_attack/attack_adapter.py` | PGD 攻击适配器（prefix-locked loss 已修复） |
| `scripts/vis_rollout_adaptive_v3.py` | 主 rollout 脚本 |
| `scripts/vis_phase_conditioned_attack.py` | 相位条件攻击 wrapper |
| `scripts/diagnostics/evaluate_phase_selector_windows.py` | 窗口 proposal 生成器 |
| `scripts/diagnostics/audit_phase_conditioned_vis.py` | 相位条件 audit（claim gate） |
| `scripts/diagnostics/build_clean_phase_dataset.py` | 干净 rollout 相位标注 |
| `scripts/train_phase_selector_scaffold.py` | 相位选择器训练 scaffold（未实现训练循环） |

### 数据

| 文件 | 描述 |
|------|------|
| `tables/phase_alignment_clean_rollouts_4tasks_seed012.csv` | 4-task phase dataset |
| `tables/phase_selector_window_proposals_4tasks_seed012_*.csv` | 各 policy 的窗口 proposals |
| `tables/phase_event_summary_4tasks_seed012.csv` | 每个 rollout 的 phase events |

### 报告

| 文件 | 状态 |
|------|------|
| `reports/VIS_PHASE_OFFSET_BOUNDARY_KETCHUP_SEED0.md` | **待生成** |
| `reports/VIS_ONLINE_DETECTOR_READINESS_AUDIT.md` | **待生成** |
| `reports/VIS_PREFIX_MARGIN_REPAIR_STATUS.md` | 已存在 |
| `reports/VIS_PHASE_ALIGNMENT_AUDIT.md` | 已存在 |

---

## 10. VIS 完成后操作

```bash
cd /data/liuyu/repos/openvla-gripper-dutycycle-attack-clean-main-20260524

# 1. audit 每个 output dir
for d in /data/liuyu/outputs/right_boundary_vis_20260604/vis_Tplus*/; do
  python -u scripts/diagnostics/audit_phase_conditioned_vis.py \
    --run-dirs "$d" --output-csv "$d/provenance.csv" --summary-csv "$d/summary.csv"
done

# 2. 收集所有 summary
find /data/liuyu/outputs/right_boundary_vis_20260604 -name summary.csv

# 3. 更新 offset-boundary 表
```

### 如果 T+25–T+40 VIS 完成

**结果 A（全 strong）**: strong coupling extends at least to T+40/[40,57]
**结果 B（部分 weak）**: 识别右边界位置
**结果 C（denominator polluted）**: 标记，不 claim
**结果 D（OOM/Xid）**: 标记 infrastructure_failure，排除

---

## 11. Claim Boundaries

### Allowed

- phase-offset boundary ablation
- early-grasp physical coupling band
- generated OPEN is necessary but not sufficient
- late ProprioNoStep windows are action-positive physical-negative
- heuristic early-grasp trigger plus positive delay recovers strong physical vulnerability
- online-feasible trigger+delay candidate

### Forbidden

- online detector solved
- trained selector works
- LIBERO-wide generalization
- window independence
- ProprioNoStep-guided VIS established
- cream/salad VIS-specific claim (if random polluted)
- pre-release drop as main VIS vulnerability

### Preferred interpretation

> VIS 攻击本质上不是 pre-release drop 攻击，而是 **early-grasp disruption** 攻击。
> 失败机制: VIS 诱导 generated OPEN → 若 OPEN 落在 early-grasp physical coupling band，qpos 强烈打开 → stable grasp 失败 / task timeout。
> 晚期 generated OPEN 单独不会引发 qpos opening 或任务失败。

---

## 12. 立即下一步

1. **等待四个右边界 VIS 完成**（~60 min）
2. 完成后 audit，更新 offset-boundary 表
3. 如果 GPU 空闲: 跑 ketchup seed1/2 T+10 CHAIN
4. 如果仍有时间: tomato_sauce seed0 T+10 CHAIN
5. **不训练 detector，不跑 cream/salad VIS**
