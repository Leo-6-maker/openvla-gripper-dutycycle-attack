# OpenVLA Gripper Duty-Cycle Attack — Detector-v3 / D8 接管文档

**更新时间：** 2026-07-31  
**目标分支：** `deepseek/detector-v3-d8-continuation-20260730`  
**仓库：** `Leo-6-maker/openvla-gripper-dutycycle-attack`  
**服务器 Python 环境：** `/mnt/sdc/dty_user/openvla_attack/envs/openvla-official-a800`  
**主要服务器输出根：** `/mnt/sdc/dty_user/openvla_attack_outputs/n5/phase3_student/`

> 本文档用于新 GPT / DeepSeek / Codex 窗口直接接管。接管者不得仅依赖本摘要宣称 Gate 通过；必须先核对 GitHub、服务器源码和 sealed artifacts。最新 P3-R1→P5 状态来自服务器执行报告，但该报告未给出最终完整源码 commit，因此接管后的第一项任务是解析实际 branch HEAD、tree、dirty state 和 artifact source binding。

---

## 1. 项目目标与当前科学主张

### 1.1 总目标

项目研究 OpenVLA 在 LIBERO manipulation 中的 **inference-time gripper duty-cycle vulnerability**：

```text
clean online observation/history
→ causal detector 识别 gripper-critical physical phase
→ 在短窗口内运行冻结的 gripper-targeted visual PGD
→ 增加 OPEN command duty cycle
→ 物理夹爪 qpos/width 响应
→ slip/drop/contact-quality degradation 或任务失败
```

Detector 是攻击时机选择器，不是论文最终目的。最终必须证明：

1. Detector 能从 clean、严格因果、部署安全信号中定位物理关键阶段；
2. 同一冻结 VIS payload 在 Student timing 上强于合法 random timing 和 early timing；
3. 同一 Student timing 下，targeted VIS 强于 matched random-direction perturbation；
4. 命令层 OPEN 能转化为 qpos/contact/task effect，而不只是 token flip；
5. Official LIBERO SR 只能作为兼容指标，接触质量和人工视频审计必须保留。

### 1.2 当前允许的 Detector claim

当前仅允许写：

> Detector-v3 使用 clean causal proprio/action features，学习 Teacher 定义的 gripper-critical physical opportunity，并计划作为冻结 VIS-PGD 的在线 timing scheduler。

当前不允许写：

- Detector-v3 已经优于 V4；
- Detector-v3 已经跨 suite 泛化；
- Detector-v3 已经提高攻击成功率；
- G=3 consolidation 的 59 个 bridge 是攻击结果；
- 高 step AUROC 自动等于有效在线 scheduler；
- Official LIBERO SR 单独证明或否定接触型失败。

---

## 2. 历史证据与当前 D8 主线的关系

### 2.1 已冻结的机制证据

早期 Black Bowl fixed-window 实验已经证明：

- gripper-targeted perturbation 可在 contact-critical phase 诱导 OPEN；
- matched random-direction arm perturbation通常不复现相同失败；
- pregrasp timing 明显弱于 carry/pre-place timing；
- State5 与 State7 给出 same-task reproduction；
- simulator SR 会漏报中途掉落、滑移和提前释放。

这些证据的正确定位是：

```text
fixed-window / phase-oracle controlled mechanism evidence
```

不是自动 detector 证据，也不是 broad cross-task generalization。

### 2.2 Detector-v3 相比 V4 的升级目标

V4 更接近：

```text
candidate_close hard gate
AND score threshold
AND fixed persistence
→ emit
```

其主要结构风险是 candidate gate 自身造成不可恢复的 recall ceiling。

Detector-v3 / D8 的目标是：

```text
continuous clean causal trajectory
→ physical-criticality Student score
→ deterministic gripper-state compatibility gate
→ frozen calibration / hysteresis / one-shot scheduler
```

核心升级包括：

- physical event 与 `candidate_close` 解耦；
- UNKNOWN 不作为 negative；
- right-censored / GEOM_NA / articulated 明确 mask；
- fragmented TRUE events 通过 relation-bound event consolidation 合并；
- event/episode level weights 替代纯 step-count dominance；
- 5-fold grouped CV 替代单一 split；
- 25D schema、cache、normalization、fold、weights 全部封存；
- GPU 训练只读 cache，不读 Teacher privileged JSONL。

---

## 3. 数据资产与标签闭包

### 3.1 FIT670 / Fresh670

当前正式 development corpus：

```text
Episodes / identities = 670
Raw steps             = 196,483
```

正式 step taxonomy：

| Category | Count | Training treatment |
|---|---:|---|
| included TRUE | 16,667 | target=1, mask=true |
| included FALSE | 163,007 | target=0, mask=true |
| UNKNOWN | 8,619 | mask=false, weight=0 |
| articulated | 8,100 | NOT_APPLICABLE, mask=false |
| RIGHT_CENSORED | 90 | mask=false, weight=0 |
| GEOM_NA | 0 in reported final taxonomy | mask=false, weight=0 |

Closure：

```text
16,667 + 163,007 + 8,619 + 8,100 + 90 = 196,483
```

### 3.2 Relation Sidecar

Relation sidecar用于证明被合并的 TRUE fragments属于同一 object-target-relation binding。关键修复历史：

- episode ID从 JSON entry读取，不从文件名推断；
- `selected_relation_id` 与 list position分离；
- gap每一步必须存在 matching relation detail；
- boundary relation必须是 `UNIQUE_SUPPORT`；
- object/target identity按 binding logical name显式匹配，不按数组位置猜测；
- object/target binding digest必须与左右边界一致；
- per-relation verdict必须为 `UNKNOWN`；
- per-relation reason使用显式 allowlist；
- formal模式删除 `selected_relation_index` legacy fallback；
- A/B覆盖670/670 episodes和196,483/196,483 steps。

允许 bridge 的 per-relation reasons：

```text
INSUFFICIENT_CAUSAL_PREFIX
RELATION_EVIDENCE_UNKNOWN
```

### 3.3 D8 G-sensitivity

Formal结果：

| G | Consolidated events | Bridged | Rejected | Ratio vs G=0 |
|---:|---:|---:|---:|---:|
| 0 | 734 | 0 | 572 | 100.0% |
| 1 | 707 | 27 | 545 | 96.3% |
| 2 | 687 | 47 | 525 | 93.6% |
| 3 | 675 | 59 | 513 | 92.0% |
| 5 | 671 | 63 | 509 | 91.4% |

当前正式 candidate：

```text
G = 3
Consolidated positive events = 675
Bridged gaps = 59
```

重要语义：bridge只合并event identity；gap中的UNKNOWN steps仍然保持mask=false、weight=0，不得改为TRUE。

---

## 4. D8-1 已完成状态

### 4.1 Relation consolidation

当前裁决：

```text
Relation Sidecar        = PASS_FORMAL_CONSUMABLE
G=3 Event Consolidation = PASS_FORMAL_CONSUMABLE_CANDIDATE
```

关键历史 commits（需接管者在branch ancestry中重新核对）：

```text
5fb42474  formal G runner + episode ID loader
071a6181  relation signature entity_type handling
36057f96  weight audit initial implementation
b4ddeeeb  RC/GEOM_NA exclusion
7751f8f88 R5 relation/weight closure
0e539b4ae R6 source iteration
5b688d45c R6.1 UNKNOWN verdict restoration
b4d156a9d reason allowlist
dd5bb989  formal weight audit rewrite
649619c4  fallback per-episode per-class normalization
```

### 4.2 Weight contract

最终报告指标：

```text
consumer_eligible              = true
Positive ESS                   = 355.9
Negative ESS                   ≈ 781.5
UNKNOWN weight                 = 0
GEOM_NA weight                 = 0
RIGHT_CENSORED weight          = 0
Intra-episode positive events equal = true
Per-episode class total ≈ 1.0       = true
Issues                         = 0
```

冻结语义：

- 每episode每class归一；
- positive event之间等权，再在event内部按step分配；
- negative contiguous spans之间等权，再在span内部按step分配；
- 不再叠加旧 candidate-close weight；
- 不得偷偷叠加额外 `pos_weight`，除非作为明确独立ablation。

已报告 weight audit seal：

```text
56491e22...  provenance-complete formal audit
```

接管者必须读取完整 `SHA256SUMS.sha256`，不能只接受前缀。

---

## 5. P3-R1 → P5 最新状态

以下是最新服务器报告。它们尚需新窗口做 GitHub/服务器三方复核。

### 5.1 P3-R1 25D Schema V2

状态：

```text
PASS
28/28 mutation tests
feature 0 != feature 12 confirmed
```

关键修复：

```text
feature 0  gripper_command = raw_action[6]
feature 12 action_gripper  = env_action[6]
```

禁止把二者当作重复字段。二者分别表达：

- policy raw gripper command；
- postprocessed / executed environment gripper action。

25D schema必须保持：

- strictly causal；
- 无 step index / normalized progress；
- 无 Teacher label/reason；
- 无 object/target privileged pose；
- 无 relation ID / event ID；
- 无 future state/action；
- 无 attack outcome / success / reward。

### 5.2 V3 Streaming Adapter

状态：

```text
PASS
20/20 tests
multi-close causality
windowed flip counting
```

新增文件（按最新报告）：

```text
d8_streaming_features_v3.py
load_fit670_25d_telemetry.py
FIT670_25D_SOURCE_MAPPING.json
test_d8_streaming_adapter_v3.py
run_d8_p5_25d_gpu_smoke.py
```

修改：

```text
DETECTOR_V3_25D_CAUSAL_FEATURE_SCHEMA.json
v5_r3_features.py
build_d8_25d_cache.py
audit_d8_25d_schema.py
```

接管者必须确认这些文件实际路径、commit和SHA；当前GitHub connector搜索尚未索引这些最新文件。

### 5.3 P4-0 Storage Census

报告状态：

```text
670/670 episodes
196,483/196,483 steps
0 missing telemetry
0 nonfinite telemetry
```

需要的原始字段至少包括：

```text
raw_action_7d
postprocessed/env_action_7d
robot0_eef_pos
robot0_gripper_qpos
step / episode identity
```

不得使用 nearest-step join、forward fill、backward fill、重新decode的action或filename推断identity。

### 5.4 P4-1 Strict Telemetry Loader

报告状态：

```text
PASS smoke on 3 episodes
raw gripper command != env gripper action confirmed
```

必须 fail-closed 于：

- 缺step；
- duplicate step；
- 非连续step；
- episode ID不一致；
- nonfinite；
- action维度错误；
- source manifest不匹配。

### 5.5 P4-2 Cache A/B

报告状态：

```text
PASS
A/B per-episode canonical identity
only timestamp differs
```

Cache A seal：

```text
0d4f964a065e3dabe2818d3723f281ba639bea3996c00e7f1aaae7752a0d605b
```

接管审计必须确认：

- Cache A/B独立run UUID、独立staging、独立execution receipt；
- canonical comparator覆盖670/670；
- 196,483/196,483 raw-step closure；
- feature schema digest一致；
- fold assignment一致；
- A/B normalized data未提前拟合；
- privileged-key scan为0；
- test/Eval160/protected reads为0；
- placeholder zero publishing在formal mode中立即raise。

### 5.6 P5 25D GPU Smoke

报告状态：

```text
PASS_ENGINEERING
11/11 gates
GPU 0
Fold 0
Seed 20260717
```

Reported dataset：

```text
Train identities = 507
Val identities   = 136
Train steps      = 141,694
Val steps        = 37,980
Train TRUE       = 13,386
Train FALSE      = 128,308
```

Reported optimization：

```text
Loss 398.6 → 391.1 over 5 epochs
monotonic decrease
finite non-zero gradients
checkpoint restore parity pass
continuation parity pass
privileged leaked keys = 0
```

正确解读：

> P5只证明真实25D cache能够进入训练闭环，并完成finite gradients、checkpoint和访问屏障验证。它不证明泛化、事件召回、scheduler有效或攻击效果。

---

## 6. 当前 Gate 总表

| Gate | 当前状态 | 是否允许自动前进 |
|---|---|---|
| D8-1 Relation Sidecar | PASS_FORMAL_CONSUMABLE | 是 |
| D8-1 G=3 consolidation | PASS_FORMAL_CONSUMABLE_CANDIDATE | 是，需保持claim边界 |
| Formal weight audit | PASS / consumer_eligible | 是，先核完整seal |
| P3-R1 25D schema | REPORTED PASS | 需三方复核 |
| P4 telemetry loader | REPORTED PASS | 需三方复核 |
| P4 cache A/B | REPORTED PASS | 需三方复核 |
| P5 25D smoke | REPORTED PASS_ENGINEERING | 需subagent + source audit |
| D8-2 full 5-fold CV | NOT STARTED / AWAITING AUTHORIZATION | 审计通过后允许 |
| D8-3 multi-seed selection | NOT STARTED | D8-2后 |
| Eval160 | BLOCKED | final model freeze后一次性读取 |
| Online shadow | BLOCKED | Eval160 pass后 |
| Attack canary/matrix | BLOCKED | shadow pass后；工程canary需单独标记 |

---

## 7. 新窗口第一阶段：强制接管审计

不得立即跑15个GPU jobs。先完成以下审计。

### Gate H0：GitHub / Server / Artifact 三方一致性

输出：

```text
HANDOFF_SOURCE_AUDIT.json
SERVER_SOURCE_AUDIT.json
ARTIFACT_BINDING_AUDIT.json
HANDOFF_AUDIT_REPORT.md
```

必须记录：

- branch；
- full 40-char HEAD；
- tree SHA；
- `git status --porcelain`；
- remote URL；
- server checkout HEAD/tree/dirty；
- P3-R1/P4/P5文件完整SHA256；
- tests command与结果；
- Cache A/B roots和完整seal；
- P5 root和完整seal；
- artifact manifest中的commit/tree/script SHA是否与实际执行匹配。

硬门：

```text
local/github/server source closure = PASS
artifact source binding           = PASS
uncommitted experiment code       = 0
unknown generated code            = 0
```

若最新P3-R1/P5代码尚未commit/push：

1. 保留当前artifact，不覆盖；
2. 将源码按实际执行版本提交；
3. 若源码hash无法从artifact证明，P5降级为diagnostic并重跑smoke。

### Gate H1：独立 subagent review

审查：

- 25D因果性；
- raw/env gripper语义；
- multi-close state reset；
- fold assignment；
- normalization train-only；
- D8 weights是否重复叠加；
- disabled heads是否参与优化；
- checkpoint optimizer alias；
- privileged leakage；
- test/Eval160 path barrier；
- failure staging cleanup；
- artifact source binding。

要求：

```text
SUBAGENT_REVIEW = COMMIT_SAFE
```

subagent可修复；修复后必须重跑受影响tests/cache/smoke。

---

## 8. D8-2：完整 5-fold CV 计划

### 8.1 Fold contract

必须满足：

```text
fold_count = 5
validation union = 670 identities
validation pairwise intersection = 0
each identity appears as validation exactly once
per fold val ≈ 134, train ≈ 536
episode overlap = 0
```

按state-group冻结的目标结构：

```text
Fold 0: states 0–3
Fold 1: states 4–7
Fold 2: states 8–11
Fold 3: states 12–15
Fold 4: states 16–19
```

但必须以实际 `FOLD_ASSIGNMENT.json` 为source of truth；不要根据上述文字重新生成。

每fold报告：

- suite/task分布；
- TRUE steps / consolidated events；
- FALSE spans；
- positive/negative ESS；
- zero-positive episodes；
- articulated/unknown/RC exclusions；
- train-only normalization digest。

### 8.2 配置矩阵

第一轮固定：

```text
Seed = 20260717
```

| Config | 定义 |
|---|---|
| B0 | Majority baseline |
| B1 | Frozen heuristic baseline |
| B2 | 25D physical Student + legacy step weighting |
| B3 | 25D + Teacher-event weighting |
| B4 | B3 + suite balancing + G=3 consolidation |

注意：B2/B3/B4必须共享同一架构、初始化策略、epoch budget、optimizer和scheduler；只改变预注册的weight/consolidation因素。

### 8.3 训练纪律

- 不加载full670 overfit checkpoint；
- normalization按fold train-only拟合；
- validation不参与fit；
- 不实例化Eval160/test；
- 每配置同fold使用相同初始化digest；
- 禁止按单fold结果中途改epoch、loss、threshold；
- failed job保留root，不覆盖；
- GPU只读sealed cache；
- attack outcome、Teacher reason和relation detail不得进入模型。

### 8.4 Step metrics

报告：

```text
AUROC
AUPRC
BACC
MCC
minority recall
ECE / calibration
per-suite macro
per-task distribution
```

Accuracy不能作为主指标。

### 8.5 Event / scheduler metrics

必须用OOF predictions复算：

```text
consolidated Teacher event recall
critical-episode opportunity hit rate
false-trigger episode rate
no-trigger rate
first-trigger latency
trigger-before-event rate
trigger-after-event rate
per-suite event recall
per-task trigger distribution
```

candidate-close不得作为hard gate。最终scheduler应由：

```text
physical Student score
+ deterministic gripper-state compatibility
+ validation-frozen threshold/hysteresis
+ one-shot emit
```

组成。

### 8.6 预冻结开发门槛

建议保持：

```text
mean OOF AUROC                 >= 0.80
mean OOF BACC                  >= 0.70
mean OOF MCC                   > 0
event recall                   >= 0.65
critical-episode hit rate      >= 0.60
false-trigger episode rate     <= 0.30
beats B1 event hit rate        >= +0.10
beats label-shuffle            >= +0.10
```

suite denominator >=10时，建议event recall >=0.40；小分母只报告置信区间。

若所有配置失败：停止，不得读取Eval160。

---

## 9. D8-3：多种子确认和最终选择

第一轮CV后最多选择两个配置。

种子：

```text
20260717
20260718
20260719
```

至少加入：

- label shuffle；
- linear probe；
- frozen heuristic。

稳定性门：

```text
AUROC seed std       <= 0.03
event hit seed std   <= 0.08
no catastrophic seed = true
```

最终冻结：

```text
SELECTED_CONFIG
SELECTED_FEATURE_SCHEMA
SELECTED_WEIGHT_PROTOCOL
SELECTED_NORMALIZATION
SELECTED_THRESHOLD
SELECTED_SMOOTHING
SELECTED_PERSISTENCE
SELECTED_GRIPPER_GATE
```

### 强制停止点 S1

完成D8-3后停止，提交全部CV / OOF / calibration / subagent报告。未审核前不得读取Eval160。

---

## 10. D8-4：Final Detector-v3 Freeze

GPT放行后：

1. 使用全部670 development episodes训练最终模型；
2. 所有超参数完全来自CV；
3. 不再选择threshold；
4. 冻结checkpoint、normalization和scheduler。

输出：

```text
DETECTOR_V3_CONTRACT.json
MODEL_MANIFEST.json
CHECKPOINT.pt
NORMALIZATION.json
SCHEDULER.json
FEATURE_SCHEMA.json
CV_DECISION_RECEIPT.json
ACCESS_AUDIT.json
SHA256SUMS
SHA256SUMS.sha256
```

最终在线Detector仅包括：

```text
25D causal physical Student
+ deterministic gripper compatibility gate
+ frozen calibration
+ frozen threshold/hysteresis
+ one-shot trigger
```

K10 feasibility、instability和safe-release若未通过learnability gate，必须保持Teacher-only或deterministic，不得包装为Student能力。

---

## 11. D8-5：Eval160 一次性独立测试

读取前确认 Eval160 与正式schema兼容：

- clean 25D telemetry完整；
- Teacher physical labels可用；
- relation-bound event schema一致；
- collector / preprocessing / checkpoint provenance一致。

若旧Eval160缺字段：

```text
不得猜测、插值或从旧结果恢复；使用冻结collector重新采fresh Eval160。
```

一次性规则：

- model/scheduler frozen；
-只运行一次正式test；
-不得在Eval160上调threshold或重选checkpoint；
-失败也保留并报告。

建议门槛：

```text
AUROC                    >= 0.80
BACC                      >= 0.70
MCC                       > 0
event recall              >= 0.60
critical-episode hit      >= 0.55
false-trigger episode rate <= 0.35
```

### 强制停止点 S2

Eval160结束后停止。未审核前不得运行shadow或attack。

---

## 12. D8-6：Online Shadow

仅在Eval通过后执行。

建议：

```text
40 episodes
4 suites × 10 task identities
```

Detector实时读取clean causal features，但不修改action。

必须验证：

```text
action mutation = 0
offline/live feature parity = PASS
normalization parity = PASS
checkpoint digest = MATCH
nonfinite = 0
one-shot trigger <= 1/episode
latency within control budget
protected reads = 0
```

Teacher overlap只能在episode结束后离线复算，不能影响在线emit。

---

## 13. D8-7：最小 Attack Canary / Matrix

### 13.1 工程canary（可选，非正式）

若只为打通接口，可做：

```text
2–4 development-held-out identities
× 4 arms
```

Arms：

```text
CLEAN
VIS @ Student timing
VIS @ legal random timing
matched random-direction @ Student timing
```

标记：

```text
ENGINEERING_ATTACK_CANARY
NONCONSUMABLE
NOT_USED_FOR_CLAIM
NOT_USED_FOR_THRESHOLD_SELECTION
```

### 13.2 正式最小矩阵

建议：

```text
12 identities
4 suites × 3 identities
× 6 arms
= 72 rollouts
```

Arms：

| Arm | 条件 |
|---|---|
| C0 | Clean，无攻击 |
| C1 | Frozen VIS-PGD @ Student timing |
| C2 | Frozen VIS-PGD @ random legal timing |
| C3 | Frozen VIS-PGD @ early timing |
| C4 | Frozen VIS-PGD @ Teacher Oracle timing |
| C5 | matched random-direction perturbation @ Student timing |

攻击参数必须完全冻结：

```text
same victim checkpoint
same preprocessing
same epsilon
same PGD steps
same K=10
target OPEN semantics fixed
same arm-control policy
```

不得根据Detector或canary结果改变epsilon、steps或K。

### 13.3 选择规则

Identity必须在查看attack outcome前冻结：

- clean rollout有效；
- mechanism eligible；
- suite/task分布预注册；
- 不因历史攻击成功而选择；
- 同identity各arms在同一物理GPU block运行；
- arm顺序随机化。

### 13.4 指标

Detector：

```text
trigger coverage
Teacher event overlap
Student–Oracle timing distance
no-trigger / false-trigger
```

Attack：

```text
official SR (secondary)
CQFR / CQSR (primary)
object slip/drop/contact loss
gripper OPEN duty/streak
qpos/width excursion
arm deviation
VIS vs random timing paired difference
VIS vs early paired difference
VIS vs random-direction paired difference
```

人工视频盲审必须用于校准自动contact-quality metrics；不能只依赖LIBERO SR。

### 13.5 Canary判定

小样本只允许：

```text
PASS_CANDIDATE
HOLD
FAIL_DIAGNOSTIC
```

至少需要看到：

- Student timing方向性优于random timing；
- Student timing优于early timing；
- matched random-direction不产生同等效果；
- 效果出现在至少两个suite；
- 不是仅有token OPEN而无qpos/contact response。

### 强制停止点 S3

完成最小矩阵后停止，不得自动扩样本。

---

## 14. 失败归因必须分层

Attack失败不能统一写成“Detector失败”。必须分类：

```text
A. Student未触发
B. Student触发但不在Teacher critical event
C. 时机合理但VIS未改变OPEN command
D. OPEN command增加但未形成足够duty cycle
E. duty cycle形成但qpos/width未响应
F. qpos响应但物体未失稳
G. contact effect发生但任务仍成功
H. official SR与人工接触质量不一致
I. GPU/rollout nondeterminism掩盖差异
```

这套taxonomy应进入最终attack ledger。

---

## 15. 资源和并行建议

### CPU

并行处理：

- cache/fold/schema audit；
- B0/B1；
- OOF metrics和calibration；
- seal/comparator；
- subagent静态审查。

### GPU

5-fold首轮建议：

```text
5 GPU workers，每个GPU负责一个fold，顺序运行B2/B3/B4
```

不要为了满8卡导致I/O争用或重复读取Teacher数据。

OpenVLA shadow/attack先benchmark：

```text
1 / 2 / 4 workers
```

选择episodes/hour最高的并发度，通常不建议直接8路并发。

---

## 16. Artifact与提交纪律

每阶段：

1. 输出到新root；
2. staging原子发布；
3. 写manifest；
4. 写递归SHA256SUMS；
5. 独立verify；
6. subagent review；
7. commit/push；
8. 报告完整commit/tree和完整seal。

禁止：

- 覆盖失败root；
- 删除失败evidence；
- 修改阈值后重写旧PASS；
- 使用seal前缀代替完整seal；
- dirty tree运行formal experiment；
- artifact source commit与最终代码不一致却不降级；
- 在GPU训练进程中直接读取Teacher privileged files；
- 在Eval160或attack结果后回调detector超参数。

---

## 17. 新GPT接管后的第一条回复应包含

新窗口读取本文档后，不要立即声称开始CV。先给出：

```text
1. GitHub branch full HEAD/tree
2. server checkout HEAD/tree/dirty
3. 最新P3-R1/P4/P5源码commit
4. Cache A/B完整roots和seals
5. P5完整root和seal
6. tests实际命令与结果
7. subagent review状态
8. 未闭合问题清单
9. 是否放行D8-2的Gate裁决
```

如果全部闭合，执行：

```text
H0/H1
→ D8-2 5-fold single-seed CV
→ D8-3 top≤2 multi-seed
→ STOP S1
```

不得跳过S1直接读取Eval160。

---

## 18. 当前最简状态总结

```text
Physical mechanism evidence             = established historically
FIT670 label/step closure                = established
Relation-bound D8 consolidation          = formal candidate established
Weight contract                          = formal consumer eligible
Real 25D schema/telemetry/cache pipeline = reported complete
25D single-fold GPU smoke                = reported engineering PASS
5-fold detector generalization           = not yet tested
Event-level online scheduler             = not yet established
Eval160                                  = untouched / blocked
Online shadow                            = not run
Detector-guided attack matrix            = not run
```

当前最优推进方向不是再改Teacher标签，而是：

```text
先完成源码与artifact接管审计
→ 正式5-fold OOF评估
→ 多种子冻结scheduler
→ 一次性Eval160
→ shadow
→ 最小matched attack matrix
```

文档末尾的claim boundary必须始终保留：

> Detector-v3学习的是 clean gripper-critical physical opportunity，不直接预测攻击成功。最终攻击有效性必须由冻结VIS payload在Student、random、early、Oracle和random-direction条件下的matched闭环实验独立证明。
