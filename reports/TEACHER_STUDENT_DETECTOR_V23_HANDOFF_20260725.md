# Teacher–Student Detector V2.3 Handoff

**Date:** 2026-07-25  
**Repository:** [Leo-6-maker/openvla-gripper-dutycycle-attack](https://github.com/Leo-6-maker/openvla-gripper-dutycycle-attack)  
**Purpose:** 新对话窗口的权威项目交接，覆盖 Teacher–Student detector 从 V1 到 V2.3 的迭代、有效与失效实验、当前授权边界、正在验证的实验以及下一步执行顺序。  
**Document status:** docs-only handoff；不是新的实验 seal，不替代 A800 上的机器可读 receipt、SHA256SUMS 或不可变证据 root。

---

## 0. 新对话必须首先知道的结论

当前项目**尚未形成正式 Detector**，但已经完成了从错误监督目标到可信 no-vision V2.3 候选的关键迭代。

    TEACHER_2000_EXTERNAL_K10          = PASS
    V1_PHASE_ONLY_DETECTOR             = FAILED
    V2_STARTABILITY_DETECTOR           = FAILED_BEFORE_H2
    V2.1_STUDENT                       = PASS / FROZEN
    V2.1_P3_SCHEDULER                  = FAIL_NO_FEASIBLE_POLICY
    V2.2_A0_TO_A4                      = INVALIDATED_BY_ENCODER_BUG
    V2.3_N0_REFERENCE                  = VALID
    V2.3_N4_CANDIDATE                  = SELECTED
    VISUAL_EMBEDDING_REQUIRED          = FALSE_FOR_CURRENT_CRITERION
    FORMAL_V2.3_TRAINING               = GO
    C4                                 = HOLD
    P4                                 = HOLD
    NEW_H2                             = SEALED_UNREAD
    FINAL_DETECTOR                     = NOT_AVAILABLE
    FORMAL_ATTACK                      = HOLD

当前最短关键路径：

    冻结 N4 完整 recipe
    → 单 split acceptance
    → 正式 12-split V2.3 Student 训练
    → Student freeze
    → C4 raw ranking / calibration
    → P4 scheduler freeze
    → offline/runtime parity
    → new-H2 一次性评估
    → Final Detector freeze
    → FEC attack pilot
    → formal A-pool attack matrix

---

## 1. 权威来源与冲突处理

当不同文件或对话结论不一致时，按以下顺序判断：

1. A800 服务器上最新、不可覆盖、SHA256 封存的机器 receipt 和 artifact root。
2. 对应执行 checkout 的完整 commit SHA、训练 access ledger、checkpoint manifest。
3. 最新角色 manifest、Teacher/K10 manifest、prediction bundle。
4. 本 handoff。
5. GitHub draft PR 中的代码和说明。
6. main README、历史 handoff、旧分支、旧目录名。

重要边界：

- GitHub main 目前仍主要描述较早的 SC5/25D detector 线路，不能代表本文档中的 V2.1/V2.3 实验状态。
- 大型 checkpoint、prediction、Teacher sidecar 和运行证据主要位于 A800 evidence roots，不全部提交到 GitHub。
- PR #99–#101 提供 factorized calibration、scheduler、adapter、production input 和 fail-closed 合同骨架；它们不是 V2.3 N4 实验结果本身。
- PR #102–#103 是 attack pilot tooling 的 draft，不代表 Detector 已通过 H2 或 attack 已获正式授权。
- 本文记录的 V2.3 N4 数值来自最新实验汇报；正式训练前必须将完整 recipe、source SHA 和 receipt 绑定到服务器证据 root。

GitHub相关入口：

- [PR #99](https://github.com/Leo-6-maker/openvla-gripper-dutycycle-attack/pull/99)
- [PR #100](https://github.com/Leo-6-maker/openvla-gripper-dutycycle-attack/pull/100)
- [PR #101](https://github.com/Leo-6-maker/openvla-gripper-dutycycle-attack/pull/101)
- [PR #102](https://github.com/Leo-6-maker/openvla-gripper-dutycycle-attack/pull/102)
- [PR #103](https://github.com/Leo-6-maker/openvla-gripper-dutycycle-attack/pull/103)

---

## 2. 科学问题：Teacher、Student、Detector、Attack 的边界

### 2.1 Teacher 学术角色

Physics Teacher V2.1C 只在离线阶段使用 privileged simulator state：

- MuJoCo gripper–object contact pairs；
- EEF position/quaternion；
- gripper qpos；
- object state；
- clean task trajectory；
- future K=10 corridor。

External K10 labeler 将 Teacher 输出转成每 step 的 strict-K10 startability：

    y_t = 1:
      从当前 step t 开始存在完整、Teacher 合法的 K=10 critical corridor

    y_t = 0:
      当前 step 不是合法 K10 起点

    unknown / invalid:
      Teacher 或 parser 无法给出可审计判断

Teacher 不预测：

- VIS 是否成功；
- task 是否一定失败；
- qpos 是否一定变化；
- attack outcome；
- random attack outcome。

因此 Teacher 提供的是 clean physical opportunity，不是攻击结果标签。

### 2.2 Student 学什么

V2.1/V2.3 Student 从 causal clean-only history 预测 strict-K10 startability。

V2.1输入：

- 25D proprio/action-derived features；
- 9D clean policy features；
- 9D gripper/decoder features；
- 总计 43D；
- causal history W=32。

V2.3 N4保留43D clean-only输入，但升级为：

- W=128；
- multi-scale causal encoding；
- command–qpos response proxies；
- GroupDRO worst-group training。

Student不能访问：

- future steps；
- privileged contact/object state；
- Teacher labels at runtime；
- attack result；
- H2结果。

### 2.3 完整 Detector 的组成

Student checkpoint 不是完整 Detector。

    Detector
      = runtime feature/proxy adapter
      + frozen Student
      + score transform/calibration
      + threshold/persistence scheduler
      + one-shot latch
      + runtime parity contract

当前目标 scheduler 应保持简单：

    candidate_close
    AND startability_score >= threshold
    for d consecutive steps
    AND not triggered_before
    → one-shot EMIT

V2.3不应恢复V1的复杂 grasp/manipulation/release 三头 FSM，除非新的独立证据明确支持。

### 2.4 Attack executor 不属于 Detector

Detector只回答何时 emit。Attack executor接到emit后执行冻结的K=10 gripper-targeted VIS payload。

Teacher opportunity、Detector正确触发和真实attack failure是三层不同结论。即使Detector通过H2，也仍需FEC/A实验证明Student timing优于random timing。

---

## 3. 数据与Teacher标签闭包

### 3.1 CLEAN2000

总计2000 identities，跨4个LIBERO suites、40 tasks、states 00–49。

历史消费关系曾多次调整；当前版本必须以最新 V2.3 role manifests 为准，不能继续按旧FIT/C2/P2/H1目录名推断科学角色。

### 3.2 新600条 privileged-state audit

states 35–49 共600 identities具备完整 privileged_teacher_sidecar.jsonl：

- mujoco_contact_pairs；
- robot0_eef_pos / quat；
- robot0_gripper_qpos；
- object_state；
- step、state_id、suite、task_idx、task_language。

Audit结果：

    DIRECT_LABELABLE       = 600/600
    REPLAY_RECONSTRUCTABLE = 0
    NOT_RECOVERABLE        = 0

无需 simulator replay，无需重新运行OpenVLA rollout，无需GPU重采集。

### 3.3 External K10 pipeline

正式权威管线：

    privileged sidecar
    → Physics Teacher V2.1C
    → label_k10_v122_v21c.py
    → derive_factorized_rows(require_external_k10=True)

states 35–49运行结果：

| Pipeline | Identities | Steps | Errors | Internal fallback |
|---|---:|---:|---:|---:|
| Physics Teacher V2.1C | 600/600 | 128,223 | 0 | N/A |
| External K10 | 600/600 | 128,223 | 0 | N/A |
| Factorized Teacher | 600/600 | 128,223 | 0 | 0 |

Artifacts：

- OFFICIAL_V3_DETECTOR_V5_TEACHER_PHYSICS_V21C_STATES_35_49_20260725
- OFFICIAL_V3_R7_K10_V122_V21C_STATES_35_49_20260725
- FACTORIZED_TEACHER_STATES_35_49_20260725

关键历史纠错：

- INTERNAL_SIMPLIFIED_V1 不是正式production K10来源。
- 旧materialize注释曾导致误解。
- 正式FIT/C/P/H使用Physics Teacher V2.1C → external K10。
- internal fallback在正式新600条中为0。
- 2000统一Teacher manifest已完成。

---

## 4. 数据角色演化与当前边界

### 4.1 V2.1阶段的角色

V2.1曾冻结：

| Role | Identities | Purpose |
|---|---:|---|
| DEV | 1300 | Student development/training |
| C3 | 200 | calibration/generalization |
| P3-enriched | 300 | scheduler selection |
| new-H2 | 200 | one-shot heldout |

P3由原opportunity-heavy 200条加100条Teacher-defined hard negatives组成：

- F1 = 20；
- F3 = 30；
- F4 = 30；
- other absent = 20；
- parser-invalid = 0。

P3最终闭包：

- opp = 187；
- abs = 113；
- F1 = 32；
- F3 = 30；
- F4 = 30；
- other = 21。

### 4.2 V2.2/V2.3版本化角色

V2.1 P3已被用于发现Student缺陷，因此不能继续作为V2.3正式P4。

V2.3应在H2之外的1800条中冻结新的：

| Role | Planned size | Boundary |
|---|---:|---|
| DEV2 | 1300 | N4 development/formal Student training |
| C4 | 200 | ranking/calibration only |
| P4 | 300 | scheduler selection only |
| H2 | 200 | unchanged, sealed and unread |

正式执行前必须从A800 receipt填入：

- 4个manifest完整路径；
- SHA256；
- pairwise overlap receipt；
- F1/F3/F4 quotas；
- parser-invalid count；
- role access ledger。

本handoff不代替这些机器字段。

---

## 5. Detector迭代历史

## 5.1 V1：三头phase detector

### 定义

V1使用25D causal history预测：

- grasp；
- manipulation；
- release。

Scheduler再组合：

- candidate-close；
- grasp threshold；
- manipulation threshold；
- release veto；
- persistence；
- dwell；
- D_max；
- one-shot FSM。

### Student训练

    12/12 splits PASS
    V2B CausalTCN
    W=32
    epoch=30
    approximately 54K parameters
    no collapse
    no NaN/Inf

### Calibrator V1失败

36个per-split/per-head unrestricted Platt calibrators中：

- 18/36 flat，absolute slope below 0.05；
- 11/36 negative slope；
- 多个Student raw AUROC 0.55–0.90被反转到低于0.35。

典型例子：

- o3_i0 manipulation raw 0.890 → calibrated 0.110；
- o2_i2 manipulation raw 0.897 → calibrated 0.103；
- o2_i1 grasp raw 0.695 → calibrated 0.305。

结论：

    CALIBRATOR_V1 = FAILED_RANK_INVERSION

### Calibrator V2

改为pooled monotonic Platt，约束a>0，避免ranking inversion。该修复在数学上正确，但无法修复Student没有学到的判别信息。

### Scheduler历史bug

- Scheduler V3消费了错误calibrator，作废。
- manipulation threshold恰好等于1.0，且比较使用大于号；有限sigmoid永远无法大于1.0，形成不可达gate。
- Scheduler V4.0 parallel coverage实现错误，误报12/12 coverage，作废。
- Scheduler V4.1修复后P上recall 31.4%、FS 8.8%，但H1 FS升至29.5%。

### H1失败

H1 Engineering Regression：

    Recall      = 33.3%
    False-start = 29.5%
    Precision   = 80.0%
    Emit cov    = 10/12
    K10 executable = 100%

13个false positives的深层分类：

- F3 no manipulation = 7；
- F4 no stable grasp = 4；
- F1 structural zero = 1；
- F6 parser/decoder zero = 1。

最终结论：

    V1_PHASE_ONLY_DETECTOR = FAILED_H1_GENERALIZATION

科学解释：

phase detection不等于K10 startability。V1主要学到policy intent，而不是“从现在开始是否存在完整K10物理窗口”。

---

## 5.2 V2：直接学习K10 startability

### 架构选择

比较：

| Variant | Input | Supervision | H1 recall | H1 FS |
|---|---|---|---:|---:|
| V1 | 25D | phase | 33.3% | 29.5% |
| V2-A | 25D | K10 start | 99.3% | 14.3% |
| V2-B | 43D | K10 start | 99.3% | 11.9% |
| V2-Full | 43D + bypass | K10 start | 100.0% | 14.3% |
| Phase control | 43D + bypass | phase | collapsed | failed |

Selected：

    V2-B
    43D direct concatenation
    no raw bypass
    K10 startability head

Architecture selection receipt：

- path reported under v2_architecture_selection/architecture_selection_receipt.json；
- reported SHA prefix 97b85478d6e64ad3；
- formal use must reference full SHA fromserver receipt。

### Formal V2 Student

    12/12 splits PASS
    pooled DEV AUROC = 0.982
    pooled DEV AUPRC = 0.986

### C2/P2失败

C2：

- step pooled AUROC = 0.755；
- episode pooled AUROC = 0.566；
- per-split alignment only improved episode AUROC约0.592→0.598；
- calibration不是主因，ranking drift是真实问题。

P2：

- opportunity present = 86；
- absent = 26；
- F3 = 5；
- F4 = 12；
- parser-invalid = 11；
- no feasible policy underFS≤10%。

封存状态：

    FINAL_DETECTOR_V2_STATUS = FAILED_BEFORE_H2
    seal prefix = b7a22650877f10d2
    H2 = UNREAD at that version boundary

主要贡献：

- startability supervision修复V1 estimand mismatch；
- 43D提供边际跨分布收益；
- 但旧FIT→C2 state-block drift和P2负例覆盖不足阻塞Detector。

---

## 5.3 V2.1：扩大DEV、统一external K10

### Loss消融

Leave-one-suite-out DEV CV：

| Variant | Loss | Mean AUROC | Median | Min |
|---|---|---:|---:|---:|
| E0 | step BCE | 0.982 | 0.988 | 0.950 |
| E1 | + episode smooth-max | 0.937 | 0.947 | 0.852 |
| E2 | + F3/F4 top-k | 0.976 | 0.992 | 0.921 |
| E3 | E1 + E2 | 0.949 | 0.951 | 0.895 |

选择E0。正确论文表述是“附加loss没有一致收益”，不是在缺少多seed显著性时声称所有附加loss显著有害。

### Formal V2.1 Student

    FORMAL_V21_STUDENT_TRAINING = PASS
    FORMAL_V21_STUDENT_FREEZE   = PASS
    12/12 splits
    pooled AUROC = 0.985
    pooled AUPRC = 0.991
    seal prefix = b1e3e7bc525c8899

### C3成功但校准不可识别

C3：

| Metric | V2 C2 | V2.1 C3 |
|---|---:|---:|
| Step AUROC | 0.755 | 0.673 |
| Episode AUROC | 0.566 | 0.989 |

C3 episode AUROC 95% CI：

    0.973–1.000

Per-suite：

- libero_goal = 0.949；
- libero_object = 1.000；
- libero_spatial = 1.000；
- libero_10 = mono-class，50 opp / 0 abs。

因为C3 step positive rate约99.3%，Platt拟合退化，最终使用RAW_IDENTITY。

正确状态：

    C3_EPISODE_RANKING       = PASS
    C3_STEP_LOCALIZATION     = PARTIAL
    C3_RAW_SCORE_FREEZE      = PASS
    PROBABILITY_CALIBRATION  = NOT_IDENTIFIABLE

不能把RAW_IDENTITY称为充分校准的概率。

### P3无可行policy

P3硬门：

- F3 FS ≤10%；
- F4 FS ≤10%；
- all-absent FS ≤10%。

结果：

    P3_SCHEDULER = NO_FEASIBLE_POLICY

### V2.1尸检

Episode max：

| Comparison | AUROC |
|---|---:|
| opp vs F1 | 0.992 |
| opp vs F3 | 0.775 |
| opp vs F4 | 0.791 |

Overlap：

- 21/30 F3 max高于positive p10；
- 14/30 F4 max高于positive p10。

Positive episode定位：

- Top-1 corridor hit = 97.3%；
- Top-3 corridor hit = 100%；
- inside score > outside score = 97.3%；
- argmax相对first feasible start median +31 steps。

解释：

- Student基本知道正例corridor在哪里；
- 主要失败是F3/F4 absent episodes产生伪高分峰值；
- +31 argmax只表示偏向corridor后部，只要仍strict-K10 feasible，不自动构成label错误；
- scheduler不能凭空创造positive与F3/F4之间不存在的margin。

正式状态：

    V2.1_STUDENT            = VALID_EPISODE_RANKER
    V2.1_P3_POLICY         = FAILED
    FINAL_DETECTOR_V2.1    = FAILED_BEFORE_H2
    H2                     = SEALED_UNREAD

---

## 5.4 V2.2：整组实验因encoder bug失效

V2.2曾报告：

| Variant | Mean AUC | Mean recall | Min recall |
|---|---:|---:|---:|
| A0 | 0.935 | 0.657 | 0.435 |
| A1 | 0.935 | 0.680 | 0.396 |
| A2 | 0.929 | 0.723 | 0.399 |
| A3 | 0.929 | 0.673 | 0.209 |
| A4 | 0.942 | 0.724 | 0.389 |

这些数字不得继续作为科学比较依据。

根因闭包：

| Dimension | V2.2 A0 | V2.3 N0 |
|---|---|---|
| Model | LocalizationStudentV22 | validated CausalTCNEncoder |
| Training data | old DEV | DEV2 with F3/F4 quotas |
| Evaluator | same | same |
| Independent smoke | absent | validated |

LocalizationStudentV22 encoder存在实现缺陷，且在直接进入ablation前没有独立smoke test。

必须保留但标记：

    V2.2_A0_TO_A4_STATUS = INVALIDATED_BY_ENCODER_BUG
    scientific_comparison_authorized = false

禁止在论文或新对话中写：

    V2.3 N4把V2.2 A4 min recall从0.389提升到0.890

这是无效对比。

---

## 5.5 V2.3：可信no-vision候选

V2.3重新使用经过验证的CausalTCNEncoder，并扩展非视觉时序和响应特征。

| Variant | Architecture | Mean recall | Min recall | Every suite ≥0.50 |
|---|---|---:|---:|---|
| N0 | W32 baseline | 0.886 | 0.830 | PASS |
| N1 | W64 multiscale | 0.886 | 0.851 | PASS |
| N2 | W128 multiscale | 0.895 | 0.826 | PASS |
| N3 | W128 + response proxies | 0.897 | 0.851 | PASS |
| N4 | W128 + proxies + GroupDRO | 0.933 | 0.890 | PASS |

可信同协议增益：

    N4 vs N0:
      mean recall       +4.7 percentage points
      worst-suite recall +6.0 percentage points

N3→N4约反映GroupDRO边际作用：

    min recall 0.851 → 0.890
    approximately +3.9 percentage points

前提是N3/N4除GroupDRO外完全相同，并且multi-seed或receipt确认该收益稳定。

当前允许结论：

- W32下43D非视觉信号本身已经较强；
- 更长causal history、response proxies和worst-group optimization带来可信边际改善；
- no-vision observability ceiling尚未耗尽；
- 对当前预注册工程criterion，不需要视觉embedding。

当前不允许结论：

- 视觉在一般情况下无价值；
- N4已经在P4/H2上解决F3/F4；
- N4已经是最终Detector；
- N4 timing一定优于random timing。

正确英文表述：

> Visual embeddings were not required to meet the preregistered detector engineering criterion on DEV2.

---

## 6. N4候选的正式定义

正式训练前必须生成唯一V23_N4_RECIPE.json，不能只写“N4”。

### 6.1 Architecture

必须绑定：

- validated CausalTCNEncoder exact source SHA；
- W=128；
- multiscale branch窗口；
- channel widths；
- dilation/layer结构；
- fusion方式；
- dropout；
- output head；
- parameter count；
- causal padding/mask语义。

### 6.2 Inputs

基础输入：

- 原43D exact feature order；
- feature-order SHA；
- per-feature dtype；
- normalization rule。

Response proxies必须逐项列公式，不能只写response_features=true。至少核对：

- command–qpos residual；
- close-response lag；
- trailing qpos slope；
- trailing qpos variance；
- close dwell；
- trailing EEF displacement；
- policy/gripper response consistency。

所有proxy必须：

- 只使用当前及过去；
- trailing window only；
- 无centered window；
- 无future delta；
- 无episode-global normalization；
- 无Teacher、privileged contact或object state；
- offline/runtime完全一致。

### 6.3 GroupDRO

必须绑定：

- group = suite × Teacher stratum 或正式采用的exact定义；
- group mapping source；
- missing/empty group处理；
- weight update；
- step size；
- clipping；
- normalization；
- loss aggregation；
- checkpoint selection metric。

Teacher strata允许在训练期定义group，但runtime Student不得消费Teacher strata。

### 6.4 W128 cold start

必须冻结：

- left padding；
- history-valid mask；
- step 0–127处理；
- 短episode处理；
- early opportunity recall；
- runtime adapter行为。

禁止模型必须等满128步后才工作，除非预注册Teacher统计证明此前没有需要触发的窗口。

### 6.5 Training

必须绑定：

- optimizer；
- LR；
- weight decay；
- batch size；
- epoch budget；
- early stopping；
- checkpoint selection；
- seed set；
- 12-split mapping；
- normalization fit identities；
- gradient clipping；
- precision/dtype。

---

## 7. 正在验证与下一步授权

### 7.1 已闭合

- External K10 2000 identity Teacher chain。
- V2.2 A0/N0基线矛盾根因。
- V2.2整组invalidated状态。
- V2.3 N0可信reference。
- N4相对N0的真实同协议增益。
- 当前不需要视觉embedding的工程判断。
- H2保持未读。

### 7.2 正式训练前必须在receipt中闭合

如果已有receipt，下一agent应读取并核对；如果没有，必须生成：

1. N4 source/config SHA。
2. Response proxy contract。
3. Future perturbation causality test。
4. Offline/runtime proxy parity。
5. W128 padding/history mask/cold-start test。
6. N3/N4 seed stability或预注册单seed选择依据。
7. DEV2 exact manifest与F3/F4 quotas。
8. C4/P4/H2 access=0。
9. Scheduler-constrained recall定义确认：
   - emit step必须strict-K10 feasible；
   - mistimed emit不能算TP；
   - F3/F4/all-absent FS约束必须真实应用；
   - threshold不得在outer fold上选择。

### 7.3 Formal V2.3 Student

授权状态：

    FORMAL_V23_TRAINING = GO

执行顺序：

1. CPU smoke。
2. 单split GPU acceptance。
3. 12-split multi-GPU正式训练。
4. selected checkpoint reload。
5. checkpoint/normalization/feature/proxy SHA closure。
6. Student freeze。
7. 到此停止，不自动打开C4。

硬门：

    splits completed        = 12/12
    NaN/Inf                 = 0
    constant output         = 0
    reload parity           = PASS
    causality               = PASS
    feature/proxy order     = PASS
    DEV2 identity exact     = PASS
    C4/P4/H2 access         = 0

输出建议：

- FORMAL_V23_STUDENT_TRAINING_RECEIPT.json
- FORMAL_V23_STUDENT_FREEZE_RECEIPT.json
- V23_N4_RECIPE.json
- V23_RESPONSE_PROXY_CONTRACT.json
- V23_CAUSALITY_RECEIPT.json
- 12_CHECKPOINT_MANIFEST.json
- FORBIDDEN_ROLE_ACCESS_RECEIPT.json
- SHA256SUMS

---

## 8. Student冻结后的正式Detector链

### 8.1 C4 raw ranking first

C4先运行raw Student inference，不先拟合calibrator。

报告：

- step AUROC/AUPRC；
- episode AUROC/AUPRC；
- opp vs F1/F3/F4 episode-max AUROC；
- F3/F4 peak overlap；
- Top-1/Top-3 corridor hit；
- first threshold crossing timing；
- per-suite/per-split score distributions；
- constant-output/inversion；
- positive/negative denominator。

如果raw ranking失败，calibration不得被用来掩盖。

### 8.2 C4 calibration

只在raw ranking通过后进行。

由于V2.1 C3为99.3% step-positive并导致Platt不可识别，C4必须验证：

- positive和negative known steps均存在；
- absent episode覆盖；
- ranking preserved；
- monotonic slope positive；
- NLL/Brier/ECE；
- calibrated score非恒定。

如果C4仍无法识别概率，可以冻结raw ranking score，但论文必须称为score而非well-calibrated probability。

### 8.3 P4 scheduler

P4必须与DEV2/C4/H2不交叉，并包含足量F3/F4。

建议仅搜索：

- global score threshold；
- persistence d；
- one-shot latch；
- runtime candidate-close guard。

选择目标：

    maximize valid strict-K10 trigger recall

约束：

    F3 false-start       <= 10%
    F4 false-start       <= 10%
    all-absent FS        <= 10%
    valid emit           > 0
    threshold reachable  = true
    mistimed emit        is not TP

P4若无可行policy：

    FINAL_DETECTOR_V23 = FAIL_BEFORE_H2
    H2 remains unread

禁止为了进入H2临时增加复杂FSM、修改Student或放宽约束。

### 8.4 Runtime parity

必须证明：

- offline 43D/proxy = runtime 43D/proxy；
- offline logit = runtime logit；
- offline score transform = runtime；
- offline scheduler state = runtime；
- offline emit step = runtime；
- reset/one-shot/cold-start行为一致。

### 8.5 H2

只有以下全部PASS才能一次性打开：

    Student freeze
    C4 score/calibration freeze
    P4 scheduler freeze
    runtime parity
    H2 authorization receipt

H2结果必须报告：

- valid opportunity recall；
- absent false-start；
- F1/F3/F4 conditional FS；
- emit precision；
- timing offset；
- K10 executable；
- suite/task coverage；
- abstention。

已知风险：H2来自opportunity-heavy states 35–49，可能缺少F3/F4。如果分母不足：

    H2_RECALL_TIMING      = VALID
    H2_F3_F4_FS          = NOT_ESTIMABLE

不能用少量或零FP声称跨分布FS≤10%。

---

## 9. Final Detector之后才允许的Attack实验

Teacher机会不等于攻击成功。

正式Detector通过后，先运行FEC pilot：

- CLEAN；
- COMMAND_OPEN_ORACLE；
- TRUE_T10；
- RAND_T10；
- RANDOM_TIME_T10。

需要证明：

- runtime/evidence closure；
- K=10 executed；
- arm parity；
- oracle actuation bridge；
- TRUE > RAND；
- TRUE > RANDOM_TIME；
- video/telemetry完整。

FEC不能用于重新训练Student、校准或scheduler。

正式A矩阵仍需独立parent/job/seed/attack seal。不要因GitHub PR #102/#103存在就视为attack已授权。

---

## 10. 有效、失效与禁止引用的结果

### 10.1 有效且可继续使用

- Physics Teacher V2.1C + external K10 V1.2.2正式标签链。
- V1 phase-only失败及其estimand mismatch结论。
- V2/V2.1 startability supervision相对phase supervision的改善。
- Formal V2.1 Student训练与C3 episode ranking结果。
- V2.1 P3 no-feasible-policy及F3/F4 overlap尸检。
- V2.3 N0和N1–N4同协议结果。
- N4相对N0 mean +4.7pp、worst-suite +6.0pp。
- 当前工程criterion下视觉embedding非必需。

### 10.2 必须标记为历史失败

- V1 negative-slope calibrator。
- V1 manipulation threshold=1.0 impossible gate。
- Scheduler V3。
- Scheduler V4.0 coverage bug。
- V2 C2/P2正式失败。
- V2.1 P3正式失败。

### 10.3 完全禁止科学引用

- V2.2 A0–A4作为模型优劣证据。
- N4 vs V2.2 A4的0.389→0.890比较。
- LocalizationStudentV22结果。
- 任何使用internal simplified K10替代external K10的production结论。

---

## 11. 当前最重要的风险

1. **N4 recipe未完整机器绑定。** W128/proxy/GroupDRO必须具体到公式和SHA。
2. **Response proxy runtime parity。** 离线统计若使用未来或episode全局信息，会形成隐藏泄漏。
3. **Cold start。** W128不得造成早期机会不可触发。
4. **Metric denominator。** 历史出现过opp=0显示路径bug、coverage并行统计bug。
5. **Calibration identifiability。** opportunity-heavy C4会再次产生RAW-only。
6. **P4 hard-negative generalization。** DEV2通过不等于P4可行。
7. **H2 hard-negative coverage。** H2可能无法正式估计F3/F4 FS。
8. **Main/PR/server漂移。** 正式执行必须记录checkout完整SHA。
9. **No-vision claim scope。** 只能说满足当前criterion，不是视觉普遍无价值。
10. **Attack authorization。** Detector通过前不得启动正式A-pool。

---

## 12. 新对话Agent的第一轮任务

新Agent不要重新设计Teacher，不要打开H2，不要从V2.2失效结果继续推理。

第一轮应执行：

1. 读取本handoff。
2. 获取最新A800状态，不假设formal V2.3 training尚未/已经开始。
3. 定位：
   - V23_N4_RECIPE；
   - DEV2/C4/P4/H2 manifests；
   - response proxy contract；
   - causality/cold-start receipts；
   - formal training authorization；
   - 当前checkout SHA。
4. 检查C4/P4/H2 access ledger。
5. 若正式训练未开始：
   - 完成单split acceptance；
   - 多GPU执行12-split；
   - freeze后停止。
6. 若训练正在运行：
   - 监控12-split完成度；
   - 不调整超参数；
   - 完成reload parity与seal；
   - freeze后停止。
7. 若Student已经freeze：
   - 先申请C4 raw inference授权；
   - 不得直接打开P4/H2。

---

## 13. 可直接粘贴到新对话的启动提示

请继续 OpenVLA gripper duty-cycle attack 的 Teacher–Student Detector V2.3 工作。先读取 GitHub reports/TEACHER_STUDENT_DETECTOR_V23_HANDOFF_20260725.md，并以 A800 最新机器 receipt、SHA256SUMS 和 access ledger 为最高权威。

必须保留以下边界：

- V2.2 A0–A4 因 LocalizationStudentV22 encoder bug 全部失效，禁止科学比较。
- V2.3可信baseline是N0：mean recall 0.886、min recall 0.830。
- N4为当前候选：W128 multiscale + response proxies + GroupDRO，mean recall 0.933、min recall 0.890；可信增益仅为相对N0的+4.7pp/+6.0pp。
- 当前不需要视觉embedding，但只能解释为满足DEV2工程criterion。
- H2必须保持sealed/unread。
- Formal V2.3 Student训练获GO；C4/P4/H2仍HOLD。
- Student freeze后必须先C4 raw ranking，再calibration，再P4 scheduler，再runtime parity，最后才H2。
- P4必须满足F3、F4、all-absent FS均不超过10%，否则Detector在H2前失败。
- 不得根据C4/P4/H2结果回头修改同版本Student并继续称为独立正式结果。
- Detector通过H2前不得启动formal attack matrix。

请先汇报：当前checkout SHA、N4 recipe/manifest/receipt路径、12-split训练状态、C4/P4/H2访问计数，然后再决定下一步，不要重新从V1开始。

---

## 14. 最终状态摘要

    TEACHER:
      Physics V2.1C                  PASS
      External K10 V1.2.2            PASS
      Unified 2000 manifest           PASS
      Internal fallback               0

    HISTORICAL:
      V1 phase detector               FAILED
      V2 detector                     FAILED_BEFORE_H2
      V2.1 Student                    PASS
      V2.1 scheduler                  FAIL_NO_FEASIBLE_POLICY
      V2.2 A0–A4                      INVALIDATED_ENCODER_BUG

    CURRENT:
      V2.3 N0                         VALID_BASELINE
      V2.3 N4                         SELECTED
      Visual embedding                NOT REQUIRED FOR DEV2 CRITERION
      Formal V2.3 training            GO
      C4                              HOLD
      P4                              HOLD
      H2                              SEALED_UNREAD
      Final Detector                  NOT YET
      Formal Attack                   HOLD
