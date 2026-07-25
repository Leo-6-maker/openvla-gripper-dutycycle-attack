# FEC Smoke / Detector Contract Gap Handoff

**Date:** 2026-07-25  
**Repository:** [Leo-6-maker/openvla-gripper-dutycycle-attack](https://github.com/Leo-6-maker/openvla-gripper-dutycycle-attack)  
**Document branch:** docs/fec-detector-gap-handoff-20260725  
**Code base:** codex/fec-smoke-runner-p0-fix-20260725  
**Last verified FEC code commit in the conversation:** [68a8af0dc73ddb54c31fef57fa49597200b09533](https://github.com/Leo-6-maker/openvla-gripper-dutycycle-attack/commit/68a8af0dc73ddb54c31fef57fa49597200b09533)  
**Purpose:** 给下一个 GPT 对话窗口的完整执行交接。覆盖 Teacher–Student / N4 Detector 到五臂 FEC GPU smoke 的最新进度、已闭合工程合同、失败的16-worker诊断、刚发现的 Detector emit GAP、当前禁止事项、下一步审计顺序，以及最终100-cell attack matrix的原子化领取设计。  
**Document status:** docs-only handoff。它不是新的实验 seal，不替代 A800 服务器上的不可覆盖 evidence root、SHA256SUMS、机器 receipt、access ledger 或 manifest。

---

## 0. 新对话必须首先知道的结论

项目不能继续按“Provider parity 已通过，所以当前 Detector 可直接冻结跑完整矩阵”的旧判断推进。

最新证据表明，一条 LIBERO-10、520 policy-step 轨迹上，candidate_close 与高 calibrated score 几乎不重叠，导致当前 D_PERSIST=6 latch 在该轨迹上不可达。同时，最新诊断报告对 LIBERO gripper OPEN/CLOSE 符号的解释与 GitHub FEC runner 明确写死的动作合同冲突。

当前状态必须冻结为：

    PROVIDER_PARITY                 = PASS_IMPLEMENTATION_REPORTED
    ATTACK_EXECUTOR_PATH            = PASS_CAPABLE
    TARGET_TOKEN_31744              = OPEN_SEMANTICS_REPORTED_PASS_4_SUITES
    WAIT_HORIZON_ACCOUNTING         = PASS_REPORTED
    FOUR_WORKERS_PER_GPU            = REJECTED_OOM
    TWO_WORKERS_PER_GPU             = PENDING_CAPACITY_C2
    CURRENT_16W_DIAGNOSTIC          = FAIL_FINAL_DIAGNOSTIC
    CURRENT_16W_COUNTS_TOWARD_SMOKE = false
    DETECTOR_LATCH_STATE36          = FAIL_FEASIBILITY
    ACTION_SIGN_CONTRACT            = CONTRADICTORY_UNRESOLVED
    CALIBRATION_TRANSFER            = UNRESOLVED
    DETECTOR_FORMAL_ELIGIBLE        = HOLD
    SMOKE_V3                        = HOLD
    FORMAL_GO                       = HOLD
    FORMAL_MATRIX                   = NOT_STARTED
    CS200_ACCESS                    = 0

最短关键路径已经变化：

    封存当前运行快照
    → 动作符号端到端 truth table
    → 冻结训练/验证记录的 label × candidate_close × score × persistence 联合审计
    → 历史成功 emit provenance 审计
    → calibration / raw-logit 分布迁移审计
    → Detector eligibility 决策
    → 若 Detector 仍合法，再恢复 C1/C2 与 Smoke V3
    → Formal Seal
    → 原子化100-cell矩阵

在 Detector 合同审计完成前，不得启动新的 C2、Smoke V3、formal manifest seal 或正式 attack jobs。

---

## 1. 权威来源和冲突处理

当本 handoff、GitHub、聊天汇报和服务器结果不一致时，按以下顺序判断：

1. A800 上不可覆盖 evidence root 中的机器 receipt、SHA256SUMS 和实际 step records。
2. 生成对应证据的准确 checkout SHA、clean-worktree receipt、外部 N4 adapter SHA、checkpoint/normalizer SHA。
3. 冻结 config、identity manifest、parent manifest、random-time manifest。
4. 本 handoff。
5. GitHub draft branch / PR 的代码与说明。
6. main README、历史 handoff、旧对话和旧目录名。

特别注意：

- main README 仍主要描述较早 SC5/25D 和历史 C2f 线路，不代表最新 FEC smoke 状态。
- 最新 FEC runner 位于 codex/fec-smoke-runner-p0-fix-20260725，不在 main。
- N4 runtime module 当前由 runner 的 --n4-module 参数外部加载，聊天中使用的路径是 /tmp/n4_detector_adapter.py。该文件和 norm data 不一定提交到 GitHub，因此仅检查仓库代码不足以完成 Detector source pin。
- 本文中的 GPU/worker完成数是最后一次聊天快照，不是实时监控结果。新窗口必须先检查服务器，禁止重复启动同一任务。

---

## 2. 项目科学问题与当前实验架构

项目研究 OpenVLA 在接触关键阶段受到短时 gripper-targeted visual perturbation 时的选择性失败。

当前逻辑分三层：

    Layer 1: clean privileged Teacher
      → 定义物理关键阶段 / strict K10 opportunity

    Layer 2: causal clean-only Student / N4 Detector
      → 从25D + 9D + 9D历史特征产生 score
      → candidate-close gate
      → threshold + persistence
      → one-shot emit

    Layer 3: FEC attack executor
      → Detector emit 后执行 K=10 visual PGD
      → 或 matched RAND / command OPEN oracle
      → 比较 timing、direction和physical failure

必须保持三个命题分离：

1. Teacher criticality：当前是否是物理上有意义的干预阶段。
2. Detector localization：clean-only causal signal能否在合法时机 emit。
3. Visual exploitability：短时视觉PGD能否让OpenVLA输出OPEN并造成失败。

Provider parity只证明 runtime 忠实复现冻结 provider，不证明冻结 Detector 的 label、动作语义、calibration 或 latch 设计科学正确。

---

## 3. 当前 FEC 五臂协议

GitHub权威配置：

- configs/fec_attack_v3.yaml
- scripts/fec/run_gpu_smoke.py
- scripts/fec/launch_smoke_workers.py

五个 arms：

| Arm | Trigger | Intervention |
|---|---|---|
| CLEAN | logging only | none |
| TRUE_T10 | N4 first emit | gripper-targeted token-prefix visual PGD |
| RAND_T10 | matched N4 first emit | same PGD contract with Rademacher gradient transform |
| COMMAND_OPEN_ORACLE | N4 first emit | force canonical env OPEN, preserve arm dims 0–5 |
| RANDOM_TIME_T10 | preregistered random time | TRUE visual payload at detector-independent timing |

冻结 attack 参数：

    epsilon                 = 0.03
    step_size               = 0.006
    optimizer_steps         = 5
    attack_frames           = 10
    random_start            = false
    strict_route            = true
    allow_fallback          = false
    temporal_init           = none
    objective               = autoregressive_prefix_gripper_target_token_logratio_arm_v3
    target_token_id         = 31744
    target_execution_class  = CLIP_MEDIATED_OPEN
    RAND gradient_transform = rademacher

GitHub runner的动作常量：

    CANONICAL_ENV_OPEN = -1.0
    DUMMY_WAIT_ACTION gripper = -1.0

runner中的默认 candidate-close fallback：

    candidate_close = clean_raw_action[-1] <= 0.5

但外部 N4 provider 可显式返回 candidate_close；若 provider 返回该字段，runner使用 provider 值。因此必须同时核对：

- 外部 provider 对 candidate_close 的定义；
- runner fallback；
- OpenVLA raw action；
- normalize/binarize/invert 后的 env action；
- 真实 gripper qpos/width变化方向。

---

## 4. Wait-step 与 horizon 合同

最后报告的正式合同：

| Suite | Policy horizon | Wait steps | Env horizon |
|---|---:|---:|---:|
| libero_10 | 520 | 10 | 530 |
| libero_goal | 300 | 10 | 310 |
| libero_object | 280 | 10 | 290 |
| libero_spatial | 220 | 10 | 230 |

已报告完成：

- num_steps_wait = 10；
- smoke runner按 policy horizon循环，并将 env_step 记录为 wait + policy_step；
- 20/20 formal parent success_step 都在修正后的 policy horizon 内；
- 最差 margin：L10 +64、Goal +63、Object +62、Spatial +53；
- target token 31744在四suite均解码为OPEN；
- formal random-time manifest尚未生成；
-旧 smoke 的 random-time start 来自运行时/硬编码逻辑，不能自动升级为正式矩阵合同。

Formal random-time manifest必须在 Formal Seal 前生成，并同时记录：

    random_time_start_policy_step
    random_time_start_env_step = 10 + policy_step
    K = 10
    valid range audit
    seed derivation
    manifest SHA

不得在 worker 内重新采样或根据episode进度改变 start。

---

## 5. Provider parity 与代码链闭合进度

最后报告的 Provider 证据：

    20 parents
    3474 frames
    g9d == p9d
    max_abs_diff = 0.0

另外报告：

- EEF proxy parity PASS：robot0_eef_pos == grip_site；
- response proxy parity PASS_AS_FROZEN_DEGENERATE；
- cmd<0相关 close indicator/duration 在冻结训练实现中恒为0；
- 该特征必须描述为 frozen degenerate feature，不能宣称提供独立close-duration信息；
- provider、attacker、suite/unnorm-key接口 P0 bug 已在 68a8af0 线路修复；
- strict route、no fallback、attack re-decode/env.step链已运行；
- RAND Rademacher路径已观察到成功；
- RANDOM_TIME独立触发路径已观察到成功；
- attack_errors与fallback在已完成正常workers中报告为0。

必须重新从服务器 receipt 核对以下字段，不能只引用聊天总结：

    unique_parent_count
    exact parent identities
    suite coverage
    source checkout SHA
    external N4 module SHA
    norm-data SHA
    checkpoint SHA
    feature order SHA
    raw logit parity
    calibrated score parity
    candidate_close equality
    persistence trajectory equality
    emit-step equality

如果20-parent receipt只证明 g9d/EEF，而未证明 score和emit parity，handoff状态必须拆成 FEATURE_PROVIDER_PARITY_PASS 与 FULL_DETECTOR_PARITY_PENDING。

---

## 6. 旧16-worker GPU smoke诊断

### 6.1 最后报告快照

    planned workers = 16
    completed        = 12/16
    PASS             = 8
    FAIL             = 4
    counts_toward_smoke/formal = false

失败分类：

| Failure | Count | Last reported detail |
|---|---:|---|
| CUDA OOM | 1 | CLEAN arm，GPU2，4 workers/GPU配置 |
| Partial TRUE K10 | 2 | 8/10、4/10，Spatial，任务自然终止 |
| Partial RANDOM_TIME | 1 | 5/10，已归类为 terminal-censored Class C |

正信号：

- RANDOM_TIME 10/10 至少4个worker；
- RAND 10/10；
- COMMAND_OPEN_ORACLE 10/10；
- spatial自然 emit，step约64–75；
- RAND gradient transform报告为Rademacher；
- 五臂代码链均已被执行过；
- terminal-censored RANDOM_TIME 5/10不是已知的horizon/protocol bug。

### 6.2 正确终态

无论剩余旧worker最终是否完成：

    CURRENT_16W_DIAGNOSTIC = FAIL_FINAL_DIAGNOSTIC
    counts_toward_smoke    = false
    counts_toward_formal   = false

原因：

1. launcher硬编码4 workers/GPU；
2. 4/GPU已经发生CUDA OOM；
3. Spatial存在partial K10；
4. 没有预注册spare pool；
5. 原始16个accepted identities无法后处理成PASS；
6. 当前 Detector 合同又出现新的不可达GAP。

禁止：

- 将剩余成功结果补成PASS_AT_16；
- 临时搜索更容易emit或有长corridor的identity；
- 覆盖旧receipt；
- 把diagnostic结果计入Formal GO。

---

## 7. Capacity 状态

最后GPU快照：

| GPU | Last reported state |
|---|---|
| GPU2 | 旧16w诊断收尾，曾约65GB、95% util |
| GPU3 | 空闲，原计划用于C2 |
| GPU6 | 旧诊断剩1 worker |
| GPU7 | C1 capacity canary，libero_10五臂 |

这只是历史快照。新窗口第一步必须重新执行 nvidia-smi、ps和输出目录审计。

容量结论：

    4 workers/GPU = FAIL
    2 workers/GPU = candidate only
    1 worker/GPU  = C2失败后的安全回退

C1只有在真实PGD backward完成后才能PASS，不能用16GB模型常驻显存判定。所需receipt至少包含：

    attack_executed_frames
    optimizer_steps
    expected/observed backward calls
    peak torch allocated
    peak torch reserved
    peak NVML used
    minimum free memory
    OOM
    fallback
    attack errors
    Linf budget

C2原设计要求两个worker同步进入PGD backward、各完成10/10，重叠backward至少3帧，且保留至少8GiB余量。

由于Detector合同现已HOLD：

- 已经在运行的C1可以自然完成，但只能记作 CAPACITY_ONLY；
- 不得声称C1验证Detector；
- 在完成Detector合同审计前，不启动新的C2；
- 如果服务器上C2已经被其他agent启动，先封存其状态，不重复启动。

---

## 8. 最新 Detector 未 emit GAP

### 8.1 观测数据

某LIBERO-10、520 policy-step轨迹：

| Condition | Steps | cal_prob >= 0.855 | Conditional rate |
|---|---:|---:|---:|
| candidate_close=True | 75 | 1 | 1.33% |
| candidate_close=False | 445 | 44 | 9.89% |

总高分步数45，只有1步同时满足candidate_close。该步为policy_step=223。

当前 latch：

    candidate_close
    AND calibrated_prob >= 0.855
    for 6 consecutive policy steps
    → emit

因此该轨迹上：

    max consecutive joint streak = 1
    required persistence          = 6
    emit feasibility              = impossible

这应描述为强负关联 / joint-gate不可达，不应写成绝对逻辑互斥，因为仍有1个交集步。

### 8.2 动作语义矛盾

最新诊断文本声称：

    raw <= 0.5 → env +1 → OPEN

但GitHub FEC runner明确写：

    CANONICAL_ENV_OPEN = -1.0

此前修复历史也使用force_open_raw_gripper=1.0来获得最终OPEN行为。当前GitHub的 normalize_and_invert_gripper 逻辑是：

    raw
    → 2*raw - 1
    → sign
    → multiply by -1
    → env action

因此当前聊天报告很可能把env gripper正负方向写反。

必须先用物理qpos/width实测封闭 truth table，才能解释candidate_close与score的关系。不能依据变量名、注释或旧报告直接判定OPEN/CLOSE。

### 8.3 Calibration 数学

报告的Platt映射：

    p = sigmoid(0.519 * raw_logit + 0.813)

正确数值：

    raw_logit = 0   → p ≈ 0.693
    p = 0.5         → raw_logit ≈ -1.57
    p = 0.855       → raw_logit ≈ 1.85

因此“sigmoid中点在raw=0”是错误表述。raw=0只是得到0.693。

当前轨迹报告raw_logit约在：

    positive mode >= +27
    negative mode <= -30

若属实，calibrated score会接近0或1，阈值退化为近似硬开关。但这不能单独证明Platt mismatch，必须与冻结train/validation raw-logit分布比较。

### 8.4 另一个数据一致性问题

报告同时称：

    longest cc=True streak = 14 steps
    range = policy_step 86–119

86–119若连续应是34步。因此需要回看step records确认：

- 是区间内累计14个cc=True；
- 还是实际连续段端点写错；
- 是否存在policy/env step混用；
- 是否丢失/过滤了中间记录。

### 8.5 当前不能下的结论

不能直接写：

- checkpoint损坏已排除；
- state36只是OOD；
- Detector学到了“闭合期间”或“张开期间”；
- candidate_close gate一定方向错误；
- 删除candidate_close即可修复；
- 调低threshold或D_PERSIST即可进入formal；
- 挑选历史能emit的states即可解决。

Provider parity只能排除部分runtime漂移，不能排除冻结label、动作符号、score target和latch之间的合同错位。

---

## 9. Detector合同审计：下一窗口的P0任务

### P0-1：端到端动作符号 truth table

对至少以下raw gripper值执行：

    0.0
    0.49
    0.50
    0.51
    1.0

逐项记录：

    raw_gripper
    normalized value
    binarized value
    inverted env action
    runner semantic label
    external provider candidate_close
    runner fallback candidate_close
    decoded action token
    actual gripper qpos delta
    actual gripper width delta
    visual/physical OPEN or CLOSE

必须使用真实qpos/width方向确认物理行为。

输出建议：

    FEC_ACTION_SIGN_TRUTH_TABLE_V1.json
    FEC_ACTION_SIGN_TRUTH_TABLE_V1.md

硬门：

    ACTION_SIGN_CONTRACT = PASS

### P0-2：外部 N4 source pin

封存：

    /tmp/n4_detector_adapter.py or actual module path
    module SHA256
    checkpoint SHA256
    norm-data SHA256
    feature-order SHA
    proxy formula/version
    calibration parameters
    threshold
    persistence
    candidate_close implementation
    reset/padding/cold-start behavior

GitHub runner允许provider覆盖candidate_close，因此必须确认实际运行时走的是provider字段还是fallback。

### P0-3：冻结train/validation联合分布

在正式冻结的train/validation records上，按suite/task/state计算：

    P(label=1 | candidate_close=True)
    P(label=1 | candidate_close=False)
    P(score>=tau | candidate_close=True)
    P(score>=tau | candidate_close=False)
    count(candidate_close AND score>=tau)
    max consecutive joint streak
    emit count under D=6
    opportunity recall
    absent false-start

必须报告整体和分组结果。仅列“哪些state emit”不够。

关键判定：

- 若训练正标签主要出现在candidate_close=False，而latch要求True，则是训练目标与scheduler gate合同冲突；
- 若train/validation普遍存在joint streak>=6，仅state36缺失，才支持state/task domain shift；
- 若训练数据本身几乎没有joint streak>=6，则当前latch结构性不可达。

### P0-4：历史成功emit provenance

对所有历史成功emit逐个记录：

    suite/task/state/parent
    source run and commit
    t_candidate_start
    t_score_threshold
    t_joint_gate
    t_emit
    t_success
    max joint streak
    operational corridor
    raw-logit quantiles
    physical phase
    provider candidate source

检查emit是否：

- 真实自然出现；
- 依赖padding/reset异常；
- 只集中在少数Spatial identities；
- 位于正确物理阶段；
- 因动作符号误命名而被误解释。

这些states只用于诊断，不得据此事后挑选Formal parents。

### P0-5：calibration / distribution shift

对冻结train、validation、20-parent parity、state36分别报告：

    min, p01, p05, p25, median, p75, p95, p99, max raw_logit
    fraction(raw_logit > 10)
    fraction(raw_logit < -10)
    fraction(cal < 1e-4)
    fraction(cal > 1 - 1e-4)
    score by label
    score by candidate_close
    score by suite/state

如果train/validation logits正常而state36到达±30，才支持迁移失配。若所有域都饱和，则需要判断这是否是hard-separation模型特征，还是calibrator/input head错误。

### P0-6：离线counterfactual gate诊断

仅在冻结记录上比较，不得直接改正式配置：

    A: candidate_close AND score>=tau, D=6
    B: score>=tau, D=6
    C: NOT candidate_close AND score>=tau, D=6
    D: candidate_close AND score>=tau, D=1

目的只是区分：

- candidate方向错位；
- persistence过严；
- state36域外；
- 整个冻结设计不可达。

不得用该诊断在Formal parents上调参。

---

## 10. Detector审计后的决策树

### Case A：动作符号只是报告解释反了

处理：

- 修正所有报告中的OPEN/CLOSE标注；
- 保留原始step records；
- 重新解释cc/score时序；
- 继续P0-3至P0-6，不能因为文字修正就自动恢复Detector资格。

### Case B：冻结标签与candidate gate结构冲突

状态：

    DETECTOR_FORMAL_ELIGIBLE = FAIL_CURRENT_VERSION

由于Formal matrix尚未开始，可以合法重新设计/训练并重新预注册，但必须：

- 使用独立train/validation角色；
- 不读取H2/Formal attack结果调参；
- 生成新checkpoint、calibration、scheduler和runtime parity；
- 旧Detector结果保留为失败版本；
- 重跑Smoke V3。

禁止静默翻转candidate或删除gate后继续使用旧version名。

### Case C：训练/验证可emit，state36不emit

可能属于合法abstention或cross-state/suite迁移失败。需要报告：

    emit coverage across frozen validation
    per-suite/task/state coverage
    timing correctness
    F3/F4/absent false-start
    no-emit ITT disposition

如果预注册Formal pool大部分不能emit，虽然NO_EMIT在统计上有效，但完整矩阵将不能有效测试Student timing，必须先重新评估实验价值，不能仅为完成100 cells而烧GPU。

### Case D：calibration transfer失败

不能在state36或未来Formal parents上重新拟合Platt。必须使用独立calibration role，重新冻结版本，并重做runtime parity和Smoke。

---

## 11. Smoke V3恢复要求

只有Detector合同重新获得PASS后才恢复。

### 11.1 新的capacity-safe launcher

当前 scripts/fec/launch_smoke_workers.py 硬编码：

    GPU_LAYOUT = {2:L10, 3:Goal, 6:Object, 7:Spatial}
    WORKERS_PER_GPU = 4
    TOTAL_WORKERS = 16

该launcher已经被OOM证据否决，不能用于Smoke V3。

Smoke V3需要：

    max_concurrent_per_gpu = 1 or 2, from capacity receipt
    dynamic queue
    16 primary identities
    2 preregistered spares per suite
    deterministic replacement order
    immutable attempt directories
    no output overwrite
    all failed/superseded attempts preserved

Smoke replacement规则：

| Failure | Action |
|---|---|
| CUDA OOM | 降并发后重跑同一identity，不换identity |
| terminal-censored K10 | 仅Smoke允许使用同suite下一预注册spare |
| provider/schema/hash error | global stop |
| fallback/attack exception/budget violation | global stop |
| scientific task failure | valid outcome，不替换 |
| detector no-emit | valid abstention，不替换，除非Smoke工程覆盖合同事先另有冻结规则 |

Spatial clean-only corridor筛选如果使用，必须在attack运行前冻结，且只用于工程Smoke，不得用于正式科学矩阵。

### 11.2 Smoke V3硬门

    accepted identities       = 16/16
    unresolved OOM           = 0
    partial K10 accepted      = 0
    fallback                 = 0
    attack errors            = 0
    budget violations        = 0
    output collisions        = 0
    provider/source mismatch = 0
    TRUE full K10            >= 1
    RAND full K10            >= 1
    ORACLE full K10          >= 1
    RANDOM_TIME full K10     >= 1
    formal cells executed    = 0
    CS200 access             = 0

旧16-worker diagnostic不能替代Smoke V3。

---

## 12. Formal Seal 与100-cell矩阵

计划结构：

    20 preregistered parents
    × 5 arms
    = 100 formal cells

Formal Seal至少绑定：

    final clean git commit
    worktree clean receipt
    FEC config SHA
    detector checkpoint SHA
    external provider SHA
    norm-data / feature-order SHA
    calibration/scheduler SHA
    action sign truth-table receipt
    20-parent parity receipt
    wait/horizon receipt
    token31744 semantics receipt
    capacity receipt
    Smoke V3 receipt
    parent manifest SHA
    random-time manifest SHA
    per-cell seeds
    CS200 access = 0
    formal cells executed before GO = 0

Formal结果必须区分：

    NO_EMIT
    FULL_K10
    TERMINAL_CENSORED_K10
    SCIENTIFIC_TASK_SUCCESS
    SCIENTIFIC_TASK_FAILURE
    INFRA_OR_PROTOCOL_FAILURE

NO_EMIT和terminal-censored是合法ITT disposition，不能在正式矩阵中换parent。Formal只允许对预注册基础设施失败重跑同一个cell。

---

## 13. 用户要求的原子化正式worker设计

核心原则：

- Formal原子任务单位是一个 parent × arm cell。
- worker进程不应每领一个cell就退出。
- 每个GPU slot常驻一个已加载模型的worker，循环：
  1. 原子领取一个cell；
  2. 执行到独立attempt目录；
  3. 原子提交receipt；
  4. 再领取下一个cell。
- 这样兼顾负载均衡和避免反复加载7B模型。

### 13.1 推荐队列

单机A800服务器优先使用本地SQLite：

    journal_mode = WAL
    synchronous = FULL
    busy_timeout >= 30s
    claim transaction = BEGIN IMMEDIATE

队列数据库必须位于本地可靠文件系统。若输出root是NFS，先检测文件系统类型；不得假设SQLite/NFS锁安全。多机执行应改用单一coordinator或PostgreSQL/Redis。

建议表：

    run_meta
    tasks
    attempts
    events

tasks字段至少：

    cell_id
    parent_id
    suite
    arm
    manifest_sha
    state
    priority
    estimated_cost
    lease_owner
    lease_token
    lease_epoch
    lease_expires_at
    accepted_attempt_id
    attempt_count

状态：

    PENDING
    LEASED
    RUNNING
    RETRY_READY
    DONE_VALID
    FATAL_HOLD

### 13.2 Claim与fencing

claim必须在单事务中：

1. 检查GLOBAL_RUN_STATE=RUNNING；
2. 检查manifest/source/config SHA；
3. 选择一个PENDING或RETRY_READY cell；
4. 递增lease_epoch；
5. 写worker UUID、GPU、slot、随机lease token和expiry；
6. 创建attempt row；
7. commit后才开始执行。

heartbeat建议每20–30秒，lease约180–300秒。reaper至少等待3次heartbeat缺失，并检查hostname/PID/boot-id。

commit必须携带：

    cell_id
    attempt_id
    lease_token
    lease_epoch

如果token或epoch不是当前值，旧worker不得提交。系统不能保证物理上exactly-once执行，但必须保证：

    at-least-once attempts
    exactly one accepted attempt per cell

### 13.3 不可变输出提交

建议目录：

    outputs/fec/formal_atomic_v1/
      manifests/
      queue/
      attempts/<cell_id>/<attempt_id>.inprogress/
      attempts/<cell_id>/<attempt_id>/
      accepted/
      receipts/
      logs/

每个attempt先写同一文件系统下的.inprogress目录，完成后：

1. 写完整result/step/attack logs；
2. 计算SHA；
3. fsync；
4. 原子rename为不可变final attempt目录；
5. 使用lease token/epoch事务标记accepted。

crash发生在rename后、DB commit前时，recovery只能在epoch仍为当前且无新attempt时接受该receipt；否则标记superseded/orphan，永不覆盖。

### 13.4 GPU和suite affinity

worker应绑定GPU slot。并发上限来自C2 receipt，不得由队列自动提高。

为避免频繁切换suite checkpoint：

- worker优先领取当前loaded suite的cell；
- 当前suite队列耗尽后，coordinator选择estimated remaining cost最大的suite；
- worker卸载并重新加载该suite模型；
- 不得在一个cell中途切换模型；
- 任务排序可用预注册estimated compute做longest-processing-time优先；
- estimated cost只能来自canary timing和horizon，不能来自科学结果。

### 13.5 Smoke与Formal复用边界

可以复用同一queue core，但任务粒度不同：

- Smoke V3保持现有工程语义：一个identity的五臂bundle可作为原子任务；
- Formal matrix：单独的parent × arm cell为原子任务。

不要为了代码复用而改变Smoke的五臂配对receipt口径。

### 13.6 Formal retry政策

| Result | Formal action |
|---|---|
| task success/failure | valid，no retry |
| NO_EMIT | valid，no retry |
| TERMINAL_CENSORED_K10 | valid，no retry |
| transient simulator/EGL init error | same cell，最多一次预注册retry |
| process crash/stale lease | same cell retry，preserve old attempt |
| OOM at 2/GPU | global stop-claiming，降至1/GPU，再跑同cell |
| OOM at 1/GPU | global HOLD |
| hash/provider/config mismatch | global HOLD |
| fallback/attack exception/Linf violation | global HOLD |
| duplicate accepted result | global FATAL |

Formal不得使用spare parent，不得因attack效果不好或没有emit而替换identity。

### 13.7 Wave 0

Formal GO后先只解锁一个预注册parent的5个cells：

    FORMAL_WAVE_0 = 1 parent × 5 arms

该parent必须在manifest seal前确定，不能按预览结果挑选。Wave 0通过协议审计后，5个cells直接计入最终矩阵，再解锁剩余95个。

---

## 14. 下一窗口的第一轮操作

第一轮只做read-only审计和状态封存，不启动新实验。

必须报告：

    SERVER SNAPSHOT
    - checkout path:
    - git HEAD:
    - git status:
    - active GPU processes:
    - C1 status:
    - C2 status:
    - old 16w completed count:
    - output roots:
    - latest receipts:

    SOURCE PIN
    - run_gpu_smoke SHA:
    - FEC config SHA:
    - external N4 module path/SHA:
    - checkpoint SHA:
    - norm data SHA:
    - provider candidate_close source:

    DETECTOR GAP
    - action truth table status:
    - state36 contingency recomputed:
    - max joint streak:
    - training/validation joint feasibility:
    - historical emit provenance:
    - calibration distribution shift:

    GO/HOLD
    - C2:
    - Smoke V3:
    - Formal Seal:
    - Formal matrix:

如果任何服务器原始artifact缺失，明确列出缺失项并停止，不得用聊天数字填补receipt。

---

## 15. 建议交付物顺序

1. FEC_ACTION_SIGN_TRUTH_TABLE_V1.json / .md
2. FEC_N4_RUNTIME_SOURCE_PIN_V1.json
3. FEC_N4_TRAIN_VAL_JOINT_GATE_AUDIT_V1.json / .md
4. FEC_HISTORICAL_EMIT_PROVENANCE_V1.json / .md
5. FEC_CALIBRATION_TRANSFER_AUDIT_V1.json / .md
6. FEC_DETECTOR_ELIGIBILITY_DECISION_V1.json / .md
7. 若Detector PASS：FEC_CAPACITY_CANARY receipt审计
8. Smoke V3 scheduler + CPU self-tests
9. Smoke V3 primary/spare manifests与overlap audit
10. FEC_GPU_SMOKE_RECEIPT_V3.json
11. Formal parent/random-time/cell manifests
12. Atomic queue self-test receipt
13. FORMAL_GO receipt
14. FORMAL_WAVE_0 audit
15. 100-cell final receipt

---

## 16. 可直接粘贴给下一个GPT的启动提示

请继续 OpenVLA gripper duty-cycle attack 的 FEC Detector / GPU smoke 工作。首先读取：

    reports/FEC_SMOKE_DETECTOR_GAP_HANDOFF_20260725.md

以A800服务器不可覆盖receipt、SHA256SUMS、实际step records和对应checkout SHA为最高权威。不要信任main README代表当前FEC状态。

必须保留以下边界：

- 当前FEC代码基线是codex/fec-smoke-runner-p0-fix-20260725，最后已知修复commit为68a8af0。
- 旧16-worker运行是FAIL_FINAL_DIAGNOSTIC，不能后处理成PASS。
- 4 workers/GPU已因OOM被否决；2/GPU尚未由C2正式证明。
- 20-parent provider parity、EEF parity和冻结退化response proxy已报告PASS，但必须从服务器receipt核对覆盖字段和source SHA。
- 当前GitHub runner定义CANONICAL_ENV_OPEN=-1.0；最新聊天诊断把env +1称为OPEN，二者冲突。
- state36上candidate_close=True共75步，其中仅1步cal>=0.855；candidate_close=False共445步，其中44步cal>=0.855；D_PERSIST=6下最大joint streak为1，因此该轨迹不可能emit。
- 在动作符号truth table、train/validation joint-gate feasibility、historical emit provenance和calibration transfer审计完成前，Detector formal eligibility为HOLD。
- 不得启动新的C2、Smoke V3、Formal Seal或正式矩阵。
- 已运行中的C1若完成，只能作为CAPACITY_ONLY，不验证Detector。
- Formal matrix最终应为20 parents × 5 arms = 100 atomic cells。
- 正式worker采用常驻GPU slot + 原子领取单cell + lease/fencing + immutable attempts；Formal不允许spare parent，只允许同cell的预注册基础设施重试。
- CS200保持0访问，formal cells保持0，直到FORMAL_GO。

第一轮只做read-only状态封存和Detector P0审计。先汇报checkout SHA、active GPU jobs、C1/C2状态、外部N4 module/checkpoint/norm SHA、candidate_close实际来源和action sign truth table计划；不要从旧对话直接继续跑矩阵。

---

## 17. 最终状态摘要

    CLOSED / REPORTED:
      FEC config/code path                 AVAILABLE
      wait-step accounting                 PASS
      parent horizon revalidation          20/20 PASS
      target token 31744 OPEN semantics     PASS across 4 suites
      feature provider parity              PASS reported, verify receipt
      EEF proxy parity                      PASS reported
      response proxy                        PASS_AS_FROZEN_DEGENERATE
      attack executor arms                  CODE/RUNTIME CAPABLE
      random-time 5/10 classification       TERMINAL_CENSORED CLASS C

    FAILED:
      old 16-worker smoke                   FAIL_FINAL_DIAGNOSTIC
      4 workers/GPU                         FAIL_OOM
      state36 detector latch feasibility    FAIL

    UNRESOLVED:
      action sign interpretation            CONTRADICTORY
      external N4 source pin                MUST VERIFY
      train/validation joint feasibility    NOT AUDITED
      calibration transfer                  NOT AUDITED
      historical emit provenance            NOT AUDITED
      2 workers/GPU                         PENDING
      Smoke V3 scheduler                    NOT IMPLEMENTED
      formal random-time manifest           NOT GENERATED
      atomic formal queue                    DESIGN ONLY

    AUTHORIZATION:
      C1 existing run                       MAY COMPLETE AS CAPACITY_ONLY
      C2                                    HOLD
      Smoke V3                              HOLD
      Formal Seal                           HOLD
      Formal Matrix                         HOLD
      CS200                                 0 access
