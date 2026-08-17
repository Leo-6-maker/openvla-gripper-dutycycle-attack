# OpenVLA Gripper Duty-Cycle Vulnerability — GPT 科学审核接管 Handoff

**日期：2026-08-18（Asia/Singapore）**  
**仓库：** `Leo-6-maker/openvla-gripper-dutycycle-attack`  
**基线：** PR #122 head `6160d3b47138166a9159453d463ac062f8df4f95`  
**角色：** GPT = 科学性/claim/gate 审核；Codex = 服务器端实现、执行、封存与工程审计  
**官方 Python：** `/mnt/sdc/dty_user/openvla_attack/envs/openvla-official-a800/bin/python`

配套历史与踩坑附录：`docs/handoffs/GPT_SCIENTIFIC_REVIEW_HISTORY_AND_FAILURES_20260818.md`。

> 接管第一原则：不要把“clean trajectory 看起来 critical”“command-OPEN 物理上有害”“视觉 PGD 能推 OPEN token”合成一个变量。过去几个月最重要的科学进展，是逐步证明这三件事有关但不等价。

---

## 1. 当前科学状态：一页摘要

### 1.1 最强正结果：gripper duty-cycle 物理机制

Stage X X0 当前是 paper 最稳的机制证据：

- 40 个 Stage V parent + 16 个 Stage VI-B2 parent；
- 1,344 个 four-arm probe groups，5,376 branches；
- consumable `V_phys`：T3=1,245，T5=1,191，T10=1,126；
- positive rate：`0.39438 -> 0.67758 -> 0.87300`；
- 1,126 个完整三剂量 pattern 全部属于 `000/001/011/111`，无 non-monotone pattern；
- exact mediator telemetry 支持：commanded OPEN -> aperture excess -> contact loss -> object displacement 随 dose 增强。

冻结结论：

```text
STAGE_X_PHYSICAL_DUTY_CYCLE_MECHANISM_SUPPORTED
```

但只能称 **descriptive mechanism evidence**，不是 formal mediation。

### 1.2 最重要负结果：clean-only timing detector 没有建立跨 suite 泛化

**Stage VI-B2 fresh held-out M4**：16x25D causal Student，threshold=0.69：

- T5 consumable=333，abstain=51；
- AUROC=0.62464；AUPRC lift=1.191；top-decile lift=1.449；
- ECE=0.46064；emission=0.432；
- `libero_10` emission=0；`libero_spatial` emission≈0.958；
- 结论 `STUDENT_HELDOUT_GENERALIZATION_NOT_ESTABLISHED`。

**Stage VII context candidates**：

- S7-A 25D：AUROC 0.7236；
- S7-B 25D+language+clean policy intent：AUROC 0.8380；
- S7-C 再加 frozen visual：AUROC 0.8259；
- 但 Spatial AUROC 仍约 0.05–0.07，S7-B/C emission 接近饱和；
- 全部 fail frozen cross-suite/selectivity gates；
- `STAGE_VII_DEVELOPMENT_NO_GENERALIZABLE_DETECTOR`。

**Stage VIII relative selector**：

- R1-A parent-macro AUC 0.5776；
- R1-B context 0.6586；
- top-k / zero-regret / worst-suite 仍失败；
- `STAGE_VIII_R1_NO_GENERALIZABLE_RELATIVE_SELECTOR`。

正确解释不是“feature 完全没信息”，而是：**overall discrimination != cross-suite selectivity != within-parent actionable timing**。

### 1.3 Model-side PGD targetability 也不等于 physical timing utility

Stage IX F0（no env step）1,344/1,344 rows：

| score | DEVTEST model AUROC | factorized parent-macro AUC | factorized top1 lift | LOSO mean/worst |
|---|---:|---:|---:|---:|
| E0 | 0.8707 | 0.4837 | 0.9697 | 0.4842 / 0.2238 |
| E1 | 0.9005 | 0.5211 | 0.9697 | 0.5592 / 0.3129 |
| E3 | 0.8972 | 0.5234 | 0.9697 | 0.6007 / 0.2363 |

冻结结论：

```text
STAGE_IX_NO_MODEL_SIDE_TIMING_SIGNAL
```

因此“某时刻容易被推 target”不能自动成为“某时刻物理上最值得攻击”。

### 1.4 Historical Stage X X1 不可作为跨 suite PGD->V_phys primary evidence

PR #120 forensic audit 发现：

1. historical X1 用单一 `libero_10` checkpoint 对 Goal/Spatial snapshots 做 PGD；
2. 121/121 sequence starts 的 snapshot-policy path 与 PGD victim path 不匹配；
3. runner 只跑 `eligible_starts["3"]`=121，而完整 nonempty starts=1,344，漏 1,223 starts；
4. `mean_loso_auc` 实际只是 within-suite AUC 的均值，不是真 LOSO；
5. 没有完整 all-start -> gate -> PGD -> metric funnel；
6. direct X1 test gap。

所以 historical `STAGE_X_SEQUENTIAL_PGD_SIGNAL_WEAK` 只能保留为 immutable diagnostic evidence，不能 promotion。

### 1.5 当前最新 T0：未来 PGD authority 仍在 HOLD

PR #122：

```text
STAGE_X_X1R_T0_REVIEW_REQUIRED
specific hold = STAGE_X_X1R_T0_HOLD_TEACHER_FORCED_ROW_PARITY
```

它没有运行 PGD、env.step、physical intervention、V_phys、Eval160 或 protected evaluation。

T0 已确认：

- native checkpoint `ActionTokenizer` 才是 authority；
- project nearest-center helper 在 endpoint/bin-edge 上有 general defect；
- differential mismatches：L10 2927、Goal 3136、Object 3052、Spatial 2843；
- 但 8 个现有 Q00 raw reference rows native/helper 8/8 exact；
- 31744 -> native decode center -> re-encode 31745 是 endpoint non-bijective roundtrip，不等于 gradient failure；
- causal row formula `[-8,-7,-6,-5,-4,-3,-2]`，toy 7/7；
- actual clean rows 165/168 exact，3 个 mismatch 全是 `libero_goal/stage_v/dim=3` 的 reproducible numerical near-tie；
- CW sign、fp32 master、Linf projection、fp16/bf16 cast correction、strict fallback tests pass；
- 旧 production path 只使用 final iterate，`best_restart_metric` 并没有真正选择 best candidate，因此 best-iterate semantics 是 `FROZEN_PROTOCOL_GAP`；
- historical direct generated token IDs=`NOT_IDENTIFIABLE`。

---

## 2. 建议永远使用的三个变量

### `C_t` — clean criticality / opportunity

完全 clean trajectory 上，当前是否处于 gripper 对 manipulation 关键的阶段。可以由 causal proprio/action/history、clean policy intent、frozen language/visual context估计；不能读取 intervention outcome、attacked state、future reward/done/info 或 protected outcome。

### `V_t(d)` — physical vulnerability at intervention dose d

在同一 parent/state 上施加 duration=d 的 command-OPEN counterfactual，是否引起 aperture/contact/object/task effect。它必须由 matched physical intervention 建立。

### `E_t` — visual exploitability / PGD realizability

在 frozen perturbation budget/objective 下，视觉 PGD 是否能改变真实 autoregressive action，使 OPEN command 出现，同时满足 arm/budget constraints。

### 三层窗口

1. clean critical opportunity window；
2. command-OPEN-sensitive physical window；
3. VIS-exploitable window。

论文真正值得研究的是三者什么时候重合、什么时候错位、为什么错位。

---

## 3. Stage V 以后必须保留的因果架构

Stage V 最重要的设计不是某个 checkpoint，而是数据流：

```text
clean rollout
-> privileged clean Teacher
-> deployment-facing causal Student
-> Teacher/Student freeze
-> pre-M4 lock
-> held-out M4 counterfactual validation
```

明确禁止：

```text
M4 outcome -> Teacher/Student tuning
```

已封存过的核心信息：

- Teacher：670 parents / 196,483 clean records；
- Student：deployment-facing causal 25D；
- active heads：physical_criticality / k10_feasibility / instability / gripper_closing_state；
- safe_release=`HOLD_COVERAGE`，零 gradient；
- threshold 在 test read 前冻结；
- test read count=1；
- M4/protected outcome未用于 Teacher/Student selection。

Stage V formal M4 曾因为 clean repeatability、source-plane mismatch、exact replay、queue/worker authority 等工程问题反复 HOLD。这些不是 scientific negative。真正的 held-out Student negative 来自后续 Stage VI-B2 fresh population。

---

## 4. Paper 主线建议

### A. 当前首选：Mechanism-first + factorization gap

建议一句话 thesis：

> **Gripper-targeted manipulation failures exhibit a dose- and phase-dependent physical duty-cycle mechanism, while clean criticality, physical vulnerability, and visual PGD exploitability are only partially aligned; high model-side targetability does not guarantee a reliable physical timing selector.**

贡献可拆为：

1. **Physical mechanism**：OPEN duty、aperture、contact loss、object displacement 和 dose response；
2. **Scientific factorization**：显式区分 `C_t`, `V_t(d)`, `E_t`；
3. **Negative generalization**：Stage VI/VII/VIII 说明 clean/context timing selector 未建立跨 suite/selectivity 泛化；
4. **Model-to-physics gap**：Stage IX 说明 model-side targetability 高并不产生 factorized physical timing utility；
5. **Reproducibility/authority**：suite checkpoint、tokenizer、exact replay、UNKNOWN/censoring、fail-closed provenance。

这是目前证据最尊重正负结果、最不需要 post-hoc tuning 的路线。

### B. 条件升级：若 repaired X1R/X2 成功，可升级为攻击论文

只有 prospective evidence 同时满足：

- suite-matched victim；
- native tokenizer authority；
- direct generated-token evidence；
- TRUE PGD > matched RAND/shuffled；
- arm preservation；
- perturbation budget；
- oracle confirms physical susceptibility；
- TRUE PGD 在 paired physical branch 增加 command/aperture/contact/displacement effect；
- timing control 优于 random-time/early shift；

才能升级成：

> visual perturbation exploits duty-cycle-sensitive phases and causes physically consequential premature opening。

即使成功，也不建议把“universal learned detector”放在主贡献，因为 Stage VI–VIII 已经给出强负证据。

### C. 不推荐：Detector paper

不能写成“clean-only vulnerability detector 已跨 LIBERO generalize”。Stage VI-B2、Stage VII worst-suite、Stage VIII actionable ranking 都不支持。

Detector 最适合作为 negative/generalization result：**高 AUROC/上下文增益不等于 physical timing utility**。

---

## 5. 当前 claim boundary

### 可写

- gripper OPEN duty-cycle physical susceptibility 有清晰 dose structure；
- descriptive mediator chain consistent with duty-cycle mechanism；
- frozen clean/context detectors 没有建立跨 suite generalizable vulnerability selection；
- model-side targetability 与 physical timing utility 明显错位；
- token authority / victim provenance 会实质影响 PGD 有效性；
- early Black Bowl fixed-window experiments 支持 phase-specific mechanism，但只能作为 bounded mechanism-development evidence。

### 不可写

- cross-suite VIS-PGD physical failure 已建立；
- historical X1 可用于 PGD->V_phys promotion；
- 31745 是全局唯一 OPEN token；
- overall AUROC 0.84 就证明 generalization；
- Stage X X0 是 formal mediation；
- historical checkpoint/generated-token identity 已恢复；
- Eval160/protected evaluation 已验证——它们仍 UNREAD。

---

## 6. 新 GPT 接手后先审核，不要先跑 GPU

### Q1. Future PGD authoritative score path

T0 已发现 actual cached autoregressive generation 与 full `use_cache=False` teacher-forced forward 在一个 low-margin row 上会 top1 switch。

必须在任何 PGD outcome 前冻结：

- actual cached autoregressive score path；或
- full teacher-forced surrogate；或
- 明确的 differentiable surrogate + 预注册 rank/margin agreement rule。

不能读结果后再放宽 parity。

### Q2. Native tokenizer V2

必须 checkpoint-local：

- native `ActionTokenizer`；
- suite-specific norm stats；
- raw->native token；
- native token->decoded action；
- OPEN/CLOSE class validation；
- endpoint/boundary/nextafter regression；
- historical helper保持 immutable。

### Q3. Best iterate vs final iterate

未来规则必须 predeclare：final，或 best objective，或 best satisfying arm gate；同时冻结 tie-break 与 extra-generation budget，并让 RAND/shuffled 使用同等 selection budget。

### Q4. X1R primary endpoint

至少同时记录：

- direct generated target/open tokens；
- target/open margin；
- OPEN duty over sequence；
- arm prefix match；
- saved perturbation Linf；
- TRUE vs RAND vs shuffled；
- suite/parent grouping；
- full eligibility funnel。

### Q5. X2 真正回答什么

不是再训练 selector，而是：

> TRUE visual perturbation 在 physically susceptible state 中，是否产生比 matched controls 更强的 command -> aperture -> contact -> displacement effect？

---

## 7. 推荐下一步 gate

### T1 — X1R-V2 authority repair（CPU + clean-only）

必须完成：

1. native tokenizer adapter；
2. suite-matched checkpoint binding；
3. generated IDs first-class logging；
4. authoritative score-path protocol；
5. final/best iterate rule；
6. exact budget accounting；
7. strict no fallback；
8. 4 suites x >=3 fresh clean processes；
9. tests覆盖 7 dims、endpoint、bin edge、nextafter、dtype budget、route、candidate selection；
10. protected counters=0。

**禁止 PGD outcome、M4/V_phys、Eval160/protected read。**

### T2 — X1R no-env PGD realizability（只有 T1 pass 后）

- prospective suite-matched clean snapshots；
- 枚举 all nonempty starts，不再只 length>=3；
- 显式 funnel：all -> nonempty -> gate -> PGD -> horizon support -> metric -> split -> identifiable；
- TRUE / matched RAND / shuffled-gradient / clean；
- same epsilon/steps/selection budget；
- TRUE 必须在预注册 endpoint 上稳定优于 controls 且 arm/budget 通过，才讨论 X2。

### T3 — 小规模 fresh paired physical X2（只有 X1R pass 后）

最小 conditions：

1. CLEAN；
2. COMMAND_OPEN_ORACLE；
3. TRUE_PGD；
4. MATCHED_RAND；
5. SHUFFLED_GRAD；
6. 若预算允许，RANDOM_TIME_TARGETED / EARLY_SHIFT。

逐层 endpoints：

```text
model token/probability
-> command OPEN fraction
-> gripper qpos/aperture
-> finger-object contact
-> object displacement/slip/drop
-> contact-quality failure
-> official task SR
```

official SR 不能作为唯一 failure 指标；至少保留 blinded video subset。

停止规则：

- TRUE model-side 不优于 RAND/shuffled：停止 physical expansion；
- oracle 强、TRUE 不产生 command/aperture：VIS realizability bottleneck；
- token flip 有但 physical effect 无：policy-to-actuator/duty bottleneck。

### T4 — Protected Eval160 最后读

只有 claim、method/version、threshold、candidate 全部 freeze，且接受现有 negative 后，才考虑 protected read。

---

## 8. GitHub/evidence 路线图

- **PR #116 Stage VII**：`STAGE_VII_DEVELOPMENT_NO_GENERALIZABLE_DETECTOR`
  - `docs/handoffs/STAGE_VII_DEVELOPMENT_NEGATIVE_HANDOFF_20260816.md`
- **PR #117 Stage VIII**：`STAGE_VIII_R1_NO_GENERALIZABLE_RELATIVE_SELECTOR`
  - `docs/handoffs/STAGE_VIII_R1_RELATIVE_SELECTOR_NEGATIVE_HANDOFF_20260817.md`
- **PR #118 Stage IX**：`STAGE_IX_NO_MODEL_SIDE_TIMING_SIGNAL`
- **PR #119 Stage X**：X0 mechanism PASS；historical X1 weak/nonpromotional
  - `docs/handoffs/STAGE_X_X0_RESULT_20260817.md`
  - `docs/handoffs/STAGE_X_X1_RESULT_20260817.md`
- **PR #120 X1R forensic**：`STAGE_X_X1R_HOLD_VICTIM_PROVENANCE_MISMATCH`
  - `docs/handoffs/STAGE_X_X1R_PREEXPERIMENT_AUDIT_20260817.md`
- **PR #121 victim closure**：`STAGE_X_X1R_HOLD_CLEAN_FORWARD_TOKEN_PARITY`
  - `docs/handoffs/STAGE_X_X1R_VICTIM_PROVENANCE_CLOSURE_20260817.md`
- **PR #122 latest T0**：`STAGE_X_X1R_T0_REVIEW_REQUIRED`
  - `docs/handoffs/STAGE_X_X1R_T0_PGD_ALIGNMENT_HANDOFF_20260818.md`
  - `reports/STAGE_X_X1R_T0_ROOT_SEAL.json`
  - `reports/STAGE_X_X1R_T0_CLEAN_FORWARD_DETERMINISM_V1.json`

PR #122 head 的 `cpu-stageb`、`cpu-b3-official-v3`、`cpu-detector-v5` CI 均 success，但 CI green **不是 X1R authorization**。

---

## 9. 建议 paper 图表

1. **Fig 1**：`C_t / V_t(d) / E_t` factorization + command->aperture->contact->displacement chain；
2. **Fig 2**：Black Bowl State5/7 historical fixed-window mechanism case：clean/oracle/TRUE/random + qpos/contact/video；
3. **Fig 3**：Stage X X0 T3/T5/T10 dose response + parent-bootstrap CI；
4. **Table 1**：Stage VI B2-C、S7-A/B/C、R1-A/B，列 overall、worst-suite、emission、calibration；
5. **Fig/Table 4**：Stage IX model AUROC high vs factorized timing near chance；
6. **Fig/Table 5**：未来 repaired X1R/X2 TRUE vs RAND/shuffled（若通过）；
7. **Supplement**：OPEN/CLOSE semantics、tokenizer endpoint、source provenance、UNKNOWN/censoring、exact replay、M4 fail-closed history。

---

## 10. GPT/Codex 工作协议

GPT 负责：claim hierarchy、因果结构、controls、data firewall、metric naming、promotion/stop、paper narrative。

Codex 负责：branch/PR、source/tree、server worktree、tests、GPU admission、执行、SHA/ROOT_SEAL、counter、first-failure reporting。

Codex 不得因为 GPU 空闲自行启动下一个 scientific stage；每个 gate 都要停下来等 GPT/owner review。

推荐每次给 Codex 的 prompt 模板：

```text
Scientific question:
Authoritative inputs:
Frozen source/base SHA:
Allowed reads:
Forbidden reads/actions:
Exact implementation change:
Unit tests:
Server clean-only checks:
Promotion criteria:
Failure statuses:
Artifacts + hashes required:
Protected counters required:
GitHub branch / PR target:
Do not proceed beyond this gate without GPT/owner review.
```

---

## 11. 接管 checklist

新 GPT 第一轮必须：

- [ ] 读本 handoff 和历史附录；
- [ ] 用 GitHub核对 PR #116–#122 live state/head；
- [ ] 重点审 PR #122 T0 handoff/root seal/clean-forward report；
- [ ] 确认 Eval160/protected仍 UNREAD；
- [ ] historical X1 标 non-promotional；
- [ ] 接受 Stage VI/VII/VIII negative，不 rerun-to-pass；
- [ ] 审核 future score path；
- [ ] 审核 native tokenizer V2；
- [ ] 冻结 best/final iterate rule；
- [ ] 冻结 T1 pass/fail；
- [ ] 然后才给 Codex 最小 T1 execution prompt。

---

## 12. 最终判断

截至 2026-08-18，已经足够稳的 paper 资产：

1. gripper OPEN duty-cycle physical mechanism / dose response；
2. `C_t`, `V_t(d)`, `E_t` 三者分离；
3. detector/context/relative-selector 的跨 suite negative；
4. model-side PGD targetability 与 physical timing utility 的错位；
5. fail-closed provenance / causal validation methodology。

仍缺：

1. prospective suite-matched native-token-authority true PGD realizability；
2. fresh TRUE > RAND/shuffled；
3. 若 X1R pass，再做 paired physical PGD chain；
4. 最后才是 protected evaluation。

**当前 paper 推荐定位：Mechanism-first / factorization-gap。**  
**攻击闭环：等待 repaired X1R/X2 后再决定是否升级。**  
**Detector：作为重要 negative/generalization result，不作为当前核心正贡献。**
