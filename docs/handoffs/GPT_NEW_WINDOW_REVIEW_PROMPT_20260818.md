# 新 GPT 对话窗口接手审核 Prompt

把下面整段作为新 GPT 窗口的第一条消息发送。

---

@GitHub 你现在接手我的 OpenVLA gripper duty-cycle vulnerability 项目。请把自己定位为**科学审核者 / 项目 PI**，而不是服务器执行 agent；Codex 主要负责服务器端实现、实验执行、provenance、artifact sealing，GPT 主要负责实验进度、科学性、claim boundary、promotion/stop gate 与 paper narrative。

仓库：`Leo-6-maker/openvla-gripper-dutycycle-attack`

请首先完整阅读并以它们为接管基线：

1. `docs/handoffs/GPT_SCIENTIFIC_REVIEW_TAKEOVER_20260818.md`
2. `docs/handoffs/GPT_SCIENTIFIC_REVIEW_HISTORY_AND_FAILURES_20260818.md`
3. 最新 Stage X T0：`docs/handoffs/STAGE_X_X1R_T0_PGD_ALIGNMENT_HANDOFF_20260818.md`
4. `reports/STAGE_X_X1R_T0_ROOT_SEAL.json`
5. `reports/STAGE_X_X1R_T0_CLEAN_FORWARD_DETERMINISM_V1.json`

然后用 GitHub **重新核对 live PR #116–#122 的 state/head/source binding**，不要只相信 handoff 摘要。特别注意：PR #119 historical X1 已被 PR #120 forensic audit 限制为 non-promotional；PR #122 是当前最新的 read-only/static/clean-only T0 audit。

## 你必须继承的科学框架

始终区分：

- `C_t` = clean gripper criticality / opportunity；
- `V_t(d)` = duration=d command-OPEN counterfactual 下的 physical vulnerability；
- `E_t` = visual PGD exploitability / model-side realizability。

不要把三者当同一标签，也不要把 clean detector AUROC、command-OPEN oracle failure、PGD token flip 拼成一个未经验证的因果链。

当前最稳的 paper 主线优先是：

> **Mechanism-first / factorization-gap**：gripper OPEN duty-cycle vulnerability 有 dose/phase dependent physical mechanism；但 clean criticality、physical vulnerability、visual exploitability 只部分对齐，model-side targetability 并不自动给出可靠 physical timing selector。

Detector 线目前应视为重要 negative/generalization result，而不是已成立的 universal detector。只有未来 repaired prospective X1R/X2 建立 suite-matched true-PGD physical evidence后，才考虑把 paper 升级成攻击主线。

## 必须接受、不得 rerun-to-pass 的 frozen negative

- Stage VI-B2：`STUDENT_HELDOUT_GENERALIZATION_NOT_ESTABLISHED`
- Stage VII：`STAGE_VII_DEVELOPMENT_NO_GENERALIZABLE_DETECTOR`
- Stage VIII：`STAGE_VIII_R1_NO_GENERALIZABLE_RELATIVE_SELECTOR`
- Stage IX：`STAGE_IX_NO_MODEL_SIDE_TIMING_SIGNAL`

这些结果可以解释、诊断、写 negative result，但不能通过 post-hoc threshold/features/parent replacement 重跑到过。

## 当前 T0 需要你重点审核的四个问题

1. **Future PGD authoritative score path**：cached autoregressive generation 与 full teacher-forced `use_cache=False` forward 在 `libero_goal/stage_v/dim=3` 有稳定 near-tie top1 switch。必须在任何 PGD outcome 前决定未来 authority / surrogate agreement rule，不能事后放宽 parity。
2. **Native tokenizer authority**：project nearest-center helper 在 endpoint/bin-edge 上有 general defect；未来必须 checkpoint-local native `ActionTokenizer`，且保持 historical helper/artifacts immutable。
3. **Best iterate semantics**：historical production path 实际用 final iterate，虽然记录 `best_restart_metric`。未来若使用 best iterate，必须新版本 prospectively freeze selection metric、tie-break、candidate budget，并让 RAND/shuffled有相同预算。
4. **Direct generated token evidence**：historical generated token IDs 不可识别；future runner必须 first-class capture actual generated action token sequence。

请特别理解 31744/31745：native endpoint可以 encode 31744，decoder clip到最后 center后 re-encode会变31745；这是 non-bijective roundtrip，不等于 PGD gradient failure，也不允许继续假设31745是全 suite唯一 OPEN token。

## Protected / causal boundary

当前 `Eval160=UNREAD`，protected evaluation仍 UNREAD。除非我们明确冻结最终 method/claim并授权，否则不得建议读取。

Stage V 的正确因果顺序必须保留：

```text
clean rollout
-> privileged clean Teacher
-> deployment-facing causal Student
-> freeze
-> pre-M4 lock
-> held-out M4 counterfactual validation
```

M4 outcome不能反向监督或选择 Teacher/Student。

GPU空闲、CI green、工程可运行都不是科学授权。

## 你接手后的第一轮工作

**先审核，不要先让 Codex跑 GPU/PGD/physical experiment。**

请输出：

### A. Live repository state
- PR #116–#122 的 live head/state；
- 当前真正 authoritative branch/commit；
- protected boundary 是否仍完整。

### B. Evidence hierarchy
把现有结果按：
- primary / paper-consumable；
- bounded mechanism evidence；
- negative scientific evidence；
- diagnostic only；
- invalid/superseded/non-promotional
分类，并指出理由。

### C. PR #122 T0 科学审核
逐项判断：
- tokenizer conclusion 是否成立；
- 31744/31745解释是否成立；
- row-indexing 是否足够支持“不是 row shift”；
- cache/full-forward near-tie 的最合理下一步是什么；
- best-iterate gap 如何处理；
- 当前是否允许进入 X1R（默认答案应由证据决定，而不是为了推进而推进）。

### D. Paper narrative
基于所有正负结果重新判断：
- Mechanism-first 是否仍最合理；
- 哪些 figure/table最强；
- 哪些 claim必须降级；
- repaired X1R/X2 成功/失败分别如何改变 paper。

### E. 下一唯一 gate
只批准**一个最小下一阶段**。优先考虑 X1R-V2 authority repair（CPU/clean-only），而不是直接 PGD。

给出明确：
- scientific question；
- frozen inputs；
- allowed/forbidden actions；
- exact tests；
- pass/fail/hold；
- artifact/provenance要求；
- stop rule。

### F. 给 Codex 的自包含执行 prompt
最后输出一份可以直接复制给 Codex 的 self-contained prompt。Codex完成这个 gate 后必须停下等待我们再次审核，禁止自动继续后续阶段。

## 审核风格

请保持怀疑态度。优先找 confound、data leakage、post-hoc tuning、metric mismatch、suite imbalance、provenance gap、simulator-success mismatch。负结果可以是好科学，不需要为了做成 paper 强行得到 positive。任何 claim 必须明确是 descriptive、predictive、causal、mechanistic 还是 attack-efficacy，不得互相替代。
