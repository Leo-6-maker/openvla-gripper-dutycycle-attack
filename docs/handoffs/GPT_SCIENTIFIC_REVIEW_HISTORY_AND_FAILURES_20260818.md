# GPT 科学审核接管附录：实验演化、探索分支与失败模式

**日期：2026-08-18**  
**主 handoff：** `GPT_SCIENTIFIC_REVIEW_TAKEOVER_20260818.md`

本文专门回答两个问题：

1. 我们是怎样从早期 Black Bowl fixed-window attack，一步步走到 Stage X mechanism/factorization 的？
2. 中间哪些 bug、confound、data-gap、runtime failure 使得历史结果不能直接升级成 final paper claim？

---

## A. 早期 fixed-window Black Bowl：先发现机制，再意识到 timing/confound 问题

最早观察到的是 gripper-specific temporal vulnerability，而不是 detector。

典型现象：

- post-grasp / carry / pre-place 的 OPEN duty 干预更危险；
- pregrasp timing 弱；
- targeted VIS 与 command-OPEN oracle 能产生 slip/drop/contact disturbance；
- random-direction 通常弱很多；
- State5、State7 有 same-task reproduction；
- official task success 有时仍判 success，但视频里已经发生掉落/滑移/提前释放。

用户当时记录的 State7 汇总：

```text
VIS_margin_prevdelta        4/5 failure
CTRL_same_gate_zero_margin  3/5
CTRL_random_direction       0/5
ORACLE_continuous           3/3
CONSTANT_DELTA_pregrasp     0/1
```

unsafe consecutive OPEN streak 在 offline guard 后可从 VIS 44、zero-margin 17、oracle 14 降到 0；这说明 duty-cycle 本身是有结构的，但 guard 只做过 counterfactual/offline analysis，不能写成已验证 defense。

### A.1 早期最严重的 gripper action semantics bug

OpenVLA raw 和 LIBERO env gripper 方向反转：

```text
raw OpenVLA OPEN > 0.5
postprocess后 LIBERO env OPEN = -1
raw CLOSE ~= 0
LIBERO env CLOSE = +1
```

历史 `gripper_logit_margin_cw` 一度没有显式传 `force_open_raw_gripper=1.0`，默认 0.0 实际是 CLOSE。后来 commit `faa7fc7...` 才把 OPEN target 修正。

**教训：** 所有 future attack / oracle / classifier都必须写清楚 raw-space、token-space、env-space、physical execution class，禁止只写“positive/negative gripper”。

### A.2 Simulator SR 不是接触型 failure 的可靠唯一指标

早期手工视频审查发现：对象可以中途滑落、失去抓取质量、提前释放，但最终 predicate 仍可能给 success。

所以最终 paper 必须保留：

- official SR；
- contact-quality；
- object displacement/height；
- gripper qpos/aperture；
- 有争议样本的 blinded manual video。

---

## B. V4 formal：自动 detector 第一次系统失败

V4 做过 20/20 formal parents：

```text
libero_10     5/5 emit
libero_object 5/5 emit
libero_goal   0/5 emit
libero_spatial 2/5 emit（其中一个 K10 truncation）
```

表面看是 Goal detector 不工作，后来 root-cause audit 证明核心是 label definition。

### B.1 `candidate_close` 被错误放进 label funnel

旧 pipeline：

```text
candidate_close
AND physical/score condition
-> label
-> Student
-> candidate_close gate again
-> scheduler emit
```

Goal 中大量 `cc=False` 被直接变成 label=0。Student 学到的是“候选 gate 截断后的标签”，而不是 physical criticality。

结果：

- Goal raw logit≈-11；
- calibrated probability max≈0.005；
- 即使拿掉 scheduler gate，score仍几乎全负。

最终根因判断：

> V4 implementation 本身可复现，但 supervision target 定义错了。

### B.2 V4 TRUE_VIS 也不是正确的 premature-open attack

2026-07-26 forensic handoff 发现：

- emission 时 clean env grip 全是 +1 CLOSE；
- `target_action=clean_raw_action`，没把 gripper改成 OPEN；
- 当时 `TARGET_TOKEN_ID=31744` 被当成 clip-mediated/invalid class；
- V4 TRUE_VIS 更接近 release-blocking CLOSE。

所以 V4 formal attack arm 不能成为今天正确 OPEN-PGD 的 final efficacy evidence。

---

## C. V5/N5：重写 Teacher/label contract

V5/N5 的核心修复：

1. candidate_close 不再定义 physical target；
2. Teacher 从 privileged clean physical evidence 建 label；
3. UNKNOWN / articulated / unsupported 明确 abstain；
4. Student只读 causal deployment-safe input；
5. M4/attack outcome不进入训练。

2026-07-26 的 800-label seal：

```text
800 episodes
176,336 steps
Goal critical = 43.7%
Goal UNKNOWN  = 20.7%
unknown->negative = 0
```

这证明 Goal 的 V4 零覆盖不是“Goal没有 critical phase”，但还不是 held-out Student 成功。

### C.1 Safe-release 与 instability head 的 coverage 问题

- safe_release 极稀疏：只有约 16/800 episodes、16 positive steps；
- instability 大量 unknown；
- articulated drawer/stove缺 joint-progress telemetry。

后续 Stage V freeze 因此把 safe_release 留在 `HOLD_COVERAGE`，零训练 gradient。

**教训：** label head coverage不足时应 abstain/hold，而不是为了模型完整硬凑负样本。

---

## D. Clean2000 / FIT670 / D8：数据规模变大后，重点变成 leakage 与 event semantics

后续 Clean2000 corpus：2000 episodes，跨四 suite，随后形成 FIT670/Fresh670 development authority。

正式 FIT670 taxonomy：

```text
identities       670
steps            196,483
included TRUE     16,667
included FALSE   163,007
UNKNOWN            8,619
articulated         8,100
RIGHT_CENSORED         90
```

### D.1 Relation sidecar 与 event consolidation

D8 中曾出现很多“fragmented TRUE event”与 UNKNOWN gap。最终规则：

- relation sidecar显式绑定 object/target logical identity；
- bridge只能合并 event identity；
- gap step 仍 UNKNOWN、mask=false、weight=0；
- 不允许用 list position/legacy index 猜 relation；
- right-censored/unsupported必须排除。

G sensitivity：

```text
G=0 events 734
G=1 events 707
G=2 events 687
G=3 events 675  <- frozen candidate
G=5 events 671
```

这段探索的价值是把 event semantics 和 mask 规范化，而不是“G=3本身证明 attack”。

### D.2 Step AUROC 不能替代 scheduler utility

D8以后越来越明确：

- row-level random split 太乐观；
- grouped parent/episode split才合理；
- event weighting可避免长 episode dominance；
- final scheduler必须看 parent-level timing/ranking，而不是只看 step AUROC。

这直接预示了 Stage VII/VIII 的 negative result。

---

## E. True-PGD objective 开发：从“token flip”到“selective attack”

早期在 Tomato/fixed-frame 上连续尝试：

1. target-token CW；
2. log-ratio；
3. arm-preservation；
4. prefix-locked/generated-prefix variants；
5. hard feasible candidate selection。

典型现象：

- TRUE 能推 target token，但 RAND20 有时也能达到同 token/margin；
- log-ratio margin改善，但 arm drift到 2/6；
- target-only objective也有 arm 2/6；
- 后来 development frame上才找到 target + arm 6/6 + 优于 RAND/shuffled 的配置。

### E.1 科学风险

这是多次看单一 frame outcome 后的 method development，不能当 held-out attack claim。

最终 evidence必须：

- fresh population；
- frozen objective；
- same budget TRUE/RAND/shuffled；
- arm preservation；
- direct generated token；
- physical validation。

---

## F. Exact replay / simulator restore：最难的因果基础设施问题之一

项目尝试过直接恢复 MuJoCo/controller state：

- first policy inference 可以 exact；
- 下一 env transition仍可能 divergence；
- 补 controller goal/internal state也没有完全消除 one-step mismatch。

后续 exact action-prefix replay canary 在一个 Goal parent 上证明：

- exact prefix；
- exact action/token；
- post-branch qpos/qvel/sim state可闭合。

但这只是 infrastructure canary。

**未来 paired causal experiment 必须把 exactness 写成 pre-intervention gate。** 任何“restore成功”但 primary window 前已经 divergence 的 branch 都应 structural invalid，而不是 physical negative。

---

## G. Stage V：科学架构正确，但 formal M4 工程 authority 多次失败

Stage V 冻结的正确因果顺序：

```text
clean data
-> privileged clean Teacher
-> causal 25D Student
-> freeze
-> pre-M4 lock
-> held-out M4
```

Teacher/Student freeze 曾封存：670 parents、196,483 records、四 active heads，G7 test read once，M4 outcomes未参与训练。

### G.1 Corridor repeatability HOLD

第一版 current-source final40 clean corridor：

```text
PASS/PASS 32
PASS/CLEAN_FAILURE 3
CLEAN_FAILURE/CLEAN_FAILURE 4
INELIGIBLE/INELIGIBLE 1
```

不能把失败 parent反复 rerun 到过。后来通过 outcome-blind replenishment和 exact55 firewall重新组建 final40。

### G.2 Formal M4 source-plane mismatch

frozen exact plan来自 snapshot selection source，而 formal runtime是另一个 source plane。旧 authority未把这两个 plane原子绑定。

Parent 00：

- 24 probes / 96 branches；
- 96/96 在 primary window 前 runtime exactness fail；
- 0 treatment primary step；
- 0 consumable V_phys。

Parent 01：

- first counterfactual branch前 structural abort；
- 0 branch/label/Vphys；
- conservative `outcomes_read_uncertain=true`保留。

这些是 engineering invalidation，不是 science negative。

### G.3 Queue contract问题

旧 formal runner 是 one-parent CLI，需要外部传 `--parent-index/--gpu`；缺：

- global dispatcher；
- rolling replenishment；
- dynamic GPU admission；
- atomic worker/GPU/PID/source/authority/attempt binding。

因此 successor authority即使 snapshot compatibility pass，仍必须 HOLD到 scheduler contract修好。

### G.4 Official interpreter与GPU provenance

后续又专门补：

- child interpreter必须 exact official Python；
- `free_memory_mib > 20480`；
- one project worker/GPU；
- foreign processes untouched。

**GPU空闲是资源事实，不是科学授权。**

---

## H. Stage VI：第一次明确区分“data gap”与“negative science”

### H.1 初始 root-cause：Teacher/M4 identity intersection=0

Stage V Teacher 670 identities 与当时 formal M4 40 parents 无交集。

因此：

```text
Teacher -> Vphys T5 = 0/858
heldout Student->Teacher = NOT_IDENTIFIABLE
```

当时却有高 development probes：

- dev Student->Teacher physical AUROC≈0.968；
- clean 25D parent probe≈0.929；
- 16-step≈0.905；
- privileged state≈0.875；
- Student M4 physical≈0.536。

正确裁决：**不可识别的数据缺口，不是 Teacher negative。**

### H.2 Fresh Stage VI-B2：真正 held-out negative

fresh 16 parents / 384 probes / 1,536 planned branches：

```text
1,152 treatment labels
1,004 binary consumable
148 abstain/censored
T5 333 consumable + 51 abstain
```

B2-C frozen 16x25D Student：AUROC 0.6246，ECE 0.4606，suite emission严重失衡。

这才是可以写成 scientific negative 的结果。

### H.3 Fresh M4 clean telemetry不完整

fresh M4 clean file一度没有 frozen R3 25D materializer要求的全部 telemetry。项目用了 diagnostic-only clean replay reconstruction：

- 16/16 parents覆盖；
- 不做 policy outcome read；
- 不读 reward/done/info/intervention/label/protected outcome。

重建结果只能用于恢复 frozen clean feature，不能当新的 post-outcome feature engineering。

---

## I. Stage VII：context能提高 overall，但 suite domain shift巨大

Frozen candidates结果：

```text
S7-A AUROC 0.7236
S7-B AUROC 0.8380
S7-C AUROC 0.8259
```

但：

```text
S7-B libero_spatial AUROC ~0.0719, emission ~0.942
S7-C libero_spatial AUROC ~0.0547, emission ~0.986
```

结论：language/policy/visual context确实有整体信息，但没有建立稳定 cross-suite calibration/selectivity。

**教训：** overall AUROC 不能掩盖 suite反向排序或 near-saturated emission。

---

## J. Stage VIII：descriptive relative signal != deployable selector

R0 说明 within-parent相对 timing signal存在。

R1-B 比 R1-A parent-macro AUC提升约 +0.081，但：

```text
R1-B macro AUC 0.6586 < frozen 0.72
Top1 lift ~1.05
Top3 lift ~0.98
zero-regret ~0.556
worst-suite ~0.478
```

suite：Goal≈0.0125，Spatial≈0.0625。

因此不能因为“有 improvement”就 promotion。

---

## K. Stage IX：PGD targetability的高 AUROC 是一个误导性“容易指标”

Stage IX F0：E1 model AUROC≈0.9005，但 factorized parent-macro≈0.521、top1 lift≈0.97。

说明：

> 能预测“哪里PGD目标分数高”不代表能预测“同一个 parent 内哪里最容易产生物理后果”。

这对 paper 很重要：把 `E_t` 与 `V_t` 分开不是理论洁癖，而是数据明确要求。

---

## L. Stage X historical X1 forensic：一个结果可以因为 provenance 失去 promotion 资格

historical X1 121 sequences / 577 frame PGD results，无 env.step。

后续发现：

### L.1 Victim mismatch

Stage IX PGD固定加载：

```text
/mnt/sdc/.../models/libero-10/openvla-7b-finetuned-libero-10
```

而 Stage V/VI snapshots 对 Goal/Object/Spatial都有各自 checkpoint path。

historical X1 population只来自 Goal/Spatial，因此 121/121 path mismatch。

### L.2 Enumeration defect

clean sequence census其实：

```text
length1 1123 starts
length2 100
length3-4 69
length5-9 43
length>=10 9
TOTAL 1344
```

runner却只遍历 length>=3 census里的121 starts，漏掉1,223。

### L.3 “LOSO”命名错误

`mean_loso_auc` 是 suite-wise AUC arithmetic mean，没有 fit-three/test-one。

### L.4 Outcome mapping文字错误

frozen mapping A/B/C里 `STAGE_X_SEQUENTIAL_PGD_SIGNAL_WEAK` 实际对应 B，但 handoff写成 X1=C。

不影响 no-X2 stop，但说明科学 status和文字标签必须机器可校验。

---

## M. X1R victim provenance：历史 identity不能后补

PR #121 对 current suite-matched model dirs做 prospective hashes，但明确：

- historical launch-time exact weights未保存到足以识别；
- historical victim identity=`NOT_IDENTIFIABLE`；
- current directory hashes不能 retroactively替代历史。

8 Q00 clean canary：processor/attention/pixels/action parity都很好，但 generated gripper tokens表面出现 31744 vs reference-derived 31745，由此触发 T0 tokenizer forensic。

---

## N. T0 tokenizer与causal-row forensic：当前接管最关键

### N.1 Native endpoint roundtrip非 bijective

Native tokenizer endpoint可直接给 token 31744；native decoder把 endpoint clip到最后 center；center再encode得到31745。

所以：

```text
31744 -> decoded center -> 31745
```

不是“generated token错误”。

### N.2 Project helper general defect

helper：nearest center。  
native：`np.digitize(action, bins)`。

一般 mismatch：

```text
L10 2927
Goal 3136
Object 3052
Spatial 2843
```

但 8 个 Q00 raw reference rows仍8/8一致。

所以两个结论必须同时保留：

1. helper不能做future authority；
2. general helper defect不自动推翻现有所有Q00 raw reference evidence。

### N.3 Causal LM row

7D：

```text
dim0 -8
dim1 -7
dim2 -6
dim3 -5
dim4 -4
dim5 -3
dim6(gripper) -2
```

`logits[-1]` 预测 action prefix之后的token，不是gripper。

Toy 7/7；actual 168 rows有165 exact。

### N.4 Cached generation vs full forward near-tie

唯一稳定 mismatch：Goal/stage_v/dim3：

```text
AR top1 31918 = 26.375
AR top2 31932 = 26.25
TF top1 31932 = 26.25
TF expected 31918 = 26.0
```

更像 numerical path difference，不像 row shift。

**未来必须先冻结score authority，再看PGD结果。**

### N.5 Best-iterate protocol gap

旧 attacker：记录 `best_restart_metric`，但最后保存 final iterate。

若未来引入 best iterate，属于新 method version；必须冻结 selection metric/tie-break/candidate budget，并给 RAND/shuffled同样预算。

### N.6 Historical direct generation token IDs缺失

旧 artifacts很多是 reference-derived/re-encoded IDs，不能证明当时 generate() 直接输出什么。

标记：

```text
historical_generated_token_ids = NOT_IDENTIFIABLE
```

future runner必须first-class capture raw generated action token sequence。

---

# O. 汇总：16类必须防止复发的问题

1. **Raw/env gripper sign inversion**；
2. **target_action未真正改OPEN**；
3. **global fixed token assumption**；
4. **nearest-center helper代替native tokenizer**；
5. **candidate gate污染label**；
6. **UNKNOWN/censored转negative**；
7. **official SR漏 contact failure**；
8. **sim-state restore假exact**；
9. **snapshot/runtime source-plane mismatch**；
10. **Teacher/M4 identity join缺失**；
11. **fresh telemetry不满足frozen feature contract**；
12. **overall AUROC掩盖worst-suite/emission collapse**；
13. **suite mean冒充LOSO**；
14. **sequence enumeration漏short starts**；
15. **cached AR和full TF数值路径混用**；
16. **best-iterate/candidate selection没有预先标准化**。

另外：official env user-site曾导致启动hang，historical X1用了 `PYTHONNOUSERSITE=1`；这是runtime hygiene而非scientific parameter change。GPU foreign workloads必须 untouched。

---

# P. 历史证据在 paper 中应该如何分层

## Primary / strongest

- Stage X X0 dose/mechanism；
- Stage VI-B2 fresh held-out negative；
- Stage VII/VIII frozen generalization negatives；
- Stage IX model-vs-factorized gap；
- future repaired X1R/X2（若完成）。

## Secondary / bounded mechanism evidence

- Black Bowl State5/7 fixed-window；
- command-OPEN oracle same-window；
- early matched random controls；
- exact action-prefix replay canary。

## Method-development / non-promotional

- repeated Tomato fixed-frame objective search；
- V4 attack formal arm with wrong OPEN semantics；
- historical X1 cross-suite PGD；
- superseded M4 structural attempts；
- any artifact lacking exact source/token/checkpoint provenance。

---

# Q. 对新 GPT 的最终提醒

这个项目已经积累了很多“看起来很强”的局部数字：0.96 development AUROC、0.84 context AUROC、0.90 model targetability、固定窗口 failure等。最危险的做法是把这些数字拼成一个从 detector 到 attack 到 physical failure 的单一成功故事。

更科学、也更有论文价值的事实是：

> **每一层单独都可能有 signal，但跨层 alignment 很弱。**

这正是为什么推荐 paper 以 mechanism/factorization gap 为主线，并把 repaired prospective PGD physical validation当最后一块，而不是继续把 detector调到“看起来过关”。
