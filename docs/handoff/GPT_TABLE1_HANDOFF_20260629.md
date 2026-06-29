# GPT Table 1 Continuation Handoff — 2026-06-29

## 0. Read this first

This document is the authoritative conversational handoff for continuing the **OpenVLA Gripper Duty-Cycle Attack** project in a new GPT session.

Repository:

```text
Leo-6-maker/openvla-gripper-dutycycle-attack
```

Primary active branch at handoff:

```text
experiments/cross-suite-generalization-v1
```

Documentation branch containing this handoff and the Table 1 plan:

```text
docs/table1-sota-handoff-20260629
```

The new session should begin with a **read-only audit**. Do not launch experiments, retune the detector, or reinterpret provisional data until repository and server artifacts are reconciled.

---

## 1. Project identity

### Working title

**OpenVLA Inference-Time Gripper Duty-Cycle Attack**

### Scientific question

Can a short, visually induced change in the gripper OPEN/CLOSE duty cycle cause selective physical failure in a VLA manipulation policy, and can the intervention be triggered online using only clean causal robot signals?

### Core pipeline

```text
Layer 1: Offline privileged Teacher
    clean simulation trajectories
    + object / target / EEF privileged state
    → task-critical CLOSE / intervention labels

Layer 2: Deployment-safe causal Student
    clean proprioceptive and action history only
    → online trigger / abstain

Layer 3: Visual attack executor
    frozen trigger
    + K=10 targeted visual perturbation
    → gripper OPEN duty-cycle manipulation
    → physical response / contact failure / task failure
```

Random attack outcomes do **not** define Student labels. Random is a post-detection attack-specificity control.

---

## 2. Main claim hierarchy

### Claim A — established mechanism

OpenVLA has a **selective, phase-dependent gripper-channel vulnerability**. Gripper-targeted visual perturbations can induce localized OPEN behavior during contact-critical phases and cause slip, drop, or premature release while matched arm-targeted/random controls are weaker.

Strongest evidence:

- Black Bowl State7 corrected evidence;
- Black Bowl State5 same-task reproduction.

Moka is only a phase-sensitive partial extension. It is not broad task generalization because matched random has some failures and official simulator success can disagree with manual review.

### Claim B — detector timing generalization

A clean-only causal detector can localize a task-critical gripper intervention opportunity on held-out Object tasks without using privileged state at deployment.

Phase B now provides confirmatory held-out estimates. The claim is supported in timing localization, but must retain caveats for zero-emission and fold heterogeneity.

### Claim C — end-to-end detector-triggered VIS

The frozen Student can trigger a short targeted VIS attack on unseen Object tasks, producing stronger end-to-end effects than matched random or incorrect timing.

This is the current primary experimental line. It is **not closed yet**.

### Long-term claim

Cross-suite clean-only detection and targeted attack transfer across Spatial, Goal, and LIBERO-10.

This is not yet proven. CLEAN1500 is data acquisition for that future line.

---

## 3. Detector architecture and frozen Phase B

### Current detector input

The formal LOTO Student uses 25 engineered causal features derived from proprioception and action history. The exact frozen schema must be read from repository artifacts rather than reconstructed from memory.

Conceptually the features include:

- gripper command, qpos, and opening proxy;
- EEF position and velocity;
- action dxyz and action gripper;
- recent open/close streaks and flip count;
- close onset and time since close;
- EEF speed and vertical motion since close;
- qpos/opening deltas and short-window variances.

No normalized timestep, object pose, target pose, future frame, manual outcome, or attack outcome is allowed in the Student input.

### Global freeze

Historical Global Freeze reference:

```text
LOTO_GLOBAL_FREEZE_V1.json
commit originally audited: 2a6a9c93013a07b924fe05a949b2f73bd51df773
30 checkpoints = 10 folds × 3 seeds
```

Server verification previously reported:

- 30/30 checkpoint files found;
- 30/30 SHA256 matched;
- no duplicate fold-seed or path;
- `test_accessed=false`;
- held-out open event absent before authorization.

### Phase B status

```text
Phase B evaluation      CLOSED
Phase B result freeze   PASS
Post-Phase-B tuning     PROHIBITED
```

Frozen result SHA reported at handoff:

```text
LOTO_PHASE_B_RESULTS_FREEZE_V1.json
SHA256: 89911600e0bdc08e46e21f1ebfa85ab37c1d16fb741d4fe7c52409d7e76cd241
```

### Phase B pooled counts

```text
Positive emission coverage:     965/1047 = 92.2%
K10 | emitted:                  906/965  = 93.9%
K10 containment full denom:     906/1047 = 86.5%
Episode false-trigger rate:      11/303  = 3.6%
```

Macro summaries:

```text
K10 containment:             ~0.855 ± 0.160
No-corridor frame FPR:       ~0.050 ± 0.081
Episode false-trigger rate:  ~0.026 ± 0.052 in seed-level macro summary
Zero-emission rate:          ~0.277 ± 0.211
Median signed error:         ~+2.9 ± 1.5 steps
```

Known failure modes:

```text
Fold 05, 07: LOW_COVERAGE
Fold 06:     TIMING_DEVIATION
Fold 03:     HIGH_FRAME_FPR
Fold 09:     SCORE_SATURATION
```

Interpretation:

- When the detector emits, it usually lands inside K10.
- No-emission is the dominant end-to-end weakness.
- ITT metrics must include no-emission.

---

## 4. VIS validation status

### Preregistration

Reported frozen commit:

```text
LOTO_VIS_HELDOUT_PREREGISTRATION_V1.json
LOTO_VIS_STATE_SELECTION_V1.json
commit: 291f01a
```

State selection must remain frozen. Do not replace difficult states after observing attack outcomes.

### Engineering canary

Four diagnostic folds:

```text
03 cream_cheese
05 bbq_sauce
06 ketchup
09 milk
```

Canary results:

```text
48/48 total episodes completed
24 CLEAN + 24 VIS
24/24 pair completeness
24/24 prefix parity
19/24 emitted and attacked
5/24 no-emission handled correctly under ITT
19/19 executed attacks: token_open_duty=1.0, arm_duty=0.0
0 missing fields
0 NaN/Inf
all epsilon <= 0.02353
```

Canary freeze:

```text
LOTO_VIS_ENGINEERING_CANARY_V1.json
SHA256 prefix: 91f313ef...
OVERALL: PASS
```

Engineering diagnostic outcomes, not formal scientific conclusions:

```text
Fold 03: CLEAN 6/6 success, attacked VIS 0/6 success
Fold 05: CLEAN 6/6 success, only 1/6 emitted/attacked, VIS 5/6 success
Fold 06: CLEAN 6/6 success, attacked VIS 0/6 success
Fold 09: CLEAN 6/6 success, attacked VIS 0/6 success
```

Fold 05 no-attack cases are correct ITT behavior, not engineering failures.

### Formal VIS

At the latest handoff checkpoint:

```text
VIS engineering canary   CLOSED / PASS
Formal 9-fold VIS         GO
Attack evaluation         HOLD until Formal CLEAN baseline is frozen
Cross-suite VIS           HOLD
```

Formal design per condition:

```text
9 folds × 2 states × 3 detector seeds × 3 perturbation seeds
= 162 rollouts per condition
```

Formal CLEAN was reported as in progress. The new session must audit current completion, process status, artifacts, and whether a freeze bundle has already been generated.

---

## 5. Two active experiment lines

### Line A — CLEAN1500

Purpose:

- collect cross-suite clean trajectories for future detector training/evaluation;
- no attack involved.

Design:

```text
3 suites: Spatial / Goal / LIBERO-10
10 tasks per suite
50 states per task
500 per suite
1500 total
```

Each trajectory records:

- causal features;
- privileged Teacher fields;
- target binding;
- task outcome;
- provenance.

Latest integrity checkpoint reported in this conversation:

```text
692 COMPLETE
Spatial 282
Goal    231
L10     179
0 corrupted artifacts
1 collector SHA, matching the accepted erratum
2 protocol byte-level SHAs, documented as canonical + GPU7 variant
```

Live counts will have advanced. Re-audit rather than repeating these values as current.

Known provenance issue:

- collector manifest SHA drift was audited and accepted as a queue-compatibility-only change;
- protocol SHA variant on GPU7 must remain explicitly documented;
- `collector_sha256` is authoritative, while a `collector_commit` field may be misleading if it points to an unrelated server branch HEAD.

CLEAN1500 continues in the background. It must not block Object Table 1 completion.

### Line B — LOTO VIS

Purpose:

- validate the frozen detector and attack executor end to end on held-out Object tasks;
- compare Student-triggered targeted VIS with matched controls and SOTA-style baselines.

This is the current primary line.

---

## 6. Data safety and golden bundles

Reported golden bundle:

```text
/mnt/sdc/dty_user/openvla_project/freeze/loto_phase_b_v1/
```

Reported contents:

```text
LOTO_GLOBAL_FREEZE_V1.json
LOTO_GLOBAL_FREEZE_V1_VERIFY.json
LOTO_TEST_OPEN_EVENT_V1.json
LOTO_PHASE_B_RESULTS_FREEZE_V1.json
LOTO_VIS_STATE_SELECTION_V1.json
LOTO_VIS_ENGINEERING_CANARY_V1.json
SHA256SUMS.txt
INVENTORY.json
README_RESTORE.txt
```

Reported properties:

```text
6 core files
~195 KB
read-only directory permissions
```

Live data roots were intentionally not moved:

```text
CLEAN1500 live:
/mnt/sdc/dty_user/openvla_attack_evidence/sc5_cross_suite_clean1500_v1/

LOTO VIS live:
/mnt/sdc/dty_user/openvla_attack/evidence/sc5_object_privileged_loto_v1/
```

Do not rename, merge, move, or clean live directories while workers are active.

Independent physical backup was still pending at the checkpoint. A same-filesystem `backup/` directory is not an independent backup.

---

## 7. Existing Table 1 artifacts are not the final new Table 1

The branch already contains legacy/earlier Table 1 reports, including:

```text
reports/phase7_table1/final_v3/TABLE1_FINAL_V3_4.md
```

That report contains useful historical evidence such as TMA/Prefix no-lock and ArmLock comparisons, breadth results, rNAD, and CQFR preparation. It must not be silently treated as the final LOTO held-out Table 1 because:

- it belongs to an earlier experiment phase;
- its denominator and state panel differ from the frozen 9-fold LOTO VIS design;
- the current formal study must use the preregistered 162-rollout condition matrix;
- new ITT/conditional and detector no-emission semantics must be explicit.

Use legacy Table 1 results as context, implementation evidence, and sanity checks—not as substitutes for the current formal matrix.

---

## 8. Final Table 1 objective

The new Table 1 must answer four questions:

1. Does targeted VIS cause task/contact failure?
2. Is it stronger than matched random and generic attacks under the same budget?
3. Is it selective to the gripper rather than generic arm destruction?
4. Does Student timing outperform random or shifted timing?

The detailed plan is in:

```text
docs/table1/TABLE1_SOTA_COMPARISON_AND_EXECUTION_PLAN_V1.md
```

### Mandatory matched-threat rows

```text
CLEAN
RAND-Linf
Shuffled gradient
UMA / untargeted CE-PGD
UADA DoF1-3
UPA DoF1-3
Adapted TMA-OPEN
Ours: Prefix Log-Ratio OPEN
```

### Mandatory timing rows

```text
Prefix + Random-Time
Prefix + Early-Shift
Prefix + Fixed-Time Prior
Prefix + Student
Prefix + Teacher Oracle
TMA + Random-Time
TMA + Student
```

### Recommended contemporary generic baseline

```text
Adapted FreezeVLA
```

Run a four-fold engineering canary before any formal 162-run FreezeVLA condition.

### Literature context only

Do not directly rank training-time backdoors, text attacks, universal transfer patches, and online K10 VIS in one ASR column.

---

## 9. Required Table 1 metrics

Primary:

- ITT task/contact failure;
- ITT CQFR;
- clean-qualified success-to-failure conversion.

Secondary:

- conditional-on-emission failure;
- emit coverage;
- official LIBERO FR;
- CQSR and SR–CQ mismatch;
- OPEN token/command TASR and duty;
- sustained OPEN streak;
- gripper qpos/width response;
- command-to-physical-response latency;
- arm and gripper NAD/rNAD;
- attack duty cycle;
- optimization and end-to-end latency;
- GPU-hours and peak VRAM.

Official simulator success is not sufficient to establish physical/contact failure.

---

## 10. Statistical unit and reporting

Formal condition size:

```text
162 executions
```

But scientific independence is clustered by:

```text
fold/task
or fold × state
```

Required reporting:

- fold-macro estimate;
- cluster-aware 95% CI;
- paired risk difference;
- pooled micro estimate as secondary;
- per-fold table;
- best/worst fold;
- no-emission, invalid, retry, and quarantine counts.

Do not treat all 162 runs as independent Bernoulli trials.

Formal CLEAN should be described as:

```text
162 executions
54 unique parent groups
3 clean replicates per parent
```

---

## 11. Strict experimental rules

1. **No post-Phase-B detector tuning.**
2. **No attack outcome in Student labels or Student features.**
3. **No replacing difficult preregistered states after observing results.**
4. **No excluding no-emission from ITT.**
5. **No claiming physical failure from simulator SR alone.**
6. **No manual copying of final table values; generate from frozen artifacts.**
7. **No direct SOTA superiority claim from an adapted loss alone.**
8. **No mixing canary, pilot, exploratory, and confirmatory results.**
9. **No claiming live server status without checking processes, logs, ledgers, and artifacts.**
10. **No moving active output directories during collection.**

---

## 12. Immediate continuation plan

### Step 1 — repository audit

Read:

```text
docs/table1/TABLE1_SOTA_COMPARISON_AND_EXECUTION_PLAN_V1.md
reports/phase7_table1/final_v3/TABLE1_FINAL_V3_4.md
docs/gpu/LOTO_GLOBAL_FREEZE_V1.json
docs/gpu/LOTO_METRIC_SCHEMA_V2.json
reports/CROSS_SUITE_CLEAN1500_CANARY_FREEZE_V1.json
```

Search the active branch for newer Phase B, VIS preregistration, canary, and Formal CLEAN artifacts that may not have existed at the historical Global Freeze commit.

### Step 2 — server read-only audit

Verify:

- current CLEAN1500 COMPLETE / SCHEMA_FAIL / INFRA_FAIL counts;
- active workers and collectors;
- current Formal CLEAN completion;
- duplicate job keys/output directories;
- protocol/registry/collector/checkpoint SHA groups;
- whether condition freeze artifacts already exist;
- available disk and inode capacity.

Do not infer live status from this handoff.

### Step 3 — close Formal CLEAN

Before attacks:

- reach legal terminal status for all 162 executions;
- confirm 54 unique parent groups;
- audit three clean replicates per parent;
- freeze manifest, accepted jobs, provenance, checksums, and parent map.

### Step 4 — freeze baseline preregistration

Start from:

```text
docs/table1/LOTO_TABLE1_BASELINE_PREREGISTRATION_V1_DRAFT.json
```

Fill exact:

- runner and objective script paths;
- Git/SHA256 values;
- attack target semantics;
- epsilon, steps, K, initialization;
- condition manifest;
- metric implementation;
- retry policy;
- freeze directory.

The draft does not authorize experiments.

### Step 5 — run only Batch A

```text
Prefix Student
RAND Student
Adapted TMA-OPEN Student
Prefix Random-Time
```

Formal CLEAN is the fifth reference condition but should already be frozen.

Freeze each condition separately.

### Step 6 — evaluate before expansion

Produce a Batch A audit with:

- ITT and conditional effects;
- paired S→F conversion;
- OPEN/qpos/NAD mechanism evidence;
- per-fold heterogeneity;
- evidence for or against proceeding to UMA/UADA/UPA and FreezeVLA.

---

## 13. First required response from the new GPT

The new session should respond first with:

# Table 1 Continuation Audit V1

Sections:

1. **Verified GitHub facts**
2. **Verified server facts**
3. **Current experiment status matrix**
4. **Mismatch between handoff and live state**
5. **P0 / P1 / P2 risks**
6. **Formal CLEAN closure status**
7. **Baseline implementation inventory**
8. **Minimum next action list**
9. **GO / HOLD verdict**

The first session should remain read-only except for generating audit reports or documentation. It must not launch a formal baseline or modify a frozen artifact.

---

## 14. Expected critical judgment

The assistant is expected to challenge overclaiming.

Examples:

- If Prefix has lower raw FR than TMA but much lower arm NAD and stronger timing specificity, frame it as a selective/mechanistic advantage rather than attack-strength superiority.
- If wrong-time Prefix has similar failure, the timing claim is weakened even if OPEN duty is high.
- If matched random fails frequently, the attack-specificity claim is weakened and the denominator may be unstable.
- If official SR and CQFR disagree, report both; do not choose the more favorable one.
- If no-emission is concentrated in a fold, retain it in ITT and discuss detector limitations.
- If a baseline requires extensive semantic changes to fit K10, label it “adapted” and document the adaptation.

---

## 15. End-state definition

Table 1 is considered complete only when:

```text
Formal CLEAN frozen
Batch A conditions frozen
Mandatory baseline family completed or explicitly justified
Timing panel completed
ITT + conditional + clean-qualified metrics computed
OPEN/qpos/NAD/CQFR metrics frozen
cluster-aware statistics generated
per-fold results published
all table cells generated from immutable artifacts
paper wording matches evidence strength
```

Until then:

```text
TABLE1_PUBLICATION_FREEZE = HOLD
CROSS_SUITE_VIS           = HOLD
POST_PHASE_B_TUNING       = PROHIBITED
```
