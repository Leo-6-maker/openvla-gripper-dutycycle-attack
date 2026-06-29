# Table 1 SOTA Comparison and Execution Plan V1

**Date:** 2026-06-29  
**Project:** OpenVLA Gripper Duty-Cycle Attack  
**Base branch:** `experiments/cross-suite-generalization-v1`  
**Document status:** DESIGN / PREREGISTRATION INPUT — not itself an authorization to launch new formal conditions

---

## 1. Decision

The highest-priority scientific objective is to complete a credible **Table 1** before expanding the project into new detector architectures, cross-suite retraining, defenses, or real-robot experiments.

Table 1 must not be a single attack-success number. It must jointly establish:

1. **Attack effectiveness:** the attack changes task/contact outcomes.
2. **Payload specificity:** the result is not reproduced by matched random noise, shuffled gradients, or generic untargeted PGD.
3. **Action-dimension selectivity:** the perturbation primarily controls the gripper channel rather than generically destroying the arm trajectory.
4. **Timing specificity:** the same targeted payload is much more consequential when triggered by the frozen causal Student than at random or shifted times.
5. **End-to-end validity:** no-emission episodes remain in the intention-to-treat denominator.
6. **Benchmark compatibility:** official LIBERO success/failure is reported, while contact-quality failure is reported separately because official success can miss slip, premature release, or uncontrolled drop.

The intended paper claim is **not** simply “our raw failure rate is the largest.” The stronger and more defensible claim is:

> A clean-only causal trigger can align a short gripper-targeted visual intervention with failure-critical manipulation phases, producing selective gripper control and contact-related failure with substantially less arm-side collateral than generic attacks or incorrect timing.

---

## 2. Claim hierarchy

### 2.1 Already supported

- OpenVLA exhibits a **selective, phase-dependent gripper-channel vulnerability**.
- The strongest frozen mechanism evidence remains the corrected Black Bowl State7 study plus State5 same-task reproduction.
- Moka is only a phase-sensitive partial extension because matched random has nonzero fragility and simulator success can disagree with manual review.
- Phase B establishes that the frozen clean-only causal detector has meaningful held-out timing localization, but with heterogeneous folds and a nontrivial zero-emission rate.

### 2.2 Being validated by formal VIS

- A frozen Student trigger can be connected to an online visual attack on unseen Object tasks.
- Student-triggered targeted VIS produces stronger end-to-end failure than matched random and incorrect timing.
- Targeted VIS preserves arm behavior better than generic action-disruption attacks.

### 2.3 Not yet supported

- Broad cross-suite attack generalization.
- Universal superiority over every published VLA attack.
- Real-robot attack success.
- “Outperforming original TMA” unless the original TMA protocol is reproduced rather than merely adapted.
- Any post-Phase-B detector retuning.

---

## 3. Current frozen detector evidence

Phase B is closed and must not be tuned after held-out results were opened.

Key pooled counts:

- Positive emission coverage: `965 / 1047 = 92.2%`
- K10 containment conditional on emission: `906 / 965 = 93.9%`
- K10 containment, full positive denominator: `906 / 1047 = 86.5%`
- Episode false-trigger rate: `11 / 303 = 3.6%`
- Macro K10 containment: approximately `0.855 ± 0.160`
- Macro no-corridor frame FPR: approximately `0.050 ± 0.081`
- Macro zero-emission rate: approximately `0.277 ± 0.211`
- Median signed timing error: approximately `+2.9 ± 1.5` steps

Known fold-level diagnostics:

- **LOW_COVERAGE:** Folds 05 and 07
- **TIMING_DEVIATION:** Fold 06
- **HIGH_FRAME_FPR:** Fold 03
- **SCORE_SATURATION:** Fold 09, with related saturation also visible in late folds

Interpretation:

- Timing is usually inside the K10 corridor when the detector emits.
- The main end-to-end weakness is no-emission, not frequent negative-episode triggering.
- Formal attack reporting must therefore include both **ITT** and **conditional-on-emission** metrics.

---

## 4. What counts as a fair SOTA comparison

Published methods cannot all be placed in one numerical ranking because threat models differ. Comparisons are split into three levels.

### Level 1 — matched-threat-model comparison

This is the primary scientific comparison. Every method uses the same:

- victim checkpoint;
- task/state manifest;
- detector checkpoint and frozen runtime parameters;
- preprocessing and runner;
- perturbation budget;
- optimization-step budget;
- K10 intervention length;
- termination policy;
- ITT and conditional denominators;
- logging, contact-quality, and statistics pipeline.

Only this level supports statements such as:

> Ours outperforms or matches an adapted baseline under the same online threat model.

### Level 2 — literature-context comparison

Reported paper numbers may be shown in a separate reference panel with an explicit `Directly comparable: No` field.

This level provides field context but must not be used for direct significance claims when attack form, patch persistence, task denominator, or model access differs.

### Level 3 — original-protocol reproduction

A claim such as “outperforms prior SOTA” requires a reproduction close to the original paper protocol. For TMA, this means reproducing the original target definition, patch-generation regime, persistence, task denominator, and reporting conventions rather than only reusing a TMA-like loss inside the current online K10 runner.

---

## 5. Baseline taxonomy

### 5.1 Mandatory simple controls

| Condition | Purpose |
|---|---|
| `CLEAN` | Establish clean outcome and replicate stability. |
| `RAND_LINF` | Test whether arbitrary same-budget noise is sufficient. |
| `SHUFFLED_GRADIENT` | Preserve gradient-like magnitude while breaking objective direction. |
| `RANDOM_TIME` | Keep the targeted payload but remove critical-phase alignment. |
| `EARLY_SHIFT` | Test a structured but incorrect earlier intervention. |
| `FIXED_TIME_PRIOR` | Test whether a simple normalized-time prior is enough. |

### 5.2 Mandatory classical / direct attack baselines

| Method | Role |
|---|---|
| `UMA` or equivalent untargeted CE-PGD | Conventional generic gradient attack. |
| `UADA_DOF1_3` | Action-space discrepancy attack. |
| `UPA_DOF1_3` | Geometry/position-aware attack. |
| `ADAPTED_TMA_OPEN` | Closest targeted-action baseline, aimed at physical OPEN under the current threat model. |
| `PREFIX_LOG_RATIO_OPEN` | Proposed targeted gripper payload. |

Why the family matters:

- UMA tests generic untargeted optimization.
- UADA tests action-space-aware disruption.
- UPA tests geometry-aware disruption.
- TMA tests standard targeted manipulation.
- Prefix Log-Ratio tests the proposed targeted gripper objective.

Comparing only against TMA is insufficient because TMA represents only the targeted branch of the earlier VLA visual-attack family.

### 5.3 Recommended contemporary generic baseline

`ADAPTED_FREEZEVLA` is recommended as a strong generic task-disruption reference, but it should first pass a four-fold engineering canary. It must be named “adapted” unless the official implementation and protocol are reproduced exactly.

### 5.4 Literature context only

These should appear in a capability/threat-model table rather than the matched numerical Table 1:

- AttackVLA as a benchmark/evaluation framework;
- transferable or universal patch attacks;
- partially observable patch methods;
- text jailbreak attacks;
- training-time backdoors and poisoning methods.

---

## 6. TMA must be represented in two distinct forms

### 6.1 Adapted TMA-OPEN

Purpose: fair algorithmic comparison with the proposed objective.

Required matching:

- same Student trigger;
- same K10 window;
- same epsilon and optimization steps;
- same DoF7 physical OPEN target;
- same parent/state/seed manifest;
- same metrics and denominators.

This supports:

> comparison to an adapted TMA baseline under the same online threat model.

### 6.2 Original-protocol TMA

Purpose: external SOTA reproduction.

It should retain the original paper’s:

- patch-generation procedure;
- target-bin convention;
- persistence/exposure model;
- task/trial denominator;
- FR and NAD definitions.

Without this second experiment, do not claim superiority over the original TMA result.

---

## 7. Formal experimental unit

For each formal attack condition:

```text
9 held-out folds
× 2 preregistered states
× 3 detector seeds
× 3 perturbation seeds
= 162 rollouts per condition
```

Formal CLEAN is also executed 162 times, but the analysis must state:

```text
162 CLEAN executions
54 unique CLEAN parent groups
3 clean replicates per parent
```

The 162 clean executions are not 162 fully independent scientific units.

Primary generalization unit:

- fold/task, or
- fold × state cluster.

Episode-level micro totals are secondary.

---

## 8. Proposed Table 1 structure

### Panel A — Matched-threat attack effectiveness and selectivity

| Method | Timing | ITT failure | Conditional failure | Clean S→F conversion | Emit coverage | Official FR | CQFR | CQSR | OPEN TASR | qpos/width response | Arm NAD ↓ | Grip NAD ↑ | Attack duty | Latency |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CLEAN | — | — | — | — | — |  |  |  | — | — | — | — | 0 |  |
| RAND-L∞ | Student |  |  |  |  |  |  |  |  |  |  |  |  |  |
| Shuffled | Student |  |  |  |  |  |  |  |  |  |  |  |  |  |
| UMA | Student |  |  |  |  |  |  |  |  |  |  |  |  |  |
| UADA | Student |  |  |  |  |  |  |  |  |  |  |  |  |  |
| UPA | Student |  |  |  |  |  |  |  |  |  |  |  |  |  |
| Adapted TMA-OPEN | Student |  |  |  |  |  |  |  |  |  |  |  |  |  |
| **Prefix Log-Ratio OPEN** | **Student** |  |  |  |  |  |  |  |  |  |  |  |  |  |

Primary metric recommendation:

- **ITT failure / ITT CQFR** for end-to-end capability.

Mandatory secondary metrics:

- conditional-on-emission failure;
- clean-qualified success-to-failure conversion;
- OPEN command/token control;
- physical qpos/width response;
- arm/gripper normalized discrepancy;
- attack duty cycle and latency.

### Panel B — Timing specificity

Hold payload and budget fixed. Change only timing.

| Objective | Timing | ITT failure | Conditional failure | OPEN TASR | CQFR | Median actual timing error | Attack duty |
|---|---|---:|---:|---:|---:|---:|---:|
| Prefix | Random-Time |  |  |  |  |  |  |
| Prefix | Early-Shift |  |  |  |  |  |  |
| Prefix | Fixed-Time Prior |  |  |  |  |  |  |
| Prefix | Student |  |  |  |  |  |  |
| Prefix | Teacher Oracle |  |  |  |  |  |  |
| TMA | Random-Time |  |  |  |  |  |  |
| TMA | Student |  |  |  |  |  |  |

The TMA Student/Random pair is important. It tests whether timing sensitivity is a general gripper-intervention property rather than an artifact unique to Prefix Log-Ratio.

### Panel C — Execution isolation ablation

Recommended as Table 2 or a compact third panel.

| Objective | No-lock failure | ArmLock failure | Δ failure | Arm NAD | Grip NAD | Arm-lock violations |
|---|---:|---:|---:|---:|---:|---:|
| TMA-OPEN |  |  |  |  |  |  |
| Prefix Log-Ratio |  |  |  |  |  |  |

Arm Lock is a mechanistic isolation ablation, not the primary real-world threat result.

### Panel D — Literature reference

Keep literature numbers separate and explicitly non-comparable unless original protocols are reproduced.

| Method / paper | Attack form | Dataset denominator | Reported metric | Reported result | Directly comparable? |
|---|---|---|---|---:|---|
| Original TMA DoF7 | Persistent/universal patch-style protocol | Original LIBERO protocol | FR / NAD | literature value | No |
| FreezeVLA | Generic freeze attack | Original protocol | ASR | literature value | No |
| Ours | Online K10 Student-triggered VIS | Frozen 9-fold panel | ITT/CQFR/NAD | formal result | — |

---

## 9. Metric definitions to freeze before execution

### 9.1 End-to-end denominators

- **ITT:** all preregistered formal episodes, including no-emission.
- **Conditional:** only valid emitted episodes.
- **Clean-qualified:** only parent groups where the paired formal CLEAN outcome meets the preregistered clean validity rule.

No-emission must never be silently removed from ITT.

### 9.2 Outcome metrics

- Official LIBERO success and failure.
- Contact-Quality Failure Rate, including premature release, slip, detach/drop, unstable transport, or uncontrolled final drop under the frozen rubric.
- Contact-Quality Success Rate: official success and no contact-quality failure.
- SR–CQ mismatch: official success despite contact-quality failure.

### 9.3 Control metrics

- gripper OPEN token/command duty;
- longest sustained OPEN streak;
- target logit/margin where valid;
- qpos and width change;
- command-to-physical-response latency;
- object–EEF detachment evidence;
- arm and gripper NAD/rNAD in the same action space;
- attack frames, duty cycle, wall-clock latency, GPU-hours, and peak VRAM.

### 9.4 Human review

Simulator SR is not sufficient for contact-rich failure claims.

Requirements:

- blinded condition labels;
- frozen rubric;
- at least a stratified double-reviewed subset;
- reviewer agreement or adjudication log;
- video identity and checksum in the result bundle.

---

## 10. Statistical analysis

Primary reporting:

- fold-macro mean;
- fold-clustered 95% confidence interval;
- paired risk difference for matched conditions.

Secondary reporting:

- pooled episode-level micro estimate;
- risk ratio where denominator permits;
- per-fold and per-state results;
- best and worst fold;
- no-emission and invalid counts;
- exact task/state/seed ledger.

Do not treat all 162 executions as independent.

Recommended paired comparisons:

1. Prefix Student vs CLEAN
2. Prefix Student vs RAND Student
3. Prefix Student vs Prefix Random-Time
4. Prefix Student vs Prefix Early-Shift
5. Prefix Student vs Adapted TMA-OPEN Student
6. TMA Student vs TMA Random-Time
7. No-lock vs ArmLock

No universal pass/fail threshold should be added after results are observed. The formal study is confirmatory estimation unless a threshold is preregistered before execution.

---

## 11. Execution order

### Batch A — minimum paper claim

1. Finish and freeze Formal CLEAN.
2. Prefix Log-Ratio + Student Trigger.
3. RAND-L∞ + Student Trigger.
4. Adapted TMA-OPEN + Student Trigger.
5. Prefix Log-Ratio + Random-Time.

This batch answers:

- Does the full system work?
- Is it stronger than arbitrary matched noise?
- Is the proposed objective different from standard targeted CE/TMA?
- Is Student timing necessary?

### Batch B — complete baseline family

6. Shuffled gradient.
7. UMA / untargeted CE-PGD.
8. UADA.
9. UPA.
10. Prefix Early-Shift.
11. Prefix Fixed-Time Prior.
12. TMA Random-Time.
13. Teacher Oracle timing upper bound.

### Batch C — contemporary generic SOTA reference

14. Four-fold adapted FreezeVLA engineering canary.
15. Formal 162-rollout FreezeVLA condition only if the canary passes and the adaptation remains scientifically faithful.

Each condition must be independently audited and frozen before the next condition is interpreted.

---

## 12. Formal freeze bundle per condition

Each completed condition should produce:

```text
MANIFEST.jsonl
MANIFEST.sha256
accepted_job_keys.txt
RESULT_INVENTORY.json
PROVENANCE_AUDIT.json
PAIRING_AUDIT.json
ARTIFACT_SHA256SUMS.txt
CONDITION_RESULTS.json
CONDITION_FREEZE.json
README_RESTORE.txt
```

The analysis script must read only frozen artifacts. Table values must never be copied manually.

Required provenance:

- Git commit and script SHA256;
- victim checkpoint SHA256;
- detector checkpoint SHA256;
- protocol, registry, and manifest SHA256;
- exact runtime parameters;
- GPU assignment and timestamps;
- retry and quarantine ledger;
- video/telemetry checksums.

---

## 13. Success interpretation

### Strong result

- high ITT and conditional failure;
- Prefix Student clearly exceeds RAND and wrong timing;
- gripper control and physical response are strong;
- arm NAD is low relative to generic attacks;
- CQFR supports official FR or exposes meaningful SR–CQ mismatch.

Supported claim:

> A clean-only causal trigger enables selective, phase-dependent gripper attacks on unseen Object tasks.

### Moderate but publishable result

- Prefix raw failure is similar to or slightly below TMA/UADA;
- Prefix has lower arm collateral, higher gripper selectivity, or clearer timing specificity;
- Student significantly exceeds Random-Time.

Supported claim:

> The method is not uniformly stronger in generic failure rate, but provides a more selective and mechanistically interpretable gripper-channel attack.

### Weak result

- Prefix does not beat matched random or wrong timing;
- arm collateral is not reduced;
- qpos/contact evidence is weak;
- detector no-emission dominates ITT.

Required reframing:

> timing detector and vulnerability characterization, not SOTA attack superiority.

---

## 14. GO / HOLD matrix

| Activity | Status | Condition |
|---|---|---|
| CLEAN1500 collection | GO in background | Continue provenance and integrity monitoring. |
| Formal CLEAN completion | GO | Freeze immediately at 162/162 valid terminal outcomes. |
| Existing VIS formal preregistered conditions | GO after condition-specific audit | Use frozen Student and state-selection rules. |
| New baseline implementation | GO for engineering only | No formal result until baseline preregistration is frozen. |
| UADA/UPA/TMA/UMA formal launch | HOLD | Freeze exact objective, target, budget, and manifest first. |
| FreezeVLA formal launch | HOLD | Four-fold engineering canary first. |
| Post-Phase-B detector tuning | PROHIBITED | Phase B is closed. |
| Cross-suite VIS | HOLD | Complete Object Table 1 first. |
| New detector architecture search | HOLD | Do not distract from Table 1 completion. |
| Attack superiority over original SOTA | HOLD | Requires original-protocol reproduction. |

---

## 15. Immediate next actions

1. Audit the current Formal CLEAN live run and freeze it when complete.
2. Finalize `LOTO_TABLE1_BASELINE_PREREGISTRATION_V1.json` before launching any new baseline.
3. Map existing runners/losses to the exact conditions: RAND, shuffled, UMA, UADA, UPA, adapted TMA-OPEN, Prefix.
4. Verify all conditions can emit the same telemetry and contact-quality fields.
5. Run Batch A only.
6. Freeze and analyze Batch A before committing compute to Batch B.
7. Generate Table 1 directly from frozen condition bundles.
8. Keep literature-reported values in a separate non-comparable panel unless exact reproduction is completed.

---

## 16. Approved wording

Recommended:

> We compare against adapted UMA, UADA, UPA, and TMA objectives under an identical online, norm-bounded, detector-triggered threat model. We separately contextualize our results against reported patch-based attacks because their persistence, access assumptions, optimization protocol, and evaluation denominator differ.

Recommended when Prefix does not have the largest raw FR:

> Although the proposed payload is not uniformly stronger in generic task-failure rate, it achieves a more selective gripper-channel intervention with lower arm-side discrepancy and substantially stronger phase dependence.

Forbidden without additional evidence:

- “We outperform the original TMA SOTA.”
- “We achieve broad cross-suite generalization.”
- “The detector is deployment-ready.”
- “Simulator SR alone confirms physical failure.”
- “No-emission episodes were excluded because no attack occurred.”
