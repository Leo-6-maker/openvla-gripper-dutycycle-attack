# Current Experiment Handoff — OpenVLA Gripper Duty-Cycle / Counterfactual Susceptibility

**Snapshot time:** 2026-08-08 08:59 SGT  
**Purpose:** hand off the current scientific story, formal experiment status, code/control-plane state, and next actions to a fresh GPT/Codex session without relying on prior chat memory.

---

## 0. Read this first

This project is no longer best described as “PGD attacks OpenVLA” or as a generic anomaly detector. The current research thesis is:

> **VLA manipulation contains temporally concentrated, physically fragile contact states. We want to define those states counterfactually, learn to identify them online from clean observations, and then show—under matched controls—that timing and targeted gripper payload each causally contribute to physical task failure.**

The key conceptual decomposition is:

- `C_t`: clean critical phase / manipulation criticality.
- `V_t`: **counterfactual physical vulnerability** of a clean state to a short direct OPEN intervention.
- `E_t`: exploitability of that vulnerability by a visual attack.

The old Teacher mostly approximated `C_t`. The redesigned Stage V / R2 program exists to measure `V_t` directly and prevent us from confusing “critical phase” with “physically vulnerable state.”

**Important evidence rule:** code being implemented or a protocol being frozen is not the same as an experiment passing. The fresh session must re-read actual server receipts before claiming that Q2, R2A, R2B, V2, Stage O, Student, direct-OPEN pilot, or VIS matrix has completed.

---

## 1. Repository and branch to use

Repository:

`Leo-6-maker/openvla-gripper-dutycycle-attack`

### Current code line

Use:

`codex/stage-v-r2-source-binding-fix-20260808`

Verified HEAD at handoff time:

`c80eb290a5039db4ac59ccd0641a055868571070`

Commit message:

`Follow appended Stage V plan registry versions`

### Do not resume from the older mainline branch

The earlier branch:

`codex/counterfactual-susceptibility-mainline-20260807`

is **32 commits behind** `c80eb290...` at handoff time. It does not contain the full Q2/source-binding/autonomous plan-chain fixes now relied upon.

### What `c80eb290...` fixes

The local orchestrator now follows the latest append-only `PLAN_REGISTRY_V*.json` version even if it was originally started from an earlier registry path. This is necessary for the receipt-driven controller to append stages over time without the orchestrator silently staying on an obsolete plan set.

---

## 2. Historical mechanism evidence — useful, but not the current formal Stage V proof

Earlier experiments established a real and repeatable **gripper duty-cycle failure mechanism** on OpenVLA/LIBERO:

- Clean state7 succeeds.
- Direct `COMMAND_OPEN_ORACLE` causes drop/task failure, including repeated oracle failures in the original mechanism study.
- VIS temporal PGD and zero-margin same-gate attacks can produce pre-place / pre-release drops.
- Random-direction controls were negative or much weaker.
- Sustained OPEN streak and gripper qpos were substantially larger for successful attack/oracle conditions than random controls.
- Offline guard removal of unsafe OPEN segments eliminated those unsafe open segments.

Representative historical numbers from the state7 study:

- VIS temporal PGD: failures in roughly 4/5 runs in the original mechanism batch.
- Zero-margin same-gate: roughly 3/5 failures.
- Random direction: 0/5 failures in that comparison.
- OPEN streaks: VIS up to ~7, zero-margin up to ~8, random ~0–1, oracle 10.
- Gripper qpos ranges were correspondingly larger for VIS/zero-margin/oracle than random.

These experiments support the existence of a **gripper-channel physical failure mechanism**. They do **not** by themselves prove the final counterfactual opportunity-selector story.

---

## 3. Why the old detector story was insufficient

### 3.1 Old privileged Teacher

The original Teacher was derived from **clean trajectory / privileged state rules** such as grasp, carry, pre-release, safe-release, and instability. It did **not** use VIS outcomes to generate labels, which was good for leakage control.

However, those rules primarily marked clean critical phases. Therefore the old Teacher should be described as:

- `clean critical-phase Teacher`, or
- `attack-window proposal Teacher`

—not as an already established physical-vulnerability Teacher.

### 3.2 Stage 1 Student

The causal 25D Student used the frozen causal feature schema/runtime and trained stably across the formal seed/fold matrix. Step-level AUROC was about `0.95` with low seed variance.

This result means:

> the Student can stably learn the old Teacher labels.

It does **not** mean:

> the old Teacher labels are the physically optimal attack opportunities.

### 3.3 Stage 2 / R2 shadow scheduler

Shadow scheduling exposed the gap between pointwise discrimination and event-level timing utility:

- 734 raw spans.
- 675 events + 59 bridges.
- 402 thresholds.
- 32,160 scheduler candidates.
- No full scheduler candidate passed all gates.
- Best active overlap recall was only about `0.265`.
- OOF fully-negative false-onset was about `0.0902`.
- Median timing delay was around `+9` steps.

There was also a deployment mismatch between the OOF fold-specific scorer and the final full-data single checkpoint. The underlying feature/training blobs matched; the discrepancy came from orchestration/scorer construction, especially tail score drift around the emission threshold.

### 3.4 Stage 3A / A1

A1 completed 60/60 rollouts and passed engineering closure, but failed the intended scientific selectivity/timing gates.

More importantly, later audit showed that **A1 did not actually test the Student as an attack-window trigger**. Attack onset was externally controlled by N4/FEC; the R3 detector was shadow-only.

Therefore A1 supports only:

- engineering matrix closure: PASS;
- post-attack anomaly-style discrimination: FAIL;
- Student as clean-state attack-window selector: **NOT TESTED**.

This reinterpretation is central. Do not cite A1 as evidence that Student-selected timing failed.

---

## 4. The redesign: exact-state counterfactual vulnerability mapping

The redesigned experiment asks, for the **same frozen clean state**:

- What happens under clean/control continuation?
- What happens under `OPEN_T3`?
- What happens under `OPEN_T5`?
- What happens under `OPEN_T10`?

The formal branch protocol uses matched controls and exact restore, with 72 branch rows per parent in the frozen science contract.

### Physical/local vulnerability

Examples include intervention-caused changes in:

- gripper qpos/width opening,
- contact loss,
- slip / height drop,
- object displacement,
- regrasp / instability.

### Task vulnerability

The strong causal task label is:

`clean/control succeeds AND direct-OPEN branch fails`

This is much stronger than merely observing that a trajectory is in a grasp/carry phase.

### Final intended supervision

A Hybrid Teacher can be conceptualized as something like:

`clean critical opportunity ∩ counterfactual OPEN vulnerability ∩ attack-feasible horizon`

The final Student still receives **causal clean observations at inference time**. Counterfactual branch outcomes are privileged **training labels**, not online inputs.

VIS outcomes must not be used to define the Teacher labels.

---

## 5. Old Stage V formal attempt — permanently invalid / aborted

Old formal root:

`/mnt/sdc/dty_user/openvla_attack_outputs/n5/gripper_attack_detector_goal_v2_20260806/STAGE_V_COUNTERFACTUAL_MAP_b300e79b_20260806T005817Z`

Old science source:

- commit `b300e79bb0e6e754a9d384f8ea1b75034bd1d4b4`
- tree `96881b4d53f901870dd53ede39d051c0a4c83e34`

Final status:

`ABORTED_INCOMPLETE`

Reason:

`PARENT_WATCHDOG_TIMEOUT`

Observed closure state at abort:

- planned parents: 40
- `PARENT_RESULT.json`: 35
- clean-success parents: 27
- clean-failure parents: 8
- missing parents: 5
- branch rows physically produced: `35 × 72 = 2520`
- accepted: 0
- audited: 0
- closure receipt: absent
- formal root SHA seal: absent
- scientific validity: 0

Missing parents were:

- `libero_10/task_02/state_48`
- `libero_10/task_03/state_47`
- `libero_10/task_03/state_48`
- `libero_10/task_05/state_47`
- `libero_10/task_05/state_49`

Abort receipt SHA256 previously recorded:

`9e2c8a6959bb2dc4340fe1c69dec433deec18fcd79bbad0f39e96111f5963655`

### Never do this

Do **not**:

- resume the old root;
- fill the missing 5 parents;
- accept the 27 clean-success parents as a partial formal dataset;
- reseal or patch the root;
- infer Stage V2 validity from it.

The old root is useful only as engineering/postmortem evidence.

---

## 6. What the old Stage V failure taught us

### 6.1 Parent eligibility problem

Eight of the 35 completed parents had `clean_success=false`.

For causal task vulnerability, a control continuation that already fails is scientifically ineligible for attributing task failure to OPEN. Therefore the original 40-parent manifest could not reliably yield 40 accepted causal parents even if the watchdog had never fired.

This is a **parent qualification / dataset problem**, not evidence against the vulnerability hypothesis.

### 6.2 Watchdog problem

The old supervisor used root-wide artifact modification time as the decisive inactivity signal. That is not equivalent to simulator liveness or parent progress. A long but healthy branch could therefore be aborted because no new root artifact appeared.

### 6.3 Static layout inefficiency

The old fixed layout left GPUs idle when some static batches finished early. The redesigned line uses a dynamic atomic queue/work-stealing model rather than static per-GPU parent batches.

---

## 7. Q2: the current clean-control qualification protocol

The current code line has moved from the first qualification implementation to a stricter **Q2** protocol.

Relevant files include:

- `scripts/detector_v5/freeze_stage_q2_protocol.py`
- `scripts/detector_v5/run_stage_v_r2_q2_control_qualification.py`
- `scripts/detector_v5/audit_stage_v_r2_control_qualification_v2.py`
- `scripts/detector_v5/run_stage_v_r2_q2_supervisor.py`
- `scripts/detector_v5/run_stage_v_clean_replay_frozen.py`

### Q2 scope

Q2 is clean-control-only.

Forbidden scientifically during Q2:

- OPEN intervention experiments,
- VIS,
- PGD,
- attack rollouts,
- vulnerability labels,
- Eval160 / protected evaluation reads.

Q2 uses the frozen clean-only candidate universe and deterministic salted ordering.

### Frozen Q2 sampling policy

- salt: `STAGE_V_R2_Q2_CONTROL_QUALIFICATION_20260807`
- initial sample: 20 candidates per suite
- expansion: +10 for underfilled suites only
- target: 10 qualified parents per suite
- total formal target: 40 parents
- at most one infrastructure retry when no valid scientific result exists

Suites:

- `libero_10`
- `libero_goal`
- `libero_object`
- `libero_spatial`

### Q2 GPU policy

Q2 is intentionally frozen to seven GPUs:

`0,1,2,3,4,6,7`

GPU5 is not authorized during Q2. The code explicitly reserves GPU5 for the post-Q2 eight-GPU path.

### Q2 qualification semantics

A parent qualifies only when **both A and B clean replays** are engineering-valid and clean-successful with exact canonical parent identity and exact A/B initial-state identity.

Hard engineering requirements include:

- zero process/result exit code,
- clean result schema/status valid,
- snapshot restore valid,
- task identity valid,
- runtime valid,
- finite metrics,
- artifact validation pass,
- exact source commit/tree,
- exact canonical parent,
- initial-state identity present,
- all forbidden boundary counters zero.

Important current semantic correction:

- `terminal_state_sha256` equality is **descriptive only**.
- `remaining_horizon_complete` equality is **descriptive only**.
- a clean-success failure with otherwise valid artifacts is a **scientific clean-repeatability failure**, not an infrastructure error and not a retry trigger.
- initial-state identity mismatch is an engineering-invalid hard failure.

Clean failures are classified explicitly, e.g.:

- `CLEAN_REPEATABILITY_FAIL_A_SUCCESS_B_FAIL`
- `CLEAN_REPEATABILITY_FAIL_A_FAIL_B_SUCCESS`
- `CLEAN_REPEATABILITY_FAIL_BOTH_FAIL`

The independent auditor independently recomputes these decisions and no longer treats scientifically valid clean-repeatability failures as audit errors.

### Q2 outputs expected on a successful run

Key artifacts include:

- `Q2_CONTROL_QUALIFICATION_REPORT.json`
- `Q2_CONTROL_QUALIFICATION_ROWS.jsonl`
- `Q2_CONTROL_QUALIFICATION_INDEPENDENT_AUDIT.json`
- `Q2_PARENT_MANIFEST_A.json`
- `STAGE_V_FORMAL_PARENT_MANIFEST_V1.json`
- supervisor completion / checksums

**At the time of this handoff, GitHub verifies the implementation/protocol state, not the live server completion state. The next session must inspect the actual Q2 root and receipts before saying Q2 passed.**

---

## 8. Source-artifact binding fix

A major recent issue was separating:

1. fresh Q1/Q2 replay artifacts used to qualify parents, from
2. the original clean source-artifact metadata needed by the frozen Stage V science runner.

Current code binds the selected parent identities to a separate `source_clean_parent_manifest` using provenance-only metadata. It does **not** read/reuse the Q1/Q2 replay artifacts as Stage V scientific inputs.

The binding helper requires, for each selected parent:

- canonical identity match;
- a source artifact root;
- associated artifact metadata/checksum fields when available.

The derived science parent manifest records:

- `source_artifact_binding_mode = FROZEN_CANDIDATE_METADATA_ONLY`
- `q1_q2_replay_artifacts_reused = False`

This distinction must be preserved in all downstream reports.

---

## 9. Current dynamic Stage V control plane

Relevant current files include:

- `scripts/detector_v5/run_stage_v_dynamic_dispatcher.py`
- `scripts/detector_v5/run_stage_v_dynamic_worker.py`
- `scripts/detector_v5/run_stage_v_parent_aware_supervisor.py`
- `scripts/detector_v5/audit_stage_v_dynamic_queue.py`
- `scripts/detector_v5/stage_v_dynamic_common.py`

### Dynamic queue model

The formal rerun uses one worker per approved physical GPU and an atomic shared queue. Workers claim parents dynamically and pull the next parent after completion.

Each parent attempt binds:

- canonical parent key,
- manifest-row identity/hash,
- worker PID/PGID,
- GPU,
- attempt index,
- output path,
- source/science provenance,
- validation receipt.

### Parent-aware heartbeat / progress

Workers expose parent-aware status including:

- current parent,
- current branch,
- simulator step,
- branch progress,
- artifact progress,
- child PID/PGID,
- child CPU progress,
- GPU utilization/memory,
- parent/branch start times.

The redesigned watchdog is intended to avoid the old “root mtime did not change, therefore kill the whole run” failure mode.

### Strict science artifact validation

For a Stage V parent to be valid under strict provenance mode, the current validator expects, among other checks:

- exactly one `PARENT_RESULT.json`;
- parent status PASS;
- `clean_success=true`;
- branch count 72;
- exactly one branch JSONL set;
- branch rows all bound to the correct parent;
- only `OPEN_T3`, `OPEN_T5`, `OPEN_T10` arms;
- 24 rows per arm;
- `control_arm = NOOP_T10_REPLAY`;
- exact prefix replay;
- branch runtime/label valid;
- all protected/attack boundary counters zero;
- expected science source commit/tree;
- clean science worktree;
- parent-level SHA seal passes.

### Retry rule

Retries are for pre-scientific transient infrastructure only. A scientifically valid negative result, clean failure, vulnerability-negative outcome, or any partial valid science result must never be retried merely to obtain a favorable result.

---

## 10. GPU5 policy has changed by stage — do not use the old blanket rule

Older planning text treated GPU5 as globally excluded because of the protected process concern.

The current code is more precise:

- **Q2:** seven GPUs only; GPU5 excluded.
- **Post-Q2 formal GPU path:** controller/specs can authorize eight GPUs including GPU5, subject to fresh process/resource preflight and protected-PID constraints.

Current R2A/R2B spec generation explicitly supports `--allow-gpu5` and eight approved GPUs when authorized.

Do not carry forward the obsolete statement “GPU5 is always forbidden.” Instead, read the stage-specific frozen protocol/spec and the current live preflight receipt.

Protected external process identity historically tracked:

`PID 1895889`

Never signal/kill/pause/change affinity of an external protected process. Only terminate process groups owned by the current experiment control plane.

---

## 11. Receipt-driven autonomous plan chain

The current controller/orchestrator architecture is server-local and receipt-driven.

Core files:

- `scripts/monitoring/materialize_stage_v_r2_next_plan.py`
- `scripts/monitoring/run_stage_v_r2_plan_controller.py`
- `scripts/monitoring/run_stage_v_r2_mainline_orchestrator.py`

Registries are append-only and versioned:

`PLAN_REGISTRY_V0001.json`, `PLAN_REGISTRY_V0002.json`, ...

The orchestrator validates SHA-bound plans, process identity, inputs, parent manifest, resource policy, and completion receipts. It launches at most one registered stage and audits it before progressing.

Current planned downstream chain is:

1. `R2A`
2. `R2B_DECISION`
3. optional `R2B` when the support rule says `R2B_REQUIRED`
4. `STAGE_V2`
5. `STAGE_O`
6. `STUDENT_FREEZE`
7. `PILOT_QUALIFICATION`
8. `DIRECT_OPEN_PILOT`
9. conditional `VIS_SMALL_MATRIX`

A `DIRECT_OPEN_PILOT` decision of `NO_GO` ends the GPU pipeline without VIS. Only `GO` permits the VIS small matrix.

Again: this is the **implemented progression logic**, not evidence that all of these stages have already run.

---

## 12. R2A formal counterfactual map

R2A is the fresh formal Stage V map on the Q2-qualified 40-parent set.

Expected formal structure:

- 40 parents total;
- 10 per suite;
- exact frozen parent manifest;
- 72 branch rows per parent;
- total expected branch rows = `2880`;
- T3/T5/T10 direct OPEN arms;
- clean/matched replay controls;
- no VIS/PGD;
- no Eval160;
- exact science source/provenance binding;
- strict independent audit and closure receipt.

Closure should require all 40 parents accepted. Any scientifically invalid parent makes that formal root invalid/incomplete; do not patch a failed formal root in place.

### Why R2B exists

R2B is a pre-registered reserve/extension path if the R2A support rule says the first 40 formal parents do not provide adequate support for the planned downstream inference. The decision must come from the frozen decision procedure, not post-hoc preference.

---

## 13. Stage V2: audit whether the old Teacher is enriched for true vulnerability

Stage V2 is CPU/read-only and can run only after exact Stage V closure.

The conceptual question is:

> Does the old clean critical-phase Teacher actually enrich for counterfactual OPEN vulnerability relative to background/random clean states?

Important statistical design corrections already identified and reflected in the current direction:

- primary unit should be a unique candidate state, not three correlated T3/T5/T10 branch rows treated as independent samples;
- parent-clustered uncertainty/bootstrap should be used;
- specificity must be `TN/(TN+FP)` rather than a negative-row FP share mislabeled as specificity;
- background-positive-rate zero must be handled with preregistered statistics rather than an undefined-ratio auto-fail;
- independent audit should recompute the statistics rather than importing producer formulas as the only implementation;
- VIS/PGD boundaries remain zero.

Interpretation:

- V2 PASS: old Teacher is a useful broad proposal filter and can contribute to a Hybrid Teacher.
- V2 FAIL: old Teacher is mostly a phase detector; retain it as an auxiliary task, but let Stage V vulnerability become the primary supervision.

Both outcomes are scientifically useful.

---

## 14. Stage O: observability study

Stage O asks what information is required to predict counterfactual vulnerability online.

Planned comparison:

- O1: causal 25D
- O2: noncausal 25D upper bound
- O3: privileged clean-state upper bound
- O4: RGB + causal 25D

Parent-grouped splits are required to prevent branch/parent leakage.

Interpretation:

- O1 adequate → proprio/causal Student may be enough.
- O2 ≫ O1 → timing/causality limitation; use temporal hazard/TCN/GRU style model.
- O3 ≫ O1/O2 → 25D observability is fundamentally insufficient.
- O4 ≫ O1 → RGB + proprio is required.

A useful final detector target is vulnerability-centered, with phase/release/instability as auxiliary heads rather than the old Teacher label as the main target.

---

## 15. Final causal validation logic

The final paper needs two separately matched causal bridges.

### 15.1 Timing utility

Hold the direct OPEN payload and budget fixed.

Compare timing selected by:

- Student,
- Random-Time,
- heuristic,
- Teacher / oracle proposal where appropriate.

The key question is whether **Student timing alone** materially increases physical failure/vulnerability engagement.

### 15.2 Payload selectivity

Hold the Student timing fixed.

Compare:

- targeted TRUE VIS,
- matched random perturbation/control,
- optional oracle/untargeted controls when useful.

The key question is whether the **targeted gripper payload** adds causal value beyond simply perturbing the image at the same vulnerable time.

### Desired physical chain

The strongest evidence chain is:

`clean observation → online Student opportunity emission → visual perturbation → policy sustained OPEN → physical gripper opening → contact loss/slip/drop → task failure`

This is the final mechanistic story.

---

## 16. Claim boundary / novelty framing

Do not overclaim any single ingredient as new:

- adversarial attacks on VLA exist;
- targeted VLA attacks/backdoors exist;
- strategically timed attacks exist in RL;
- vulnerability mapping exists in robotics;
- stage/keyframe supervision exists;
- teacher/student detectors are standard methodology.

The strongest contribution is the **combination and causal factorization**:

1. identify a phase-dependent gripper duty-cycle physical failure mechanism;
2. define vulnerability counterfactually at an exact frozen clean state using direct OPEN dose interventions;
3. distinguish `C_t` from `V_t` and later `E_t`;
4. distill privileged counterfactual vulnerability labels into an online causal clean-observation selector;
5. prove timing utility under a matched direct-OPEN payload;
6. prove targeted VIS payload selectivity under frozen timing;
7. trace the failure all the way to physical contact loss/task failure.

A concise paper-level framing is:

> **Not all moments are equally vulnerable: VLA manipulation contains counterfactually fragile contact states where a short gripper-directed intervention causes irreversible failure, and these states can be identified online from clean observations.**

---

## 17. Hard scientific / engineering boundaries

The fresh session should preserve these principles:

- No Eval160 until the final detector/protocol is frozen and the designated final read is justified.
- No protected evaluation leakage.
- No VIS/PGD during Q2 or the direct counterfactual map.
- No Student training from VIS outcomes.
- No branch leakage across train/val/test; split by parent.
- Do not call a root formal-valid without closure receipt, independent audit, counts, provenance, and seals.
- Do not resume an `ABORTED_INCOMPLETE` formal root.
- Do not cherry-pick seeds/thresholds/parents based on scientific outcomes.
- No retry to obtain a favorable scientific label.
- Source/worktree drift is a hard stop for formal stages.
- OOM/Xid/sustained swap/resource violations are hard stops according to the frozen stage policy.
- Only terminate process groups owned by the current run.

---

## 18. What is scientifically established vs not yet established

### Established / historically supported

- A gripper OPEN duty-cycle mechanism can physically cause drops/task failure in OpenVLA/LIBERO.
- Targeted VIS/zero-margin attacks can drive harmful OPEN behavior in selected earlier states.
- Random direction was much weaker/negative in the original mechanism controls.
- The old Student can learn the old clean critical-phase Teacher labels well pointwise.
- Old event-level scheduler performance was substantially weaker than pointwise AUROC.
- A1 did not constitute a valid Student-timing experiment.
- The first Stage V formal attempt is invalid/aborted and exposed both parent-eligibility and watchdog design problems.

### Not yet safe to claim from GitHub code alone

- Q2 has passed on the live server.
- 40/40 fresh R2A parents have closed.
- the old Teacher is enriched for true counterfactual vulnerability.
- causal25D is sufficient to observe vulnerability.
- a newly trained vulnerability Student beats Random-Time.
- TRUE VIS beats matched RAND at frozen Student timing.
- the final end-to-end visual→OPEN→drop chain is formally closed under the redesigned protocol.

Those require actual receipts/results.

---

## 19. First actions for the new GPT/Codex session

Do these in order.

### A. Re-establish repository truth

1. Connect/read GitHub repo `Leo-6-maker/openvla-gripper-dutycycle-attack`.
2. Verify branch `codex/stage-v-r2-source-binding-fix-20260808`.
3. Verify HEAD `c80eb290a5039db4ac59ccd0641a055868571070` or identify a newer descendant if work continued after this handoff.
4. Read this handoff and the current versions of:
   - Q2 protocol/producer/auditor/supervisor;
   - `stage_v_dynamic_common.py`;
   - dynamic worker/dispatcher/parent-aware supervisor;
   - plan materializer/controller/orchestrator.

### B. Establish live server truth before launching anything

Locate current Q2 / R2 state roots and read receipts only. Determine:

- whether Q2 is still running, PASS, FAIL, or aborted;
- exact evaluated/qualified counts by suite;
- active worker/PID/GPU state;
- resource hard-stop state;
- whether `Q2_PARENT_MANIFEST_A.json` and `STAGE_V_FORMAL_PARENT_MANIFEST_V1.json` exist and are independently audited;
- whether C0/R2A plans have been materialized/launched/audited;
- current latest `PLAN_REGISTRY_V*.json`;
- whether any root is `ABORTED_INCOMPLETE`.

Do not infer runtime state from commit history.

### C. Only then choose the next stage

- If Q2 incomplete: continue monitoring the existing healthy Q2 control plane; do not create a second qualification root unless the existing root is formally aborted.
- If Q2 PASS but C0 not audited: finish/verify C0 dynamic control canary.
- If C0 PASS and R2A not launched: verify science snapshot/source-artifact bindings and preflight, then let the receipt-driven controller materialize R2A.
- If R2A running: monitor parent-aware progress and do not interrupt a healthy long branch for lack of root artifact mtime.
- If R2A closed: run the preregistered R2B decision, then optional R2B only if required.
- Only after formal Stage V closure should V2 and Stage O be interpreted.
- Only after Student freeze and direct-OPEN timing GO should VIS small matrix be allowed.

---

## 20. Useful historical constants / identifiers

Old aborted Stage V root:

`/mnt/sdc/dty_user/openvla_attack_outputs/n5/gripper_attack_detector_goal_v2_20260806/STAGE_V_COUNTERFACTUAL_MAP_b300e79b_20260806T005817Z`

Old Stage V science source commit:

`b300e79bb0e6e754a9d384f8ea1b75034bd1d4b4`

Old root status:

`ABORTED_INCOMPLETE`

Historically protected external PID:

`1895889`

Current handoff branch/head:

- branch: `codex/stage-v-r2-source-binding-fix-20260808`
- commit: `c80eb290a5039db4ac59ccd0641a055868571070`

Q2 salt:

`STAGE_V_R2_Q2_CONTROL_QUALIFICATION_20260807`

Q2 approved GPUs:

`0,1,2,3,4,6,7`

---

## 21. One-paragraph handoff summary

The project has moved from an old clean-phase Teacher/Student detector toward a much stronger causal story: first measure **counterfactual physical vulnerability** at exact clean states using matched direct OPEN T3/T5/T10 interventions; then test whether the old Teacher enriches for that vulnerability; then determine what online clean observations make vulnerability observable; then train/freeze a vulnerability-centered Student and separately prove **timing utility** and **targeted payload selectivity**. The first formal Stage V attempt was permanently aborted and scientifically invalid because of both clean-control parent failures and a root-mtime watchdog. The current code line implements a fresh Q2 clean A/B qualification, exact identity/source-artifact binding, dynamic parent-aware GPU execution, append-only receipt-driven stage orchestration, and downstream R2A→R2B-decision→V2→O→Student/pilot→conditional VIS progression. At the start of a new session, treat GitHub as code/protocol truth but re-audit the server receipts to determine the actual live experiment stage before making any new scientific claim or launching anything.
