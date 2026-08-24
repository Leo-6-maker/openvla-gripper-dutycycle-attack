# Stage Z — Cross-model physical OPEN duty-cycle replication goal

Date: 2026-08-24  
Role: PI scientific authorization / Codex execution contract  
Repository: `Leo-6-maker/openvla-gripper-dutycycle-attack`  
Parent scientific branch: `codex/stage-x-x1r2-q3-arm-isolation-repair-20260820`  
Parent HEAD at authorization: `7f095965df61e8065d5c76fc2be4504bed4d9ab6`

## 0. Purpose and controlling scientific question

Stage Z asks one bounded external-validity question:

> Does the command-level OPEN duty-cycle physical mechanism observed in Stage X0 replicate across distinct VLA policy families under a common official LIBERO environment and a gripper-only final-action intervention, or is the mechanism specific to the original OpenVLA policy stack?

This is **not** a visual-PGD efficacy experiment, not a Student/detector experiment, not protected evaluation, and not real-robot validation.

The scientific north star is architecture-level physical mechanism replication. Stage Z may strengthen the paper by establishing that actuator-semantic OPEN susceptibility is not confined to one OpenVLA implementation, while preserving the paper's separate conclusion that physical susceptibility does not imply visual exploitability.

The existing Stage-X0 claim remains immutable:

`STAGE_X_PHYSICAL_DUTY_CYCLE_MECHANISM_SUPPORTED`

Stage Z does not reinterpret or overwrite X0. It creates a new, prospectively frozen cross-model experiment.

---

# 1. Live pre-authorization state

At the parent HEAD, Stage Z is still in Z0R2 authority recovery.

Current root status:

`HOLD_STAGE_Z_Z0R2_OFT_CHECKPOINT_AUTHORITY_NOT_ESTABLISHED`

Already sealed / acceptable at the parent HEAD:

- frozen shared Stage-Z panel: 36 identities;
- suite denominators: `libero_10=10`, `libero_goal=6`, `libero_object=10`, `libero_spatial=10`;
- four structural-missing Goal cells remain structural missing and must never be replaced:
  - `libero_goal/task_01`
  - `libero_goal/task_04`
  - `libero_goal/task_06`
  - `libero_goal/task_09`
- M0 load-bearing bytes: sealed exact;
- M2 official OpenPI `pi05_libero`: sealed/reverified;
- official common LIBERO source/task/action authority: statically sealed;
- final LIBERO action semantics: 7-D, arm indices `0..5`, gripper index `6`, native OPEN=`-1.0`, CLOSE=`+1.0`;
- M0/M1/M2 static adapter/queue/replan contracts;
- prospective five-arm Stage-Z matrix;
- static preparation status: `STAGE_Z_MULTI_MODEL_RUNNER_PREPARATION_STATIC_PASS`;
- no scientific Stage-Z model/simulator outcome exposure has occurred.

Remaining Z0R2 blocker at authorization:

- M1 official OpenVLA-OFT server-side byte authority is incomplete.
- `libero_spatial` has already been sealed server-side.
- `libero_10`, `libero_goal`, and `libero_object` still require provenance-safe server materialization / exact byte sealing.

All pre-existing protected boundaries remain unchanged:

- Eval160 = `UNREAD`
- protected evaluation = `UNREAD`
- BRIDGE/F1 protected path = unopened

---

# 2. This authorization and relationship to prior STOP gates

This document creates a new Stage-Z continuation lane and **conditionally supersedes the old Z0R2 `STOP_FOR_PI` requirement only for this new continuation PR**.

It does **not** retroactively change any historical gate in PR #135.

Codex may autonomously advance through the following state machine only when each preceding gate is machine- and provenance-verified PASS:

1. `STAGE_Z_Z0R2_THREE_MODEL_AUTHORITY_CLOSURE_PASS`
2. `STAGE_Z_Z1_EXECUTABLE_CANARY_PASS`
3. `STAGE_Z_Z2_CLEAN_REFERENCE_AND_ANCHOR_PASS`
4. `STAGE_Z_Z3_PHYSICAL_MATRIX_COMPLETE`
5. `STAGE_Z_Z4_CROSS_MODEL_SYNTHESIS_READY_FOR_PI_REVIEW`

A gate PASS is **not** permission to weaken any frozen contract. Unknown, mismatch, incomplete authority, or unresolvable ambiguity must fail closed.

Codex must STOP immediately for PI review if any early-stop condition in Section 18 is triggered.

No automatic merge is authorized. Final terminal state is always `STOP_FOR_PI`.

---

# 3. Immutable model panel

The Stage-Z model panel is exactly three families.

## Z-M0 — suite-matched OpenVLA anchor

Use the already sealed suite-matched OpenVLA authority inherited from the Stage-X/Q3R2 runtime authority.

Rules:

- no alternate OpenVLA checkpoint;
- no checkpoint selection by outcome;
- no LoRA/fine-tuning/retraining;
- final intervention occurs only after the authoritative final LIBERO 7-D action exists;
- fresh policy decision semantics must be explicit.

## Z-M1 — official suite-specific OpenVLA-OFT

Official source authority:

- commit `e4287e94541f459edc4feabc4e181f537cd569a8`
- tree `0ae110ee28943b9e46feffad84429d2d6e026a32`

Frozen suite checkpoints:

- `libero_10`: `moojink/openvla-7b-oft-finetuned-libero-10`, rev `95220f9a3421a7ff12d4218e73d09ade830fa9a3`
- `libero_goal`: `moojink/openvla-7b-oft-finetuned-libero-goal`, rev `c2d0f9fbbd82674683b397ff923168a12f6a307b`
- `libero_object`: `moojink/openvla-7b-oft-finetuned-libero-object`, rev `4c89574e1c538b6c102f43f0526d60a9d3650148`
- `libero_spatial`: `moojink/openvla-7b-oft-finetuned-libero-spatial`, rev `6d0231af0e48c5985f1ff86908f4674b84bc049b`

Runtime boundary:

- `NUM_ACTIONS_CHUNK=8`
- `num_open_loop_steps=8`
- scientific branch starts only at a fresh official action-queue boundary;
- residual queued actions at branch start are forbidden.

## Z-M2 — official OpenPI pi0.5-LIBERO

Official source authority:

- commit `15a9616a00943ada6c20a0f158e3adb39df2ccac`
- tree `a7f18af2745255b5fa98c86d6031f858bf73d1be`
- config `pi05_libero`
- frozen checkpoint `/llm_jzm/mt/models/openpi-assets/checkpoints/pi05_libero`
- checkpoint-manifest SHA256 `d9104dfdea46eca2fadf05ec7fc478b19d39b19aa8dfda0e6adedcd6d6b6efac`

Runtime boundary:

- `replan_steps=5`
- `action_horizon=10`
- scientific branch starts only at a fresh official replan boundary;
- residual action chunk at branch start is forbidden.

## Absolute model-panel prohibitions

Do not add:

- a fourth model;
- pi0 instead of pi0.5;
- another OpenPI checkpoint;
- another OFT checkpoint;
- Octo / RT-2 variants;
- fine-tuned or newly selected checkpoints;
- model variants chosen after seeing Stage-Z outcomes.

---

# 4. Immutable environment and final-action contract

All three families must execute against the one sealed official common LIBERO authority from Z0R2.

Do not use the modified `/mnt/sdc/dty_user/pi0_openpi` fork as common environment authority.

The final environment action contract is fixed:

- action dimension = 7;
- arm = indices `[0,1,2,3,4,5]`;
- gripper = index `6`;
- native OPEN = exactly `-1.0`;
- native CLOSE = `+1.0`;
- the policy/model first produces the authoritative final LIBERO action;
- intervention modifies only index `6`;
- arm values must be copied exactly according to the frozen branch-isolation semantics;
- no decode/re-encode loop;
- no token-level target substitution;
- no actuator fallback;
- no sign guessing;
- no silent normalization drift;
- no silent fallback to a different model path.

Historical visual-token target `31745` and historical `target_action[6]=+1.0` belong to another representation layer and are **not Stage-Z action semantics**.

---

# 5. Frozen scientific population

The primary Stage-Z population is the already sealed 36 shared identities.

Per model family, the intent-to-evaluate population remains 36, giving 108 model-parent clean references maximum.

The four structural-missing Goal task cells remain structural missing globally and may not be top-upped.

No identity may be:

- replaced because of clean failure;
- replaced because no critical anchor exists;
- replaced because a runtime error occurs;
- replaced because a physical arm is censored;
- selected/ranked by model outcome;
- selected/ranked by V_phys;
- selected/ranked by task success;
- selected/ranked by manual video review.

Model-specific attrition must remain visible.

Required model-parent status classes include at least:

- `CLEAN_REFERENCE_VALID`
- `CLEAN_REFERENCE_FAILURE`
- `RUNTIME_INVALID_PRE_BRANCH`
- `NO_CRITICAL_ANCHOR`
- `NO_NONCRITICAL_ANCHOR`
- `READY_FOR_BRANCH_MATRIX`
- `BRANCH_RUNTIME_INVALID`
- `BRANCH_CENSORED`

These statuses are not interchangeable with V_phys negatives.

---

# 6. Frozen physical arms

The Stage-Z scientific matrix has exactly five named arms:

1. `CLEAN_BRANCH_CRITICAL`
2. `COMMAND_OPEN_T3_CRITICAL`
3. `COMMAND_OPEN_T5_CRITICAL`
4. `COMMAND_OPEN_T10_CRITICAL`
5. `COMMAND_OPEN_T5_NONCRITICAL_CONTROL`

Frozen dose set:

- T3 = 3 steps
- T5 = 5 steps
- T10 = 10 steps

Frozen physical horizon:

- `H_phys = 10`

Do not add/remove doses, add a sixth scientific arm, or change the noncritical control after outcome exposure.

The noncritical T5 arm is a timing-specificity control. Because the frozen protocol contains no separate `CLEAN_BRANCH_NONCRITICAL` arm, do not later reinterpret it as a fully matched noncritical causal contrast.

---

# 7. Critical requirement before any real Stage-Z science: recover exact X0 arm-isolation semantics

The key Stage-Z estimand is gripper-only physical intervention. Before Z1 canary execution, Codex must explicitly recover from sealed Stage-X0 source/artifacts the exact operational rule used to ensure arm motion was not changed by the OPEN intervention.

Produce:

`reports/STAGE_Z_Z1_X0_ARM_ISOLATION_SEMANTICS_V1.json`

It must state, with source/blob/path evidence:

- how the clean arm-action sequence was obtained;
- whether branch arms replayed clean-reference arm actions or re-queried policy actions after intervention-induced state divergence;
- how relative-step alignment was handled;
- how multi-step OPEN duration was applied;
- how T10 crossed any policy/chunk boundary in X0;
- what state, RNG, observation, and action bytes were required for an exact branch;
- how exact arm preservation was checked;
- how clean branch and intervention branches were compared.

Stage Z must use the **same scientific isolation principle** across all three model families. Family-specific queue/replan machinery may differ, but the estimand may not silently become "gripper override plus altered arm policy feedback" for one family and "gripper-only with frozen arm schedule" for another.

If the X0 isolation semantics are `NOT_IDENTIFIABLE`, or if a common implementation cannot preserve the same estimand across M0/M1/M2, STOP with:

`HOLD_STAGE_Z_Z1_ARM_ISOLATION_ESTIMAND_NOT_COMMON`

Do not improvise T10 behavior at runtime.

---

# 8. Z0R2-R3 — finish three-model authority closure

This is the first executable goal after server connectivity is restored.

## 8.1 Remaining M1 server byte seals

Complete, sequentially and provenance-safely:

1. `libero_10`
2. `libero_goal`
3. `libero_object`

`libero_spatial` is already sealed and must not be rewritten unless an actual integrity mismatch is discovered.

For each remaining suite:

- use the exact frozen repo/revision;
- materialize on durable server storage;
- verify exactly 25 files;
- verify total bytes against the already sealed local/API manifest;
- verify every SHA256;
- verify HF LFS object identity for LFS-backed files;
- verify ordinary Git blob authority for non-LFS files;
- record exact server path;
- record materialization time and command provenance;
- write an append-only server receipt;
- only after successful receipt may a newly created temporary Stage-Z cache copy be deleted if space is required;
- never delete historical/scientific evidence for space.

Use one transfer/materialization connection at a time. Do not create concurrency solely for speed if it risks partial files or ambiguous provenance.

## 8.2 Rebuild full closure against the continuation branch

After all four M1 suite manifests are sealed, rebuild:

- `reports/STAGE_Z_Z0R2_M1_OFT_CHECKPOINT_MANIFESTS_V1.json`
- `reports/STAGE_Z_Z0R2_M1_MATERIALIZATION_LEDGER_V1.json`
- `reports/STAGE_Z_Z0R2_MODEL_AUTHORITY_MAP_V1.json`
- `reports/STAGE_Z_Z0R2_ENVIRONMENT_ACTION_BRANCH_PARITY_V1.json`
- `reports/STAGE_Z_Z0R2_STORAGE_PREFLIGHT_V1.json`
- `reports/STAGE_Z_Z0R2_ARTIFACT_MANIFEST_V1.json`
- `reports/STAGE_Z_Z0R2_ROOT_SEAL_V1.json`
- SHA256 sidecars where the existing Z0R2 convention requires them.

The new root must bind the then-current continuation-branch source HEAD/tree. Do not leave the closure root bound only to the earlier `b16f1df...` source commit.

## 8.3 Z0R2 PASS criteria

All eight must pass:

1. frozen 36 + four structural-missing population exact;
2. M0 load-bearing byte authority exact;
3. all four official OFT checkpoint byte manifests established server-side;
4. existing pi0.5 source/checkpoint authority exact;
5. one common official LIBERO authority exact;
6. final 7-D action + native OPEN=`-1.0` exact;
7. M0 fresh-step, M1 fresh-queue, M2 fresh-replan contracts sealed;
8. all new Z0R2 scientific/protected execution counters remain zero.

Required transition label:

`STAGE_Z_Z0R2_THREE_MODEL_AUTHORITY_CLOSURE_PASS`

Only then may this new authorization enter Z1.

---

# 9. Z1 — excluded engineering executable qualification

Z1 is an engineering runtime qualification stage. It is not part of the 36-parent scientific denominator and must never be reported as Stage-Z science.

## 9.1 Runtime package separation

Keep `src/stage_z_preparation/` as the already audited synthetic/static contract surface.

Create a separate runtime namespace, e.g.:

`src/stage_z_runtime/`

and separate scripts/configs, e.g.:

- `scripts/stage_z/run_z1_runtime_canary.py`
- `scripts/stage_z/run_z2_clean_reference.py`
- `scripts/stage_z/run_z3_physical_matrix.py`
- `scripts/stage_z/analyze_z4_cross_model.py`

Do not weaken the static-preparation audit merely to allow runtime imports.

## 9.2 Engineering canary population

Create a deterministic, outcome-blind, permanently excluded canary set outside:

- the frozen scientific 36;
- Eval160;
- all protected evaluation populations.

Select canaries by fixed identity-only salt before any canary outcome is read.

Target matrix: one excluded canary per `model family × suite` = 12 canaries maximum.

If an appropriate excluded identity does not exist for a cell, record `NO_ENGINEERING_CANARY` and STOP before scientific execution; do not borrow a scientific 36-parent identity merely to make Z1 pass.

## 9.3 Z1 runtime checks

For every authorized model-suite canary, verify at runtime:

- exact source/checkpoint authority binding;
- exact common LIBERO authority binding;
- correct task/BDDL/init-state identity;
- clean model call works;
- authoritative final action is exactly 7-D;
- final action is finite;
- native gripper semantics are confirmed at the env-action boundary;
- model-specific decision-boundary contract is satisfied;
- M1 has no residual queue at branch start;
- M2 has no residual chunk / nonfresh replan at branch start;
- branch-state serialize/restore reproduces state identity/hash;
- clean action arm indices are preserved exactly by intervention helper;
- intervention changes only action index 6;
- forced gripper value equals exactly `-1.0`;
- no token decode/re-encode occurs after final action exposure;
- protected firewall rejects Eval160/protected paths;
- runtime receipt records model/load/inference/env-step counters at actual boundaries.

A minimal excluded `ENGINEERING_OPEN_T1_CANARY` may be used only to prove the final action intervention reaches the simulator as expected. T1 is never a Stage-Z scientific dose and must never enter Z2/Z3/Z4 analysis.

Do not read V_phys for Z1. Do not use engineering task outcome to tune code.

## 9.4 Z1 no-retry policy

No retry-to-pass after any canary has produced a simulator `env.step` under the tested runtime path.

A pre-exposure configuration failure may be repaired only if the receipt proves zero model/simulator exposure for that canary attempt. Once exposure occurs, the canary result is immutable engineering evidence.

## 9.5 Z1 PASS

All available required model-suite cells must establish the runtime contract with no unresolved action/queue/replan/state-restore mismatch.

Required artifacts include:

- runtime source authority manifest;
- dependency/environment manifest;
- canary population manifest;
- per-canary receipts;
- action-isolation audit;
- queue/replan runtime audit;
- protected-firewall audit;
- Z1 root seal.

Required label:

`STAGE_Z_Z1_EXECUTABLE_CANARY_PASS`

If any model family cannot satisfy the same gripper-only physical estimand, STOP. Do not silently drop that family and call the remaining panel cross-model replication.

---

# 10. Z2 — scientific clean references and prospective anchors

Only after Z1 PASS.

Z2 creates one clean trajectory per `model × frozen parent`.

Maximum planned clean trajectories:

- 36 parents × 3 models = 108.

Each model uses **its own clean trajectory**. Do not reuse OpenVLA timing/trajectory anchors across M1 or M2.

## 10.1 Clean trajectory execution

For each model-parent:

- use exact sealed model source/checkpoint;
- use exact common LIBERO task/init-state authority;
- use the model's official action-decoding/normalization path;
- record clean policy decisions/actions and model-specific queue/replan boundaries;
- record clean simulator telemetry required by the frozen X0-compatible privileged physical semantics;
- record official task success only as a secondary clean-reference field;
- save branchable state snapshots only at legally fresh model decision boundaries;
- record state hash, observation hash, clean-prefix hash, RNG/simulator state as required for exact restore;
- record no Stage-Z physical intervention in Z2.

## 10.2 Anchor semantics

Do **not** use Student/detector scores or emits.

Recover and reuse the exact X0-compatible privileged physical critical/noncritical semantics from sealed source. The anchor classifier must use only contemporaneous clean-trajectory physical state and prospective rules.

Forbidden anchor-selection inputs include:

- V_phys;
- future intervention outcomes;
- official future task outcome;
- contact-loss outcome after intervention;
- object displacement after intervention;
- manual video labels;
- Student output;
- detector output;
- any score chosen after Stage-Z outcomes are seen.

If exact X0-compatible anchor semantics cannot be recovered without post-hoc invention, STOP with:

`HOLD_STAGE_Z_Z2_ANCHOR_SEMANTICS_NOT_IDENTIFIABLE`

## 10.3 Deterministic selection

Use the already frozen salts:

- critical: `STAGE_Z_Z2_CRITICAL_ANCHOR_V1_20260823`
- noncritical: `STAGE_Z_Z2_NONCRITICAL_CONTROL_V1_20260823`

For each eligible clean model-parent, rank legal candidate steps by the frozen identity/model/step hash rule and select exactly one critical and one noncritical anchor.

If no candidate exists:

- abstain;
- do not replace the parent;
- do not search another task/state;
- do not loosen the physical rule.

## 10.4 Z2 denominator discipline

The fixed intent-to-evaluate denominator remains 36 per model.

Report separately:

- 36 planned identities;
- valid clean references;
- clean failures;
- runtime invalids;
- critical-anchor availability;
- noncritical-anchor availability;
- full five-arm matrix eligibility.

Do not report only the conditional eligible subset without also showing the 36-parent denominator.

Clean failure / no-anchor / runtime-invalid statuses are not V_phys negatives.

## 10.5 Freeze before Z3

Before any Stage-Z intervention outcome is read, seal:

- all 108 planned model-parent identities and statuses;
- all clean trajectory manifests;
- selected critical/noncritical anchor steps or explicit abstentions;
- branch snapshot hashes;
- clean relative arm-action schedule / isolation data required by the X0-compatible branch estimand;
- model queue/replan state at each selected anchor;
- fixed Z3 arm order;
- no-replacement rule;
- analysis schema.

Required label:

`STAGE_Z_Z2_CLEAN_REFERENCE_AND_ANCHOR_PASS`

---

# 11. Z3 — five-arm scientific physical matrix

Only after Z2 PASS.

## 11.1 Branch isolation

For critical arms, all branches for a model-parent must start from the exact same sealed critical branch snapshot.

For the noncritical T5 control, start from the exact sealed noncritical snapshot selected in Z2.

Across critical branches:

- prebranch simulator state is identical;
- clean-prefix identity is identical;
- arm-action schedule follows the common X0-compatible isolation rule;
- the only intended intervention difference is gripper index 6 for the declared OPEN duration.

Before each arm:

- restore and verify branch snapshot hash;
- verify model/action authority;
- verify relative-step alignment;
- verify no residual queue/chunk contamination at the branch decision boundary;
- verify the requested arm is the next frozen arm in order.

## 11.2 Fixed arm order

Use one prospectively fixed order for every eligible model-parent:

1. `CLEAN_BRANCH_CRITICAL`
2. `COMMAND_OPEN_T3_CRITICAL`
3. `COMMAND_OPEN_T5_CRITICAL`
4. `COMMAND_OPEN_T10_CRITICAL`
5. `COMMAND_OPEN_T5_NONCRITICAL_CONTROL`

Do not reorder after seeing any arm result.

## 11.3 Dose execution

For intervention arms:

- requested duration must equal exactly 3, 5, 10, or 5 according to the arm;
- each intervention step must preserve the frozen arm-action values exactly according to the sealed isolation schedule;
- gripper index 6 must equal exactly `-1.0` on every requested OPEN step;
- outside the requested OPEN dose, do not extend OPEN because a failure did not occur;
- no adaptive early stop based on V_phys;
- no retry until a positive occurs.

T10 handling across M1 queue and M2 replan/horizon semantics must already have been resolved and audited in Z1/Z2 under the common isolation estimand. Z3 may not invent a new cross-boundary behavior.

## 11.4 Runtime failure / resume rule

Every model-parent-arm is once-only after first simulator exposure.

If an arm has executed any `env.step`, never rerun that arm to obtain a cleaner result.

If a job stops after some arms:

- completed arms remain immutable;
- only never-started arms may continue later;
- continuation must restore the sealed branch snapshot and respect the same fixed arm order among remaining arms;
- runtime-invalid/censored arms remain visible;
- no replacement parent.

Do not silently discard incomplete model-parents from denominators.

## 11.5 Required physical telemetry

For each branch collect, at minimum:

- requested OPEN duration;
- commanded OPEN steps/fraction/duty;
- executed OPEN duty;
- final 7-D actions;
- exact arm-preservation check;
- gripper qpos / width / aperture;
- aperture excess relative to the clean aligned reference where defined;
- contact state / contact loss timing;
- object displacement;
- branch-state identity and relative-step alignment;
- model authority and action authority;
- runtime-valid / censored status;
- official task success as secondary only;
- V_phys only under the frozen Stage-X0-compatible operational definition.

Do not reconstruct the historically unavailable downstream task-failure taxonomy.

Do not call descriptive telemetry a formal causal mediation analysis.

## 11.6 Qualitative video / paper-figure support

Stage Z should also improve reviewer-facing evidence without cherry-picking.

Before outcome inspection, deterministically select a small visualization subset by identity-only salt, preferably one eligible parent per `model × suite` where available.

For this preselected subset, save synchronized videos / frame sequences for the clean/T3/T5/T10 critical arms and the T5 noncritical control.

These qualitative assets are secondary illustration only. They may not redefine V_phys or select successful examples after outcome exposure.

If storage permits, retaining compact videos for all scientific branches is preferred, but storage pressure must never justify deleting historical sealed evidence.

## 11.7 Manual video audit

After automated outcomes are sealed, perform a pre-specified manual audit on the deterministic visualization subset (or a prospectively frozen larger subset).

Audit only observable physical behavior such as:

- object retained/lost;
- aperture opening visibly consistent with telemetry;
- contact-quality degradation;
- gross object displacement;
- simulator/runtime anomaly.

Manual review is secondary and must not overwrite automated V_phys labels.

---

# 12. Z4 — analysis and cross-model synthesis

Z4 is analysis only. No new scientific rollout may be launched to improve a result.

## 12.1 Primary units

Do not treat steps, candidate rows, or telemetry samples as iid scientific units.

Primary reporting unit is the model-parent / parent-level branch experiment.

The shared identity panel supports paired/descriptive comparison across models only where the corresponding model-parent is valid; all model-specific attrition remains explicit.

## 12.2 Required per-model summaries

For each of M0, M1, M2 report:

- planned denominator = 36;
- clean-reference validity/attrition;
- critical/noncritical anchor availability;
- five-arm completion / censoring;
- consumable V_phys counts for T3/T5/T10;
- raw positive rates for T3/T5/T10;
- complete three-dose pattern counts (`000`, `001`, `010`, `011`, `100`, `101`, `110`, `111`);
- nonmonotone-pattern count, without hiding it if nonzero;
- T10-T3 difference with parent/model-parent bootstrap uncertainty;
- commanded OPEN duty by dose;
- aperture excess by dose;
- contact-loss incidence/timing by dose;
- object displacement by dose;
- official task success as secondary;
- noncritical T5 control behavior separately.

Use parent/model-parent bootstrap, not iid-row confidence intervals.

Do not pool all models into one iid-row p-value.

## 12.3 Cross-model claim categories

Use bounded wording.

Possible terminal scientific classifications:

### `STAGE_Z_CROSS_MODEL_PHYSICAL_MECHANISM_REPLICATION_SUPPORTED`

Use only if all three model families show a coherent dose-dependent command-OPEN physical susceptibility pattern under the frozen common estimand, with exact command delivery and mechanism-consistent downstream telemetry. This does not imply identical effect sizes.

### `STAGE_Z_CROSS_MODEL_PHYSICAL_MECHANISM_PARTIALLY_SUPPORTED`

Use if the mechanism is supported in more than one family but one family is non-identifiable, runtime-limited, or directionally inconsistent. State exactly which family and why.

### `STAGE_Z_CROSS_MODEL_PHYSICAL_MECHANISM_NOT_ESTABLISHED`

Use if evidence does not establish replication beyond the original anchor family.

Do **not** convert `NOT_ESTABLISHED` into `DISPROVED`.

Do not claim universal VLA vulnerability, real-robot vulnerability, visual attack efficacy, causal mediation, or defense effectiveness.

## 12.4 No outcome-driven rescue

After Z3 outcomes exist, forbidden actions include:

- changing model panel;
- changing 36-parent panel;
- changing structural-missing cells;
- adding parents;
- changing doses;
- changing H_phys;
- changing OPEN sign/value;
- adding/removing an arm;
- changing anchor salts;
- changing criticality semantics;
- changing V_phys definition;
- selecting a new checkpoint;
- rerunning negatives;
- excluding runtime-valid negatives;
- choosing qualitative examples by success.

---

# 13. Recommended Stage-Z outputs

Use append-only, machine-readable artifacts. Exact filenames may be versioned, but the following logical outputs are required.

## Z0R2 closure

- M1 suite server receipts/manifests;
- rebuilt model authority map;
- rebuilt environment/action parity map;
- rebuilt storage audit;
- rebuilt root seal + sidecar.

## Z1

- `STAGE_Z_Z1_RUNTIME_AUTHORITY_V1.json`
- `STAGE_Z_Z1_ENGINEERING_CANARY_PANEL_V1.json`
- `STAGE_Z_Z1_X0_ARM_ISOLATION_SEMANTICS_V1.json`
- `STAGE_Z_Z1_QUEUE_REPLAN_RUNTIME_AUDIT_V1.json`
- `STAGE_Z_Z1_ACTION_ISOLATION_AUDIT_V1.json`
- per-canary receipts;
- `STAGE_Z_Z1_ROOT_SEAL_V1.json`

## Z2

- clean-reference manifest;
- per-model clean trajectory receipts;
- anchor candidate/selection manifest;
- abstention/attrition ledger;
- branch snapshot manifest;
- frozen relative arm-action/isolation schedule manifest;
- Z2 root seal.

## Z3

- branch-matrix manifest;
- per-model-parent-arm receipts;
- physical telemetry rows;
- branch videos / deterministic visualization-subset manifest;
- runtime-invalid/censor ledger;
- Z3 root seal.

## Z4

- per-model summary JSON/CSV;
- complete-dose pattern table;
- parent-bootstrap output with frozen seed recorded;
- telemetry dose summaries;
- noncritical-control summary;
- cross-model synthesis JSON/MD;
- figure-ready deterministic export package;
- final Stage-Z root seal.

---

# 14. Deterministic paper-support outputs

Stage Z is motivated partly by the current paper's external-validity limitation. If Z4 reaches a scientific terminal classification, prepare a deterministic paper delta package, but do not edit the already merged paper main branch automatically.

Produce an export package suitable for later Paper V2.1 integration containing:

- model-panel authority table;
- per-model T3/T5/T10 dose-response table;
- per-model attrition/denominator table;
- monotonic pattern table;
- telemetry mechanism table;
- deterministic qualitative visualization manifest;
- suggested multi-panel figure data;
- exact allowed claim text;
- exact forbidden claim text;
- source artifact paths + SHA256 digests.

Suggested future paper figure if Stage Z succeeds:

- panel A: model families / common intervention contract;
- panels B-D: per-model dose-response curves;
- panel E: normalized mechanism telemetry / contact-loss comparison;
- panel F: deterministic qualitative frames across model families.

No Stage-Z number enters the paper until PI reviews Z4.

---

# 15. Execution resources and concurrency

Server connectivity has been restored, but provenance takes priority over throughput.

Codex may use available GPUs only after the relevant gate allows real execution.

Rules:

- never kill or modify unrelated users' GPU processes;
- preflight free VRAM and environment before model load;
- use explicit output locks;
- never run the same scientific model-parent-arm concurrently in two workers;
- a model-parent five-arm matrix must be serialized under one owner/receipt namespace;
- M1 checkpoint materialization remains one suite at a time if storage requires sequential handling;
- up to one active worker per model family is acceptable after Z1 if resource/provenance isolation is proven;
- if concurrency creates ambiguous output ownership, reduce to one worker;
- do not use `/dev/shm` as evidence authority storage.

No scientific result is worth a provenance ambiguity.

---

# 16. Protected firewall

The following remain prohibited throughout Stage Z unless a future explicit PI authorization says otherwise:

- Eval160;
- protected evaluation;
- BRIDGE/F1 protected identities/outcomes;
- any population chosen from protected outcomes;
- any protected read performed merely to decide whether Stage Z looks promising.

Every Z1-Z4 root seal must record protected counters.

Expected values:

- Eval160 reads = 0
- protected reads = 0

A nonzero protected read is an immediate STOP condition.

---

# 17. Claims explicitly forbidden even if Stage Z is positive

Even a strong three-model replication does **not** authorize claims of:

- universal VLA vulnerability;
- universal manipulation vulnerability;
- visual adversarial attack success;
- matched visual-PGD physical efficacy;
- real-world/real-robot failure;
- causal mediation;
- validated defense;
- detector impossibility;
- guaranteed monotonicity outside the tested panel;
- all architectures sharing identical effect size.

The strongest intended claim is bounded architecture external validity in simulation:

> A gripper-only OPEN duty-cycle physical susceptibility observed in the original OpenVLA setting replicates, under a common LIBERO intervention contract, across the tested VLA policy families.

Only use that sentence if Z4 actually supports it.

---

# 18. Mandatory early-stop conditions

STOP and return to PI if any of the following occurs:

1. any remaining M1 checkpoint cannot be byte-sealed exactly;
2. common LIBERO authority drifts or cannot be reproduced;
3. M0/M1/M2 checkpoint/source authority mismatches;
4. X0 arm-isolation semantics are not identifiable;
5. a common gripper-only estimand cannot be implemented across all three families;
6. M1 T10 queue-crossing semantics or M2 replan semantics would change the estimand and cannot be resolved prospectively;
7. Z1 reveals arm drift, decode/re-encode, residual queue/chunk contamination, or state-restore mismatch;
8. exact X0-compatible critical/noncritical anchor semantics cannot be recovered;
9. any proposal arises to change panel/model/dose/OPEN/H_phys/anchor rule after outcome exposure;
10. protected/Eval160 data is accessed;
11. an existing scientific arm is proposed for retry-to-pass after env exposure;
12. a runtime bug makes previously generated scientific rows semantically ambiguous;
13. a code change after partial Z3 exposure changes action semantics for remaining arms;
14. output provenance cannot distinguish workers/checkpoints/parents/arms;
15. scientific conclusions would require silent complete-case deletion or replacement.

When stopping, preserve partial evidence and write a HOLD root. Do not delete or rerun to make the gate green.

---

# 19. Autonomous non-stop conditions

The following ordinary engineering issues do **not** require PI review by themselves, provided no scientific exposure has occurred under the faulty configuration and all authority remains exact:

- SSH reconnect before a file transfer completes;
- resumable transfer of an incomplete unsealed checkpoint file;
- temporary local disk cleanup of newly created Stage-Z cache after its durable receipt exists;
- formatting/lint fixes;
- CPU/mock-test fixes;
- output-directory creation;
- non-semantic logging improvements;
- pre-exposure package/environment repair;
- retrying a model download/materialization that has not yet been accepted as authority.

Codex should fix these, document them, and continue.

---

# 20. Goal-level Codex execution instructions

Treat this document as the controlling long-horizon goal for the new Stage-Z continuation PR.

Do not stop after each successful substep merely because the previous PR historically said `STOP_FOR_PI`. This document is the new conditional authorization for the continuation branch.

Operational loop:

1. read the current continuation-branch HEAD and all existing Stage-Z authority artifacts;
2. execute only the currently legal gate;
3. seal machine-readable receipts and update handoff status;
4. run static/unit/CI checks;
5. if the gate PASS conditions are satisfied, commit/push and autonomously enter the next gate defined here;
6. if any early-stop condition occurs, write a fail-closed HOLD artifact, commit/push, and stop;
7. never merge the PR automatically.

Do not infer success from CI alone. CI is engineering evidence; scientific promotion requires the gate-specific authority and outcome criteria.

---

# 21. Final terminal state

The desired terminal deliverable is:

`STAGE_Z_Z4_CROSS_MODEL_SYNTHESIS_READY_FOR_PI_REVIEW`

At that point STOP_FOR_PI with:

- live PR HEAD/tree;
- all CI states;
- Z0R2/Z1/Z2/Z3/Z4 root seals;
- exact model/parent/arm counts;
- attrition/censoring ledger;
- per-model dose-response and telemetry summaries;
- complete-dose pattern tables;
- deterministic qualitative/video audit manifest;
- protected counters;
- one of the bounded Stage-Z scientific classifications;
- deterministic paper-delta export package.

Do not merge and do not modify the merged paper repository until PI reviews the Stage-Z synthesis.
