# GPT Scientific Review / PI Handoff — Stage Z takeover

Date: 2026-08-22
Repository: `Leo-6-maker/openvla-gripper-dutycycle-attack`
Primary live coordination PR: #135

## 0. Purpose and role split

This document is for a **new GPT conversation window** taking over as the project's **scientific reviewer / project PI**.

Do **not** behave like the server execution agent.

- **Codex**: server implementation, runtime qualification, experiment execution, provenance, counters, manifests, root seals, durable artifacts.
- **GPT / PI**: independently audit live GitHub evidence, define scientific gates, distinguish science from engineering holds, enforce denominators and claim boundaries, decide promotion/stop, and maintain the paper narrative.

The new GPT must **not trust this handoff blindly**. Its first task is to independently re-read the live GitHub state listed in Section 2 and report discrepancies before authorizing or interpreting any new Codex execution.

---

# 1. Live repository snapshot at handoff time

Independently rechecked immediately before this handoff:

- PR #135: `OPEN / DRAFT / MERGEABLE`
- PR head branch: `codex/stage-x-x1r2-q3-arm-isolation-repair-20260820`
- live HEAD: `85af6cfac905b06a17aec26096c092e07a45a949`
- current CI at this HEAD:
  - `cpu-stageb`: SUCCESS
  - `cpu-b3-official-v3`: SUCCESS
  - `cpu-detector-v5`: SUCCESS
- Paper V1 is immutable.
- F1 experimental track is terminally closed.
- BRIDGE_V3 remains sealed/unopened.
- Eval160/protected evaluation remain `UNREAD`.
- **Stage Z has been authorized by PR comment but has not yet produced any committed Stage-Z artifact at this HEAD.**

Important: the **PR body is historical/stale** and still describes the old Q3 repair. It is not the current scientific authority. Use latest PI comments + sealed artifacts instead.

Latest important PR comments:

1. F1T sealed synthesis: `#issuecomment-5376799409`
2. Paper V2 submission-readiness gate: `#issuecomment-5376823733`
3. Current superseding bounded experiment authorization: **Stage Z** `#issuecomment-5380973050`

Current experiment authority is therefore:

`STAGE_Z_CROSS_MODEL_OPEN_DUTY_GENERALIZATION_V1`

Stage Z supersedes only the “no further experiment” portion of the Paper-V2 readiness gate. It does **not** reopen F1/F1-C/F1-D/BRIDGE_V3/R0/R1/R2/protected evaluation.

---

# 2. Mandatory first read / independent verification checklist for new GPT

Before talking to Codex about further execution, independently verify the following from GitHub.

## 2.1 Live state

1. Fetch PR #135 metadata and verify state/head SHA/branch.
2. Fetch workflow runs for the live head and verify the three CPU workflows.
3. Fetch the latest PR conversation comments and confirm the chronological authority order around:
   - `5376799409` — F1T terminal synthesis
   - `5376823733` — Paper V2 submission readiness
   - `5380973050` — Stage Z authorization
4. Check whether any comment/commit after this handoff supersedes Stage Z.
5. Search the PR changed-file list for `STAGE_Z` artifacts. At handoff time there are none; if new ones exist, do not assume PASS—review them.

## 2.2 Canonical scientific evidence

Read at minimum:

- `docs/handoffs/STAGE_X_X0_RESULT_20260817.md`
- `docs/handoffs/STAGE_VI_B2_FRESH_M4_AND_NEGATIVE_CAUSAL_HANDOFF_20260816.md`
- `docs/handoffs/STAGE_VII_DEVELOPMENT_NEGATIVE_HANDOFF_20260816.md`
- `docs/handoffs/STAGE_IX_F0_RESULT_20260817.md`
- `reports/STAGE_X1R2_E4_FACTORIZATION_FAILURE_DECOMPOSITION_20260821/STAGE_X1R2_E4_FINAL_CLAIM_LEDGER_V1.json`
- `paper/PAPER_V1_EVIDENCE_AUTHORITY_MAP_V1.json`
- `paper/PAPER_V1_CLAIM_LEDGER_V1.json`
- `paper/PAPER_V1_FINAL_ROOT_SEAL_V1.json`
- `reports/STAGE_X1R2_F1C4_FRESH_CANARY_RESULT_V1_R3_20260822/F1C4_TERMINAL_DECISION_V1.json`
- `reports/STAGE_X_X1R2_F1T_TERMINAL_SYNTHESIS_V1.json`
- `reports/STAGE_X_X1R2_F1T_CLAIM_LEDGER_DELTA_V1.json`
- `reports/STAGE_X_X1R2_F1T_EVIDENCE_AUTHORITY_MAP_V1.json`
- `reports/STAGE_X_X1R2_F1T_ROOT_SEAL_V1.json`
- `paper/PAPER_V2_F1_DELTA_FROM_V1.md`

Then read the full current Stage-Z PI comment `#issuecomment-5380973050` before allowing Codex to interpret its next action.

## 2.3 Required takeover response

The new GPT should first return a short independent audit with:

- live HEAD/tree and PR state;
- latest controlling PI comment;
- current scientific status of X0, detector line, IX, E3/E4, F1, F1T;
- whether Stage Z has started;
- protected-boundary state;
- any discrepancy found between this handoff and GitHub.

Only after that should it coordinate with Codex.

---

# 3. Frozen scientific framing

The project originally pursued a gripper-targeted inference-time visual attack, but the mature scientific result is **mechanism-first / factorization-gap**, not a universal attack or detector.

Keep these quantities separate:

- `C_t`: clean timing criticality / opportunity.
- `V_t(d)`: physical vulnerability at clean state `t` under a duration-`d` command-OPEN counterfactual.
- `E_t(single)`: existence of a strict model-side/selective visual realization at a single state.
- `E_t(T5)`: sustained strict selective delivery across five attempted intervention steps.
- `Y_t(vis)`: physical response of the full visual attack pipeline.

Do **not** draw or claim a demonstrated causal chain `C -> V -> E -> Y`.

Preferred Paper-V2 thesis:

> A dose- and phase-dependent gripper-OPEN physical mechanism is strong, while clean timing opportunity, physical susceptibility, model-side selective realizability, and sustained executable delivery only partially align. A positive result at one layer does not promote an untested claim at the next.

F1 added an important execution-layer distinction:

`single-step strict realizability != sustained T5 selective delivery != matched physical efficacy`

---

# 4. Evidence hierarchy and experiment status

## 4.1 MAIN POSITIVE — Stage X X0 physical duty-cycle mechanism

Status:

`STAGE_X_PHYSICAL_DUTY_CYCLE_MECHANISM_SUPPORTED`

Key sealed evidence:

- 40 Stage V + 16 Stage VI-B2 parents.
- 1,344 probe groups / 5,376 four-arm branches.
- consumable `V_phys` rows:
  - T3 = 1,245
  - T5 = 1,191
  - T10 = 1,126
- raw positive rates:
  - T3 = 0.39438
  - T5 = 0.67758
  - T10 = 0.87300
- 1,126 complete three-dose patterns; all monotone `000/001/011/111`.
- mechanism-consistent telemetry: exact command delivery -> aperture excess -> contact loss -> object displacement with increasing dose.
- uncertainty: 2,000 parent bootstrap replicates; no iid-row CI.

Claim boundary:

- CAN: bounded dose/phase-dependent OPEN physical mechanism in evaluated LIBERO/OpenVLA setting.
- CANNOT: formal mediation, universal physical law, real-robot validation, visual-PGD physical efficacy.

This is the strongest positive result and remains Paper V1/V2 anchor.

## 4.2 Historical Black Bowl fixed-window evidence

Corrected State5/State7 material is bounded same-task/phase mechanism context only.

Do not promote it over X0. Earlier versions include simulator-SR mismatch and historical protocol limitations; manual video evidence is important. It is useful context showing that official task SR can miss contact-quality failures, but the current authority map keeps it contextual rather than a primary broad denominator.

## 4.3 Detector / timing-selector line — valid negative scientific evidence

### Stage VI-B2

Status:

`STUDENT_HELDOUT_GENERALIZATION_NOT_ESTABLISHED`

Fresh 16-parent population; T5 primary 333 consumable + 51 abstain/censored.

- AUROC 0.6246432939
- AUPRC 0.7976720489
- AUPRC lift 1.1911425664
- top-decile lift 1.4493537325
- ECE-10 0.4606357016
- threshold emission 0.4324324324
- `libero_10` emission 0; `libero_spatial` 0.9583; Object AUROC non-identifiable due to no negative consumables.

This is a valid negative scientific result, not a runtime hold.

### Stage VII

Status:

`STAGE_VII_DEVELOPMENT_NO_GENERALIZABLE_DETECTOR`

Frozen S7-A/B/C all fail at least one cross-suite generalization/selectivity promotion gate. None promoted.

### Stage VIII R1

Status:

`STAGE_VIII_R1_NO_GENERALIZABLE_RELATIVE_SELECTOR`

Direct R1 handoff is recovered from immutable Git history in the Paper V1 authority map.

- R1-A parent-macro AUC 0.577615 — FAIL
- R1-B parent-macro AUC 0.658572 — FAIL

Claim boundary for VI-B2/VII/VIII:

- CAN: frozen evaluated timing selectors did not establish stable deployment-facing cross-suite generalization.
- CANNOT: no detector can ever work; every feature is uninformative; detector failure caused later attack results.

## 4.4 Stage IX — model-side targetability vs timing utility gap

Status:

`STAGE_IX_NO_MODEL_SIDE_TIMING_SIGNAL`

No environment / no physical intervention. 1,344/1,344 sealed rows.

Model-side DEVTEST AUROC:

- E0 0.870743
- E1 0.900510
- E3 0.897157

Factorized parent-macro timing AUC:

- E0 0.483698
- E1 0.521112
- E3 0.523390

Worst LOSO remains weak. Therefore high model-side targetability scores did not establish factorized physical timing utility.

Claim boundary:

- CAN: model-side targetability is not sufficient for useful physical timing.
- CANNOT: physical attack efficacy, attack impossibility, formal mediation.

## 4.5 Historical X1 / X1R-V1 — invalid/incomplete, non-promotional

Preserve for governance/provenance only.

Important historical facts:

- earlier tokenizer endpoint/bin-edge defect created non-bijective 31744/31745 behavior;
- historical X1 contained victim/provenance and sequence-enumeration defects;
- X1R-V1 ITT N=7, scientific evaluable N=0;
- two consumed runtime-invalid; five never started;
- no accepted adversarial result, attacked env.step, `V_phys`, or attack outcome.

Never call X1R-V1 a negative attack experiment and never recycle untouched identities into later scientific cohorts.

## 4.6 Q3 / Q3-AR / Q3R2 / Q3R3 — engineering qualification history

These are important engineering/reproducibility diagnostics but **not physical efficacy experiments**.

Highlights:

- Q3 first real-model fixture failed exact arm isolation before attacked env.step.
- Q3-AR-F01 stopped after clean screening because expected Student emit 133 replayed as None; no attack began.
- RCA1 localized primary `CLEAN_TRAJECTORY_DRIFT`, secondary `UNRESOLVED_MULTI_FACTOR`; Student stochasticity not established.
- Q3R2-C exposed visual-prefix nondeterminism on some suites.
- Q3R3 introduced branch-controlled estimand/replay; four-suite branch replay eventually passed as engineering infrastructure.
- D2 exact-arm selector attempts all structural-invalid; did not prove method infeasibility.
- E0 showed candidate-level failure evidence had not been sealed sufficiently.
- E1 repaired failure-path evidence persistence only.

Do not convert these holds into negative attack results.

## 4.7 E2 — bounded timing/scheduler hold

Status:

`HOLD_E2_FOUR_SUITE_BRANCH_QUALIFICATION_INCOMPLETE_NO_LEGAL_GOAL_EMIT`

Three Goal successor clean identities had no legal Student emit. TRUE attack feasibility never started.

Use only as bounded Goal timing/scheduler non-emission evidence. It is not a strict visual-method negative and does not prove detector invalidity globally.

## 4.8 E3 — fresh timing-decoupled strict visual realizability

Status:

`E3_STRICT_SELECTIVE_REALIZABILITY_PARTIAL_SUITE_DEPENDENT`

Fixed engineering-only denominator:

- 12 parents = 3 per suite
- 12/12 clean valid
- 12/12 probe available
- 12/12 TRUE invocation reached
- 12/12 six-candidate audits complete
- 72 candidate slots (diagnostic, non-iid)

Strict-valid parent result:

- libero_10: 1/3
- libero_goal: 0/3
- libero_object: 0/3
- libero_spatial: 1/3
- total 2/12

No attacked env.step, physical intervention, Vphys, protected, or attack outcome.

Correct claim:

> strict model-side/selective realizability under the frozen method was sparse/partial and suite-state dependent; cross-suite capability was not established.

Do not call 2/12 attack success rate or physical efficacy. Do not call Goal/Object 0/3 impossibility.

## 4.9 E4 — factorization decomposition / Paper lock

Status:

`STAGE_X_X1R2_E4_PAPER_LOCK_READY`

Offline CPU-only decomposition of E3.

72 candidate classes:

- `ARM_EXACT_AND_NATIVE_OPEN`: 4
- `ARM_EXACT_BUT_NOT_NATIVE_OPEN`: 29
- `NATIVE_OPEN_BUT_ARM_DRIFT`: 1
- `NEITHER_OPEN_NOR_ARM_EXACT`: 38
- evidence missing: 0

Parent categories:

- `STRICT_REALIZABLE`: 2
- `JOINT_LIMITED`: 1
- `TARGETABILITY_LIMITED`: 9
- `SELECTIVITY_LIMITED`: 0

Every parent had some exact-arm candidate; only 3/12 parents had any native-OPEN candidate. This motivated one bounded prospective targetability-development namespace, F1.

## 4.10 Paper V1 — sealed immutable paper package

Status:

`PAPER_V1_MECHANISM_FACTORIZATION_DRAFT_BUNDLE_READY_FOR_PI`

Final Paper V1 root seal SHA-256:

`830c55dee96477e87c36437a33760a9dced0d1217ea01b3e72905f26fd142336`

Paper V1 includes authority map, 48-claim ledger/audit, manuscript, figures/tables, reproducibility supplement, manifests/root seals.

Do not edit `paper/PAPER_V1_*` to anticipate later results.

## 4.11 F1-A/A2/A3 — prospective population/source governance

F1 was a new exploratory-then-prospective attempt to improve model-side native-OPEN targetability and later test a matched same-state C/V/E physical bridge.

Important progression:

- original G10 source became capacity-insufficient after stronger exclusion authorities, especially Goal.
- F1-A2 correctly HOLDed rather than recycling/top-up.
- F1-A3 introduced a prospective V3 source split before attack-development outcomes:
  - `DEV_V3`: 24 = 6/suite, engineering-only
  - `C_CANARY_V3`: 8 = 2/suite, reserved F1-C
  - `BRIDGE_V3`: 20 = 5/suite, prospective scientific cohort
- all roles disjoint and exposure-classified; Paper V1 remained unchanged.

BRIDGE_V3 was never opened.

## 4.12 F1-B — bounded engineering method development

Result:

`F1B_NEW_METHOD_SELECTED_FOR_F1C`

24 DEV parents, 6/suite. Parent-level frozen lexicographic comparison:

- M0-10: min-suite success 0; successful parents 4; strict-valid probes 5
- M1-10: min-suite success 1; successful parents 5; strict-valid probes 8
- M2-10: min-suite success 1; successful parents 5; strict-valid probes 9

M1-10 selected because M1/M2 tied on first two preregistered parent criteria and M1 won the frozen lower-Linf/lower-complexity tie break.

M1 objective: native-OPEN-set aligned, rather than old single-target-token objective.

This is engineering/model-side method-development evidence only. It does not upgrade E3 and is not physical efficacy.

## 4.13 F1-C historical canary and repair

Initial F1-C T5 canary namespace failed executability before meaningful attack attempts due schema/replay issues. Historical canary identities were consumed and not rerun/top-up.

Contract-preserving CPU/static repairs were sealed later, but these did not retroactively qualify the consumed canaries.

## 4.14 F1-C4 fresh namespace — terminal executable-evidence HOLD

Terminal status:

`HOLD_F1C4_EXECUTABLE_EVIDENCE_INSUFFICIENT_TERMINAL`

Fresh denominator:

- 8 parents
- 7 completed / 1 replay-HOLD
- 14/16 temporal arms completed
- replay-HOLD parent: `libero_spatial/task_02/state_13`, both temporal arms
- 70 attempted steps
- 70/70 candidate audits complete
- 770 diagnostic candidate rows
- 69 clean fallbacks
- 1 strict-valid executed/attacked step
- 1 PGD call
- 1 attacked `env.step`
- no `V_phys` read
- no physical intervention/outcome read
- no task-outcome read
- Eval160/protected unread

Interpretation:

- at least one strict selective execution exists at the single-step execution layer;
- reliable sustained T5 delivery was **not established**;
- this is not a physical negative and not a T5 qualification PASS;
- no temporal-init superiority claim.

Known bookkeeping subtlety:

- canonical F1-C4 terminal aggregate/root records `attacked_env_steps = 1`;
- one historical arm receipt's nested `protected_boundary.attacked_env_steps` mirrored `0` despite its raw counter showing 1;
- treat terminal aggregate/root + raw execution counter as authority; preserve historical receipt immutably and do not rewrite it.

## 4.15 F1T — canonical terminal synthesis

Status:

`F1T_TERMINAL_SYNTHESIS_SEALED_FOR_PI`

F1T is static/offline only and binds F1-A3 -> F1-B -> F1-C history -> F1-C4.

F1T root sidecar SHA-256:

`697fe0584b55c676c964836471581d5f7cea05cdae6b5346a3e8b9a85ef10c10`

Canonical F1T claim:

`single-step strict realizability != sustained T5 selective delivery != matched physical efficacy`

Promotable F1 additions:

- M1-10 improved over M0-10 on preregistered DEV parent-level advancement criteria.
- strict selective visual execution was observed at least once in fresh C4 canaries.
- reliable full T5 executable qualification was not established.
- 69 other attempted steps fail-closed to clean action.
- BRIDGE/F1-D was never opened because qualification did not pass.

F1 experimental track is closed under finite stop-loss.

## 4.16 Paper V2 submission-readiness gate

After F1T, PI authorized static Paper V2 assembly/readiness in a new namespace while keeping Paper V1 immutable.

Paper V2 should integrate F1 only as an append-only extension, not rewrite historical evidence.

Current `paper/PAPER_V2_F1_DELTA_FROM_V1.md` preserves the correct boundaries.

## 4.17 CURRENT ACTIVE PROSPECTIVE EXTENSION — Stage Z cross-model physical mechanism

Authorized by PR #135 comment:

`#issuecomment-5380973050`

Namespace:

`STAGE_Z_CROSS_MODEL_OPEN_DUTY_GENERALIZATION_V1`

At handoff time this is **authorized but not yet started/committed**.

Scientific question:

> Does the strongest architecture-independent positive result—the dose/phase-dependent command-OPEN physical mechanism—replicate across substantially different VLA policy families, or is it specific to the original OpenVLA policy/action architecture?

Primary fixed model panel:

1. current OpenVLA-7B anchor;
2. OpenVLA-OFT using official LIBERO suite-specific checkpoints;
3. `pi0.5-LIBERO` / OpenPI family.

Do not add/drop models after seeing Stage-Z results.

Stage Z deliberately does **not** copy OpenVLA token-PGD across architectures. It tests the architecture-independent environment/action-level command-OPEN counterfactual after each model's official decoder/action interface.

Planned shared prospective panel:

- 4 LIBERO suites
- 10 tasks per suite
- 1 deterministic fresh identity/state per task after complete historical exclusion
- 40 identities shared across model families
- 120 model-parent clean references maximum

For each eligible model-parent, planned five-arm matrix:

1. `CLEAN_BRANCH_CRITICAL`
2. `COMMAND_OPEN_T3_CRITICAL`
3. `COMMAND_OPEN_T5_CRITICAL`
4. `COMMAND_OPEN_T10_CRITICAL`
5. `COMMAND_OPEN_T5_NONCRITICAL_CONTROL`

Maximum nominal branch scale ~600 before clean/no-anchor exclusions.

Anchor rule:

- use the model's own clean trajectory plus simulator privileged physical state / X0-compatible semantics;
- do **not** use frozen Student detector to choose Stage-Z timing;
- freeze critical and noncritical anchors outcome-blind;
- if multiple legal anchors exist, use frozen deterministic hash;
- permit abstain/no-anchor without replacement.

Chunked-policy fairness hard gate:

- branch only at fresh policy decision/replan boundary;
- Z0 must explicitly audit policy-state/action-queue/chunk/replan semantics for OFT and pi0.5 before scientific execution;
- if a common causal branch contract cannot be established, HOLD rather than silently comparing unlike decision states.

Primary physical readouts must remain X0-compatible:

- commanded OPEN fraction/delivery
- aperture excess
- contact loss
- object displacement
- bounded physical-vulnerability readout
- official LIBERO success only as secondary/compatibility metric, never the sole physical truth.

Manual video audit is mandatory because historical project evidence shows official simulator SR can miss contact-quality failures.

Primary scientific unit: **model-parent pair**. Candidate/step/telemetry rows are not iid samples.

Per-model report:

- T3/T5/T10 physical-positive rates and denominators;
- complete-dose monotonic pattern fraction;
- 2,000 parent bootstrap uncertainty;
- mechanism telemetry by dose;
- critical-T5 vs noncritical-T5 phase comparison;
- manual-audit agreement/disagreement with automatic/official metrics.

Do not pool three model families into one iid significance test.

Predeclared Stage-Z interpretation:

- `...REPLICATED_3_OF_3`: all three families support the frozen dose/phase mechanism criteria.
- `...PARTIAL_MODEL_DEPENDENT`: some but not all families replicate.
- `...GENERALIZATION_NOT_ESTABLISHED`: only original OpenVLA anchor supports the mechanism under the frozen cross-model panel.
- `HOLD_STAGE_Z_EXECUTABLE_EVIDENCE_INSUFFICIENT`: engineering/runtime/branch authority prevents interpretable execution.

Never tune a model after its result to turn partial evidence into 3/3.

Stage-Z state machine:

`Z0 static model/env/action authority -> Z1 excluded engineering executable qualification -> Z2 freeze/execute clean references and anchors -> Z3 five-arm physical matrix -> Z4 static synthesis -> STOP_FOR_PI`

Codex may repair contract-preserving implementation/provenance/storage/GPU orchestration issues, but must stop before changing model panel, checkpoint, source population, doses, OPEN semantics, anchor rule, branch estimand, physical endpoint definitions, or protected boundary.

---

# 5. Frozen action/token/runtime authorities for historical OpenVLA attack evidence

These remain important when interpreting E3/F1, even though Stage Z itself is architecture-independent.

- Native checkpoint-local `ActionTokenizer` is action-token authority.
- Historical nearest-center helper has endpoint/bin-edge defect; do not mutate historical evidence to “fix” it.
- 31744 can encode but endpoint decoding/clipping can re-encode to 31745; roundtrip is non-bijective. No universal OPEN token may be assumed from surrogate re-encoding.
- Actual cached deterministic `model.generate(... do_sample=False, return_dict_in_generate=True, output_scores=True)` is behavior authority.
- Differentiable cached generated-prefix path is optimization surrogate only.
- Full `use_cache=False` teacher-forced forward is diagnostic only.
- Historical/current strict selective OpenVLA candidate validity requires direct generation and exact arm preservation under the relevant frozen namespace.
- No actuator overwrite, decode->re-encode substitution, or fallback can be silently introduced into historical strict attack evidence.

Runtime nuance:

A named “official environment” path is not sufficient authority by itself. Historical audit found the executable could resolve through another base prefix/site-package surface. Any new runtime must bind actual import paths/package/source hashes, not just an environment name.

---

# 6. Protected-evaluation firewall

At handoff time:

- `Eval160 = UNREAD`
- protected evaluation = `UNREAD`

Do not read protected/Eval160 to decide whether a method is promising, to select Stage-Z parents/models, or to strengthen Paper V2.

No historical engineering HOLD should be upgraded by protected evidence.

Any future protected read requires an explicit new prospective PI authorization after method/claim freeze.

---

# 7. Current paper claim ledger in plain language

## Can claim

1. In the evaluated OpenVLA/LIBERO setting, command-OPEN duty-cycle exposure shows a strong dose-dependent physical susceptibility and mechanism-consistent telemetry.
2. Frozen clean/context timing selectors evaluated in VI-B2/VII/VIII did not establish stable cross-suite actionable generalization.
3. Stage IX shows that strong model-side targetability scores do not automatically yield useful factorized physical timing utility.
4. Under the frozen E3 method, strict single-state/selective model-side realization was sparse and suite/state dependent (2/12 engineering parents).
5. F1's native-OPEN-set-aligned M1 improved engineering DEV targetability over M0 under the preregistered parent-level selection rule.
6. Fresh F1-C4 canaries demonstrated at least one strict selective executed visual step, while reliable sustained T5 executable delivery was not established.
7. F1-D/matched visual physical efficacy was never opened because the qualification gate did not pass.
8. The mature synthesis is a factorization gap across physical susceptibility, timing opportunity, model-side/selective realization, and sustained delivery.

## Cannot claim

- universal/generalizable detector;
- cross-suite visual-PGD physical efficacy;
- reliable sustained T5 visual delivery;
- Goal/Object visual-attack impossibility;
- detector caused E3/F1 failures;
- formal X0 mediation;
- attack efficacy from E3/F1 single-step realization;
- real-robot validation;
- cross-model mechanism generalization **before Stage Z result exists**;
- protected/Eval160 validation;
- simulator official SR alone as proof of contact-quality success/failure.

---

# 8. How the new GPT should interact with Codex

## 8.1 Before every new scientific gate

GPT should independently fetch:

- live PR head/tree/state;
- latest controlling PI comment(s);
- changed files / relevant commits;
- CI status;
- protocol/root seal/claim ledger;
- exposure/read counters.

Then classify the result as one of:

- scientific PASS;
- valid scientific negative;
- engineering HOLD;
- invalid/superseded/non-promotional.

Do not use CI green as scientific authorization.

## 8.2 Codex autonomy

Allow Codex to autonomously fix implementation bugs only when they preserve the written scientific contract. Examples:

- import/path/runtime binding;
- crash-safe receipts;
- deterministic pool/hash scripts;
- storage/root seals;
- counters/manifests;
- model adapter plumbing that does not change the model/checkpoint/action semantics;
- GPU scheduling under existing resource policy.

Codex must return to PI before semantic changes to populations, endpoints, attacks, doses, models, estimands, or protected boundaries.

## 8.3 Fail closed

Never rescue a result by:

- rerun-to-pass of consumed identities;
- replacement/top-up after exposure;
- post-hoc threshold/objective/epsilon/model search;
- weakening exact arm/selectivity rules in historical OpenVLA attack experiments;
- selecting only successful candidate slots as scientific denominator;
- treating candidate/step rows as iid;
- reading protected outcomes to decide next steps.

---

# 9. Paper strategy at takeover

Paper V1 is already a complete sealed mechanism/factorization draft package.

Paper V2 should preserve the same central thesis and add F1 as bounded execution-layer evidence.

Stage Z is a prospective external-validity extension. If successful, preferred paper structure becomes:

1. **Cross-model physical mechanism**: does OPEN dose/phase susceptibility replicate across OpenVLA / OFT / pi0.5?
2. **OpenVLA deep factorization case study**:
   - timing-selector negatives (VI-B2/VII/VIII)
   - model-side timing gap (IX)
   - sparse selective realization (E3/E4)
   - bounded targetability improvement and execution gap (F1/F1T)
3. Discussion: physical susceptibility may generalize more readily than reliable exploitability.

Even a Stage-Z 3/3 replication does **not** replace real-robot validation. Multi-model evidence addresses model-family external validity; real hardware addresses sim-to-real external validity.

Do not delay/overwrite the sealed Paper V1 package while Stage Z runs. Preserve an ICRA-ready static path in parallel.

---

# 10. Canonical files / comments to keep bookmarked

## Main physical mechanism

- `docs/handoffs/STAGE_X_X0_RESULT_20260817.md`

## Detector/timing negatives

- `docs/handoffs/STAGE_VI_B2_FRESH_M4_AND_NEGATIVE_CAUSAL_HANDOFF_20260816.md`
- `docs/handoffs/STAGE_VII_DEVELOPMENT_NEGATIVE_HANDOFF_20260816.md`
- Stage VIII immutable Git binding: see `paper/PAPER_V1_EVIDENCE_AUTHORITY_MAP_V1.json`

## Model-side factorization

- `docs/handoffs/STAGE_IX_F0_RESULT_20260817.md`

## E3/E4

- `reports/STAGE_X1R2_E3_FACTORIZED_SELECTIVE_REALIZABILITY_20260821/E3_DECISION_TABLE_V1.json`
- `reports/STAGE_X1R2_E4_FACTORIZATION_FAILURE_DECOMPOSITION_20260821/STAGE_X1R2_E4_E3_CANDIDATE_FAILURE_DECOMPOSITION_V1.json`
- `reports/STAGE_X1R2_E4_FACTORIZATION_FAILURE_DECOMPOSITION_20260821/STAGE_X1R2_E4_FINAL_CLAIM_LEDGER_V1.json`

## Paper V1

- `paper/PAPER_V1_EVIDENCE_AUTHORITY_MAP_V1.json`
- `paper/PAPER_V1_MANUSCRIPT_DRAFT.md`
- `paper/PAPER_V1_CLAIM_LEDGER_V1.json`
- `paper/PAPER_V1_CLAIM_AUDIT_V1.md`
- `paper/PAPER_V1_FINAL_ROOT_SEAL_V1.json`

## F1/F1T

- `reports/STAGE_X_X1R2_F1B_DEV_RESULT_AGGREGATION_V3_20260821/F1B_DEV_DECISION_V3.json`
- `reports/STAGE_X1R2_F1C4_FRESH_CANARY_RESULT_V1_R3_20260822/F1C4_TERMINAL_DECISION_V1.json`
- `reports/STAGE_X_X1R2_F1T_TERMINAL_SYNTHESIS_V1.json`
- `reports/STAGE_X_X1R2_F1T_CLAIM_LEDGER_DELTA_V1.json`
- `reports/STAGE_X_X1R2_F1T_EVIDENCE_AUTHORITY_MAP_V1.json`
- `reports/STAGE_X_X1R2_F1T_ROOT_SEAL_V1.json`
- `paper/PAPER_V2_F1_DELTA_FROM_V1.md`
- `docs/handoffs/STAGE_X_X1R2_F1T_TERMINAL_SYNTHESIS_20260822.md`

## Current prospective authority

- PR #135 `#issuecomment-5380973050` — Stage Z cross-model OPEN-duty physical-mechanism generalization.

---

# 11. Immediate next action after takeover audit

If the new GPT confirms that no later comment/commit supersedes this handoff and no Stage-Z artifact has yet appeared:

1. tell Codex to begin **Z0 only** under `STAGE_Z_CROSS_MODEL_OPEN_DUTY_GENERALIZATION_V1`;
2. require static binding of exact model checkpoints, official model->LIBERO action semantics, policy decision/chunk/replan behavior, environment/runtime imports, OPEN intervention mapping, common branch estimand, shared fresh identity source/exclusion union, and all zero protected counters;
3. do not permit scientific model execution until Z0 is sealed and independently reviewed;
4. if Z0 PASS, allow the predeclared excluded engineering qualification Z1 according to the Stage-Z comment;
5. preserve the Paper V2 static submission-readiness work in parallel without allowing it to reinterpret Stage Z before results exist.

If Stage-Z artifacts have appeared by the time the new GPT takes over, **do not tell Codex to rerun Z0**. Audit the existing Stage-Z state first, identify the last valid gate, and continue only from the next authorized transition.

---

# 12. Final scientific north star

The project is strongest when it demonstrates disciplined separation between:

- a real, dose/phase-dependent physical OPEN mechanism;
- the difficulty of identifying deployment-facing timing opportunity;
- the difficulty of realizing selective model-side OPEN behavior;
- the additional difficulty of sustaining that behavior through an executable closed-loop pipeline.

Do not convert failed promotion gates into successful attacks, and do not convert engineering failures into scientific negatives.

Stage Z should answer one narrow external-validity question: **does the strongest physical mechanism replicate across different VLA policy families?** It must not become a new open-ended search for a positive attack.
