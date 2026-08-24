% Paper V1 manuscript draft

# From OPEN Duty-Cycle Mechanism to Factorization Gap: Timing, Physics, and Visual Selectivity in OpenVLA Manipulation

Status: `PAPER_V1_FULL_DRAFT_READY`

## Abstract

<!-- CLAIM:C001 -->
Gripper-targeted manipulation failures are often discussed as if clean timing
criticality, physical vulnerability, and visual exploitability were one
quantity. We separate them into three estimands: clean criticality/opportunity
`C_t`, physical vulnerability under a duration-`d` command-OPEN counterfactual
`V_t(d)`, and strict visual exploitability `E_t`. The evidence supports a
different picture. In Stage X0, the physical OPEN mechanism shows a dose
response: the raw positive rates at T3, T5, and T10 are 0.39438, 0.67758, and
0.87300 among the source-defined consumable rows, and the complete three-dose
patterns are all monotone. The associated telemetry chain is consistent with
command delivery, aperture excess, contact loss, and displacement.

<!-- CLAIM:C002 -->
Clean/context timing selectors do not establish stable deployment-facing
generalization under the frozen gates: the held-out VI-B2 result has overall
AUROC 0.6246432939 and ECE-10 0.4606357016, all three Stage VII candidates fail
at least one promotion gate, and both Stage VIII R1 selectors fail their
parent-grouped gate.

<!-- CLAIM:C003 -->
Stage IX makes the factorization gap explicit. Model-side scores have
DEVTEST AUROC 0.870743--0.900510, while factorized parent-macro AUC is
0.483698--0.523390; the former does not establish physical timing utility.

<!-- CLAIM:C004 -->
When Student timing is removed, the frozen strict visual audit remains sparse
and suite/state dependent: E3/E4 find two strict-valid parents among twelve
engineering-only parents, with one in `libero_10`, one in `libero_spatial`, and
none in `libero_goal` or `libero_object`. The primary E3/E4 unit is the parent;
the 72 candidate slots are diagnostic and non-iid.

<!-- CLAIM:C005 -->
The resulting contribution is a mechanism-first factorization account, not a
universal detector or a demonstrated cross-suite visual attack. X0 provides
bounded physical mechanism evidence; VI-B2 through VIII provide frozen
timing-selector negatives; IX separates model-side targetability from timing
utility; and E3/E4 bound the sparse model-side realizability regime.

<!-- CLAIM:C006 -->
No physical attack efficacy is established by this paper package, and Eval160
and protected evaluation remain unread. The paper therefore treats negative
results, censoring, provenance failures, and fail-closed governance as part of
the result rather than as invitations to tune the method.

## 1. Introduction

OpenVLA manipulation failures involving the gripper have at least three
distinct sources of difficulty. A clean trajectory can contain a phase in
which a gripper command matters. A command that increases the OPEN duty cycle
can then alter contact or object motion. Finally, a visual perturbation may or
may not be able to cause the policy to emit the desired gripper behavior while
preserving the non-gripper action coordinates. Treating these as a single
“vulnerability” score makes it difficult to tell whether a negative result is
a timing failure, a physical non-response, or a failure to realize a selective
visual intervention.

<!-- CLAIM:C101 -->
This study uses a mechanism-first design to keep those questions separate. The
positive physical result is an OPEN duty-cycle mechanism with dose and phase
structure. The negative results test whether clean timing selectors generalize
and whether model-side timing scores retain physical timing utility. The final
structural audit asks a narrower question: after removing Student timing, can
the frozen visual method realize exact gripper-selective OPEN at all on a
bounded engineering population?

<!-- CLAIM:C102 -->
The paper makes six bounded contributions:

<!-- CLAIM:C102A -->
1. a dose- and phase-dependent physical OPEN duty-cycle mechanism;
<!-- CLAIM:C102B -->
2. an explicit separation of `C_t`, `V_t(d)`, and `E_t`;
<!-- CLAIM:C102C -->
3. negative held-out and cross-suite evidence for frozen timing selectors;
<!-- CLAIM:C102D -->
4. a model-to-physics timing factorization gap;
<!-- CLAIM:C102E -->
5. sparse, suite/state-dependent strict selective visual realizability under
   the frozen method; and
<!-- CLAIM:C102F -->
6. a reproducibility record covering tokenizer semantics, victim provenance,
   censoring, branch replay, immutable histories, and fail-closed paper
   governance.

These contributions do not include a universal detector, cross-suite physical
visual-PGD efficacy, or protected validation.

## 2. Problem formulation: three estimands, not one score

Let `t` index a clean observation/history and let `d` denote the duration of a
command-OPEN counterfactual.

### 2.1 Clean criticality and opportunity: `C_t`

`C_t` denotes a clean, causal timing quantity: whether the current phase is a
plausible opportunity for a gripper-directed intervention to matter. It is a
timing/opportunity construct, not a physical outcome and not a proof that a
visual perturbation can be realized.

### 2.2 Physical vulnerability: `V_t(d)`

`V_t(d)` denotes the physical response under a duration-`d` command-OPEN
counterfactual. Stage X0 measures command delivery, aperture excess, contact
loss, and displacement in sealed probe data. This quantity is downstream of
the physical command semantics and is not interchangeable with clean timing
criticality.

### 2.3 Visual exploitability: `E_t`

`E_t` denotes model-side visual exploitability under a strict selective audit.
In E3, Student timing is deliberately absent from the selection path. A valid
candidate must preserve the exact arm action coordinates and move the gripper
from a clean non-OPEN state to native `NATIVE_OPEN` under the frozen method.
This is a structural model-side event. It is not a physical rollout, a task
outcome, or a `V_phys` label.

### 2.4 What is and is not inferred

The conceptual relation among these estimands is not asserted as a demonstrated
causal chain. In particular, the evidence does not establish `C_t -> V_t(d) ->
E_t`, formal mediation, or a universal relationship among the three. The paper
instead asks where alignment breaks.

## 3. Evidence hierarchy and experimental design

### 3.1 Evidence levels and primary units

<!-- CLAIM:C103 -->
The evidence map distinguishes primary bounded evidence, negative scientific
evidence, diagnostic-only evidence, and invalid/superseded histories. Each
stage keeps its source-declared population and denominator. E3/E4 use twelve
engineering-only parents as the primary unit. Their six ordered candidates per
parent yield 72 candidate slots, but those slots are not independent samples
and are never used as a parent-level inferential denominator.

There are no identity-level joins across X0, the bounded Black Bowl context,
VI-B2, VII, VIII, IX, E2, E3, and E4. The synthesis is stage-level and
claim-safe.

### 3.2 X0 physical mechanism

<!-- CLAIM:C104 -->
X0 contains 40 Stage V and 16 Stage VI-B2 parents, 1,344 probe groups, and
1,126 complete three-dose rows. The source-defined consumable counts are 1,245
at T3, 1,191 at T5, and 1,126 at T10. The downstream task-failure taxonomy
was not reconstructed from `V_phys`; the mechanism readout is therefore kept
separate from any task-failure interpretation.

The earlier Black Bowl fixed-window material is retained only as bounded
historical mechanism context. Its repository configuration specifies seeds and
reference windows, but the raw sealed outcome denominator is not identifiable
in the current checkout. It is not used as a primary quantitative population.

### 3.3 Timing-selector evidence

<!-- CLAIM:C105 -->
VI-B2 evaluates a frozen Student on fresh held-out parents. Its T5 primary
closure has 333 consumable rows and 51 abstain/censored rows. VII evaluates
three frozen development candidates under predeclared promotion gates. VIII R1
evaluates two relative-selector candidates with parent-grouped and leave-one-
suite-out requirements. These stages retain abstains and failed gates rather
than converting missing or censored evidence into negative labels.

### 3.4 Stage IX factorized no-environment audit

<!-- CLAIM:C106 -->
Stage IX contains 1,344 sealed no-environment rows. It compares model-side
scores with factorized parent-macro timing utility. No perturbed action is
stepped in the environment, no physical intervention is performed, and no
protected result is read.

### 3.5 E2, E3, and E4 structural boundaries

<!-- CLAIM:C107 -->
E2 is a bounded clean scheduler feasibility audit. Three Goal successor
identities had no legal Student emit, so no TRUE probe was started. E2 is not a
strict visual-method negative.

<!-- CLAIM:C108 -->
E3 evaluates twelve fresh engineering-only identities, with twelve clean
runtimes valid, twelve probes available, twelve TRUE invocations reached, and
twelve six-candidate audits complete. The frozen method uses the pre-registered
strict route, exact arm coordinates `[0:6]`, native `NATIVE_OPEN`, epsilon
0.03, step size 0.006, and five PGD iterations. E3 does not execute attacked
environment steps.

<!-- CLAIM:C109 -->
E4 performs only offline decomposition of the sealed E3 rows. It classifies
candidate-level structural outcomes and aggregates them at the parent level.
Its `attack_efficacy` field is false.

## 4. Results

### 4.1 X0: a dose- and phase-dependent OPEN mechanism

<!-- CLAIM:C201 -->
The X0 raw positive rates rise from 0.39438 at T3 to 0.67758 at T5 and 0.87300
at T10 among the source-defined consumable rows. Of the 1,126 complete
three-dose patterns, all are monotone and fall in `000`, `001`, `011`, or
`111`; no non-monotone pattern is observed.

<!-- CLAIM:C202 -->
The associated telemetry is mechanism-consistent: command delivery is exact
for eligible rows, aperture excess increases with dose, contact-loss incidence
increases, and object displacement increases. This supports a descriptive,
mechanistic interpretation of OPEN duty-cycle exposure.

<!-- CLAIM:C209 -->
The wording is intentionally limited. X0 does not provide a formal mediation
analysis. It does not establish that a clean timing score causes the physical
response, and its task-failure taxonomy remains unavailable rather than being
reconstructed from a downstream label.

### 4.2 VI-B2 through VIII: timing selectors do not generalize reliably

<!-- CLAIM:C203 -->
The held-out VI-B2 Student result has overall AUROC 0.6246432939, AUPRC lift
1.1911425664, and ECE-10 0.4606357016. Suite-level failures are material: the
`libero_10` emission rate is zero, `libero_object` has no negative consumable
examples for an identifiable AUROC, and overall calibration exceeds the frozen
limit. The frozen causal/actionable generalization conclusion is therefore not
established.

<!-- CLAIM:C204 -->
Stage VII reaches a negative development decision for all three frozen
candidates S7-A, S7-B, and S7-C: each fails at least one predeclared
cross-suite generalization or selectivity gate, and none is promoted. This is
not a claim that every runtime feature is uninformative.

<!-- CLAIM:C205 -->
Stage VIII R1 preserves a narrower negative. R1-A has DEVTEST parent-macro AUC
0.577615 and R1-B has 0.658572; both fail the frozen gate. Relative timing may
be descriptively identifiable within some source data, but the two frozen
selectors do not establish a generalizable deployment-facing relative
vulnerability selector.

Together these stages support the distinction between overall discrimination,
cross-suite generalization, and within-parent actionable timing. A score that
looks useful in one aggregation is not automatically an actionable scheduler.

### 4.3 Stage IX: model-side signal versus factorized timing utility

<!-- CLAIM:C206 -->
Stage IX model-side DEVTEST AUROC is 0.870743 for E0, 0.900510 for E1, and
0.897157 for E3. The corresponding factorized parent-macro AUC is 0.483698,
0.521112, and 0.523390. The factorized top-k and LOSO gates also remain
unsatisfied. Thus high model-side targetability scores did not establish
reliable physical timing utility.

This is a model-side/no-environment factorization result. It does not measure
physical attack efficacy and does not imply that a visual attack is impossible.

### 4.4 E3/E4: strict selective realizability without Student timing

<!-- CLAIM:C207 -->
E3 completes all twelve clean runtimes, probes, TRUE invocations, and
six-candidate audits. Two of twelve engineering-only parents have at least one
strict-valid candidate: one in `libero_10` and one in `libero_spatial`. The
`libero_goal` and `libero_object` suites have zero strict-valid parents.

<!-- CLAIM:C208 -->
E4 aggregates the same denominator into nine targetability-limited parents,
one joint-limited parent, and two strict-realizable parents. Every parent has
at least one exact-arm candidate, but only three of twelve parents have any
native-OPEN candidate. The candidate-slot decomposition is 4
`ARM_EXACT_AND_NATIVE_OPEN`, 29 `ARM_EXACT_BUT_NOT_NATIVE_OPEN`, 1
`NATIVE_OPEN_BUT_ARM_DRIFT`, and 38 `NEITHER_OPEN_NOR_ARM_EXACT` over 72
ordered slots; these counts are diagnostic secondary evidence, not iid
scientific samples.

The dominant bounded limitation is therefore targetability under the frozen
method, with one Goal parent showing a joint targetability/selectivity
conflict. This conclusion is model-side and structural. It does not say that
Goal or Object physical attacks are impossible, nor does it say that a
different method could not realize them.

## 5. Discussion

### 5.1 The factorization gap is the result

<!-- CLAIM:C302 -->
The positive and negative evidence is coherent when the layers are kept
separate. X0 shows that the physical system can respond to increased OPEN
duty-cycle exposure in a dose- and phase-dependent way. VI-B2 through VIII show
that clean timing selectors do not automatically yield stable deployment-facing
generalization. IX shows that a strong model-side score can coexist with near-
chance factorized timing utility. E3/E4 show that even when Student timing is
removed, strict visual realization is sparse and suite/state dependent under a
frozen method.

<!-- CLAIM:C301 -->
The evidence therefore supports a factorization-gap thesis: physical
vulnerability, clean timing criticality, and visual exploitability are related
questions but are not interchangeable evidence. A positive result at one layer
does not promote an untested claim at another layer.

### 5.2 Why the negative results are informative

The negative stages were not used to tune toward a desired attack outcome.
Thresholds, Student features, attack objective, epsilon, PGD steps, candidate
selection semantics, arm-isolation criterion, and parent pools remained frozen
within their respective authorizations. E2 was preserved as a scheduler hold;
it was not relabeled as a method failure. E3/E4 were stopped after the
pre-registered structural audit rather than expanded until a positive parent
appeared.

### 5.3 Engineering evidence and scientific evidence

Branch replay, direct-token equality, state equality, and candidate receipts
are valuable engineering evidence. They establish that the paired-branch
infrastructure and E3/E4 evidence persistence were sufficiently explicit for
the bounded audits. They do not convert a structurally valid model-side
candidate into a physical efficacy result. The same separation applies to
historical X1/X1R-V1 attempts: runtime-invalid or provenance-mismatched
histories are preserved for governance, not promoted as negative attack data.

## 6. Limitations

1. X0 provides a bounded physical mechanism result, not formal mediation and
   not a universal physical law.
2. VI-B2, VII, and VIII are source-specific frozen negative studies; they do
   not prove that every detector architecture fails.
3. Stage IX is a no-environment model-side audit. Its factorized utility is
   not a physical outcome.
4. E3/E4 use twelve engineering-only parents. The counts are descriptive and
   parent-level; candidate slots are not iid and no significance test is used.
5. E3/E4 do not establish physical efficacy, prevalence, or impossibility in
   any suite. Goal/Object remain scientifically unresolved for other methods.
6. The Black Bowl raw sealed outcome denominator is not identifiable in the
   current checkout, so it remains contextual rather than a primary table.
7. Stage VIII was recovered from immutable Git history rather than the current
   checkout; its direct handoff and blob binding are recorded in the authority
   map.
8. Historical X1 and X1R-V1 contain provenance/runtime defects and supply no
   efficacy estimate.
9. Eval160 and protected evaluation were not read. No claim about protected
   validation is made.

## 7. Reproducibility and governance

The paper bundle is generated from the authority map and sealed stage roots.
The official runtime environment is
`/mnt/sdc/dty_user/openvla_attack/envs/openvla-official-a800`, but P0--P4 paper
assembly uses CPU/static operations only.

The action-token boundary is recorded explicitly. Native generated-token
authority uses the repository's native endpoint; the tokenizer audit preserves
the non-bijective 31744/31745 boundary and does not silently round-trip a
decoded action through a surrogate token. Cached autoregressive behavior is
distinguished from any optimization surrogate. Candidate selection is frozen
before any outcome and the selected candidate is not re-encoded for execution.

The sealed sources and hashes are listed in:

- `paper/PAPER_V1_EVIDENCE_AUTHORITY_MAP_V1.json`;
- `paper/PAPER_V1_FIGURE_TABLE_ROOT_SEAL_V1.json`;
- `reports/STAGE_X1R2_E4_FACTORIZATION_FAILURE_DECOMPOSITION_20260821/E4_ROOT_SEAL_V1.json`;
- the supplement `paper/PAPER_V1_SUPPLEMENT_REPRODUCIBILITY.md`.

The governance rule is fail-closed: a missing source, denominator mismatch,
protected-boundary violation, invalid historical status, or claim requiring new
evidence blocks promotion rather than triggering a rerun or replacement.

## 8. Conclusion

<!-- CLAIM:C401 -->
OpenVLA gripper-targeted manipulation exhibits a measurable dose- and
phase-dependent OPEN duty-cycle mechanism, but that mechanism does not by itself
establish a generalizable clean timing selector or a reliable visual route to
physical intervention. The earned evidence instead shows a factorization gap:
model-side targetability and physical timing utility can diverge, and strict
timing-decoupled visual realizability is sparse and suite/state dependent under
the frozen method.

<!-- CLAIM:C402 -->
The strongest claim supported by the current evidence is therefore bounded and
mechanism-first. It does not include universal detection, cross-suite visual
attack efficacy, Goal/Object impossibility, formal mediation, or protected
validation. Further attack escalation would require a new scientific authority
and is outside this paper lock.

## Appendix map

- **Appendix A:** evidence hierarchy and claim boundary —
  `paper/tables/PAPER_V1_EVIDENCE_HIERARCHY.csv` and
  `paper/tables/PAPER_V1_CLAIM_BOUNDARY.csv`.
- **Appendix B:** X0 dose/mechanism data —
  `paper/data/PAPER_V1_FIGURE2_X0_DOSE_RESPONSE.csv`.
- **Appendix C:** timing-selector negative cascade —
  `paper/data/PAPER_V1_FIGURE3_TIMING_NEGATIVE_CASCADE.csv`.
- **Appendix D:** Stage IX factorization gap —
  `paper/data/PAPER_V1_FIGURE4_FACTORIZATION_GAP.csv`.
- **Appendix E:** E3/E4 parent and diagnostic candidate decomposition —
  `paper/tables/PAPER_V1_E3_E4_PARENT_REALIZABILITY.csv` and the sealed E4
  decomposition JSON.
- **Appendix F:** full reproducibility and immutable-history ledger —
  `paper/PAPER_V1_SUPPLEMENT_REPRODUCIBILITY.md`.

## Internal source references

This draft intentionally does not fabricate external bibliography. The
repository handoffs, sealed roots, source commits/trees, and hashes in the
authority map are the evidence references for this internal draft. External
literature citations should be added only after bibliographic verification
(`CITATION_TODO`), without changing the empirical claim ledger.
