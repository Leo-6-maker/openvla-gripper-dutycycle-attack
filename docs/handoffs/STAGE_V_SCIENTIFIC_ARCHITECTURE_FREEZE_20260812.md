# Stage V scientific architecture freeze — 2026-08-12

Status: `FROZEN_PROSPECTIVE_SCIENTIFIC_ARCHITECTURE`

This is an append-only scientific correction. It does not delete or rewrite
historical M4, Counterfactual/Hybrid Teacher, or R3 Teacher/Student evidence.
Those artifacts remain historical and are not silently promoted to the primary
paper pipeline.

## Primary pipeline

```text
CLEAN rollout
  -> privileged clean Teacher C_t
  -> causal Student C_hat_t trained only on clean Teacher labels
  -> held-out matched counterfactual V_t(d)
  -> timing / VIS / defense protocols later
```

The variables are separated as `C_t != V_t(d) != E_t`:

- `C_t` is clean-derived physical criticality from clean state and clean
  telemetry only. It is not a causal vulnerability label.
- `V_t(d)` is a held-out physical counterfactual outcome under frozen
  CONTROL/T3/T5/T10 conditions. Unknown or abstained outcomes are not
  negatives.
- `E_t` is visual exploitability and requires a separate VIS protocol.

The privileged Teacher may use clean physical state. The deployment-facing
Student may use only causal, deployment-visible observation/action/history,
gripper/eef/contact/phase/timing features. It may not use privileged Teacher
fields, M4 outcomes, attack/VIS/oracle/random outcomes, future or post-treatment
variables, identity/task leakage, or M4 dose as a training feature.

## Claim boundary

The main claim is two-tiered:

1. Physical phenomenon: a completed held-out M4 can test dose/state-dependent
   physical susceptibility inside the frozen critical-opportunity corridor.
2. Clean localization: a clean Teacher and clean-supervised causal Student may
   localize that susceptibility, but this is not established until their fresh
   primary evidence is evaluated against held-out `V_phys`.

The current 24-probe design contains only contact-positive
`CONTACT_MANIPULATION`, `CARRY`, and `ENGAGED_LIFT` states. It contains no
`PRE_CONTACT` or safe/background panel. Therefore it cannot support a
high-risk-versus-low-risk or critical-versus-safe enrichment claim. A new
negative-control panel would require a separate prospective protocol and was
not launched.

## Current evidence and gates

- Formal clean corridor: `29/40` stable parents; status
  `HOLD_FORMAL_M4_CORRIDOR_INSUFFICIENT`.
- Valid replenishment additions: `1 libero_10 + 6 libero_goal`; four
  `libero_spatial` parents remain.
- M4 intervention labels/outcomes: not started and not read.
- Historical R3 Teacher/Student artifacts exist, but their reports are
  development-only or coverage-HOLD and are `HISTORICAL_PRE_FREEZE_SECONDARY_OR_UNCLASSIFIED`.
- Current live audit found no project process, no compute app, and all eight
  A800 GPUs idle. No scientific runtime was started for this correction.
- Protected counters remain zero: protected reads, Eval160 reads, attack
  rollouts, and VIS/PGD attack rollouts.

## Required firewall and sequence

The final M4 parent set must be excluded from FIT, CAL, CHECK, threshold
selection, model selection, and outcome-informed redesign. The current known
36 identities are disjoint from FIT670 and all six G1 train/validation/test
manifests. They overlap the historical G10 identity registry in `36/36`, but
the G10 protocol records `G10_READ=false`; this is quarantined identity-level
overlap, not usable outcome evidence.

The legal sequence is:

1. finish the architecture and claim freeze;
2. obtain the four remaining spatial clean qualifications;
3. freeze one exact 40-parent manifest, split, and exact clean probe manifest;
4. rerun the complete firewall audit against the final primary FIT/CAL/CHECK
   manifests;
5. build and lock the clean Teacher/Student primary evidence package;
6. only then read held-out M4 outcomes and report the predeclared analysis.

No Teacher threshold or Student feature may be revised from M4 outcomes. The
old historical designs may be reported as secondary ablations only.

## Machine-readable artifacts

- [Architecture freeze](../../configs/STAGE_V_SCIENTIFIC_ARCHITECTURE_FREEZE_V1.json)
- [Claim/evidence matrix](../../reports/STAGE_V_CLAIM_EVIDENCE_ALIGNMENT_AUDIT_V1.json)
- [Primary data firewall audit](../../reports/STAGE_V_PRIMARY_DATA_FIREWALL_OVERLAP_AUDIT_V1.json)
- [M4 probe-support audit](../../reports/STAGE_V_M4_PROBE_SUPPORT_AUDIT_V1.json)
- [Current takeover update](STAGE_V_M4_CURRENT_TAKEOVER_UPDATE_20260812.md)

The repository source binding at freeze is commit
`eb05ae20ead190e91221ede6ae6d18fca70c2b30`, tree
`092f55c3fbce82421bfa8bca67cd192f321048d7`.

Semantic artifact SHA256 bindings:

- `configs/STAGE_V_SCIENTIFIC_ARCHITECTURE_FREEZE_V1.json`: `ef601cdfe5a4eadca07c2a7e372c8b2d66710c323a3697ef8da5eeadb17b6ee4`
- `reports/STAGE_V_CLAIM_EVIDENCE_ALIGNMENT_AUDIT_V1.json`: `896fbdb48b2a206846bf1dc171271d8a13bdaad2759a3a14133672ef4ca5b486`
- `reports/STAGE_V_PRIMARY_DATA_FIREWALL_OVERLAP_AUDIT_V1.json`: `0cadb59d01b860cfd9995499bb036e39918c1e84aaf1b8f8650e2a5d80a3ba3f`
- `reports/STAGE_V_M4_PROBE_SUPPORT_AUDIT_V1.json`: `98e977d5d22f294cc71b65ca5ff87b3c0c9749c0dfdd3d9524118087af8d2e59`

## Goal Addendum: Teacher–Student → Small Attack Matrix

This is an append-only extension of the scientific freeze. It extends the
Goal Mode endpoint from a clean detector to one small end-to-end validation,
but it does not authorize an attack run by itself.

### Clean-only Student must be frozen first

No attack rollout is legal until the primary Student is sealed from clean
Teacher supervision and deployment-visible causal inputs only. The freeze must
bind the Teacher semantic SHA, Teacher-label manifest SHA, Student FIT/CAL/TEST
identities, exact feature schema and ordering, normalization, architecture,
checkpoint SHA, threshold, persistence/hysteresis, cooldown/lockout, one-shot
emit behavior, abstention/no-emit semantics, runtime feature implementation
SHA, and source commit/tree. M4, attack, VIS, oracle, random, future,
post-treatment, identity, task, and privileged fields are excluded from
design and model selection. If any M4 outcome was read before this seal, the
affected model is exploratory and a new clean-only primary must be built.

The Student offline gate must separately check causal replay, no future input,
no privileged runtime state, finite inputs, preprocessing/checkpoint parity,
determinism, and scheduler semantics. Teacher localization metrics and
held-out causal checks are evaluated only against the frozen definitions. If
the current M4 support cannot identify the desired comparison, report
NOT_IDENTIFIABLE_FROM_CURRENT_M4_PROTOCOL; do not turn an underpowered M4
comparison into a causal claim. This does not by itself block an explicitly
exploratory engineering canary, but it limits the claim.

### Protocol provenance is a hard gate

Attack parameters must be obtained mechanically from the current
owner-approved GitHub/receipt chain. They must not be reconstructed from
memory, chat text, old reports, copied commands, or historical performance.
The binding must include the protocol artifact and SHA256, budget seal,
executor and route-contract SHAs, model/checkpoint and preprocessing SHAs,
target action/token and gripper semantics, temporal/K/history rules,
random-control construction and salt/seed, decode configuration, and runtime
authorization. The current audit is recorded in
STAGE_V_SMALL_ATTACK_FROZEN_PROTOCOL_BINDING_V1.json and is deliberately
HOLD_AUTHORITATIVE_OWNER_APPROVAL_UNRESOLVED. No attack protocol has been
selected and no attack runtime may start until this ambiguity is resolved.

Once bound, the payload is immutable. Do not retune epsilon, norm, steps, step
size, K, target, loss, arm treatment, preprocessing, decode path, temporal
history, random budget, or perturbation mask from Student or attack outcomes.
A weak or negative frozen payload result is valid evidence.

### Small held-out matrix

| Arm | Timing | Payload | Purpose |
| --- | --- | --- | --- |
| C0 CLEAN_REPLAY | registered clean branch | none | clean baseline |
| C1 TRUE @ STUDENT_TIME | Student emit state | frozen targeted visual payload | end-to-end method |
| C2 RAND @ STUDENT_TIME | same Student emit state | matched random payload | payload specificity |
| C3 TRUE @ RANDOM_TIME | prospective random eligible state | same frozen targeted payload | timing specificity |

Add C4 COMMAND_OPEN_ORACLE @ STUDENT_TIME only when a compatible direct-OPEN
oracle is already frozen; it is a physical ceiling reference, not a replacement
for either matched control. Do not add Early/Late, shuffled-gradient, arm-only,
TMA/UADA/UPA, new objectives, budgets, or K values to this canary.

The preferred sample is eight held-out parents, ideally two per suite when
mechanism eligibility supports it, yielding 32 primary branches (40 only with
C4). Do not force unsupported tasks to fill quotas. Select parents before any
attack outcome by the frozen deterministic salt plus canonical identity rule,
with suite/mechanism stratification only when preregistered. Keep Student FIT,
CAL, threshold-tuning, M4 development/replenishment, and protected Eval160
identities out of the pool.

All preregistered parents remain in the disposition denominator. Record
STUDENT_EMIT, NO_EMIT, ABSTAIN, and RUNTIME_INVALID; never lower a threshold
or replace a no-emit parent after inspection. Report both ITT policy-level
accounting and the separate conditional-on-valid-emission matched analysis.

Random timing must be prospectively generated, legal, and outcome-blind, with
the matching dimensions explicitly recorded. Prefer same-prefix/same-state
branching. If the frozen executor cannot support exact snapshot branching
without changing its semantics, use a protocol-preserving adapter or an
authorized matched independent rollout and label it honestly; independent
rollouts are not byte-exact counterfactuals. Before delivery, require parent,
initial state, source commit/tree, model/checkpoint, prompt, preprocessing,
Student checkpoint, emit/branch state, clean prefix, and required RNG parity.
Hold any branch that fails parity.

The TRUE arm must call the existing frozen visual executor through the
adversarial re-decode and environment path. It must not be replaced by direct
actuator mutation. A direct OPEN oracle remains a separate control.

### Telemetry, estimands, and claim boundary

Each branch records Student score and scheduler disposition; protocol SHA,
norm, iteration/delivery count, K and active frames; clean/adversarial
gripper action or token, OPEN count/duty/streak; aperture/qpos and contact,
support, object-motion/release and latency evidence; official task result and
physical-failure taxonomy; and full provenance including commit/tree,
checkpoint, runner, parent/branch identity, runtime/GPU receipt, and protected
counters. Official LIBERO success is not the sole physical outcome.

The preregistered estimands are C1−C2 (payload specificity), C1−C3 (timing
specificity), and C1−C0 (end-to-end effect), with C4 descriptive only. After
outcomes are observed, exposed parents are permanently ATTACK_EVAL_EXPOSED
and cannot tune the Teacher, Student, threshold, persistence, attack
budget/objective, or random-time rule. Any revision is a new protocol with new
development and held-out identities; no rerun-to-pass.

Canary completion is a valid PASS when engineering validity, parity, protocol
compliance, protected-data safety, and evidence audit pass and the result is
reported faithfully. A negative or contradictory canary is a valid scientific
completion, not a reason to retune. A consistent adverse direction may
support design of a larger matrix, but a larger matrix is not auto-launched.
The matrix does not establish broad generalization, SOTA, universal
vulnerability, final defense effectiveness, or causal Student validity
without the relevant evidence.

### Required artifact sequence

Before launch, seal:

- STAGE_V_SMALL_ATTACK_MATRIX_PROTOCOL_V1.json
- STAGE_V_SMALL_ATTACK_PARENT_MANIFEST_V1.json
- STAGE_V_SMALL_ATTACK_FROZEN_PROTOCOL_BINDING_V1.json
- STAGE_V_SMALL_ATTACK_STATIC_AUDIT_V1.json
- STAGE_V_SMALL_ATTACK_RUNTIME_AUTHORIZATION_V1.json

After execution, seal per-branch receipts/audits, the aggregate and
independent audit, report, SHA256SUMS, and a handoff update. The terminal
state must preserve preregistered N, emits, no-emits, valid matched sets,
runtime-invalid branches, task/contact failures, and all denominators. Current
terminal state is ATTACK_PROTOCOL_BINDING=HOLD; no Teacher/Student attack
runtime, M4 outcome read, VIS/PGD rollout, or protected Eval160 read has been
started for this correction.
