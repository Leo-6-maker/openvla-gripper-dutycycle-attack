# R10.4E Two-Hour Autonomous Handoff

## Current scientific state

The one-parent R10.4D passive smoke completed as `PASS_RUNTIME_NO_EMIT` on
`libero_10/task_00/state_20` with 520 policy steps, exactly one generation per
step, valid finite causal 25D features, zero action mutation, zero structural
FSM violations, and zero Student emits. Task failure is descriptive and is not
a runtime-contract failure.

R4C established that the prior replay mismatch was direct physical-state
replay divergence, not a Student, feature-adapter, or training-source binding
failure. Teacher, Student, checkpoint, threshold, and FSM therefore remain
frozen.

## Autonomous authorization objective

During the bounded authorization window, DeepSeek may progress without asking
for an intermediate approval through the following sequence only:

1. prepare and statically audit the frozen ten-task passive panel;
2. execute the nine new passive episodes on physical GPU0;
3. combine them with the sealed task00 R10.4D smoke;
4. run an offline privileged-Teacher versus online-Student matched audit;
5. only when every command-open preparation gate passes, create static
   command-OPEN canary preparation code and a Draft PR.

Actual command-OPEN, VIS, RAND, attacks, retraining, threshold changes, FSM
changes, parent substitution, retries, PR merge, and Ready-for-review remain
prohibited.

## Phase A — evidence and implementation closure

### A0. Revalidate the consumed R10.4D smoke

Verify the sealed task00 output root, `SHA256SUMS`, sidecar, source HEAD,
authorization receipt, model tree, detector checkpoint, parent BDDL and init
state. Do not rerun task00.

Failure status: `HOLD_R10_4D_EVIDENCE_BINDING`.

### A1. Build the panel runner

Use branch `codex/r10-4e-passive-panel-20260720`, stacked on the exact R10.4D
preparation HEAD. The runner must reuse the R10.4D official OpenVLA adapter,
official image helper, official postprocess, canonical
`SC5StreamingFeatureAdapterV2`, frozen dual-head Student, and frozen FSM.

The model may be loaded once and retained for the nine new episodes. Each
episode must construct a fresh LIBERO environment and reset all of:

- feature history;
- GRU hidden state;
- persistence counters;
- FSM state and event ID;
- privileged sidecar buffers.

The execution order is canonical and immutable:

```text
libero_10/task_01/state_20
libero_10/task_02/state_20
...
libero_10/task_09/state_20
```

No task00 rerun, retry, parent replacement, reordering based on outcomes, or
use of state21+ is permitted.

### A2. Required CPU/static tests

At minimum test:

- exact manifest of ten unique identities;
- one reused and nine newly authorized episodes;
- task00 cannot be launched;
- task01–09 can each launch at most once;
- one model-load call for the panel process;
- strict checkpoint load and 46,658 parameters;
- missing, boolean, zero and multiple generation passes fail closed;
- exact seven-dimensional clean/executed action equality;
- feature, GRU and FSM reset between episodes;
- no hidden-state leakage across tasks;
- unsupported parent and route rejection;
- non-overwrite per-episode and aggregate roots;
- privileged information cannot enter Student/FSM inputs;
- no command-OPEN, VIS, RAND or attack path;
- panel execution cannot begin without a machine-built receipt bound to the
  exact clean HEAD and Issue authorization comment.

Create a stacked Draft PR. All relevant CPU workflows must pass at the exact
execution HEAD before model load.

## Phase B — nine new real passive episodes

### B0. Runtime scope

- physical GPU0 and render GPU0 only;
- pinned OpenVLA checkpoint and upstream checkout;
- one resident 7B model load;
- nine new episodes, each exactly once;
- official horizon 520 and ten dummy-open steps;
- passive observation only;
- no action mutation;
- no attack or override code;
- task success is not required;
- Student emit is not required.

### B1. Per-step hard invariants

Every policy step must have:

- exactly one measured generation pass;
- raw uint8 official image input;
- finite float32 25D vector in frozen order;
- one Student forward update;
- one FSM update;
- `executed_action == official_postprocessed_clean_action` in all seven
  dimensions with maximum absolute error exactly zero;
- privileged sidecar collection after Student/FSM execution only.

### B2. Global hard-stop conditions

Immediately stop the panel, preserve the failing root, and do not start the
next task on any of:

- source, receipt, model, detector, parent, BDDL or init-state mismatch;
- worktree dirty or wrong HEAD;
- checkpoint strict-load or parameter-count failure;
- generation pass missing, boolean, zero or not one;
- invalid or non-finite feature;
- any nonzero action mutation;
- unsupported route, pre-anchor emit, release-overlap emit or duplicate emit;
- privileged leakage;
- OOM, EGL, filesystem, environment, serialization or checksum-seal failure;
- output overwrite attempt.

No automatic retry is authorized, including infrastructure failures.

### B3. Runtime gate

The passive panel runtime gate passes only when the reused task00 evidence and
all nine new episodes provide:

```text
runtime-valid episodes                 = 10/10
action mutations                       = 0
generation-contract failures           = 0
invalid/non-finite feature steps        = 0
route/FSM/privileged structural errors  = 0
maximum emits per episode               <= 1
```

Failure status: `HOLD_RUNTIME_CONTRACT`.

## Phase C — offline Teacher–Student matched audit

Teacher labels must be materialized only after each rollout from its isolated
privileged sidecar, using the frozen R10 `grasp_established` Teacher contract.
No privileged values may be reintroduced into runtime records or used to
change the Student, threshold, FSM, parent set, or execution order.

### C1. Per-task outputs

Produce for all ten tasks:

- runtime status, policy steps and task success;
- Teacher grasp-established intervals and release boundaries;
- Student probability trace, threshold persistence and maximum probability;
- Student emit count and step;
- whether each emit lies inside a Teacher active interval and before release;
- onset latency for matched emits;
- structural violations;
- a no-emit category:
  - `NO_TEACHER_EVENT`;
  - `TEACHER_EVENT_STUDENT_BELOW_THRESHOLD`;
  - `STUDENT_POSITIVE_FSM_NO_LIFT`;
  - `RELEASE_OR_ROUTE_BLOCK`;
  - `EMIT_MATCHED`.

### C2. Aggregate outputs

Report:

- tasks with Teacher events;
- tasks with Student emits;
- Teacher event count and Student emit count;
- event-level recall and precision;
- latency distribution;
- clean-success tasks;
- clean-success tasks with a matched Student emit;
- task-level no-emit taxonomy;
- all runtime and safety invariants.

### C3. Conditional command-OPEN preparation gate

Static command-OPEN engineering-canary preparation may begin only when all are
true:

```text
runtime-valid episodes                      = 10/10
Teacher-event tasks                         >= 3
Student-emit tasks                          >= 2
Student emit precision inside Teacher event = 100%
clean-success matched-emit tasks             >= 1
FSM structural violations                    = 0
action mutations                             = 0
```

Gate outcomes:

- Teacher-event tasks < 3:
  `HOLD_PANEL_OPPORTUNITY_SPARSE`.
- Teacher events in at least three tasks but Student emits in fewer than two:
  `HOLD_STUDENT_DEPLOYMENT_RECALL`.
- Any unmatched, pre-anchor, release-overlap or duplicate emit:
  `HOLD_TIMING_SAFETY`.
- No clean-success task with a matched emit:
  `HOLD_NO_CLEAN_SUCCESS_ATTACK_PARENT`.
- All conditions pass:
  `PASS_COMMAND_OPEN_STATIC_PREP_GATE`.

A gate pass permits only source code, protocol, manifests, fake tests, static
audits and a Draft PR for one engineering canary. It does not permit a real
command-OPEN execution.

## Phase D — optional static command-OPEN preparation

This phase may start during the authorization window only after
`PASS_COMMAND_OPEN_STATIC_PREP_GATE`.

Freeze the engineering-canary parent as the lexicographically smallest
canonical identity among the clean-success tasks with a matched Student emit.
This is explicitly an engineering selection rule, not a formal scientific
panel selection.

Preparation must include:

- a passive-clean versus Student-timed command-OPEN engineering protocol;
- exact command amplitude, duration and postprocess layer;
- a random-timing negative-control design for later authorization;
- physical qpos/opening response metrics;
- action-dimension mutation ledger;
- stop-on-first-execution contract;
- CPU fake tests and an independent auditor;
- explicit `command_open_execution_authorized=false`.

Do not execute it in this authorization window.

## Authorization-window semantics

- No new phase, episode, model load, or rollout may start after the expiry time
  written in the controlling Issue #88 authorization comment.
- Work already executing at the expiry may finish only its current episode and
  seal that episode; it may not begin another parent afterward.
- Offline hashing, sealing, aggregation and report writing for already completed
  work may finish after expiry.
- Any ambiguity is a HOLD, not permission.

## Final report

Return one report containing:

```text
exact source HEAD and clean worktree
Draft PR and workflow results
model tree and detector/bundle digests
R10.4D reused evidence digest
panel receipt and manifest digests
per-task runtime table
per-task Teacher–Student table
aggregate gates and exact numerators/denominators
all output roots and SHA256SUMS digests
final status
```

The final status must be exactly one of:

```text
PASS_COMMAND_OPEN_STATIC_PREP_GATE
PASS_PASSIVE_PANEL_ONLY
HOLD_R10_4D_EVIDENCE_BINDING
HOLD_RUNTIME_CONTRACT
HOLD_PANEL_OPPORTUNITY_SPARSE
HOLD_STUDENT_DEPLOYMENT_RECALL
HOLD_TIMING_SAFETY
HOLD_NO_CLEAN_SUCCESS_ATTACK_PARENT
```

## Long-range roadmap after this window

1. **R10.4F command-OPEN engineering canary** — one pre-registered parent,
   physical opening verification only.
2. **R10.5 formal command-OPEN causal panel** — frozen clean, Student-timed,
   Teacher-timed and random-timing controls; induced failure and physical
   response as separate endpoints.
3. **R10.6 targeted VIS canary** — verify that visual optimization changes the
   gripper command through the intended mechanism.
4. **R10.7 formal VIS specificity panel** — targeted VIS versus random direction
   and random timing, with no attack-outcome-driven tuning.
5. **Defense evaluation** — frozen command-layer guard or abstention policy,
   measured against clean utility and physical attack suppression.
6. **Paper closure** — separate opportunity detection, physical actuation,
   induced task failure and visual specificity claims; retain all negative and
   no-emit results.
