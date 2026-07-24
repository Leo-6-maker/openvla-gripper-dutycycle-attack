# Codex Experiment Plan V1

Status: REVIEWED_PLANNING_ONLY

This document plans repository-side implementation and later scientific
execution for the OpenVLA gripper duty-cycle attack project. It does not
activate the Label V2 build, detector training, OpenVLA inference, simulator
rollout, attack execution, or GPU jobs.

## 1. Current Gate State

```text
LABEL_V2_AUTHORIZATION_RECORD_GIT_COMMIT = PASS
LABEL_V2_AUTHORIZATION_RECORD_FILE_SHA256 = PASS
LABEL_V2_SERVER_BINDING_HANDOFF = PASS_BOUND
LABEL_V2_ONE_SHOT_ACTIVATION_READINESS = PASS_PREFLIGHT
LABEL_V2_BUILD_EXECUTION_AUTHORIZATION = NOT_AUTHORIZED

CODEX_REPOSITORY_IMPLEMENTATION = AUTHORIZED_AFTER_PLAN_REVIEW
CODEX_SERVER_EXECUTION = NOT_AUTHORIZED
A800_GPU_EXPERIMENT_EXECUTION = NOT_AUTHORIZED
GATE_A2_DETECTOR = HOLD
GATE_A3_ATTACK = HOLD
EXPERIMENT_AUTHORIZATION_STATUS = NOT_AUTHORIZED
```

## 2. Scientific Objective

The confirmatory claim is intentionally narrow:

> A clean-only causal detector can identify a deployment-facing gripper-contact
> phase, and a matched inference-time visual intervention triggered at that
> phase can selectively increase gripper OPEN duty, produce a measurable
> gripper qpos/width response, degrade contact quality, and increase task failure
> more than timing- and payload-matched controls under the frozen OpenVLA/LIBERO
> protocol.

The evidence chain is:

```text
clean-only Label V2
  -> causal detector/FSM
  -> exact-prefix trigger
  -> gripper-targeted visual attack
  -> OPEN-duty increase
  -> qpos/width physical response
  -> contact-quality degradation
  -> task outcome
```

## 3. Non-goals

The following are outside the confirmatory scope:

- universal VLA generalization;
- arbitrary robot embodiments;
- training-time backdoors;
- text attacks or physical patch attacks in the main ASR leaderboard;
- post-hoc tuning on test suites;
- replacing the frozen primary detector after inspecting attack results;
- using privileged teacher state during deployment;
- treating official LIBERO success as the only outcome;
- treating frame-level samples or perturbation seeds as independent parents.

## 4. Immutable Scientific Inputs

Codex must treat these documents as controlling specifications:

```text
CLEAN2000_LABEL_V2_SPEC.md
POPULATION_DEFINITION_V1.md
SPLIT_AND_LEAKAGE_SPEC.md
DETECTOR_PROTOCOL_V1.md
EXACT_PREFIX_BRANCHING_SPEC_V1.md
ATTACK_PROTOCOL_V1.md
BASELINE_PROTOCOL_V1.md
CONTACT_QUALITY_PROTOCOL_V1.md
METRIC_DEFINITIONS_V1.md
STATISTICAL_ANALYSIS_PLAN_V1.md
EXPERIMENT_MATRIX_V2.csv
```

Codex may identify conflicts among them but may not silently change a frozen
scientific choice. Any scientific change requires a separate planning commit and
review before implementation.

## 5. Phase and Gate Sequence

```text
C0 repository audit and implementation plan
  -> C1 Label V2 artifact freeze
  -> C2 detector dataset/split closure
  -> C3 detector implementation and CPU tests
  -> C4 A800 topology qualification
  -> C5 detector training/evaluation
  -> C6 Gate A2 freeze
  -> C7 exact-prefix attack canary
  -> C8 confirmatory attack matrix
  -> C9 contact-quality/manual audit
  -> C10 statistical analysis and figures
```

No phase may start server-side execution before its explicit authorization.

---

# C0. Repository Audit and Implementation Plan

## Codex tasks

1. Audit all scripts referenced by the frozen protocol.
2. Produce a path inventory for:
   - detector training/evaluation;
   - feature and Label V2 joins;
   - split generation;
   - exact-prefix snapshot/restore;
   - attack objectives and matched controls;
   - runtime telemetry;
   - CQ evaluation;
   - artifact manifests and validators.
3. Mark every path as one of:

```text
EXISTS_AND_REVIEWED
EXISTS_NEEDS_HARDENING
MISSING_IMPLEMENTATION
LEGACY_NOT_FORMAL
```

4. Produce a dependency graph from source artifact to paper table cell.
5. Do not run a server command.

## Required deliverables

```text
docs/paper_driven_multisuite_confirmatory_v1/CODEX_REPOSITORY_AUDIT_V1.md
docs/paper_driven_multisuite_confirmatory_v1/CODEX_IMPLEMENTATION_GAP_MATRIX_V1.csv
```

## Gate C0

```text
all formal paths enumerated
all missing components identified
no scientific setting changed
no server execution
```

---

# C1. Label V2 Artifact Freeze

This phase is not a Codex implementation task. It is the separately bound
CPU-only one-shot operation controlled by PR #49.

## Fixed sequence

```text
read-only preflight
-> exactly one formal-ledger-build
-> exactly one validate-formal-output
-> 160-row human manual audit
-> Gate A1 artifact freeze
```

## Codex role before execution

Codex may only:

- verify that downstream code accepts the documented five-file artifact;
- add parser/unit tests using synthetic fixtures;
- define a read-only ingestion validator;
- prepare but not execute downstream commands.

## Gate C1

```text
LABEL_V2_FORMAL_BUILD = PASS
LABEL_V2_INDEPENDENT_VALIDATION = PASS
LABEL_V2_MANUAL_AUDIT = PASS
GATE_A1_LABEL_ARTIFACT = PASS_FROZEN
```

No detector training starts before Gate C1.

---

# C2. Detector Dataset and Split Closure

## Inputs

- frozen Label V2 five-file artifact;
- frozen clean feature artifact;
- canonical 25-feature order `SC5_FEATURES`;
- frozen parent/state identities.

## Codex implementation tasks

1. Implement a deterministic join keyed by `episode_key`.
2. Verify exact-set closure and fail on missing/duplicate rows.
3. Enforce `parent_key` and initial-state hash as split units.
4. Produce the three required split manifests:

```text
parent_random_split_v1
object_leave_task_out_v1
suite_loso_split_v1
```

5. Compute normalization from training partitions only.
6. Prevent attack rollouts, task outcomes from attack conditions, or future
   telemetry from entering detector features.
7. Build detector populations:

```text
DETECTOR_ELIGIBLE
DETECTOR_SAFETY
DETECTOR_MULTI_EVENT
```

8. Emit a leakage audit and manifest SHA256 values.

## Required outputs

```text
artifacts/manifests/detector_dataset_manifest_v1.csv
artifacts/manifests/parent_random_split_v1.csv
artifacts/manifests/object_leave_task_out_v1.csv
artifacts/manifests/suite_loso_split_v1.csv
artifacts/manifests/detector_normalization_v1.json
artifacts/manifests/detector_dataset_validation_v1.json
```

Formal artifacts remain outside Git; Git stores only schema, builders,
validators, tests, and authorization records.

## Gate C2

```text
row-set closure = PASS
parent leakage = 0
state-hash leakage = 0
normalization leakage = 0
feature order = exact 25D SC5_FEATURES
future/attack telemetry leakage = 0
```

---

# C3. Detector Implementation and CPU Test Closure

## Frozen primary detector

```text
model = SC5MLPV1
features = 25 canonical causal features
hidden dimensions = 64, 64
heads = phase(9), corridor(1), release(1)
loss = phase CE + 0.5 corridor BCE(pos_weight=5.0) + 0.3 release BCE
threshold defaults = tau_corridor 0.3, tau_release 0.3
FSM = legacy_v1
guard duration = 5 steps
```

## Codex implementation tasks

1. Harden train/eval CLIs to require manifest and SHA bindings.
2. Add deterministic seed handling and environment capture.
3. Add checkpoint provenance:

```text
source_git_sha
source_file_sha256
split_sha256
normalization_sha256
feature_schema_sha256
training_config_sha256
```

4. Implement event-level metrics:
   - event precision/recall/F1;
   - false-trigger rate;
   - signed and absolute onset error;
   - +/-10-step recall;
   - ineligible abstention;
   - no-emit rate;
   - per-suite and suite-macro results.
5. Implement validation-only threshold selection.
6. Add negative tests for leakage, target-suite normalization, test threshold
   tuning, unknown schema, missing manifest, and dirty producer identity.
7. Add synthetic CPU end-to-end training/evaluation smoke.

## Baselines

```text
fixed normalized time
close-onset heuristic
rule-based proprio
logistic regression or shallow MLP
one lightweight temporal model
privileged teacher upper bound, offline only
```

TCN or revocable FSM variants are ablations and cannot replace the primary
model after results are inspected.

## Gate C3

```text
all unit tests PASS
synthetic detector smoke PASS
manifest identity PASS
threshold-selection isolation PASS
no OpenVLA inference
no simulator rollout
```

---

# C4. A800 Topology Qualification

This phase requires a separate server/GPU authorization.

## Preferred topology

```text
8 independent one-GPU workers
```

If a single worker cannot fit OpenVLA plus attack gradients, the only permitted
fallback is:

```text
4 fixed ordered two-GPU workers: (0,1), (2,3), (4,5), (6,7)
```

The topology must be selected before scientific outcomes are inspected.

## Qualification checks

Per GPU/worker:

```text
GPU UUID and model
CUDA, driver, PyTorch, transformers versions
victim checkpoint SHA
clean forward checksum/tolerance
clean rollout smoke
backward/PGD smoke
peak allocated/reserved memory
step latency
ECC/Xid status
worker output isolation
```

## Assignment rule

Matched branches for the same `parent_key` must execute on the same worker.
A deterministic assignment is required, for example:

```text
worker_id = int(sha256(parent_key)[:8], 16) % worker_count
```

Conditions must not be assigned to different GPUs in a way that confounds
method with hardware.

## Gate C4

```text
A800_TOPOLOGY_QUALIFICATION = PASS
A800_WORKER_PARITY = PASS
A800_OUTPUT_ISOLATION = PASS
```

---

# C5. Detector Training and Evaluation

This phase requires Gate C1, C2, C3, and explicit detector-training
authorization. It performs no attack rollout.

## Formal run matrix

```text
Object-only x 3 seeds
Pooled four-suite x 3 seeds
LOSO x 4 held-out suites x 3 seeds
Object leave-one-task-out x 10 folds x 3 seeds
```

Total: 48 lightweight detector runs.

## Suggested eight-GPU scheduling

Detector runs are lightweight; parallelize by run identity, not by data-parallel
training:

```text
GPU0: Object-only and pooled seed 0 queue
GPU1: Object-only and pooled seed 1 queue
GPU2: Object-only and pooled seed 2 queue
GPU3: LOSO Object / Spatial queue
GPU4: LOSO Goal / LIBERO-10 queue
GPU5: Object LOTO folds 0-3
GPU6: Object LOTO folds 4-6
GPU7: Object LOTO folds 7-9
```

The scheduler must enforce one run per GPU unless measured memory permits a
reviewed change.

## Gate C5 / Gate A2 thresholds

Use the frozen detector protocol thresholds:

```text
Object held-out event recall >= 0.70
Object held-out +/-10 recall >= 0.65
Object held-out false-trigger rate <= 0.20
Object held-out median timing error <= 10 steps
Cross-suite macro event recall >= 0.60
Each eligible suite recall >= 0.50
Cross-suite false-trigger rate <= 0.30
```

A suite that fails may not enter formal detector-triggered attack. It may only
enter a separately authorized teacher-timing mechanism smoke.

## Frozen detector outputs

```text
selected checkpoint
training/evaluation config
split and normalization SHA
threshold/FSM config
per-episode predictions
per-suite metrics
checkpoint SHA256
environment manifest
```

---

# C6. Gate A2 Freeze

A separate review must decide:

```text
GATE_A2_DETECTOR = PASS_FROZEN
or
GATE_A2_DETECTOR = HOLD_WITH_FAILED_SUITES
```

No attack protocol, threshold, detector architecture, or normalization may be
changed after test results are inspected without restarting the confirmatory
freeze.

---

# C7. Exact-prefix Attack Canary

This phase requires explicit attack-smoke authorization and Gate A2 for each
suite using Student timing.

## Population

Eight held-out primary parents:

```text
Object: 2
Spatial: 2
Goal: 2
LIBERO-10: 2
```

Each must satisfy:

```text
clean success
mechanism eligible
V2 positive event
valid exact-prefix snapshot
Student causal emission valid
```

## Branch family per parent

```text
CLEAN_EXACT_PREFIX_REPLAY
OURS_STUDENT_GRIPPER_TARGET
RAND_DIRECTION
RANDOM_TIME
ADAPTED_TMA_OPEN
```

Total initial canary budget:

```text
8 parents x 5 branches = 40 suffix branches
```

## Canary abort conditions

```text
prefix or simulator-state hash mismatch
off-by-one attack onset
clean replay parity failure
missing actual Linf telemetry
attack frame count != K
matched budget mismatch
worker crash or incomplete branch family
CQ telemetry missing
```

A failed family is preserved as failed evidence and is not silently replaced.

---

# C8. Confirmatory Attack Matrix

## Population

Use `PRIMARY_ATTACK` only:

```text
clean success
mechanism eligible
V2 positive event
exact-prefix reproducible
```

Target per suite:

```text
20 parents
maximum 3 parents per task
minimum 5 eligible tasks where available
```

Use all legal parents and report the shortfall if a suite cannot reach 20.

## Main conditions

```text
Clean exact-prefix replay
Ours: detector-triggered gripper target
RAND_DIRECTION
RANDOM_TIME
Adapted TMA-OPEN
```

## Frozen attack parameters

```text
K = 10
PGD steps = 20
one preprocessing backend
one gripper-only objective
one threshold/FSM
one global attack parameter set across suites
```

Epsilon may be selected once on an independent calibration split from:

```text
2/255, 4/255, 6/255
```

Choose the smallest epsilon satisfying the pre-registered gripper-duty effect,
weak matched RAND, arm-NAD ceiling, and complete actual-Linf telemetry. Test
suites cannot retune epsilon.

## Primary contrasts

```text
Ours vs RAND_DIRECTION
Ours vs RANDOM_TIME
Ours vs Adapted TMA-OPEN
```

Ours vs Clean is an attack-effect baseline outside the primary
multiple-testing family.

## ITT rule

```text
detector no emit
-> attack not executed
-> parent remains in ITT with its observed outcome
```

Emitted-only results are auxiliary.

---

# C9. Contact-quality and Manual Audit

Official simulator success is a compatibility metric, not the sole primary
outcome.

## Automatic CQ flags

```text
premature_release
drop_after_lift
object_eef_detach
unstable_transport
uncontrolled_final_drop
CQ_TELEMETRY_MISSING
```

## Blind manual review

Review:

```text
all automatic CQ positives
all Official SR / CQ disagreements
all Ours and Oracle failures
20% random CQ negatives for every suite/method
```

At least 20% of reviewed videos require an independent second reviewer. Report
Cohen's kappa. If kappa < 0.80, Table 1 CQFR uses manual labels or the audit
sample expands.

---

# C10. Statistical Analysis and Figure Production

## Binary outcomes

Report:

```text
numerator / denominator
paired risk difference
95% task-and-parent cluster bootstrap CI
exact McNemar test
Holm correction for the three primary contrasts
```

## Continuous outcomes

Use paired bootstrap or paired nonparametric tests for:

```text
OPEN duty
longest OPEN streak
gripper NAD
qpos/width response
command-to-qpos latency
arm NAD
actual Linf
runtime
```

Perturbation seeds are not independent parent samples.

## Planned main outputs

```text
Figure 1: threat model and end-to-end pipeline
Table 1: matched task/contact and mechanism results
Figure 2: Student -> OPEN duty -> qpos -> contact timeline
Figure 3: timing and payload specificity
Figure 4: cross-suite paired forest plot
Figure 5: command-to-physical bridge
Table 2: detector performance
Table 3: deployment overhead
```

---

# 6. Storage and Evidence Policy

Because `/mnt/sdc` is already highly utilized:

## Always retain

```text
configs and manifests
per-step numeric telemetry
terminal outcome records
artifact SHA256 files
stderr/stdout and exit status
selected compressed videos required for audit
```

## Do not retain by default

```text
full uncompressed PNG sequences
multiple duplicate checkpoint caches
unbounded debug dumps
artifacts inside the Git worktree
```

## Stop rule

The scheduler must stop dispatching new work when the bound free-space threshold
is reached. Existing evidence must not be deleted to create space without a
separate evidence-retention review.

---

# 7. Codex Repository-only Authorization

After review of this plan, Codex is authorized to perform only the following on
the planning/implementation branch:

```text
repository inspection
schema and manifest builders
validators
CLI hardening
unit and synthetic integration tests
CPU CI
planning and authorization documents
```

Codex is not authorized to:

```text
run the formal Label V2 build
read or mutate live scientific artifacts
train a detector on the server
load OpenVLA for inference
run LIBERO rollouts
execute attacks
reserve or use A800 GPUs
change frozen scientific settings without review
```

## Required Codex commit discipline

Each implementation batch must:

1. cite its task IDs from `CODEX_TASK_MATRIX_V1.csv`;
2. modify the smallest coherent file set;
3. include exact tests and outputs;
4. keep all server/GPU execution statuses `NOT_AUTHORIZED`;
5. request review before advancing to the next gate.

## Final planning state

```text
CODEX_EXPERIMENT_PLAN_REVIEW = PASS
CODEX_REPOSITORY_IMPLEMENTATION = AUTHORIZED_CPU_CI_ONLY
LABEL_V2_BUILD_EXECUTION_AUTHORIZATION = NOT_AUTHORIZED
DETECTOR_TRAINING_EXECUTION = NOT_AUTHORIZED
A800_GPU_EXPERIMENT_EXECUTION = NOT_AUTHORIZED
GATE_A2_DETECTOR = HOLD
GATE_A3_ATTACK = HOLD
EXPERIMENT_AUTHORIZATION_STATUS = NOT_AUTHORIZED
```
