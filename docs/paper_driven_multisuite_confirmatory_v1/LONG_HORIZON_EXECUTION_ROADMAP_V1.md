# Long-Horizon Execution and A800 Utilization Roadmap V1

Status: PLANNING_ONLY_NOT_AUTHORIZED

This roadmap converts the reviewed Codex task matrix into a long-horizon,
parallelized implementation and execution program. It does not authorize the
formal Label V2 build, server access, detector training, OpenVLA inference,
LIBERO rollout, attack execution, or GPU use.

## 1. Current Binding

```text
producer planning base = af8217c934e5894c87d3db73b031a93f2536624d
C0 audit commit = 59ba119901a1019e37c69cde7ae68a9fa2f530ad
C1 implementation under review = 44d0837ea1828d0727300439c0636d4986bdbcc2
C1 review state = HOLD_INTERNAL_CLOSURE_GAPS
Label V2 one-shot readiness = PASS_PREFLIGHT
Label V2 execution = NOT_AUTHORIZED
A800 experiment execution = NOT_AUTHORIZED
```

The roadmap assumes the frozen scientific protocol remains unchanged. Any
change to populations, split units, checkpoint-selection rule, threshold
selection, attack conditions, epsilon selection, K, PGD steps, primary outcome,
or statistical family requires a separate planning review.

## 2. Program Structure

The work is divided into four tracks that converge at explicit gates:

```text
Track A: Label/data closure
Track B: detector implementation and freeze
Track C: exact-prefix/attack/CQ engineering
Track D: A800 infrastructure and scientific execution
```

```text
A1 Label V2 freeze
  -> A2 detector dataset closure
  -> B1 detector CPU closure
  -> D1 A800 qualification
  -> B2 detector training and Gate A2
  -> C1 exact-prefix canary
  -> D2 confirmatory attack matrix
  -> C2 CQ/manual audit
  -> final statistics and figures
```

## 3. Immediate Critical Path

### Milestone M0 — close C1 ingestion review

Required fixes on PR #50:

```text
manifest schema_version binding
formal manual requested-priority set closure
event/disposition semantic closure
positive synthetic-mode test
invalid-window summary recomputation
concise parse/I/O failure handling
root-symlink and malformed/missing SHA coverage
validated-object reuse or post-read hash recheck
```

Exit condition:

```text
C1_01_LABEL_V2_DOWNSTREAM_INGESTION_SCHEMA = PASS
C1_02_LABEL_V2_INGESTION_VALIDATOR = PASS
```

No C2 or GPU work begins before M0 review.

### Milestone M1 — one-shot Label V2 build and Gate A1

Separate PR #49 activation only:

```text
read-only preflight
-> exactly one formal-ledger-build
-> exactly one validate-formal-output
-> five-file loader validation
-> 160-row human audit
-> artifact SHA freeze
```

Exit condition:

```text
LABEL_V2_FORMAL_BUILD = PASS
LABEL_V2_INDEPENDENT_VALIDATION = PASS
LABEL_V2_FIVE_FILE_INGESTION = PASS
LABEL_V2_MANUAL_AUDIT = PASS
GATE_A1_LABEL_ARTIFACT = PASS_FROZEN
```

Because `/mnt/sdc` is already highly utilized, M1 must complete before any
large GPU outputs are created.

## 4. Repository Implementation Waves

After M0, repository-only engineering may proceed in small reviewed branches.
No branch reads real artifacts or executes server jobs.

### Wave R1 — detector dataset closure

Task IDs:

```text
C2_01 detector dataset join builder
C2_02 parent/state split builders
C2_03 train-only normalization builder
C2_04 detector dataset closure validator
```

Deliverables:

```text
frozen feature artifact reader
exact episode-set Label V2 x feature join
25D SC5 feature-contract binding
DETECTOR_ELIGIBLE / DETECTOR_SAFETY / DETECTOR_MULTI_EVENT populations
parent_key and initial_state_hash leakage checks
parent_random / object_loto / suite_loso split schemas
train-only normalization manifest
synthetic negative tests
```

Exit condition:

```text
row-set closure = PASS
parent leakage = 0
state-hash leakage = 0
normalization leakage = 0
future/attack telemetry leakage = 0
```

Formal dataset construction remains separately authorized after Gate A1.

### Wave R2 — detector formalization

Task IDs:

```text
C3_01 train CLI identity hardening
C3_02 eval CLI identity hardening
C3_03 event/timing metrics
C3_04 validation-only threshold selection
C3_05 synthetic detector end-to-end smoke
C5_01 detector run scheduler dry-run
```

Required scientific clarification before implementation:

```text
freeze one primary checkpoint-selection rule
```

The rule must be selected before any real detector result is viewed.

Exit condition:

```text
formal manifest required by train/eval
checkpoint provenance complete
exclusive window semantics tested
no-emit and ineligible abstention tested
validation-only threshold selector tested
synthetic e2e detector smoke PASS
```

### Wave R3 — exact-prefix and branch engineering

Task IDs:

```text
C7_01 exact-prefix snapshot schema
C7_02 restore/parity validator
C7_03 matched branch queue builder
C7_04 runtime telemetry validator
C8_01 primary parent selector
C8_02 formal attack matrix generator
```

Deliverables:

```text
simulator/RNG/observation/action/FSM/history identity schema
off-by-one-safe restore contract
same-parent same-worker deterministic assignment
atomic five-condition branch-family specification
actual Linf/frame-count/terminal-status validation
result-blind primary parent selection
```

No OpenVLA or LIBERO execution occurs in this wave.

### Wave R4 — CQ and analysis engineering

Task IDs:

```text
C9_01 formal CQ evaluator
C9_02 blind audit manifest builder
C10_01 paired statistics
C10_02 paper table builders
C10_03 figure-data builders
```

Deliverables:

```text
five frozen CQ flags plus CQ_TELEMETRY_MISSING
explicit task-object ontology binding
condition-blinded audit IDs
second-reviewer overlap and Cohen kappa
parent-level paired risk difference
exact McNemar and Holm correction
task/parent cluster bootstrap
ITT and emitted-only table separation
manifest-bound Table 1/2/3 and Figure 2-5 data
```

## 5. Safe Early Use of Idle A800 Resources

GPU work is split into infrastructure qualification and scientific execution.
They require separate authorization records.

### Level G0 — hardware-only inventory

May be proposed immediately after M0, but remains not authorized here.

Scope:

```text
nvidia-smi GPU index, UUID, model, memory, temperature, power, ECC
NVLink/topology inventory
driver and CUDA runtime identity
PyTorch/CUDA import identity
read-only filesystem/output-root checks
synthetic tensor allocation and deterministic matmul checksum
```

Prohibited in G0:

```text
real Label V2 or feature artifacts
OpenVLA weights
LIBERO
victim forward pass
attack gradients
scientific outcomes
```

G0 uses no more than a few minutes per GPU and creates only a small manifest.
It can establish whether all eight A800s are healthy without consuming the
scientific experiment budget.

### Level G1 — victim execution qualification

Prerequisites:

```text
M0 complete
R3 exact-prefix/telemetry contracts reviewed
separate clean checkpoint/environment binding
explicit GPU qualification authorization
```

Run sequence:

```text
one-GPU checkpoint-load smoke
one clean forward checksum
one synthetic backward/PGD memory smoke
one non-scientific runner output-isolation smoke
repeat on all eight GPUs
```

Topology decision is frozen before scientific outcomes:

```text
preferred = 8 independent one-GPU workers
fallback = 4 fixed ordered two-GPU workers
```

No condition may be assigned by GPU. All branches for one parent remain on one
worker.

### Level G2 — detector training

Prerequisites:

```text
Gate A1 PASS
formal detector dataset PASS
R2 PASS
G0/G1 qualification PASS as applicable
explicit detector-training authorization
```

Formal run matrix:

```text
Object-only x 3 seeds
Pooled four-suite x 3 seeds
LOSO x 4 suites x 3 seeds
Object leave-one-task-out x 10 folds x 3 seeds
```

These 48 detector runs are lightweight. Use independent run queues, not data
parallelism. One process per GPU is the default. Actual wall time and memory
budget must be bound from a one-run pilot before releasing the full queue.

### Level G3 — exact-prefix attack canary

Prerequisites:

```text
Gate A2 detector freeze
R3 PASS
G1 worker parity PASS
explicit attack-canary authorization
```

Canary:

```text
8 held-out parents
2 per suite
5 matched branches per parent
40 suffix branches total
```

All eight GPUs can be used concurrently with one parent family per worker.
Every branch family is preserved even when failed.

### Level G4 — confirmatory attack matrix

Prerequisites:

```text
canary PASS
formal parent selection frozen
attack parameters frozen
CQ/telemetry validators PASS
explicit confirmatory authorization
```

Primary scale:

```text
4 suites
up to 20 parents per suite
5 matched conditions
up to 400 primary branches
```

Seed robustness and mechanism controls are separate, later subsets. They do not
inflate the primary parent denominator.

Queue rule:

```text
worker_id = int(sha256(parent_key)[:8], 16) % worker_count
```

The scheduler dispatches parent families, never isolated conditions.

## 6. GPU Scheduling Policy

### Detector phase

Use all eight GPUs only after a pilot establishes that independent runs are
stable. Suggested queue partition:

```text
GPU0-2: Object-only and pooled seeds
GPU3-4: LOSO queues
GPU5-7: Object LOTO folds
```

### Attack phase

Use one parent family per worker:

```text
CLEAN_EXACT_PREFIX_REPLAY
OURS_STUDENT_GRIPPER_TARGET
RAND_DIRECTION
RANDOM_TIME
ADAPTED_TMA_OPEN
```

The five branches execute serially on the same worker unless an exact-prefix
snapshot format is proven safe for concurrent isolated processes.

### Failure isolation

```text
one branch failure -> preserve evidence, mark family incomplete
one worker parity failure -> stop assigning new parents to that worker
clean replay mismatch -> stop the entire scientific wave
Xid/ECC/OOM instability -> stop the worker and review topology
```

## 7. Storage and Evidence Envelope

Before M1, preserve the existing 100 GiB free-space threshold.

After Gate A1, each GPU wave requires a separately measured storage budget based
on a canary. Default retention:

```text
retain all numeric telemetry, manifests, hashes, logs, and terminal outcomes
retain all canary videos
retain all CQ positives and SR/CQ disagreements
retain matched controls for claimed Ours failures
retain a blinded stratified audit sample
```

Do not retain by default:

```text
full uncompressed PNG sequences
duplicate model caches per worker
unbounded debug tensors
artifacts inside Git
```

The scheduler stops dispatching new jobs before the bound free-space threshold
is crossed. Existing evidence is never deleted merely to keep a queue running.

## 8. Review and Release Gates

```text
Review R0: C1 loader closeout
Review A1: formal Label V2 + manual audit freeze
Review R1: detector dataset/split implementation
Review R2: detector formalization and synthetic smoke
Review G0: hardware inventory authorization and result
Review G1: victim execution qualification
Review A2: detector metrics and checkpoint freeze
Review G3: 40-branch attack canary
Review G4: confirmatory matrix release
Review CQ: blind manual audit and kappa
Review Final: statistics, tables, figures, and claims
```

No review automatically activates the next execution gate.

## 9. Recommended Parallelization

After C1 passes:

```text
Lane 1: M1 Label V2 activation and human audit
Lane 2: R1 detector dataset implementation
Lane 3: R3 exact-prefix/queue/telemetry implementation
Lane 4: R4 CQ/statistics implementation
```

After Gate A1 and R1/R2 pass:

```text
Lane 1: formal detector dataset build
Lane 2: G0/G1 A800 qualification
Lane 3: detector training scheduler dry-run and environment freeze
```

After Gate A2:

```text
all qualified workers -> G3 attack canary
then all qualified workers -> G4 confirmatory matrix
```

This ordering minimizes idle GPU time without allowing hardware availability to
bypass scientific gates.

## 10. Current State

```text
LONG_HORIZON_ROADMAP_REVIEW = PENDING
C1_01/C1_02 = HOLD_PENDING_FIXES
LABEL_V2_BUILD_EXECUTION_AUTHORIZATION = NOT_AUTHORIZED
A800_INFRA_QUALIFICATION_EXECUTION = NOT_AUTHORIZED
DETECTOR_TRAINING_EXECUTION = NOT_AUTHORIZED
A800_GPU_EXPERIMENT_EXECUTION = NOT_AUTHORIZED
GATE_A2_DETECTOR = HOLD
GATE_A3_ATTACK = HOLD
EXPERIMENT_AUTHORIZATION_STATUS = NOT_AUTHORIZED
```
