# M3 GPU45 Long-Run Root-Cause Preregistration - 2026-06-16

## Authority Boundary

This is an infrastructure/root-cause stage for the OpenVLA Gripper Duty-Cycle
Attack Layer3 M3 line.  It is not a scientific attack stage.

Authorized:

```text
development-only fixed-frame forward diagnostics
development-only pixel-gradient diagnostics
strict determinism qualification
explicit device-map experiments
same-process and fresh-process repeatability tests
direct-forward vs autoregressive-generation isolation
layer/step divergence localization
model-load and memory profiling
zero-perturbation harness dry-run
one-step debug-only gradient update on development inputs
candidate artifact writer and auditor development
CPU/mock tests, static checks, reports, tables, manifests
```

Not authorized:

```text
final frozen 8-frame panel
seed428198
seed85 or seed86
TRUE_PGD21
RAND21
SHUFFLED_GRAD21
aggregate scientific comparison
LIBERO env.step or closed-loop rollout
Layer2-triggered attack
modifying frozen scientific parameters
claiming TRUE_PGD > RAND
claiming official-token attack effect
using GPU0 or GPU7
using GPU1,3 or GPU2,6
```

## Fixed Start State

```text
BRANCH: exp/m3-arm-v5-clean-close-event-panel-20260616
EXPECTED_START_HEAD: d4b1f7c96269d347ea892f5cc496a992238ddb39
CUDA_VISIBLE_DEVICES: 4,5
GPU4: GPU-d0a54f5d-938c-a148-fff9-c135201e3f61
GPU5: GPU-9794d733-042f-46a2-fc86-5a3fe32a158a
```

Existing qualification outputs are frozen separately under:

```text
/data/liuyu/outputs/m3_gpu45_longrun_provenance_freeze_20260616_023749_r2
```

## Development Input

The primary input is the already-authorized development step78 fixed frame:

```text
/data/liuyu/outputs/m3_arm_v4_panel_capture_f41ab1a_r2/step78
```

No final V5 frozen panel frame may be used.  Any additional development input
must be preregistered by path and SHA before execution.  If the development
input pool is not already frozen, multiframe qualification stops with
`DEVELOPMENT_INPUT_POOL_NOT_FROZEN`.

## Stage A - Qualification Runner R2

Runner R2 must:

```text
freeze all model parameters
set model.eval()
disable sampling
use fixed generation config
bind exact saved input_ids and pixel_values
record prompt length
record model bundle SHA manifest
record processor/preprocess SHA inputs
hard-check physical GPU index to UUID order
create a new output directory per run
clone inputs before every repeat
clear input gradients between repeats
synchronize CUDA at controlled boundaries
not reload/mutate the model during same-process repeats
record canonical hf_device_map and SHA
record model shard dtype/device summary
record attention implementation and software versions
```

Strict deterministic profile:

```text
PYTHONHASHSEED=0
CUDA_DEVICE_ORDER=PCI_BUS_ID
CUDA_VISIBLE_DEVICES=4,5
CUBLAS_WORKSPACE_CONFIG=:4096:8
TOKENIZERS_PARALLELISM=false

random.seed(0)
numpy.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.use_deterministic_algorithms(True, warn_only=False)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.set_float32_matmul_precision("highest")
```

Any deterministic-operation exception is evidence and must not be downgraded
to warn-only.

## Stage B - Step78 Root-Cause Matrix

### B1 Same-Process Clean Direct Forward

Load the model once.  Repeat 10 times:

```text
clone identical inputs
no generation
direct model forward over prompt + clean generated arm prefix
collect gripper-row logits
collect top-k action tokens
collect 31744 and 31872 scores
collect target/native margins
collect full score-row SHA
collect memory state
```

Output:

```text
direct_forward_same_process.csv
```

Gate:

```text
exact score hash 10/10
exact top-k ordering 10/10
no NaN/Inf
```

### B2 Same-Process Autoregressive Generation

Load the model once.  Repeat 10 times:

```text
official generation route
exact 7 generated tokens
per-token argmax
per-token score-vector SHA
gripper token
prompt length
generated sequence SHA
```

Output:

```text
generation_same_process.csv
```

The analysis records the first autoregressive token index where score hash
or argmax diverges and whether direct forward was stable while generation
diverged.

### B3 Same-Process Pixel Gradient

Only if B1 completes.  Repeat 5 times:

```text
same frozen objective
identical input clone
model parameters frozen
zero input gradient
one forward/backward
no perturbation update
```

Output:

```text
gradient_same_process.csv
```

### B4 Fresh-Process Repeatability

Launch 5 independent fresh processes.  Each process:

```text
loads identical model bundle
uses the same explicit or production device-map profile
runs one direct forward
runs one official generation
runs one pixel gradient
exits cleanly
```

Output:

```text
fresh_process_repeatability.csv
```

## Stage C - Device-Map and GPU-Order Isolation

Diagnostic profiles:

```text
C0: CUDA_VISIBLE_DEVICES=4,5 production-like auto device_map
C1: CUDA_VISIBLE_DEVICES=4,5 explicit frozen module-to-device map
C2: CUDA_VISIBLE_DEVICES=5,4 same logical module assignment with physical order reversed
C3: GPU4-led explicit split, if constructible without changing precision
C4: GPU5-led explicit split, if constructible without changing precision
```

For every runnable profile:

```text
5 direct-forward repeats
5 generation repeats
3 gradient repeats
3 fresh processes if memory/time permits
```

Output:

```text
gpu45_device_map_matrix.csv
```

Interpretation:

```text
C1 stable while C0 unstable -> AUTO_DEVICE_MAP_IS_PRIMARY_CONFOUNDER
one physical order stable -> GPU_ORDER_OR_SHARD_PLACEMENT_SENSITIVE
direct stable / generation unstable -> GENERATION_CACHE_OR_AUTOREGRESSIVE_PATH_SENSITIVE
forward stable / gradient unstable -> BACKWARD_PATH_NONDETERMINISM
all token-unstable -> GPU45_FIXED_FRAME_NUMERICALLY_UNQUALIFIED
```

## Stage D - Divergence Localization

If direct forward is unstable, use lightweight forward hooks.  Record only:

```text
shape
dtype
device
min/max/mean/std
finite count
canonical tensor SHA
max absolute difference versus repeat0
```

Suggested checkpoints:

```text
vision encoder output
multimodal projector output
language model input embedding
every fourth transformer block
final four transformer blocks
final norm
action-token logits
```

If direct forward is stable but generation diverges, record each
autoregressive step, score-row SHA, argmax, and generated token.  Diagnostic
use_cache=True/use_cache=False profiles may be run but cannot replace the
frozen production route.

Outputs:

```text
first_divergent_module.json
activation_divergence_summary.csv
```

## Stage E - Multiframe Development Qualification

This stage is skipped unless a development-only input pool is frozen by path
and SHA before execution.

Required frame classes:

```text
near-boundary step78
clearly native-OPEN frame
clearly native-CLOSE frame
non-boundary neutral diagnostic frame
```

Output:

```text
gpu45_multiframe_development_qualification.csv
```

Tiering:

```text
TIER A: exact token, score hash, gradient hash stable on all development frames and fresh processes
TIER B: token/gripper stable and gradients finite, but score or gradient hashes not exact and no boundary flips
TIER C: any clean token flip, NaN/Inf, deterministic exception, repeated OOM, input/model/provenance mismatch
```

## Stage F - Layer3 Harness Dry-Run

Proceed only if at least Tier B is achieved.  Allowed dry-runs:

```text
zero perturbation candidate0 artifact
one-step debug gradient update with development seed only
candidate bookkeeping, ledger, retry state machine, no overwrite
```

Not allowed:

```text
21 candidates
official PGD budget
random control
shuffled-gradient control
final-panel aggregation
```

## Stage G - Independent Auditor

The auditor must not reuse producer selection or validation logic.  It must
independently verify branch/commit, clean worktree, GPU UUID order, model and
input hashes, device-map SHA, deterministic environment, repeat counts, token
and score hashes, gradient hashes, development-only input status, forbidden
seed absence, no PGD21/RAND21/shuffled, no LIBERO rollout, and artifact hash
manifests.  Producer PASS with auditor FAIL is overall FAIL.

## Stop Rules

Stop the affected GPU stage immediately on:

```text
HEAD/branch/remote mismatch
dirty worktree
GPU UUID mismatch
target GPU has an existing compute process
new Xid event on GPU4/5
unauthorized input
final frozen frame
forbidden seed
model/prompt/input mismatch
NaN/Inf
deterministic-operation exception
repeated OOM after frozen-parameter fix
artifact overwrite
hash mismatch
auditor failure
```

## Required Final Classification

The final report must state exactly one:

```text
GPU45_FIXED_FRAME_TIER_A_RECOMMENDATION
GPU45_DEVELOPMENT_ONLY_TIER_B
GPU45_LAYER3_DISQUALIFIED_TIER_C
```

No final scientific claim is allowed from this stage without external audit.

