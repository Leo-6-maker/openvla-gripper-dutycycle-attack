# A800 G0 Hardware Qualification Handoff V1

Status: READY_FOR_READ_ONLY_SERVER_BINDING_NOT_AUTHORIZED

This handoff prepares the first safe use of the idle eight-A800 server. G0 is
infrastructure qualification only. It does not read project data, load OpenVLA,
launch LIBERO, train a detector, execute an attack, or produce a scientific
outcome.

## Known Host Binding

```text
server_host = pm-364c0001
scientific producer checkout = /mnt/sdc/dty_user/openvla_attack_pr48_af8217c
legacy checkout mutation = PROHIBITED
A800 hardware inventory execution = NOT_AUTHORIZED
synthetic CUDA allocation execution = NOT_AUTHORIZED
```

The producer checkout does not need to be modified or checked out to a new
branch for G0. G0 evidence must be written outside all Git worktrees.

## G0 Stages

### G0A — read-only identity capture

The next permissible server action is read-only capture of:

```text
hostname and UTC time
nvidia-smi GPU index / UUID / model / memory / driver
ECC current/pending mode and volatile/aggregate error counts
retired pages and row-remapping status where supported
GPU temperature / power / utilization at capture time
nvidia-smi topo -m
NVLink status where supported
CUDA compiler/runtime discovery
available Python/conda environments
PyTorch/CUDA/transformers versions without loading model weights
current GPU processes
candidate evidence parent ownership/mode/free space
```

G0A must not create an evidence directory yet. Return the exact captured values
for authorization-record binding.

Recommended read-only command family:

```bash
set -euo pipefail
hostname
date -u +%Y-%m-%dT%H:%M:%SZ
nvidia-smi -L
nvidia-smi --query-gpu=index,uuid,name,memory.total,driver_version,temperature.gpu,power.draw,utilization.gpu,memory.used,ecc.mode.current,ecc.mode.pending --format=csv,noheader
nvidia-smi -q -d ECC,PAGE_RETIREMENT,ROW_REMAPPER
nvidia-smi topo -m
nvidia-smi nvlink --status || true
command -v nvcc || true
nvcc --version || true
command -v python3 || true
python3 - <<'PY'
try:
    import torch
    print('torch', torch.__version__)
    print('torch_cuda', torch.version.cuda)
    print('cuda_available', torch.cuda.is_available())
    print('device_count', torch.cuda.device_count())
except Exception as exc:
    print('torch_probe_error', type(exc).__name__, str(exc))
try:
    import transformers
    print('transformers', transformers.__version__)
except Exception as exc:
    print('transformers_probe_error', type(exc).__name__, str(exc))
PY
nvidia-smi pmon -c 1 || true
stat -c '%U:%G %a %n' /mnt/sdc/dty_user/openvla_attack_evidence
findmnt -T /mnt/sdc/dty_user/openvla_attack_evidence
_df=$(df --output=avail -B1 /mnt/sdc | tail -1); echo "sdc_available_bytes=${_df}"
test ! -e /mnt/sdc/dty_user/openvla_attack_evidence/a800_g0
```

Import probes must not call `torch.cuda.init()`, allocate tensors, or load any
checkpoint.

### G0B — synthetic CUDA smoke

G0B remains blocked until G0A binds:

```text
exact Python executable or conda environment
PyTorch and CUDA versions
all eight GPU UUIDs
candidate evidence path
resource envelope
expiration
```

A later one-shot authorization may allow, per GPU:

```text
small synthetic tensor allocation only
deterministic matmul checksum
one forward/backward synthetic operation
peak allocated/reserved memory capture
synchronization and latency capture
no project imports except a dedicated G0 utility
no project data or model weights
```

Default proposed envelope for later review:

```text
8 independent processes, one per GPU
maximum 2 GiB allocated memory per GPU
maximum 5 minutes total wall time
no NCCL collective in the first smoke
maximum 100 MiB new evidence
no persistent daemon or reservation
```

These are proposed values, not an active authorization.

## G0 Pass Criteria

```text
exactly 8 expected A800 devices
stable UUID/index mapping
no Xid, uncorrectable ECC, retired-page, or row-remap failure
consistent driver/runtime visibility
all qualified devices pass the same synthetic checksum
no unexpected process interference
per-GPU output isolation
resource envelope respected
```

A device with an error is excluded only after review; G0 must not silently
continue with seven devices and still claim eight-worker qualification.

## G0 Evidence Contract

Future evidence path proposal:

```text
/mnt/sdc/dty_user/openvla_attack_evidence/a800_g0/<authorization_id>/
```

Expected small files:

```text
identity_capture.txt
gpu_inventory.csv
topology.txt
ecc_report.txt
environment.json
synthetic_cuda_smoke.jsonl
SHA256SUMS
```

No videos, datasets, checkpoints, PNGs, or rollout artifacts are allowed.

## Explicitly Prohibited

```text
formal Label V2 build or validator
real Label V2 / feature artifact reads
OpenVLA checkpoint loading
LIBERO launch
victim inference
attack gradient computation on project models
detector training
rollout or attack
NCCL multi-GPU training
scientific result generation
legacy checkout modification
```

## Current State

```text
C1_LABEL_V2_INGESTION = PASS
C2_REPOSITORY_IMPLEMENTATION = AUTHORIZED_CPU_CI_ONLY
A800_G0A_READ_ONLY_BINDING = READY_NOT_EXECUTED
A800_G0B_SYNTHETIC_CUDA_SMOKE = NOT_AUTHORIZED
A800_HARDWARE_ONLY_QUALIFICATION = NOT_AUTHORIZED
A800_GPU_EXPERIMENT_EXECUTION = NOT_AUTHORIZED
LABEL_V2_BUILD_EXECUTION_AUTHORIZATION = NOT_AUTHORIZED
GATE_A2_DETECTOR = HOLD
GATE_A3_ATTACK = HOLD
EXPERIMENT_AUTHORIZATION_STATUS = NOT_AUTHORIZED
```
