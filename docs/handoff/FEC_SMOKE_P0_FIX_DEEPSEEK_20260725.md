# DeepSeek Handoff — FEC Smoke P0 Fix

Execution branch:

```text
codex/fec-smoke-runner-p0-fix-20260725
HEAD = 86f8eaa21a40e25f78e83f8fa526fde682dd002d
```

This branch replaces the invalid `35c5001` smoke runner. The old outputs are capacity diagnostics only and must be marked:

```text
SUPERSEDED_INVALID_SMOKE_RUNNER
counts_toward_smoke_pass = false
counts_toward_fec = false
```

## 1. Checkout and CPU self-test

```bash
cd /mnt/sdc/dty_user/openvla_attack
git fetch origin codex/fec-smoke-runner-p0-fix-20260725
git checkout codex/fec-smoke-runner-p0-fix-20260725
git reset --hard origin/codex/fec-smoke-runner-p0-fix-20260725

PY=/mnt/sdc/dty_user/openvla_attack/envs/openvla-official-a800/bin/python
$PY -m py_compile scripts/fec/run_gpu_smoke.py scripts/fec/launch_smoke_workers.py
$PY scripts/fec/selftest_gpu_smoke_contract.py --repo-root /mnt/sdc/dty_user/openvla_attack
```

Expected:

```text
FEC smoke CPU contract self-test: PASS
```

## 2. Bind the exact frozen N4 feature provider

The hardened runner prohibits the previous zero-filled `f25d/p9d/g9d` path. The server-local N4 module must expose `N4DetectorAdapter` and one exact provider callable named one of:

```text
build_n4_inputs
extract_n4_inputs
build_inputs
make_n4_inputs
```

The provider must return either:

```python
{
    "f25d": np.ndarray(shape=(25,)),
    "p9d": np.ndarray(shape=(9,)),
    "g9d": np.ndarray(shape=(9,)),
    "candidate_close": bool,
    "meta": {...},
}
```

or:

```python
(f25d, p9d, g9d, candidate_close)
```

Use the exact provider from the completed 20-parent N4 validation. Do not invent features and do not fill missing fields with zero. If the function has a non-standard name, pass `--n4-provider-name <name>`.

Run a provider preflight before model loading:

```bash
$PY scripts/fec/run_gpu_smoke.py \
  --gpu-id 2 \
  --suite libero_10 \
  --task-index 0 \
  --state-index 0 \
  --output-root /tmp/fec_contract_unused \
  --model-path /mnt/sdc/dty_user/openvla_attack/models/libero-10/openvla-7b-finetuned-libero-10 \
  --repo-root /mnt/sdc/dty_user/openvla_attack \
  --config configs/fec_attack_v3.yaml \
  --n4-module /tmp/n4_detector_adapter.py \
  --n4-norm-data /mnt/sdc/dty_user/openvla_attack_evidence/fec_implementation_v1/n4_norms_o0i0.pt \
  --expected-attacker-sha256 26cfb9f5d8a5a29e7ac2729f5c9cdd58dadfd75e45eebe935ee66214cc9402be \
  --seed 11001 --rand-direction-seed 21001 --random-time-seed 31001 \
  --dry-run-contract
```

This validates source/config/module hashes. The real provider is exercised in the live canary.

## 3. Build a frozen 16-worker identity manifest

Do not reuse formal FEC parents, matrix IDs, DEV2/C4/P4/H2, or CS200. Each row must contain:

```json
{
  "worker_id": 0,
  "gpu_id": 2,
  "suite": "libero_10",
  "task_index": 0,
  "init_state_npy": "/absolute/path/to/approved_smoke_only_state.npy",
  "model_path": "/absolute/path/to/suite/checkpoint",
  "seed": 11001,
  "rand_direction_seed": 21001,
  "random_time_seed": 31001
}
```

Use exactly four rows per GPU with the frozen mapping:

```text
GPU2 = libero_10
GPU3 = libero_goal
GPU6 = libero_object
GPU7 = libero_spatial
```

`state_index` may replace `init_state_npy` only when it is a real valid index in `get_task_init_states(task_index)`. Values such as 111 are forbidden unless provided as an explicit exported `.npy` state.

## 4. Rolling replacement of the invalid workers

Do not mix old and new artifacts. Use a new output root. Replace workers one GPU at a time so at least 12 old capacity-test processes remain until the new workers on that GPU have loaded successfully:

```text
GPU2 old -> GPU2 hardened
GPU3 old -> GPU3 hardened
GPU6 old -> GPU6 hardened
GPU7 old -> GPU7 hardened
```

Never kill another user's process.

## 5. Launch the hardened 16-worker smoke

```bash
OUT=/mnt/sdc/dty_user/openvla_attack_outputs/fec_gpu_smoke_v2_hardened
rm -rf "$OUT"   # only when confirmed to be a disposable, never-started hardened output root

$PY scripts/fec/launch_smoke_workers.py \
  --identity-manifest /absolute/path/FEC_SMOKE_IDENTITY_MANIFEST_V2.json \
  --output-base "$OUT" \
  --python "$PY" \
  --repo-root /mnt/sdc/dty_user/openvla_attack \
  --config configs/fec_attack_v3.yaml \
  --n4-module /tmp/n4_detector_adapter.py \
  --n4-norm-data /mnt/sdc/dty_user/openvla_attack_evidence/fec_implementation_v1/n4_norms_o0i0.pt \
  --expected-attacker-sha256 26cfb9f5d8a5a29e7ac2729f5c9cdd58dadfd75e45eebe935ee66214cc9402be \
  --wave-stagger-seconds 20
```

## 6. Hard smoke gates

`FORMAL_GO` remains forbidden unless:

```text
FEC_GPU_SMOKE_RECEIPT_V2.status = PASS_AT_16
16/16 worker exit codes = 0
16/16 smoke summaries valid = true
TRUE natural attack coverage >= 10 frames
RAND natural attack coverage >= 10 frames
ORACLE natural attack coverage >= 10 frames
RANDOM_TIME attack coverage >= 10 frames
strict_route = true
allow_fallback = false
fallback_used = false
TRUE objective = arm_v3
RAND objective = arm_v3
TRUE gradient_transform = none
RAND gradient_transform = rademacher
RAND gradient seed = manifest-bound
actual Linf <= 0.030001
K10 executed exactly
wait steps = 10 with no policy/detector update
env reset -> exact init state -> wait -> detector reset
CLEAN/adv action use official LIBERO gripper postprocess
formal matrix cells executed = 0
CS200 access = 0
```

Scientific outcomes such as task success, Goal no-emit, TRUE not causing failure, or TRUE not beating RAND do not block the complete formal FEC matrix. Engineering/provenance failures do block it.

## 7. Stop point

After hardened smoke passes, generate and seal the formal random-time manifest, execution envelope, matrix manifest, job queue, and `FORMAL_GO` token. Do not execute CS200. Do not reuse any artifact produced by the superseded runner.
