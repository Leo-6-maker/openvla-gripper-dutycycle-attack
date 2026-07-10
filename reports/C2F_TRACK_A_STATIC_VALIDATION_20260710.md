# C2F Track A Static Validation - 2026-07-10

STATUS: PASS_STATIC_VALIDATION

CPU-only validation. No GPU episodes, Goal smoke, Object replication, Spatial expansion, or D7 parity jobs were launched.

## Test Command

```bash
PYTHONPATH=. /mnt/sdc/dty_user/openvla_attack/envs/openvla-official-a800/bin/python3.10 tests/test_c2f_track_a_static.py
```

Result: 5 tests passed.

## Static Validation Statuses

- deterministic RAND seed and identical noise reproduction: PASS
- runtime error writes `runtime_valid=false`, `success=null`, and returns nonzero: PASS
- strict frozen condition names: PASS
- authentic Goal `norm_stats` key resolution: PASS
- worker-derived git/file provenance and clean-tree enforcement: PASS

## Track A Protocol

- `protocol_name`: `C2F_TRACK_A_CMDOPEN_ACTION_SPACE`
- `protocol_version`: `2026-07-10.v1`
- `attack_space`: `action_space_command_intervention`
- conditions: `CLEAN`, `TRUE_CMDOPEN_T10_C2F`, `RAND_ACTION_NOISE_T10_C2F`

## RAND Seed Formula

```python
seed_material = f"{PROTOCOL_VERSION}|{parent_key}|{condition}|{attack_start}"
seed = int.from_bytes(sha256(seed_material.encode("utf-8")).digest()[:8], "big") % (2**32)
```

Noise is generated from `np.random.default_rng(seed)`, then L2-normalized and scaled by `EPSILON = 6/255`.

## Runtime Error Behavior

On any worker exception:

- append a `step=-1` record with `runtime_valid=false` and error fields;
- write `episode_metadata.json` with `runtime_valid=false`, `success=null`, `error_type`, `error_message`;
- exit nonzero.

Runtime-invalid episodes must not be counted as `success=false` outcomes.

## Provenance / Clean Tree

The worker records `git_provenance`, `worker_sha256`, `runtime_sha256`, detector checkpoint hash, selected policy model file hashes, processor path, `norm_stats_keys`, and `unnorm_key`.

The worker refuses to run when `git status --porcelain` is non-empty.

## Boundary Confirmation

- GPU episodes: NOT_LAUNCHED
- Goal smoke: NOT_LAUNCHED
- Object replication: NOT_LAUNCHED
- Spatial expansion: NOT_LAUNCHED
- D7 image-PGD parity: NOT_LAUNCHED