# Detector-v3 D8 H1 Final Rebuild

## Status before execution

- Previous Cache A/B and P5 roots are superseded diagnostics.
- Do not delete them, but do not promote or consume them.
- D8-2, Eval160, shadow, and attack rollout remain blocked.
- Only `SUBAGENT_REVIEW.json: review_verdict == COMMIT_SAFE` authorizes H0/H1 release.

## Frozen branch

```text
branch: deepseek/detector-v3-d8-continuation-20260730
```

Resolve the exact HEAD and tree after checkout. Do not paste an old commit/tree into any artifact.

## Fail-closed shell discipline

```bash
set -euo pipefail
umask 022
```

Every formal root must be new and absent before its command starts. Never overwrite or reuse a failed root.

## 1. Clean checkout and CPU contract tests

```bash
REPO=/path/to/openvla-gripper-dutycycle-attack
cd "$REPO"
git fetch --all --prune
git checkout deepseek/detector-v3-d8-continuation-20260730
git reset --hard origin/deepseek/detector-v3-d8-continuation-20260730
test -z "$(git status --porcelain)"

HEAD=$(git rev-parse HEAD)
TREE=$(git show -s --format=%T HEAD)
printf 'HEAD=%s\nTREE=%s\n' "$HEAD" "$TREE"

python -m py_compile \
  scripts/detector_v5/d8_source_contract.py \
  scripts/detector_v5/make_d8_source_snapshot.py \
  scripts/detector_v5/d8_train_core.py \
  scripts/detector_v5/build_d8_25d_cache.py \
  scripts/detector_v5/run_d8_p5_25d_gpu_smoke.py \
  scripts/detector_v5/compare_d8_25d_caches.py \
  scripts/detector_v5/audit_d8_h1_r9.py

pytest -q \
  tests/detector_v5/test_d8_source_contract.py \
  tests/detector_v5/test_d8_train_core_parity.py \
  tests/detector_v5/test_d8_streaming_adapter_v3.py
```

Any failure: stop and report. Do not generate a source snapshot from a dirty or failing checkout.

## 2. Generate external SOURCE_SNAPSHOT_V2

The snapshot must be outside the Git worktree.

```bash
RUN_ROOT=/new/formal/root/d8_h1_${HEAD}
mkdir -p "$RUN_ROOT"
SOURCE_SNAPSHOT="$RUN_ROOT/SOURCE_SNAPSHOT_V2.json"

python scripts/detector_v5/make_d8_source_snapshot.py \
  --repo-root "$REPO" \
  --output "$SOURCE_SNAPSHOT" \
  --branch deepseek/detector-v3-d8-continuation-20260730 \
  --remote https://github.com/Leo-6-maker/openvla-gripper-dutycycle-attack.git

sha256sum "$SOURCE_SNAPSHOT"
```

The snapshot generator must report the same `HEAD` and `TREE` printed above.

## 3. Archive deployment binding

If the server cannot access GitHub, deploy a deterministic archive of the exact clean checkout together with the external snapshot. On the server:

```bash
python scripts/detector_v5/make_d8_source_snapshot.py --help >/dev/null
python - <<'PY'
import json
from pathlib import Path
import sys
sys.path.insert(0, "scripts/detector_v5")
from d8_source_contract import REVIEW_REQUIRED_SOURCE_FILES, load_and_validate_source_snapshot
snap = load_and_validate_source_snapshot(
    Path("/absolute/path/SOURCE_SNAPSHOT_V2.json"),
    Path("/absolute/path/repo"),
    REVIEW_REQUIRED_SOURCE_FILES,
)
print(json.dumps({
    "commit": snap["executable_source_commit"],
    "tree": snap["executable_source_tree"],
    "snapshot_sha256": snap["source_snapshot_sha256"],
    "validated_files": len(snap["file_sha256_map"]),
}, indent=2))
PY
```

Mismatch or missing file: stop.

## 4. Rebuild Cache A and Cache B

Use the exact same sidecar, Teacher, telemetry, and external source snapshot for A and B.

```bash
CACHE_A="$RUN_ROOT/cache_A"
CACHE_B="$RUN_ROOT/cache_B"

python scripts/detector_v5/build_d8_25d_cache.py \
  --sidecar-root "$SIDECAR_ROOT" \
  --teacher-root "$TEACHER_ROOT" \
  --telemetry-root "$TELEMETRY_ROOT" \
  --source-snapshot "$SOURCE_SNAPSHOT" \
  --output-root "$CACHE_A" \
  --run-label A \
  --workers 64

python scripts/detector_v5/build_d8_25d_cache.py \
  --sidecar-root "$SIDECAR_ROOT" \
  --teacher-root "$TEACHER_ROOT" \
  --telemetry-root "$TELEMETRY_ROOT" \
  --source-snapshot "$SOURCE_SNAPSHOT" \
  --output-root "$CACHE_B" \
  --run-label B \
  --workers 64
```

Expected hard closures include:

```text
670 raw identities
196,483 raw steps
179,674 effective steps
643 included identities
27 articulated_task identities
675 validation-union consolidated positive events
0 Eval160 reads
consumer_eligible=false
```

## 5. Canonical A/B comparison

```bash
CACHE_COMPARE="$RUN_ROOT/cache_ab_compare"
python scripts/detector_v5/compare_d8_25d_caches.py \
  --cache-a "$CACHE_A" \
  --cache-b "$CACHE_B" \
  --output-root "$CACHE_COMPARE"
```

Required result:

```text
status=PASS
670/670 per-episode files byte-identical
canonical manifests identical after removing only run_label/run_uuid/timestamp_utc
input seals identical
source binding identical
FOLD_ASSIGNMENT byte-identical
IDENTITY_DISPOSITION byte-identical
```

Failure: stop. Do not run P5.

## 6. P5 GPU smoke

Run P5 against exactly one of the two verified caches.

```bash
P5_ROOT="$RUN_ROOT/p5_gpu_smoke"
CUDA_VISIBLE_DEVICES=0 python scripts/detector_v5/run_d8_p5_25d_gpu_smoke.py \
  --cache-root "$CACHE_A" \
  --source-snapshot "$SOURCE_SNAPSHOT" \
  --output-root "$P5_ROOT" \
  --run-label H1_FINAL
```

Required gates include source/cache binding, train-only normalization, full mask contract, full privileged-key scan, finite nonzero gradients, loss decrease, model+optimizer checkpoint roundtrip, true disk-restored continuation parity including post-step logits/loss and RNG, and finite validation.

Failure: stop. Do not run H1-R9.

## 7. Independent H1-R9 audit

```bash
H1_REVIEW="$RUN_ROOT/h1_r9_review"
python scripts/detector_v5/audit_d8_h1_r9.py \
  --cache-a "$CACHE_A" \
  --cache-b "$CACHE_B" \
  --p5-root "$P5_ROOT" \
  --source-snapshot "$SOURCE_SNAPSHOT" \
  --output-root "$H1_REVIEW"
```

The only release verdict is:

```text
review_verdict=COMMIT_SAFE
```

Anything else is a hard stop. Preserve all roots and send `SUBAGENT_REVIEW.json`, `SUBAGENT_REVIEW.md`, all four package seals, HEAD, TREE, source snapshot SHA, command logs, and return codes for review.

## 8. H0/H1 release boundary

Generate `H0_H1_RELEASE.json` only after H1-R9 returns `COMMIT_SAFE`. The release receipt must bind:

- exact HEAD and TREE;
- external source snapshot SHA;
- Cache A and Cache B seals;
- A/B comparator seal;
- P5 seal;
- H1-R9 review seal;
- `protected_reads=0`;
- `eval160_reads=0`;
- `attack_rollouts_started=0`.

After that receipt is sealed, D8-2 may be authorized separately. This handoff does not authorize Eval160, shadow, or any attack rollout.
