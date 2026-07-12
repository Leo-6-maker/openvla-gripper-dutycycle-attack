#!/usr/bin/env python3
"""Build the identity-preserving R8Y L10-520 horizon-repair plan.

Extracts all libero_10 rows from a frozen R8W plan, keeps every identity
unchanged, and only updates:
  * max_steps: 300 → 520
  * protocol_generation: R8Y (new)
  * assigned shard/worker mapping: 20 logical shards (5/GPU)

Does NOT change cohort, split, task, state, seed, or GPU assignment.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from tools.multisuite_detector.c2g_official_suite_horizons import (
    OFFICIAL_DUMMY_WAIT_STEPS,
    OFFICIAL_MAX_POLICY_STEPS,
    validate_official_suite_horizon,
)
from tools.multisuite_detector.plan_c2g_scientific_corpus import (
    ATTACK_EVAL,
    COHORTS,
    COHORT_TO_SPLIT,
    DETECTOR_TEST,
    DETECTOR_TRAIN,
    DETECTOR_VAL,
    SUITES,
)

SCHEMA = "c2g.r8y.l10_520_plan.2026-07-12.v1"
PASS_STATUS = "PASS_C2G_R8Y_L10_520_PLAN"
PREVIEW_STATUS = "PASS_C2G_R8Y_L10_520_PLAN_PREVIEW"
AUTHORIZATION_TOKEN = "R8Y_L10_520_IMPLEMENTATION_AND_CANARY_AUTHORIZED"
PURPOSE = "L10_HORIZON_REPAIR_520"
TARGET_SUITE = "libero_10"
GPUS = (4, 5, 6, 7)

# 20 logical shards: 5 per GPU, 25 episodes each
LOGICAL_SHARDS_PER_GPU = 5
EPISODES_PER_LOGICAL_SHARD = 25
EPISODES_PER_GPU = 125

# GPU 4,6: val=13, test=12
GPU46_COHORT_QUOTA = {
    0: {DETECTOR_TRAIN: 15, DETECTOR_VAL: 3, DETECTOR_TEST: 2, ATTACK_EVAL: 5},
    1: {DETECTOR_TRAIN: 15, DETECTOR_VAL: 2, DETECTOR_TEST: 3, ATTACK_EVAL: 5},
    2: {DETECTOR_TRAIN: 15, DETECTOR_VAL: 3, DETECTOR_TEST: 2, ATTACK_EVAL: 5},
    3: {DETECTOR_TRAIN: 15, DETECTOR_VAL: 2, DETECTOR_TEST: 3, ATTACK_EVAL: 5},
    4: {DETECTOR_TRAIN: 15, DETECTOR_VAL: 3, DETECTOR_TEST: 2, ATTACK_EVAL: 5},
}

# GPU 5,7: val=12, test=13
GPU57_COHORT_QUOTA = {
    0: {DETECTOR_TRAIN: 15, DETECTOR_VAL: 2, DETECTOR_TEST: 3, ATTACK_EVAL: 5},
    1: {DETECTOR_TRAIN: 15, DETECTOR_VAL: 3, DETECTOR_TEST: 2, ATTACK_EVAL: 5},
    2: {DETECTOR_TRAIN: 15, DETECTOR_VAL: 2, DETECTOR_TEST: 3, ATTACK_EVAL: 5},
    3: {DETECTOR_TRAIN: 15, DETECTOR_VAL: 3, DETECTOR_TEST: 2, ATTACK_EVAL: 5},
    4: {DETECTOR_TRAIN: 15, DETECTOR_VAL: 2, DETECTOR_TEST: 3, ATTACK_EVAL: 5},
}

EXPECTED_L10_COHORTS = {
    DETECTOR_TRAIN: 300,
    DETECTOR_VAL: 50,
    DETECTOR_TEST: 50,
    ATTACK_EVAL: 100,
}

CANONICAL_MAX_STEPS = OFFICIAL_MAX_POLICY_STEPS[TARGET_SUITE]
SOURCE_MAX_STEPS = 300


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def require_sha256(value: str, name: str) -> str:
    value = str(value).strip().lower()
    if len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
        raise ValueError(f"{name} must be a lowercase SHA256 hex string")
    return value


def require_full_sha(value: str, name: str) -> str:
    value = str(value).strip().lower()
    if len(value) != 40 or any(ch not in "0123456789abcdef" for ch in value):
        raise ValueError(f"{name} must be a 40-char lowercase git SHA")
    return value


def assert_hash(path: Path, expected: str, name: str) -> str:
    expected = require_sha256(expected, name)
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"{name} hash mismatch: {actual} != {expected}")
    return actual


def identity(row: Mapping[str, Any]) -> tuple[str, int, int]:
    return str(row["suite"]), int(row["task_index"]), int(row["state_id"])


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_no} must be an object")
        rows.append(value)
    if not rows:
        raise ValueError(f"empty JSONL: {path}")
    return rows


def json_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(dict(value), indent=2, sort_keys=True) + "\n").encode("utf-8")


def jsonl_bytes(rows: Iterable[Mapping[str, Any]]) -> bytes:
    return "".join(json.dumps(dict(row), sort_keys=True) + "\n" for row in rows).encode("utf-8")


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(json_bytes(value))


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(jsonl_bytes(rows))


def stable_rank(*parts: object) -> bytes:
    payload = "|".join(str(p) for p in parts)
    return hashlib.sha256(payload.encode("utf-8")).digest()


def logical_worker_id(gpu: int, shard: int) -> str:
    return f"g{gpu}_l10_s{shard}"


def quota_for_gpu(gpu: int) -> dict[int, dict[str, int]]:
    if gpu in (4, 6):
        return GPU46_COHORT_QUOTA
    return GPU57_COHORT_QUOTA


def extract_l10_rows(source_manifest: Path) -> list[dict[str, Any]]:
    """Extract only libero_10 rows from source R8W manifest."""
    all_rows = read_jsonl(source_manifest)
    l10_rows = [dict(row) for row in all_rows if row.get("suite") == TARGET_SUITE]
    if len(l10_rows) != 500:
        raise ValueError(f"expected 500 L10 rows in source manifest, got {len(l10_rows)}")
    identities = {identity(row) for row in l10_rows}
    if len(identities) != 500:
        raise ValueError(f"duplicate L10 identities in source: {500 - len(identities)} dupes")
    return l10_rows


def validate_l10_identity_closure(rows: list[dict[str, Any]]) -> None:
    """Verify every identity is preserved exactly once."""
    if len(rows) != 500:
        raise ValueError(f"L10 closure: expected 500, got {len(rows)}")
    ids = [identity(row) for row in rows]
    if len(set(ids)) != 500:
        raise ValueError("L10 identity closure: duplicates detected")

    suites = Counter(str(row["suite"]) for row in rows)
    if suites != Counter({TARGET_SUITE: 500}):
        raise ValueError(f"suite closure: {dict(suites)}")

    for row in rows:
        if row.get("max_steps") != SOURCE_MAX_STEPS:
            raise ValueError(f"source row has unexpected max_steps: {row.get('max_steps')}")
        if row.get("assigned_physical_gpu") not in GPUS:
            raise ValueError(f"source row has invalid GPU: {row.get('assigned_physical_gpu')}")

    cohorts = Counter(str(row["cohort"]) for row in rows)
    if cohorts != Counter(EXPECTED_L10_COHORTS):
        raise ValueError(f"L10 cohort closure: {dict(cohorts)}")


def assign_20_shards(l10_rows: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    """Assign 500 L10 rows to 20 logical shards with per-GPU cohort quotas.

    Uses deterministic cohort-level slicing: for each (GPU, cohort), items across
    all tasks are sorted by a stable key then sliced into fixed-quota shards.
    Task balance (within-GPU, per task, across 5 shards difference <= 1) is
    verified after assignment.
    """
    assigned: list[dict[str, Any]] = []

    for gpu in GPUS:
        gpu_rows = [row for row in l10_rows if int(row["assigned_physical_gpu"]) == gpu]
        if len(gpu_rows) != EPISODES_PER_GPU:
            raise ValueError(f"GPU {gpu}: expected {EPISODES_PER_GPU} rows, got {len(gpu_rows)}")

        quotas = quota_for_gpu(gpu)
        gpu_total = sum(sum(q.values()) for q in quotas.values())
        if gpu_total != EPISODES_PER_GPU:
            raise ValueError(f"GPU {gpu}: quota sum {gpu_total} != {EPISODES_PER_GPU}")

        # Phase 1: compute exact per-task-per-shard allocation matrix
        # for each cohort, using floor(N/S)+1 for remainder shards.
        allocation: dict[tuple[str, int], dict[int, int]] = {}
        for cohort in COHORTS:
            cohort_rows = [r for r in gpu_rows if str(r["cohort"]) == cohort]
            by_task: dict[int, list[dict[str, Any]]] = defaultdict(list)
            for row in cohort_rows:
                by_task[int(row["task_index"])].append(row)

            shard_remaining = {s: quotas[s][cohort] for s in range(LOGICAL_SHARDS_PER_GPU)}
            task_order = sorted(
                by_task.keys(),
                key=lambda t: stable_rank(seed, "r8y_a", gpu, cohort, t),
            )

            for task in task_order:
                n = len(by_task[task])
                base = n // LOGICAL_SHARDS_PER_GPU
                extra = n % LOGICAL_SHARDS_PER_GPU

                # Every shard gets at least `base` items
                allocation[(cohort, task)] = {
                    s: base for s in range(LOGICAL_SHARDS_PER_GPU)
                }
                for s in range(LOGICAL_SHARDS_PER_GPU):
                    shard_remaining[s] -= base
                    if shard_remaining[s] < 0:
                        raise ValueError(
                            f"GPU{gpu} {cohort} task {task}: "
                            f"base allocation {base} exceeds shard {s} quota"
                        )

                # Allocate `extra` items to the shards with highest remaining quota
                if extra > 0:
                    candidate_order = sorted(
                        range(LOGICAL_SHARDS_PER_GPU),
                        key=lambda s: (
                            -shard_remaining[s],
                            stable_rank(seed, "r8y_x", gpu, cohort, task, s),
                        ),
                    )
                    for s in candidate_order[:extra]:
                        allocation[(cohort, task)][s] += 1
                        shard_remaining[s] -= 1

        # Phase 2: distribute items according to the allocation matrix
        shard_accum: dict[int, list[dict[str, Any]]] = {
            s: [] for s in range(LOGICAL_SHARDS_PER_GPU)
        }
        for cohort in COHORTS:
            cohort_rows = [r for r in gpu_rows if str(r["cohort"]) == cohort]
            by_task: dict[int, list[dict[str, Any]]] = defaultdict(list)
            for row in cohort_rows:
                by_task[int(row["task_index"])].append(row)

            task_order = sorted(
                by_task.keys(),
                key=lambda t: stable_rank(seed, "r8y_d", gpu, cohort, t),
            )

            for task in task_order:
                items = sorted(
                    by_task[task],
                    key=lambda r: stable_rank(
                        seed, "r8y_e", gpu, cohort, task, r["parent_key"]
                    ),
                )
                item_idx = 0
                for s in range(LOGICAL_SHARDS_PER_GPU):
                    count = allocation[(cohort, task)][s]
                    for _ in range(count):
                        output = dict(items[item_idx])
                        output.update(_r8y_fields(gpu, s, items[item_idx]))
                        shard_accum[s].append(output)
                        item_idx += 1

        # Verify and collect
        gpu_assigned: list[dict[str, Any]] = []
        for s in range(LOGICAL_SHARDS_PER_GPU):
            if len(shard_accum[s]) != EPISODES_PER_LOGICAL_SHARD:
                raise ValueError(
                    f"GPU{gpu} shard {s}: expected {EPISODES_PER_LOGICAL_SHARD}, "
                    f"got {len(shard_accum[s])}"
                )
            gpu_assigned.extend(shard_accum[s])

        assigned.extend(gpu_assigned)

    if len(assigned) != 500:
        raise ValueError(f"total assigned {len(assigned)} != 500")

    # Verify per-GPU task balance: within each GPU, each task spread across
    # 5 shards with count difference <= 1.
    for gpu in GPUS:
        gpu_assigned = [r for r in assigned if int(r["assigned_physical_gpu"]) == gpu]
        task_shard_counts: dict[int, Counter] = defaultdict(Counter)
        for row in gpu_assigned:
            task_shard_counts[int(row["task_index"])][
                int(row["assigned_shard_index"])
            ] += 1
        for task, counts in task_shard_counts.items():
            values = list(counts.values())
            if len(values) < 5:
                # Some shards may have 0 for this task — that's fine as long as
                # the spread across occupied shards is <= 1
                all_values = [counts.get(s, 0) for s in range(LOGICAL_SHARDS_PER_GPU)]
                if max(all_values) - min(all_values) > 1:
                    raise ValueError(
                        f"GPU{gpu} task {task}: shard imbalance "
                        f"{max(all_values)} - {min(all_values)} > 1: {dict(counts)}"
                    )
            elif max(values) - min(values) > 1:
                raise ValueError(
                    f"GPU{gpu} task {task}: shard imbalance "
                    f"{max(values)} - {min(values)} > 1: {dict(counts)}"
                )

    return assigned


def _r8y_fields(gpu: int, shard_index: int, source: Mapping[str, Any]) -> dict[str, Any]:
    wid = logical_worker_id(gpu, shard_index)
    return {
        "max_steps": CANONICAL_MAX_STEPS,
        "dummy_wait_steps": OFFICIAL_DUMMY_WAIT_STEPS,
        "protocol_generation": "R8Y",
        "assigned_worker_id": wid,
        "assigned_shard_id": f"{TARGET_SUITE}__r8y_gpu{gpu}_s{shard_index}",
        "assigned_shard_index": shard_index,
        "source_r8w_parent_key": source.get("parent_key", ""),
        "source_r8w_max_steps": SOURCE_MAX_STEPS,
        "canonical_max_steps": CANONICAL_MAX_STEPS,
        "horizon_repair": True,
        "horizon_repair_reason": "OFFICIAL_LIBERO10_520",
        "collection_purpose": PURPOSE,
        "materializable": False,
    }


def verify_assigned_shard_closure(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Verify 20-shard closure and build shard index."""
    if len(rows) != 500:
        raise ValueError(f"closure: {len(rows)} != 500")
    ids = [identity(r) for r in rows]
    if len(set(ids)) != 500:
        raise ValueError("duplicate identities in assigned rows")

    # Verify max_steps
    for row in rows:
        if int(row.get("max_steps", 0)) != CANONICAL_MAX_STEPS:
            raise ValueError(f"row has wrong max_steps: {row.get('max_steps')}")
        if str(row.get("suite", "")) != TARGET_SUITE:
            raise ValueError(f"non-L10 row in assigned: {row.get('suite')}")

    shard_index: list[dict[str, Any]] = []
    for gpu in GPUS:
        for s in range(LOGICAL_SHARDS_PER_GPU):
            wid = logical_worker_id(gpu, s)
            members = [r for r in rows if r["assigned_worker_id"] == wid]
            if len(members) != EPISODES_PER_LOGICAL_SHARD:
                raise ValueError(
                    f"{wid}: expected {EPISODES_PER_LOGICAL_SHARD}, got {len(members)}"
                )
            counts = Counter(str(r["cohort"]) for r in members)
            expected = Counter(quota_for_gpu(gpu)[s])
            if counts != expected:
                raise ValueError(f"{wid}: cohort counts {dict(counts)} != {dict(expected)}")
            shard_index.append({
                "worker_id": wid,
                "suite": TARGET_SUITE,
                "physical_gpu": gpu,
                "shard_index": s,
                "shard_id": f"{TARGET_SUITE}__r8y_gpu{gpu}_s{s}",
                "episode_count": len(members),
                "cohort_counts": dict(sorted(counts.items())),
                "max_steps": CANONICAL_MAX_STEPS,
            })

    if len(shard_index) != 20:
        raise ValueError(f"shard closure: {len(shard_index)} != 20")
    return shard_index


def build_r8y_plan(
    *,
    mode: str,
    repo: Path,
    expected_git_commit: str,
    source_r8w_plan_report: Path,
    expected_source_r8w_plan_report_sha256: str,
    source_r8w_master_manifest: Path,
    expected_source_r8w_master_manifest_sha256: str,
    output_root: Path,
    authorization: str,
    selection_seed: int = 20260712,
) -> dict[str, Any]:
    """Build the R8Y L10-520 plan."""
    if mode not in {"preview", "run"}:
        raise ValueError("mode must be preview or run")

    expected_git_commit = require_full_sha(expected_git_commit, "expected_git_commit")
    repo = repo.resolve()
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True
    ).strip()
    if head != expected_git_commit:
        raise ValueError(f"HEAD {head[:10]} != expected {expected_git_commit[:10]}")

    source_r8w_plan_report = source_r8w_plan_report.resolve()
    source_r8w_master_manifest = source_r8w_master_manifest.resolve()

    # SHA verification (fail-closed)
    assert_hash(
        source_r8w_plan_report,
        expected_source_r8w_plan_report_sha256,
        "source R8W plan report",
    )
    assert_hash(
        source_r8w_master_manifest,
        expected_source_r8w_master_manifest_sha256,
        "source R8W master manifest",
    )

    # Validate source plan report
    r8w_report = read_json(source_r8w_plan_report)
    r8w_schema = r8w_report.get("schema", "")
    if "r8w" not in r8w_schema.lower():
        raise ValueError(f"source plan report is not R8W: schema={r8w_schema}")
    if r8w_report.get("status", "").startswith("HOLD"):
        raise ValueError("source R8W plan is on HOLD")

    output_root = output_root.resolve()
    if mode == "run":
        if authorization != AUTHORIZATION_TOKEN:
            raise PermissionError("R8Y plan authorization mismatch")
        if output_root.exists():
            raise FileExistsError(f"output root already exists: {output_root}")
    elif output_root.exists():
        raise FileExistsError(f"output root already exists: {output_root}")

    # Extract and validate L10 rows
    l10_rows = extract_l10_rows(source_r8w_master_manifest)
    validate_l10_identity_closure(l10_rows)

    # Verify horizon is 300 in source
    for row in l10_rows:
        validate_official_suite_horizon(TARGET_SUITE, CANONICAL_MAX_STEPS)

    # Assign to 20 logical shards
    assigned = assign_20_shards(l10_rows, selection_seed)
    shard_index = verify_assigned_shard_closure(assigned)

    # Materialize outputs
    manifest_path = output_root / "c2g_r8y_l10_520_master_manifest.jsonl"
    shard_index_path = output_root / "c2g_r8y_l10_520_shard_index.json"
    report_path = output_root / "c2g_r8y_l10_520_plan_report.json"
    shards_dir = output_root / "shards"

    manifest_bytes = jsonl_bytes(assigned)
    manifest_sha256 = sha256_bytes(manifest_bytes)

    shard_entries: list[dict[str, Any]] = []
    for shard in shard_index:
        members = [r for r in assigned if r["assigned_worker_id"] == shard["worker_id"]]
        path = shards_dir / f"{shard['worker_id']}.jsonl"
        shard_bytes = jsonl_bytes(members)
        entry = {
            **shard,
            "manifest": str(path),
            "manifest_sha256": sha256_bytes(shard_bytes),
        }
        shard_entries.append(entry)

    shard_index_bytes = json_bytes({"schema": SCHEMA, "shards": shard_entries})
    shard_index_sha256 = sha256_bytes(shard_index_bytes)

    # Per-GPU cohort verification
    per_gpu_cohorts: dict[int, dict[str, int]] = {}
    for gpu in GPUS:
        members = [r for r in assigned if int(r["assigned_physical_gpu"]) == gpu]
        per_gpu_cohorts[gpu] = dict(
            sorted(Counter(str(r["cohort"]) for r in members).items())
        )

    report = {
        "schema": SCHEMA,
        "status": PREVIEW_STATUS if mode == "preview" else PASS_STATUS,
        "mode": mode,
        "plan_kind": PURPOSE,
        "target_suite": TARGET_SUITE,
        "canonical_max_steps": CANONICAL_MAX_STEPS,
        "dummy_wait_steps": OFFICIAL_DUMMY_WAIT_STEPS,
        "source_max_steps": SOURCE_MAX_STEPS,
        "expected_git_commit": expected_git_commit,
        "authorization_token": AUTHORIZATION_TOKEN,
        "selection_seed": selection_seed,
        "source_r8w_plan_report": str(source_r8w_plan_report),
        "source_r8w_plan_report_sha256": sha256_file(source_r8w_plan_report),
        "source_r8w_master_manifest": str(source_r8w_master_manifest),
        "source_r8w_master_manifest_sha256": sha256_file(source_r8w_master_manifest),
        "episode_count": 500,
        "unique_identity_count": 500,
        "suite_counts": {TARGET_SUITE: 500},
        "cohort_counts": dict(
            sorted(Counter(str(r["cohort"]) for r in assigned).items())
        ),
        "logical_shard_count": 20,
        "shards_per_gpu": LOGICAL_SHARDS_PER_GPU,
        "episodes_per_shard": EPISODES_PER_LOGICAL_SHARD,
        "episodes_per_gpu": EPISODES_PER_GPU,
        "per_gpu_cohort_counts": {
            str(gpu): counts for gpu, counts in sorted(per_gpu_cohorts.items())
        },
        "physical_gpus": list(GPUS),
        "master_manifest": str(manifest_path),
        "master_manifest_sha256": manifest_sha256,
        "shard_index": str(shard_index_path),
        "shard_index_sha256": shard_index_sha256,
        "horizon_repair": True,
        "horizon_repair_reason": "OFFICIAL_LIBERO10_520",
        "boundaries": {
            "libero_environment_created": False,
            "openvla_models_loaded": 0,
            "clean_rollouts_launched": 0,
            "attacks_launched": 0,
            "attack_outcomes_read": False,
            "training_epochs": 0,
            "materialization_runs": 0,
            "storage_deletions": 0,
        },
    }

    # Write outputs (preview mode writes to output_root)
    output_root.mkdir(parents=True, exist_ok=False)
    write_jsonl(manifest_path, assigned)
    write_json(shard_index_path, {"schema": SCHEMA, "shards": shard_entries})
    write_json(report_path, report)
    shards_dir.mkdir(parents=True, exist_ok=True)
    for entry in shard_entries:
        members = [r for r in assigned if r["assigned_worker_id"] == entry["worker_id"]]
        write_jsonl(Path(entry["manifest"]), members)

    # SHA256SUMS
    sums_lines = []
    for path in sorted(output_root.rglob("*")):
        if path.is_file() and path.name not in ("SHA256SUMS", "SHA256SUMS.sha256"):
            sums_lines.append(f"{sha256_file(path)}  {path.relative_to(output_root)}\n")
    sums_path = output_root / "SHA256SUMS"
    sums_path.write_text("".join(sums_lines), encoding="utf-8")
    sums_sha256 = sha256_file(sums_path)
    (output_root / "SHA256SUMS.sha256").write_text(f"{sums_sha256}  SHA256SUMS\n", encoding="utf-8")

    report["SHA256SUMS"] = str(sums_path)
    report["SHA256SUMS_sha256"] = sums_sha256
    report["report"] = str(report_path)
    report["report_sha256"] = sha256_file(report_path)
    write_json(report_path, report)

    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=["preview", "run"], default="preview",
        help="preview = dry-run; run = materialize (requires authorization)",
    )
    parser.add_argument(
        "--source-r8w-plan-report", type=Path, required=True,
        help="Path to R8W plan report JSON",
    )
    parser.add_argument(
        "--expected-source-r8w-plan-report-sha256", required=True,
        help="SHA256 of source R8W plan report",
    )
    parser.add_argument(
        "--source-r8w-master-manifest", type=Path, required=True,
        help="Path to R8W master manifest JSONL",
    )
    parser.add_argument(
        "--expected-source-r8w-master-manifest-sha256", required=True,
        help="SHA256 of source R8W master manifest",
    )
    parser.add_argument(
        "--output-root", type=Path, required=True,
        help="Output directory for R8Y plan artifacts",
    )
    parser.add_argument(
        "--expected-git-commit", required=True,
        help="Expected git HEAD",
    )
    parser.add_argument(
        "--authorization", default="",
        help="Authorization token for run mode",
    )
    parser.add_argument(
        "--selection-seed", type=int, default=20260712,
        help="Deterministic seed for shard assignment",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_r8y_plan(
        mode=args.mode,
        repo=Path(__file__).resolve().parents[2],
        expected_git_commit=args.expected_git_commit,
        source_r8w_plan_report=args.source_r8w_plan_report,
        expected_source_r8w_plan_report_sha256=args.expected_source_r8w_plan_report_sha256,
        source_r8w_master_manifest=args.source_r8w_master_manifest,
        expected_source_r8w_master_manifest_sha256=args.expected_source_r8w_master_manifest_sha256,
        output_root=args.output_root,
        authorization=args.authorization,
        selection_seed=args.selection_seed,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
