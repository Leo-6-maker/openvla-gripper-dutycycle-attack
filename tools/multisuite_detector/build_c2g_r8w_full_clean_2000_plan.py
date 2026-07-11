#!/usr/bin/env python3
"""Build the frozen R8W full Clean-2000 collection plan.

The planner copies every R7 registry identity exactly once and only assigns
collection ownership.  It never redraws identities or changes cohort/split
semantics.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from tools.multisuite_detector.plan_c2g_scientific_corpus import (
    ATTACK_EVAL,
    COHORTS,
    COHORT_TO_SPLIT,
    DETECTOR_TEST,
    DETECTOR_TRAIN,
    DETECTOR_VAL,
    PASS_STATUS as R7_PASS_STATUS,
    SCHEMA as R7_SCHEMA,
    SUITES,
)

SCHEMA = "c2g.r8w.full_clean_2000_plan.2026-07-12.v1"
PASS_STATUS = "PASS_C2G_R8W_FULL_CLEAN_2000_PLAN"
PREVIEW_STATUS = "PASS_C2G_R8W_FULL_CLEAN_2000_PLAN_PREVIEW"
CANARY_PASS_STATUS = "PASS_C2G_R8W_FRESH_SHADOW_CANARY_PLAN"
CANARY_PREVIEW_STATUS = "PASS_C2G_R8W_FRESH_SHADOW_CANARY_PLAN_PREVIEW"
AUTHORIZATION_TOKEN = "R8W_FULL_CLEAN_2000_GPU4567_16WORKER_AUTHORIZED"
PURPOSE = "FULL_CLEAN_2000"
CANARY_PURPOSE = "FRESH_SHADOW_CANARY"
GPUS = (4, 5, 6, 7)
SUITE_SLUG = {
    "libero_object": "object",
    "libero_spatial": "spatial",
    "libero_goal": "goal",
    "libero_10": "l10",
}
COHORT_QUOTAS = {
    0: {DETECTOR_TRAIN: 75, DETECTOR_VAL: 13, DETECTOR_TEST: 12, ATTACK_EVAL: 25},
    1: {DETECTOR_TRAIN: 75, DETECTOR_VAL: 12, DETECTOR_TEST: 13, ATTACK_EVAL: 25},
    2: {DETECTOR_TRAIN: 75, DETECTOR_VAL: 13, DETECTOR_TEST: 12, ATTACK_EVAL: 25},
    3: {DETECTOR_TRAIN: 75, DETECTOR_VAL: 12, DETECTOR_TEST: 13, ATTACK_EVAL: 25},
}
EXPECTED_GLOBAL_COHORTS = {
    DETECTOR_TRAIN: 1200,
    DETECTOR_VAL: 200,
    DETECTOR_TEST: 200,
    ATTACK_EVAL: 400,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_sha256(value: str, name: str) -> str:
    value = str(value).strip().lower()
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"{name} must be a lowercase SHA256")
    return value


def assert_hash(path: Path, expected: str, name: str) -> str:
    expected = require_sha256(expected, name)
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"{name} hash mismatch: {actual} != {expected}")
    return actual


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
            raise ValueError(f"{path}:{line_no} must contain an object")
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


def identity(row: Mapping[str, Any]) -> tuple[str, int, int]:
    return str(row["suite"]), int(row["task_index"]), int(row["state_id"])


def stable_rank(seed: int, *parts: object) -> bytes:
    payload = "|".join((str(seed), *(str(part) for part in parts)))
    return hashlib.sha256(payload.encode("utf-8")).digest()


def worker_id(gpu: int, suite: str) -> str:
    return f"g{gpu}_{SUITE_SLUG[suite]}"


def expected_flags(cohort: str) -> dict[str, bool]:
    return {
        "eligible_for_detector_fit": cohort == DETECTOR_TRAIN,
        "eligible_for_checkpoint_selection": cohort == DETECTOR_VAL,
        "eligible_for_threshold_calibration": cohort == DETECTOR_VAL,
        "eligible_for_clean_test": cohort == DETECTOR_TEST,
        "eligible_for_attack_evaluation": cohort == ATTACK_EVAL,
    }


def _validate_full_sha(value: str, name: str) -> str:
    value = str(value).strip().lower()
    if len(value) != 40 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"{name} must be a 40-character lowercase git SHA")
    return value


def validate_registry(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if len(rows) != 2000:
        raise ValueError(f"R7 registry must contain 2000 rows, got {len(rows)}")
    normalized: list[dict[str, Any]] = []
    identities: set[tuple[str, int, int]] = set()
    parents: set[str] = set()
    for index, raw in enumerate(rows):
        row = dict(raw)
        key = identity(row)
        parent = str(row.get("parent_key", ""))
        if key in identities:
            raise ValueError(f"duplicate R7 identity: {key}")
        if not parent or parent in parents:
            raise ValueError(f"duplicate or missing parent_key at row {index}: {parent!r}")
        identities.add(key)
        parents.add(parent)
        suite, task_index, state_id = key
        if suite not in SUITES or task_index < 0 or state_id < 0:
            raise ValueError(f"invalid identity at row {index}: {key}")
        cohort = str(row.get("cohort", ""))
        if cohort not in COHORTS:
            raise ValueError(f"invalid cohort at row {index}: {cohort!r}")
        if row.get("split") != COHORT_TO_SPLIT[cohort]:
            raise ValueError(f"cohort/split mismatch at row {index}")
        for name, expected in expected_flags(cohort).items():
            if row.get(name) is not expected:
                raise ValueError(f"eligibility mismatch at row {index}: {name}")
        max_steps = row.get("max_steps")
        if type(max_steps) is not int or max_steps <= 0:
            raise ValueError(f"invalid max_steps at row {index}: {max_steps!r}")
        normalized.append(row)

    suite_counts = Counter(str(row["suite"]) for row in normalized)
    if suite_counts != Counter({suite: 500 for suite in SUITES}):
        raise ValueError(f"suite cardinality mismatch: {dict(suite_counts)}")
    cohort_counts = Counter(str(row["cohort"]) for row in normalized)
    if cohort_counts != Counter(EXPECTED_GLOBAL_COHORTS):
        raise ValueError(f"cohort cardinality mismatch: {dict(cohort_counts)}")
    for suite in SUITES:
        counts = Counter(str(row["cohort"]) for row in normalized if row["suite"] == suite)
        expected = Counter({DETECTOR_TRAIN: 300, DETECTOR_VAL: 50, DETECTOR_TEST: 50, ATTACK_EVAL: 100})
        if counts != expected:
            raise ValueError(f"{suite} cohort cardinality mismatch: {dict(counts)}")
    return normalized


def verify_required_identity_manifests(
    registry_ids: set[tuple[str, int, int]], paths: Sequence[Path]
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for path in paths:
        path = path.resolve()
        rows = read_jsonl(path)
        ids = [identity(row) for row in rows]
        if len(ids) != len(set(ids)):
            raise ValueError(f"required identity manifest contains duplicates: {path}")
        missing = sorted(set(ids) - registry_ids)
        if missing:
            raise ValueError(f"required identity manifest has identities outside R7: {path}: {missing[:5]}")
        results.append({
            "path": str(path),
            "sha256": sha256_file(path),
            "identity_count": len(ids),
            "all_present_once": True,
        })
    return results


def assign_suite_rows(rows: Sequence[Mapping[str, Any]], suite: str, seed: int) -> list[dict[str, Any]]:
    suite_rows = [dict(row) for row in rows if row["suite"] == suite]
    assigned: dict[int, list[dict[str, Any]]] = {shard: [] for shard in range(4)}
    grouped: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in suite_rows:
        grouped[(int(row["task_index"]), str(row["cohort"]))].append(row)

    # Allocate one cohort at a time.  Every task first contributes floor(n/4)
    # rows to every shard; the remainder is a deterministic bipartite matching
    # against exact shard capacity.  This preserves the requested per-task
    # difference <= 1 without the late-quota starvation of row-wise greedy
    # assignment.
    allocation: dict[tuple[int, str], dict[int, int]] = {}
    for cohort in COHORTS:
        tasks = sorted(task for task, row_cohort in grouped if row_cohort == cohort)
        residual = {shard: COHORT_QUOTAS[shard][cohort] for shard in range(4)}
        for task in tasks:
            base = len(grouped[(task, cohort)]) // 4
            allocation[(task, cohort)] = {shard: base for shard in range(4)}
            for shard in range(4):
                residual[shard] -= base
                if residual[shard] < 0:
                    raise ValueError(f"base allocation exceeds {suite}/{cohort}/shard_{shard} quota")
        for task in tasks:
            extra = len(grouped[(task, cohort)]) % 4
            candidates = sorted(
                range(4),
                key=lambda shard: (
                    -residual[shard],
                    stable_rank(seed, "extra", suite, cohort, task, shard),
                    shard,
                ),
            )
            selected = [shard for shard in candidates if residual[shard] > 0][:extra]
            if len(selected) != extra:
                raise ValueError(f"cannot close balanced quota for {suite}/{task}/{cohort}")
            for shard in selected:
                allocation[(task, cohort)][shard] += 1
                residual[shard] -= 1
        if any(residual.values()):
            raise ValueError(f"unfilled quotas for {suite}/{cohort}: {residual}")

    for task_index, cohort in sorted(grouped, key=lambda key: (key[0], COHORTS.index(key[1]))):
        members = sorted(
            grouped[(task_index, cohort)],
            key=lambda row: stable_rank(seed, suite, task_index, cohort, row["parent_key"]),
        )
        offset = 0
        for shard in range(4):
            count = allocation[(task_index, cohort)][shard]
            for row in members[offset:offset + count]:
                gpu = GPUS[shard]
                output = dict(row)
                output.update({
                    "collection_purpose": PURPOSE,
                    "materializable": False,
                    "assigned_physical_gpu": gpu,
                    "assigned_worker_id": worker_id(gpu, suite),
                    "assigned_shard_id": f"{suite}__shard_{shard}",
                })
                assigned[shard].append(output)
            offset += count
        if offset != len(members):
            raise AssertionError(f"allocation cardinality mismatch for {suite}/{task_index}/{cohort}")
    output_rows: list[dict[str, Any]] = []
    for shard in range(4):
        members = sorted(
            assigned[shard],
            key=lambda row: (
                COHORTS.index(str(row["cohort"])),
                int(row["task_index"]),
                stable_rank(seed, "local", row["parent_key"]),
            ),
        )
        if len(members) != 125:
            raise ValueError(f"{suite} shard {shard} has {len(members)} rows")
        for local_index, row in enumerate(members):
            row["shard_local_index"] = local_index
            output_rows.append(row)
    return output_rows


def assignment_balance(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    violations: list[dict[str, Any]] = []
    groups: dict[tuple[str, int, str], Counter[int]] = defaultdict(Counter)
    for row in rows:
        groups[(str(row["suite"]), int(row["task_index"]), str(row["cohort"]))][
            int(row["assigned_physical_gpu"])
        ] += 1
    for key, counts in sorted(groups.items()):
        values = [counts[gpu] for gpu in GPUS]
        if max(values) - min(values) > 1:
            violations.append({
                "suite": key[0],
                "task_index": key[1],
                "cohort": key[2],
                "per_gpu": {str(gpu): counts[gpu] for gpu in GPUS},
                "difference": max(values) - min(values),
                "mathematically_unavoidable": False,
            })
    return violations


def build_plan_data(
    rows: Sequence[Mapping[str, Any]], selection_seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    assigned = [row for suite in SUITES for row in assign_suite_rows(rows, suite, selection_seed)]
    if len(assigned) != 2000 or len({identity(row) for row in assigned}) != 2000:
        raise AssertionError("assigned identity closure failed")
    shards: list[dict[str, Any]] = []
    for gpu in GPUS:
        for suite in SUITES:
            wid = worker_id(gpu, suite)
            members = [row for row in assigned if row["assigned_worker_id"] == wid]
            counts = Counter(str(row["cohort"]) for row in members)
            shard = GPUS.index(gpu)
            if len(members) != 125 or counts != Counter(COHORT_QUOTAS[shard]):
                raise AssertionError(f"worker quota closure failed: {wid}: {len(members)}, {dict(counts)}")
            shards.append({
                "worker_id": wid,
                "suite": suite,
                "physical_gpu": gpu,
                "shard_id": f"{suite}__shard_{shard}",
                "shard_index": shard,
                "episode_count": len(members),
                "cohort_counts": dict(sorted(counts.items())),
                "max_steps": sorted({int(row["max_steps"]) for row in members}),
            })
    if len(shards) != 16:
        raise AssertionError("worker closure failed")
    return assigned, shards


def _git_head(repo: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()


def _assert_external_new_output(path: Path, repo: Path) -> Path:
    path, repo = path.resolve(), repo.resolve()
    if path.exists():
        raise FileExistsError(path)
    if path == repo or repo in path.parents:
        raise ValueError("R8W plan output must be outside repository")
    return path


def build_plan(
    *,
    mode: str,
    repo: Path,
    expected_git_commit: str,
    registry_path: Path,
    expected_registry_sha256: str,
    plan_report_path: Path,
    expected_plan_report_sha256: str,
    output_root: Path,
    authorization: str,
    selection_seed: int = 20260712,
    required_identity_manifests: Sequence[Path] = (),
) -> dict[str, Any]:
    if mode not in {"preview", "run"}:
        raise ValueError("mode must be preview or run")
    expected_git_commit = _validate_full_sha(expected_git_commit, "expected_git_commit")
    repo = repo.resolve()
    if _git_head(repo) != expected_git_commit:
        raise ValueError("repository HEAD differs from expected_git_commit")
    output_root = output_root.resolve()
    if mode == "run":
        if authorization != AUTHORIZATION_TOKEN:
            raise PermissionError("R8W full collection plan authorization mismatch")
        _assert_external_new_output(output_root, repo)
    elif output_root.exists():
        raise FileExistsError(output_root)

    registry_path, plan_report_path = registry_path.resolve(), plan_report_path.resolve()
    registry_sha = assert_hash(registry_path, expected_registry_sha256, "R7 registry")
    plan_sha = assert_hash(plan_report_path, expected_plan_report_sha256, "R7 plan report")
    r7_plan = read_json(plan_report_path)
    if r7_plan.get("schema") != R7_SCHEMA or r7_plan.get("status") != R7_PASS_STATUS:
        raise ValueError("R7 plan report is not accepted")
    if Path(str(r7_plan.get("registry", ""))).resolve() != registry_path:
        raise ValueError("R7 plan report binds another registry")
    if r7_plan.get("registry_sha256") != registry_sha:
        raise ValueError("R7 plan report registry SHA mismatch")

    registry = validate_registry(read_jsonl(registry_path))
    required = verify_required_identity_manifests({identity(row) for row in registry}, required_identity_manifests)
    assigned, shards = build_plan_data(registry, selection_seed)
    balance_violations = assignment_balance(assigned)
    if balance_violations:
        raise ValueError(f"task/cohort balance violation: {balance_violations[:3]}")

    manifest_path = output_root / "c2g_r8w_full_clean_2000.jsonl"
    shard_index_path = output_root / "c2g_r8w_full_clean_2000_shards.jsonl"
    report_path = output_root / "c2g_r8w_full_clean_2000_plan.json"
    shard_rows: list[dict[str, Any]] = []
    for shard in shards:
        members = [row for row in assigned if row["assigned_worker_id"] == shard["worker_id"]]
        path = output_root / "shards" / f"{shard['worker_id']}.jsonl"
        row = {
            **shard,
            "manifest": str(path),
            "manifest_sha256": hashlib.sha256(jsonl_bytes(members)).hexdigest(),
        }
        shard_rows.append(row)

    manifest_bytes = jsonl_bytes(assigned)
    shard_index_bytes = jsonl_bytes(shard_rows)
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "status": PREVIEW_STATUS if mode == "preview" else PASS_STATUS,
        "mode": mode,
        "plan_kind": PURPOSE,
        "expected_git_commit": expected_git_commit,
        "authorization_token": AUTHORIZATION_TOKEN,
        "selection_seed": selection_seed,
        "r7_plan_report": str(plan_report_path),
        "r7_plan_report_sha256": plan_sha,
        "r7_registry": str(registry_path),
        "r7_registry_sha256": registry_sha,
        "required_identity_manifests": required,
        "episode_count": 2000,
        "suite_counts": dict(sorted(Counter(str(row["suite"]) for row in assigned).items())),
        "cohort_counts": dict(sorted(Counter(str(row["cohort"]) for row in assigned).items())),
        "worker_count": 16,
        "workers_per_gpu": {str(gpu): 4 for gpu in GPUS},
        "episodes_per_gpu": {str(gpu): 500 for gpu in GPUS},
        "episodes_per_worker": 125,
        "manifest": str(manifest_path),
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "shard_index": str(shard_index_path),
        "shard_index_sha256": hashlib.sha256(shard_index_bytes).hexdigest(),
        "shards": shard_rows,
        "task_cohort_balance_violations": balance_violations,
        "identity_closure": True,
        "registry_redrawn": False,
        "cohort_reassigned": False,
        "training_authorization": "HOLD",
        "materialization_authorization": "HOLD",
        "attack_authorization": "HOLD",
    }
    if mode == "preview":
        return report

    output_root.mkdir(parents=True)
    write_jsonl(manifest_path, assigned)
    for shard in shard_rows:
        members = [row for row in assigned if row["assigned_worker_id"] == shard["worker_id"]]
        write_jsonl(Path(shard["manifest"]), members)
    write_jsonl(shard_index_path, shard_rows)
    write_json(report_path, report)
    checksum_paths = [manifest_path, shard_index_path, report_path] + [Path(row["manifest"]) for row in shard_rows]
    checksums = output_root / "SHA256SUMS"
    checksums.write_text(
        "".join(f"{sha256_file(path)}  {path.relative_to(output_root).as_posix()}\n" for path in checksum_paths),
        encoding="ascii",
    )
    checksum_digest = output_root / "SHA256SUMS.sha256"
    checksum_digest.write_text(f"{sha256_file(checksums)}  SHA256SUMS\n", encoding="ascii")
    return {
        **report,
        "report": str(report_path),
        "report_sha256": sha256_file(report_path),
        "checksums": str(checksums),
        "checksums_sha256": sha256_file(checksums),
    }


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def build_shadow_canary_plan(
    *,
    mode: str,
    repo: Path,
    expected_git_commit: str,
    registry_path: Path,
    expected_registry_sha256: str,
    plan_report_path: Path,
    expected_plan_report_sha256: str,
    r8u_report_path: Path,
    expected_r8u_report_sha256: str,
    r8u_episode_ledger_path: Path,
    expected_r8u_episode_ledger_sha256: str,
    r8u_step_ledger_path: Path,
    expected_r8u_step_ledger_sha256: str,
    r8u_sha256s_path: Path,
    expected_r8u_sha256s_sha256: str,
    output_root: Path,
    authorization: str,
    selection_seed: int = 20260712,
    reference_scheduler_report_path: Path | None = None,
    expected_reference_scheduler_report_sha256: str = "",
) -> dict[str, Any]:
    if mode not in {"canary-preview", "canary-run"}:
        raise ValueError("canary mode must be canary-preview or canary-run")
    run = mode == "canary-run"
    expected_git_commit = _validate_full_sha(expected_git_commit, "expected_git_commit")
    repo = repo.resolve()
    if _git_head(repo) != expected_git_commit:
        raise ValueError("repository HEAD differs from expected_git_commit")
    if run:
        if authorization != AUTHORIZATION_TOKEN:
            raise PermissionError("R8W shadow canary authorization mismatch")
        _assert_external_new_output(output_root, repo)
    elif output_root.exists():
        raise FileExistsError(output_root)
    registry_sha = assert_hash(registry_path, expected_registry_sha256, "R7 registry")
    plan_sha = assert_hash(plan_report_path, expected_plan_report_sha256, "R7 plan report")
    r7_plan = read_json(plan_report_path)
    if r7_plan.get("schema") != R7_SCHEMA or r7_plan.get("status") != R7_PASS_STATUS:
        raise ValueError("R7 plan report is not accepted")
    if r7_plan.get("registry_sha256") != registry_sha:
        raise ValueError("R7 plan report registry SHA mismatch")
    registry = validate_registry(read_jsonl(registry_path))
    lookup = {identity(row): row for row in registry}
    r8u_report_sha = assert_hash(r8u_report_path, expected_r8u_report_sha256, "R8U replay report")
    r8u_episode_sha = assert_hash(
        r8u_episode_ledger_path, expected_r8u_episode_ledger_sha256, "R8U replay episode ledger"
    )
    r8u_step_sha = assert_hash(r8u_step_ledger_path, expected_r8u_step_ledger_sha256, "R8U replay step ledger")
    r8u_sums_sha = assert_hash(r8u_sha256s_path, expected_r8u_sha256s_sha256, "R8U SHA256SUMS")
    sums_text = r8u_sha256s_path.read_text(encoding="utf-8")
    for digest, name in (
        (r8u_report_sha, r8u_report_path.name),
        (r8u_episode_sha, r8u_episode_ledger_path.name),
        (r8u_step_sha, r8u_step_ledger_path.name),
    ):
        if f"{digest}  {name}" not in sums_text:
            raise ValueError(f"R8U SHA256SUMS does not bind {name}")
    r8u = read_json(r8u_report_path)
    expected_per_suite = {
        "libero_object": {"success": 5, "total": 6},
        "libero_spatial": {"success": 4, "total": 6},
        "libero_goal": {"success": 4, "total": 6},
        "libero_10": {"success": 2, "total": 6},
    }
    expected_report = {
        "status": "PASS_C2G_R8U_SUCCESS_REPLAY_INTEGRITY",
        "episode_count": 24,
        "replay_exact_count": 24,
        "replay_numerically_equivalent_count": 0,
        "replay_diverged_count": 0,
        "replay_failed_count": 0,
        "canonical_clean_success_count": 15,
        "per_suite_clean_success": expected_per_suite,
    }
    for key, expected in expected_report.items():
        if r8u.get(key) != expected:
            raise ValueError(f"R8U replay invariant mismatch: {key}: {r8u.get(key)!r} != {expected!r}")
    ledger = read_csv(r8u_episode_ledger_path)
    if len(ledger) != 24 or any(row.get("classification") != "REPLAY_EXACT" for row in ledger):
        raise ValueError("R8U replay episode ledger is not exact 24/24")
    selected: list[dict[str, Any]] = []
    # ── Resolve reference GPU mapping from R8T scheduler report ──
    if reference_scheduler_report_path is not None:
        if not expected_reference_scheduler_report_sha256:
            raise ValueError("expected_reference_scheduler_report_sha256 required with reference report")
        actual_ref_sched_sha = sha256_file(reference_scheduler_report_path)
        if actual_ref_sched_sha != expected_reference_scheduler_report_sha256:
            raise ValueError(f"reference scheduler report SHA mismatch: {actual_ref_sched_sha[:16]}... != {expected_reference_scheduler_report_sha256[:16]}...")
        ref_sched = json.loads(reference_scheduler_report_path.read_text(encoding="utf-8"))
        if ref_sched.get("status") != "PASS_C2G_R8T_DYNAMIC_GPU_CANARY":
            raise ValueError("reference scheduler report is not PASS")
        # Extract per-shard suite→physical_gpu from R8T shards
        r8t_shards = ref_sched.get("shards", [])
        if len(r8t_shards) != 4:
            raise ValueError("reference scheduler has != 4 shards")
        gpu_by_suite = {}
        for sh in r8t_shards:
            s = sh.get("suite", "")
            g = sh.get("physical_gpu", -1)
            if s and g >= 0:
                gpu_by_suite[s] = int(g)
        if set(gpu_by_suite.keys()) != set(SUITES):
            raise ValueError(f"reference scheduler missing suites: {set(SUITES) - set(gpu_by_suite.keys())}")
        hardware_binding_source = "R8T_DYNAMIC_GPU_SCHEDULER"
    else:
        gpu_by_suite = dict(zip(SUITES, GPUS))
        hardware_binding_source = "IMPLICIT_ZIP_ORDER"
    for suite in SUITES:
        local = [row for row in ledger if row.get("suite") == suite]
        successes = [row for row in local if row.get("canonical_success") == "True"]
        failures = [row for row in local if row.get("canonical_success") == "False"]
        if not successes or not failures:
            raise ValueError(f"R8U lacks success/failure pair for {suite}")
        for outcome, candidates in ((True, successes), (False, failures)):
            chosen = min(
                candidates,
                key=lambda row: stable_rank(selection_seed, "shadow", suite, outcome, row["parent_key"]),
            )
            key = (suite, int(chosen["task_index"]), int(chosen["state_id"]))
            if key not in lookup:
                raise ValueError(f"R8U shadow identity is outside R7: {key}")
            source = dict(lookup[key])
            if source["parent_key"] != chosen["parent_key"]:
                raise ValueError(f"R8U/R7 parent key mismatch: {key}")
            gpu = gpu_by_suite[suite]
            source.update({
                "collection_purpose": CANARY_PURPOSE,
                "materializable": False,
                "assigned_physical_gpu": gpu,
                "assigned_worker_id": f"canary_{worker_id(gpu, suite)}",
                "assigned_shard_id": f"shadow_canary__{suite}",
                "expected_canonical_success": outcome,
                "r8u_classification": chosen["classification"],
            })
            selected.append(source)
    selected.sort(key=lambda row: (SUITES.index(row["suite"]), not row["expected_canonical_success"]))
    for suite in SUITES:
        local = [row for row in selected if row["suite"] == suite]
        for index, row in enumerate(local):
            row["shard_local_index"] = index
    shards = []
    for suite in SUITES:
        gpu = gpu_by_suite[suite]
        wid = f"canary_{worker_id(gpu, suite)}"
        members = [row for row in selected if row["assigned_worker_id"] == wid]
        path = output_root / "shards" / f"{wid}.jsonl"
        shards.append({
            "worker_id": wid,
            "suite": suite,
            "physical_gpu": gpu,
            "shard_id": f"shadow_canary__{suite}",
            "shard_index": 0,
            "episode_count": 2,
            "cohort_counts": dict(sorted(Counter(row["cohort"] for row in members).items())),
            "max_steps": sorted({row["max_steps"] for row in members}),
            "manifest": str(path),
            "manifest_sha256": hashlib.sha256(jsonl_bytes(members)).hexdigest(),
        })
    manifest_path = output_root / "c2g_r8w_fresh_shadow_canary.jsonl"
    shard_index_path = output_root / "c2g_r8w_fresh_shadow_canary_shards.jsonl"
    report_path = output_root / "c2g_r8w_fresh_shadow_canary_plan.json"
    report = {
        "schema": SCHEMA,
        "status": CANARY_PASS_STATUS if run else CANARY_PREVIEW_STATUS,
        "mode": mode,
        "plan_kind": CANARY_PURPOSE,
        "expected_git_commit": expected_git_commit,
        "authorization_token": AUTHORIZATION_TOKEN,
        "selection_seed": selection_seed,
        "r7_plan_report": str(plan_report_path.resolve()),
        "r7_plan_report_sha256": plan_sha,
        "r7_registry": str(registry_path.resolve()),
        "r7_registry_sha256": registry_sha,
        "r8u_replay_report": str(r8u_report_path.resolve()),
        "r8u_replay_report_sha256": r8u_report_sha,
        "r8u_episode_ledger": str(r8u_episode_ledger_path.resolve()),
        "r8u_episode_ledger_sha256": r8u_episode_sha,
        "r8u_step_ledger": str(r8u_step_ledger_path.resolve()),
        "r8u_step_ledger_sha256": r8u_step_sha,
        "r8u_sha256s": str(r8u_sha256s_path.resolve()),
        "r8u_sha256s_sha256": r8u_sums_sha,
        "episode_count": 8,
        "worker_count": 4,
        "manifest": str(manifest_path),
        "manifest_sha256": hashlib.sha256(jsonl_bytes(selected)).hexdigest(),
        "shard_index": str(shard_index_path),
        "shard_index_sha256": hashlib.sha256(jsonl_bytes(shards)).hexdigest(),
        "shards": shards,
        "shadow_outputs_count_toward_full_2000": False,
        "training_authorization": "HOLD",
        "materialization_authorization": "HOLD",
        "attack_authorization": "HOLD",
    }
    if not run:
        return report
    output_root.mkdir(parents=True)
    write_jsonl(manifest_path, selected)
    for shard in shards:
        members = [row for row in selected if row["assigned_worker_id"] == shard["worker_id"]]
        write_jsonl(Path(shard["manifest"]), members)
    write_jsonl(shard_index_path, shards)
    write_json(report_path, report)
    checksum_paths = [manifest_path, shard_index_path, report_path] + [Path(row["manifest"]) for row in shards]
    sums = output_root / "SHA256SUMS"
    sums.write_text(
        "".join(f"{sha256_file(path)}  {path.relative_to(output_root).as_posix()}\n" for path in checksum_paths),
        encoding="ascii",
    )
    self_binding = output_root / "SHA256SUMS.sha256"
    self_binding.write_text(f"{sha256_file(sums)}  SHA256SUMS\n", encoding="ascii")
    return {**report, "report": str(report_path), "report_sha256": sha256_file(report_path)}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("preview", "run", "canary-preview", "canary-run"))
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--expected-git-commit", required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--expected-registry-sha256", required=True)
    parser.add_argument("--plan-report", type=Path, required=True)
    parser.add_argument("--expected-plan-report-sha256", required=True)
    parser.add_argument("--selection-seed", type=int, default=20260712)
    parser.add_argument("--required-identity-manifest", type=Path, action="append", default=[])
    parser.add_argument("--r8u-report", type=Path)
    parser.add_argument("--expected-r8u-report-sha256")
    parser.add_argument("--r8u-episode-ledger", type=Path)
    parser.add_argument("--expected-r8u-episode-ledger-sha256")
    parser.add_argument("--r8u-step-ledger", type=Path)
    parser.add_argument("--expected-r8u-step-ledger-sha256")
    parser.add_argument("--r8u-sha256s", type=Path)
    parser.add_argument("--expected-r8u-sha256s-sha256")
    parser.add_argument("--reference-scheduler-report", type=Path)
    parser.add_argument("--expected-reference-scheduler-report-sha256", default="")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--authorization", default="")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    common = dict(
        repo=args.repo,
        expected_git_commit=args.expected_git_commit,
        registry_path=args.registry,
        expected_registry_sha256=args.expected_registry_sha256,
        plan_report_path=args.plan_report,
        expected_plan_report_sha256=args.expected_plan_report_sha256,
        output_root=args.output_root,
        authorization=args.authorization,
        selection_seed=args.selection_seed,
    )
    if args.mode.startswith("canary-"):
        required = {
            "r8u_report_path": args.r8u_report,
            "expected_r8u_report_sha256": args.expected_r8u_report_sha256,
            "r8u_episode_ledger_path": args.r8u_episode_ledger,
            "expected_r8u_episode_ledger_sha256": args.expected_r8u_episode_ledger_sha256,
            "r8u_step_ledger_path": args.r8u_step_ledger,
            "expected_r8u_step_ledger_sha256": args.expected_r8u_step_ledger_sha256,
            "r8u_sha256s_path": args.r8u_sha256s,
            "expected_r8u_sha256s_sha256": args.expected_r8u_sha256s_sha256,
            "reference_scheduler_report_path": args.reference_scheduler_report,
            "expected_reference_scheduler_report_sha256": args.expected_reference_scheduler_report_sha256,
        }
        missing = [name for name, value in required.items()
                   if value is None and not name.startswith("reference_")]
        if missing:
            raise ValueError("canary mode missing arguments: " + ", ".join(missing))
        result = build_shadow_canary_plan(mode=args.mode, **common, **required)
    else:
        result = build_plan(
            mode=args.mode,
            required_identity_manifests=args.required_identity_manifest,
            **common,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
