#!/usr/bin/env python3
"""Run the post-HOLD corridor through the immutable science runner, per suite."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any


COUNTERS = {"protected_reads": 0, "eval160_reads": 0, "attack_rollouts": 0, "vis_pgd_attack_rollouts": 0}
SUITES = ("libero_10", "libero_goal", "libero_object", "libero_spatial")


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON_OBJECT_REQUIRED:{path}")
    return value


def _write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _utc() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _safe_key(key: str) -> str:
    return key.replace("/", "__")


def _resolve(repo_root: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (repo_root / path).resolve()


def _validate(post_path: Path, manifest_path: Path, source_commit: str, source_tree: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    post = _load(post_path)
    if post.get("schema") != "STAGE_V_M4_CORRIDOR_REPLENISHMENT_POST_32_OF_40_HOLD_V1" or post.get("status") != "FROZEN_RUNTIME_AUTHORIZED" or post.get("runtime_authorized") is not True:
        raise ValueError("POST_HOLD_PROTOCOL_NOT_AUTHORIZED")
    if post.get("source_binding", {}).get("science_commit") != source_commit or post.get("source_binding", {}).get("science_tree") != source_tree:
        raise ValueError("POST_HOLD_SOURCE_BINDING_MISMATCH")
    if post.get("protected_counters") != COUNTERS or post.get("operation", {}).get("outcomes_read") is not False:
        raise ValueError("POST_HOLD_BOUNDARY_INVALID")
    inputs = post.get("inputs", {})
    expected_manifest = _resolve(post_path.parents[1], str(inputs.get("candidate_parent_manifest_path", "")))
    if expected_manifest != manifest_path:
        raise ValueError("POST_HOLD_MANIFEST_PATH_MISMATCH")
    if inputs.get("candidate_parent_manifest_sha256") != _sha(manifest_path):
        raise ValueError("POST_HOLD_MANIFEST_HASH_MISMATCH")
    manifest = _load(manifest_path)
    parents = manifest.get("parents")
    if manifest.get("schema") != "STAGE_V_M4_POST_HOLD_CANDIDATE_PARENT_MANIFEST_V1" or manifest.get("status") != "FROZEN" or not isinstance(parents, list):
        raise ValueError("POST_HOLD_MANIFEST_NOT_FROZEN")
    if len(parents) != int(post.get("qualification", {}).get("candidate_parent_count", -1)):
        raise ValueError("POST_HOLD_MANIFEST_COUNT_MISMATCH")
    if manifest.get("source_binding", {}).get("science_commit") != source_commit or manifest.get("source_binding", {}).get("science_tree") != source_tree:
        raise ValueError("POST_HOLD_MANIFEST_SOURCE_MISMATCH")
    selection = manifest.get("selection_rule", {})
    if selection.get("current_corridor_attempts_excluded") is not True or selection.get("task_specific_blacklist") is not False:
        raise ValueError("POST_HOLD_SELECTION_RULE_INVALID")
    report_path = _resolve(post_path.parents[1], str(selection.get("current_corridor_report_path", "")))
    if selection.get("current_corridor_reconciliation_sha256") != _sha(report_path):
        raise ValueError("POST_HOLD_CURRENT_CORRIDOR_REPORT_HASH_MISMATCH")
    report = _load(report_path)
    if report.get("status") != "HOLD_FORMAL_M4_CORRIDOR_INSUFFICIENT" or report.get("sealed") is not True or report.get("immutable") is not True or report.get("retry_forbidden") is not True:
        raise ValueError("POST_HOLD_CURRENT_CORRIDOR_REPORT_NOT_SEALED")
    attempted = {str(row.get("canonical_parent_key", "")) for row in report.get("independent_audit", {}).get("receipt_rows", [])}
    by_suite: dict[str, list[dict[str, Any]]] = {suite: [] for suite in SUITES}
    seen: set[str] = set()
    for parent in parents:
        key = str(parent.get("canonical_parent_key", ""))
        suite = str(parent.get("suite", ""))
        if not key or key in seen or key in attempted or suite not in by_suite or parent.get("taxonomy_status") != "PASS":
            raise ValueError(f"POST_HOLD_PARENT_INVALID:{key}")
        seen.add(key)
        by_suite[suite].append(parent)
    targets = post["qualification"]["target_stable_by_suite"]
    if any(int(targets.get(suite, -1)) < 0 or int(targets.get(suite, -1)) > len(by_suite[suite]) for suite in SUITES):
        raise ValueError("POST_HOLD_TARGET_INVALID")
    if len(by_suite["libero_object"]) != 0:
        raise ValueError("POST_HOLD_OBJECT_POOL_MUST_BE_EMPTY")
    return post, parents


def _runner_files(root: Path, post: dict[str, Any], manifest: dict[str, Any], source_commit: str, source_tree: str) -> tuple[Path, Path]:
    runner_root = root / "_runner_contract"
    runner_manifest = runner_root / "STAGE_V_M4_CORRIDOR_RESERVE_PARENT_MANIFEST_V1.json"
    runner_protocol = runner_root / "STAGE_V_M4_CORRIDOR_QUALIFICATION_PROTOCOL_V1.json"
    runner_auth = runner_root / "STAGE_V_M4_CORRIDOR_RUNTIME_AUTHORIZATION_V1.json"
    compatibility = dict(manifest)
    compatibility.update({"schema": "STAGE_V_M4_CORRIDOR_RESERVE_PARENT_MANIFEST_V1", "compatibility_layer": "POST_HOLD_OUTER_GOVERNANCE"})
    _write(runner_manifest, compatibility)
    protocol = {
        "schema": "STAGE_V_M4_CORRIDOR_QUALIFICATION_PROTOCOL_V1",
        "version": "POST-HOLD-IMMUTABLE-RUNNER-COMPATIBILITY",
        "status": "FROZEN_RUNTIME_AUTHORIZED",
        "runtime_authorized": True,
        "source_binding": {"runtime_commit": source_commit, "runtime_tree": source_tree},
        "inputs": {"candidate_parent_manifest_path": str(runner_manifest), "candidate_parent_manifest_sha256": _sha(runner_manifest)},
        "qualification": {"candidate_parent_count": len(manifest["parents"]), "probe_count": 24, "h_phys": 10, "minimum_remaining_horizon": 20},
        "operation": {"clean_only": True, "intervention_executed": False, "outcomes_read": False, "labels_generated": False},
        "protected_counters": dict(COUNTERS),
        "outer_protocol": str(post.get("schema")),
        "outer_protocol_sha256": _sha(Path(post["_path"])),
    }
    _write(runner_protocol, protocol)
    _write(runner_auth, {"schema": "STAGE_V_M4_CORRIDOR_RUNTIME_AUTHORIZATION_V1", "status": "PASS", "protocol_sha256": _sha(runner_protocol), "protected_counters": dict(COUNTERS), "outer_protocol_sha256": protocol["outer_protocol_sha256"]})
    return runner_protocol, runner_auth


def _receipt(path: Path) -> dict[str, Any]:
    receipt = path / "M4_CORRIDOR_PREFLIGHT.json"
    if not receipt.is_file():
        raise ValueError(f"CORRIDOR_RECEIPT_MISSING:{receipt}")
    value = _load(receipt)
    if value.get("protected_counters") != COUNTERS or value.get("outcomes_read") is not False:
        raise ValueError(f"CORRIDOR_RECEIPT_BOUNDARY_INVALID:{receipt}")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--post-hold-protocol", type=Path, required=True)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--science-runner", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--official-snapshot-root", type=Path, required=True)
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--source-tree", required=True)
    parser.add_argument("--gpus", required=True)
    parser.add_argument("--owner-basis", required=True)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args(argv)
    if not args.owner_basis.strip():
        raise ValueError("OWNER_AUTHORIZATION_BASIS_REQUIRED")
    post_path = args.post_hold_protocol.resolve()
    manifest_path = args.candidate_manifest.resolve()
    post = _load(post_path)
    post["_path"] = str(post_path)
    post, parents = _validate(post_path, manifest_path, args.source_commit, args.source_tree)
    post["_path"] = str(post_path)
    targets = {suite: int(post["qualification"]["target_stable_by_suite"][suite]) for suite in SUITES}
    grouped = {suite: [p for p in parents if p["suite"] == suite] for suite in SUITES}
    plan = {"schema": "STAGE_V_M4_POST_HOLD_SEQUENTIAL_PLAN_V1", "protocol_sha256": _sha(post_path), "candidate_manifest_sha256": _sha(manifest_path), "source_commit": args.source_commit, "source_tree": args.source_tree, "suite_order": list(SUITES), "target_stable_by_suite": targets, "candidate_order": {suite: [p["canonical_parent_key"] for p in grouped[suite]] for suite in SUITES}, "outcomes_read": False, "intervention_executed": False, "protected_counters": dict(COUNTERS)}
    if args.owner_basis != str(post.get("owner_authorization_basis", "")):
        raise ValueError("OWNER_AUTHORIZATION_BASIS_MISMATCH")
    if args.plan_only:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0
    gpus = [int(item.strip()) for item in args.gpus.split(",") if item.strip()]
    if not gpus or len(gpus) != len(set(gpus)):
        raise ValueError("GPU_LIST_INVALID")
    root = args.output_root.resolve()
    if root.exists() and any(root.iterdir()):
        raise ValueError(f"REFUSE_NONEMPTY_OUTPUT_ROOT:{root}")
    root.mkdir(parents=True, exist_ok=False)
    runner_protocol, runner_auth = _runner_files(root, post, {"schema": "STAGE_V_M4_POST_HOLD_CANDIDATE_PARENT_MANIFEST_V1", "status": "FROZEN", "parents": parents}, args.source_commit, args.source_tree)
    jobs: list[dict[str, Any]] = []
    manifest = {**plan, "status": "RUNNING", "started_utc": _utc(), "admitted_gpus": gpus, "owner_authorization_basis": args.owner_basis, "runner": str(args.science_runner), "tasks": jobs}
    launch_path = root / "POST_HOLD_LAUNCH_MANIFEST.json"
    _write(launch_path, manifest)
    for suite in SUITES:
        stable = 0
        if targets[suite] == 0:
            continue
        for parent in grouped[suite]:
            key = str(parent["canonical_parent_key"])
            base = root / "parents" / _safe_key(key)
            result: dict[str, Any] = {"canonical_parent_key": key, "suite": suite, "selection_rank": parent["selection_rank"], "replicates": {}}
            for replicate in ("A", "B"):
                output = base / replicate
                log = root / "logs" / f"{_safe_key(key)}__{replicate}.log"
                log.parent.mkdir(parents=True, exist_ok=True)
                command = [str(args.python), str(args.science_runner), "--protocol", str(runner_protocol), "--authorization", str(runner_auth), "--parent-key", key, "--output-dir", str(output), "--official-snapshot-root", str(args.official_snapshot_root), "--upstream-root", str(args.upstream_root), "--model-root", str(args.model_root), "--gpu", str(gpus[len(jobs) % len(gpus)]), "--source-commit", args.source_commit, "--source-tree", args.source_tree, "--replicate", replicate]
                with log.open("w", encoding="utf-8") as handle:
                    code = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, text=True).returncode
                if code != 0:
                    raise RuntimeError(f"SCIENCE_RUNNER_FAILED:{key}:{replicate}:{code}")
                value = _receipt(output)
                result["replicates"][replicate] = {"status": value.get("status"), "reason": value.get("reason"), "receipt": str(output / "M4_CORRIDOR_PREFLIGHT.json"), "source_commit": value.get("source_commit"), "source_tree": value.get("source_tree")}
            statuses = tuple(result["replicates"][rep]["status"] for rep in ("A", "B"))
            result["status_pair"] = "/".join(statuses)
            result["stable_pass_pass"] = statuses == ("PASS", "PASS")
            jobs.append(result)
            if result["stable_pass_pass"]:
                stable += 1
            manifest["completed_candidate_count"] = len(jobs)
            manifest["stable_by_suite"] = {name: sum(1 for job in jobs if job["suite"] == name and job["stable_pass_pass"]) for name in SUITES}
            _write(launch_path, manifest)
            if stable >= targets[suite]:
                break
    stable_by_suite = {suite: sum(1 for job in jobs if job["suite"] == suite and job["stable_pass_pass"]) for suite in SUITES}
    manifest.update({"status": "COMPLETED_TARGETS_REACHED" if all(stable_by_suite[suite] >= targets[suite] for suite in SUITES) else "HOLD_POOL_EXHAUSTED", "completed_utc": _utc(), "stable_by_suite": stable_by_suite, "outcomes_read": False, "intervention_executed": False, "protected_counters": dict(COUNTERS)})
    _write(launch_path, manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
