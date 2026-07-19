#!/usr/bin/env python3
"""Independent read-only auditor for R7 K10 Opportunity Labeler V1.2.1.

This script does not trust the generator's AUDIT.json. It independently verifies:
- exact checksum/file-set closure for the Physics Teacher and K10 output roots;
- all 800 FIT identities and every Physics V2.1 source row;
- output identity/step closure and exact recomputation of every K=10 start;
- all ten steps of each positive burst are known, Student-valid, critical,
  release/regrasp-valid, nonzero-component, and in one candidate segment;
- SOURCE_BINDING matches the clean repository worktree and labeler bytes;
- summary counts agree with the recomputed labels.

The auditor writes one JSON report outside the sealed label root and exits nonzero
on any failure. It never mutates the source or label roots.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

K = 10
SUITES = ("libero_10", "libero_goal", "libero_object", "libero_spatial")
TASKS = range(10)
FIT_STATES = range(20)
TEACHER_SHA256SUMS = "18f3520351e1291e462656fb1236baa5bc1b5136848a10174e0a4010cc3d38da"
TEACHER_SCHEMA = "DETECTOR_V5_PHYSICS_TEACHER_PROTOCOL_V21"
LABEL_SCHEMA = "R7_K10_OPPORTUNITY_LABELER_V1_2_1"
LABELER_PATH = Path("scripts/detector_v4/label_k10_v121.py")

REQUIRED_TEACHER_FIELDS = {
    "known_mask", "student_valid", "candidate_close", "stable_grasp_score",
    "lift_score", "support_removed", "target_progress", "target_progress_known",
    "release_risk", "regrasp_or_instability_risk", "task_grasp_necessity",
    "component_valid_mask", "phase_name", "window_id", "step",
    "physics_protocol_schema", "canonical_parent_key", "suite", "task_idx",
    "state_id",
}
REQUIRED_VALIDITY_KEYS = {
    "relative_pose_stability", "object_eef_comotion_score", "lift_score",
    "target_progress", "support_removed", "release_risk",
    "regrasp_or_instability_risk",
}
NUMERIC_FIELDS = {
    "stable_grasp_score", "lift_score", "support_removed", "target_progress",
    "release_risk", "regrasp_or_instability_risk", "task_grasp_necessity",
}
BOOL_FIELDS = {"known_mask", "student_valid", "candidate_close", "target_progress_known"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def verify_sealed_root(root: Path, *, expected_sums_sha: str | None = None) -> dict[str, Any]:
    sums = root / "SHA256SUMS"
    sidecar = root / "SHA256SUMS.sha256"
    if not sums.is_file() or not sidecar.is_file():
        raise ValueError(f"missing checksum files: {root}")
    sums_sha = sha256_file(sums)
    if expected_sums_sha is not None and sums_sha != expected_sums_sha:
        raise ValueError(f"SHA256SUMS digest mismatch: expected {expected_sums_sha}, got {sums_sha}")
    expected_sidecar = f"{sums_sha}  SHA256SUMS"
    if sidecar.read_text(encoding="utf-8").strip() != expected_sidecar:
        raise ValueError(f"checksum sidecar mismatch: {root}")

    listed: set[str] = set()
    for raw in sums.read_text(encoding="utf-8").splitlines():
        digest, sep, name = raw.partition("  ")
        if not sep or not name or name in listed:
            raise ValueError(f"invalid checksum row: {raw!r}")
        rel = Path(name)
        if rel.is_absolute() or ".." in rel.parts or name in {"SHA256SUMS", "SHA256SUMS.sha256"}:
            raise ValueError(f"unsafe checksum path: {name}")
        target = root / rel
        if not target.is_file() or sha256_file(target) != digest:
            raise ValueError(f"checksum mismatch: {name}")
        listed.add(rel.as_posix())

    actual = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    expected = listed | {"SHA256SUMS", "SHA256SUMS.sha256"}
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(f"sealed file-set mismatch; missing={missing[:5]} extra={extra[:5]}")
    return {"sha256sums_sha256": sums_sha, "payload_file_count": len(listed)}


def expected_identity(suite: str, task: int, state: int) -> str:
    return f"{suite}/task_{task:02d}/state_{state:02d}"


def validate_teacher_root(root: Path) -> dict[str, Any]:
    seal = verify_sealed_root(root, expected_sums_sha=TEACHER_SHA256SUMS)
    row_count = 0
    label_files = 0
    for suite in SUITES:
        for task in TASKS:
            for state in FIT_STATES:
                identity = expected_identity(suite, task, state)
                path = root / "labels" / suite / f"task_{task:02d}" / f"state_{state:02d}" / "physics_teacher_v21.jsonl"
                if not path.is_file():
                    raise ValueError(f"missing Teacher identity: {identity}")
                rows = load_jsonl(path)
                if not rows:
                    raise ValueError(f"empty Teacher identity: {identity}")
                label_files += 1
                for index, row in enumerate(rows):
                    row_count += 1
                    missing = sorted(REQUIRED_TEACHER_FIELDS - set(row))
                    if missing:
                        raise ValueError(f"Teacher missing fields at {identity}:{index}: {missing}")
                    if row["physics_protocol_schema"] != TEACHER_SCHEMA:
                        raise ValueError(f"Teacher schema mismatch at {identity}:{index}")
                    if row["step"] != index or not isinstance(row["step"], int):
                        raise ValueError(f"Teacher step discontinuity at {identity}:{index}")
                    if row["canonical_parent_key"] != identity:
                        raise ValueError(f"Teacher identity mismatch at {identity}:{index}")
                    if row["suite"] != suite or int(row["task_idx"]) != task or int(row["state_id"]) != state:
                        raise ValueError(f"Teacher path metadata mismatch at {identity}:{index}")
                    for field in BOOL_FIELDS:
                        if type(row[field]) is not bool:
                            raise ValueError(f"Teacher non-bool {field} at {identity}:{index}")
                    for field in NUMERIC_FIELDS:
                        value = row[field]
                        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                            raise ValueError(f"Teacher non-finite/non-numeric {field} at {identity}:{index}")
                    mask = row["component_valid_mask"]
                    if not isinstance(mask, dict) or not REQUIRED_VALIDITY_KEYS.issubset(mask):
                        raise ValueError(f"Teacher validity mask incomplete at {identity}:{index}")
                    for key in REQUIRED_VALIDITY_KEYS:
                        if type(mask[key]) is not bool:
                            raise ValueError(f"Teacher validity key not bool: {key} at {identity}:{index}")
                    window_id = row["window_id"]
                    if not isinstance(window_id, str):
                        raise ValueError(f"Teacher window_id not string at {identity}:{index}")
                    if row["candidate_close"] and not window_id.startswith("candidate:"):
                        raise ValueError(f"candidate_close/window_id mismatch at {identity}:{index}")
    if label_files != 800:
        raise ValueError(f"expected 800 Teacher label files, got {label_files}")
    return {**seal, "identity_count": label_files, "row_count": row_count}


def recompute_start(rows: list[dict[str, Any]], start: int) -> bool:
    if start < 0 or start + K > len(rows):
        return False
    window = rows[start : start + K]
    wid = window[0].get("window_id")
    return bool(
        isinstance(wid, str)
        and wid.startswith("candidate:")
        and all(
            row.get("critical_t") is True
            and row.get("known_mask") is True
            and row.get("student_valid") is True
            and row.get("candidate_close") is True
            and row.get("window_id") == wid
            and row.get("release_risk_valid") is True
            and row.get("regrasp_risk_valid") is True
            and isinstance(row.get("component_bitmask"), int)
            and 1 <= row["component_bitmask"] <= 7
            for row in window
        )
    )


def audit_label_episode(rows: list[dict[str, Any]], identity: str) -> dict[str, Any]:
    if not rows:
        raise ValueError(f"empty label episode: {identity}")
    starts = 0
    for index, row in enumerate(rows):
        if row.get("step") != index or row.get("episode_key") != identity:
            raise ValueError(f"output identity/step mismatch at {identity}:{index}")
        for field in ("candidate_close", "known_mask", "student_valid", "critical_t", "burst_feasible_t", "is_feasible_start", "release_risk_valid", "regrasp_risk_valid"):
            if type(row.get(field)) is not bool:
                raise ValueError(f"output non-bool {field} at {identity}:{index}")
        actual = recompute_start(rows, index)
        if row["burst_feasible_t"] != actual or row["is_feasible_start"] != actual:
            raise ValueError(f"burst recomputation mismatch at {identity}:{index}")
        if actual:
            starts += 1
    return {"n_steps": len(rows), "feasible_start_count": starts, "has_feasible_k10": starts > 0}


def git_text(repo_root: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo_root), *args], text=True).strip()


def validate_binding(label_root: Path, repo_root: Path) -> dict[str, Any]:
    binding = load_json(label_root / "SOURCE_BINDING.json")
    if binding.get("labeler_schema") != LABEL_SCHEMA or int(binding.get("K", -1)) != K:
        raise ValueError("SOURCE_BINDING schema/K mismatch")
    if binding.get("teacher_sha256sums") != TEACHER_SHA256SUMS:
        raise ValueError("SOURCE_BINDING Teacher digest mismatch")
    head = git_text(repo_root, "rev-parse", "HEAD")
    if binding.get("git_commit") != head:
        raise ValueError(f"SOURCE_BINDING commit mismatch: {binding.get('git_commit')} != {head}")
    if git_text(repo_root, "status", "--porcelain"):
        raise ValueError("repository worktree is dirty during independent audit")
    labeler = repo_root / LABELER_PATH
    if not labeler.is_file():
        raise ValueError(f"labeler missing from repo root: {labeler}")
    labeler_sha = sha256_file(labeler)
    if binding.get("labeler_blob_sha256") != labeler_sha:
        raise ValueError("SOURCE_BINDING labeler SHA mismatch")
    return {"git_commit": head, "labeler_sha256": labeler_sha}


def audit_label_root(label_root: Path, repo_root: Path) -> dict[str, Any]:
    seal = verify_sealed_root(label_root)
    binding = validate_binding(label_root, repo_root)
    protocol = load_json(label_root / "PROTOCOL.json")
    if protocol.get("schema") != LABEL_SCHEMA or int(protocol.get("K", -1)) != K:
        raise ValueError("PROTOCOL schema/K mismatch")
    manifest = load_json(label_root / "MANIFEST.json")
    if manifest.get("schema") != LABEL_SCHEMA or manifest.get("teacher_sha256sums") != TEACHER_SHA256SUMS:
        raise ValueError("MANIFEST mismatch")

    recomputed: dict[str, dict[str, Any]] = {}
    for suite in SUITES:
        for task in TASKS:
            for state in FIT_STATES:
                identity = expected_identity(suite, task, state)
                path = label_root / "labels" / suite / f"task_{task:02d}" / f"state_{state:02d}" / "k10_labels_v121.jsonl"
                if not path.is_file():
                    raise ValueError(f"missing output identity: {identity}")
                recomputed[identity] = audit_label_episode(load_jsonl(path), identity)
    if len(recomputed) != 800:
        raise ValueError(f"expected 800 output identities, got {len(recomputed)}")

    n_feasible = sum(int(item["has_feasible_k10"]) for item in recomputed.values())
    total_starts = sum(int(item["feasible_start_count"]) for item in recomputed.values())
    audit = load_json(label_root / "AUDIT.json")
    if int(audit.get("n_episodes", -1)) != 800 or int(audit.get("n_feasible", -1)) != n_feasible or int(audit.get("total_starts", -1)) != total_starts:
        raise ValueError("AUDIT.json aggregate mismatch")
    gates = audit.get("gates", {})
    if not gates or not all(value is True for value in gates.values()):
        raise ValueError("generator AUDIT gates are not all true")

    summary_path = label_root / "EPISODE_SUMMARY.csv"
    with summary_path.open(newline="", encoding="utf-8") as handle:
        summary = list(csv.DictReader(handle))
    if len(summary) != 800 or len({row["identity"] for row in summary}) != 800:
        raise ValueError("EPISODE_SUMMARY identity closure failure")
    for row in summary:
        identity = row["identity"]
        if identity not in recomputed:
            raise ValueError(f"unexpected summary identity: {identity}")
        expected = recomputed[identity]
        if int(row["feasible_start_count"]) != expected["feasible_start_count"]:
            raise ValueError(f"summary start-count mismatch: {identity}")
        parsed_has = str(row["has_feasible_k10"]).lower() == "true"
        if parsed_has != expected["has_feasible_k10"]:
            raise ValueError(f"summary feasibility mismatch: {identity}")

    task_path = label_root / "TASK_GEOMETRY.csv"
    with task_path.open(newline="", encoding="utf-8") as handle:
        task_rows = list(csv.DictReader(handle))
    if len(task_rows) != 40 or len({row["task"] for row in task_rows}) != 40:
        raise ValueError("TASK_GEOMETRY closure failure")

    return {
        **seal,
        **binding,
        "identity_count": 800,
        "n_feasible": n_feasible,
        "total_starts": total_starts,
        "protocol_sha256": sha256_file(label_root / "PROTOCOL.json"),
        "manifest_sha256": sha256_file(label_root / "MANIFEST.json"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-root", type=Path, required=True)
    parser.add_argument("--label-root", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report: dict[str, Any] = {"schema": "R7_K10_V121_INDEPENDENT_AUDIT_V1", "status": "HOLD"}
    try:
        report["teacher"] = validate_teacher_root(args.teacher_root.resolve())
        report["labels"] = audit_label_root(args.label_root.resolve(), args.repo_root.resolve())
        report["status"] = "PASS"
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] != "PASS":
        sys.exit(1)


if __name__ == "__main__":
    main()
