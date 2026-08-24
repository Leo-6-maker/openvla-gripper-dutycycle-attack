#!/usr/bin/env python3
"""FIT-only V5 input availability audit.

This is a read-only census.  It never creates Teacher labels, reads a future
split, starts a model, or authorizes training.  Policy-intent and visual
sources are optional inputs so that the audit can distinguish V5-A readiness
from the conditional readiness of multimodal variants.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import uuid
from pathlib import Path
from typing import Any

from gripper_attack.b3_training_protocol import seal_directory, sha256_file, verify_sealed_directory
from gripper_attack.v5_protocol import validate_student_features


def _tree_digest(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(path.stat().st_size).encode("ascii"))
        digest.update(b"\0")
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        digest.update(b"\0")
    return digest.hexdigest()


def _read_registry(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) not in (800, 2000):
        raise ValueError(f"V5 FIT audit requires a complete 800-row FIT or 2000-row global registry, got {len(rows)}")
    seen: set[str] = set()
    fit_rows: list[dict[str, Any]] = []
    for row in rows:
        key = str(row.get("canonical_parent_key", ""))
        parts = key.split("/")
        state = int(row.get("state_id", -1))
        if len(parts) != 3 or state < 0 or state >= 50:
            raise ValueError(f"malformed identity in V5 audit: {key}")
        if key in seen:
            raise ValueError(f"duplicate global identity: {key}")
        seen.add(key)
        row["state_id"] = state
        row["task_idx"] = int(row.get("task_idx", -1))
        if state < 20:
            fit_rows.append(row)
    if len(fit_rows) != 800:
        raise ValueError(f"global registry does not yield exactly 800 FIT rows, got {len(fit_rows)}")
    return fit_rows


def _episode_dirs(root: Path) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for manifest in root.rglob("materialization_manifest.json"):
        try:
            value = json.loads(manifest.read_text(encoding="utf-8"))
            key = str(value["source_identity"]["canonical_parent_key"])
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue
        if key in result:
            raise ValueError(f"duplicate S1 episode manifest: {key}")
        result[key] = manifest.parent
    return result


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"JSON object expected in {path}")
            rows.append(value)
    return rows


def _audit_s1(root: Path, registry: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    by_key = {row["canonical_parent_key"]: row for row in registry}
    episodes = _episode_dirs(root)
    rows: list[dict[str, Any]] = []
    for key, registry_row in sorted(by_key.items()):
        episode = episodes.get(key)
        record = {"canonical_parent_key": key, "student_status": "HOLD", "step_count": 0, "error": ""}
        if episode is None:
            record["error"] = "missing materialized episode"
            rows.append(record)
            continue
        student_path = episode / "student_input_records.jsonl"
        teacher_path = episode / "teacher_retention_records.jsonl"
        try:
            students = _jsonl(student_path)
            teachers = _jsonl(teacher_path)
            if not students or len(students) != len(teachers):
                raise ValueError("student/teacher stream missing or misaligned")
            for index, student in enumerate(students):
                validate_student_features(student)
                if int(student.get("step", -1)) != index or student.get("canonical_parent_key") != key:
                    raise ValueError(f"student identity/step mismatch at {index}")
            record.update({"student_status": "PASS", "step_count": len(students)})
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            record["error"] = str(exc)
        rows.append(record)
    summary = {
        "root": str(root),
        "root_tree_sha256": _tree_digest(root),
        "registry_identity_count": len(registry),
        "episode_manifest_count": len(episodes),
        "student_pass_count": sum(row["student_status"] == "PASS" for row in rows),
        "student_hold_count": sum(row["student_status"] != "PASS" for row in rows),
        "fit_data_contract_status": "PASS" if all(row["student_status"] == "PASS" for row in rows) and set(episodes) == set(by_key) else "HOLD",
    }
    return summary, rows


def _source_inventory(root: Path | None, label: str) -> dict[str, Any]:
    if root is None:
        return {"source": label, "status": "NOT_SUPPLIED", "root": "", "file_count": 0, "tree_sha256": ""}
    if not root.is_dir():
        return {"source": label, "status": "HOLD", "root": str(root), "file_count": 0, "tree_sha256": "", "error": "root missing"}
    paths = [path for path in root.rglob("*") if path.is_file()]
    if label == "policy_intent":
        paths = [path for path in paths if path.name == "policy_intent_9d_records.jsonl"]
    elif label == "causal_visual":
        allowed = {".png", ".jpg", ".jpeg", ".npy", ".npz", ".pt", ".safetensors"}
        paths = [path for path in paths if path.suffix.lower() in allowed]
    forbidden_names = [path.relative_to(root).as_posix() for path in paths if any(token in path.name.lower() for token in ("mask", "object_pose", "contact", "privileged"))]
    return {
        "source": label,
        "status": "PASS" if paths and not forbidden_names else ("HOLD" if forbidden_names else "NOT_FOUND"),
        "root": str(root),
        "file_count": len(paths),
        "tree_sha256": _tree_digest(root),
        "forbidden_filename_count": len(forbidden_names),
        "forbidden_filename_examples": forbidden_names[:10],
    }


def audit(args: argparse.Namespace) -> dict[str, Any]:
    registry_path = Path(args.registry_csv).resolve()
    s1_root = Path(args.s1_root).resolve()
    output = Path(args.output_root).resolve()
    registry = _read_registry(registry_path)
    s1_summary, rows = _audit_s1(s1_root, registry)
    policy = _source_inventory(Path(args.policy_intent_root).resolve() if args.policy_intent_root else None, "policy_intent")
    visual = _source_inventory(Path(args.visual_root).resolve() if args.visual_root else None, "causal_visual")
    return {
        "schema": "DETECTOR_V5_INPUT_AVAILABILITY_AUDIT_V1",
        "status": "PASS_DATA_ONLY" if s1_summary["fit_data_contract_status"] == "PASS" else "HOLD",
        "formal_training_authorized": False,
        "formal_attack_authorized": False,
        "protected_splits_read": [],
        "registry": {"path": str(registry_path), "sha256": sha256_file(registry_path), "identity_count": len(registry)},
        "s1": s1_summary,
        "policy_intent": policy,
        "causal_visual": visual,
        "variant_readiness": {
            "V5_A_PROPRIO": s1_summary["fit_data_contract_status"] == "PASS",
            "V5_B_PROPRIO_POLICY_INTENT": s1_summary["fit_data_contract_status"] == "PASS" and policy["status"] == "PASS",
            "V5_C_PROPRIO_CAUSAL_VISUAL": s1_summary["fit_data_contract_status"] == "PASS" and visual["status"] == "PASS",
            "V5_D_PROPRIO_POLICY_INTENT_CAUSAL_VISUAL": s1_summary["fit_data_contract_status"] == "PASS" and policy["status"] == "PASS" and visual["status"] == "PASS",
        },
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry-csv", required=True)
    parser.add_argument("--s1-root", required=True)
    parser.add_argument("--policy-intent-root")
    parser.add_argument("--visual-root")
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    output = Path(args.output_root).resolve()
    if output.exists():
        raise SystemExit(f"refusing to overwrite output root: {output}")
    result = audit(args)
    staging = output.with_name(f".{output.name}.{uuid.uuid4().hex}.staging")
    try:
        staging.mkdir(parents=True)
        (staging / "audit.json").write_text(json.dumps({key: value for key, value in result.items() if key != "rows"}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        with (staging / "identity_rows.csv").open("w", newline="", encoding="utf-8") as handle:
            fieldnames = ["canonical_parent_key", "student_status", "step_count", "error"]
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(result["rows"])
        (staging / "source_inventory.json").write_text(json.dumps({key: result[key] for key in ("registry", "s1", "policy_intent", "causal_visual")}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        seal_directory(staging)
        os.replace(staging, output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}, sort_keys=True))
    return 0 if result["status"] == "PASS_DATA_ONLY" else 2


if __name__ == "__main__":
    raise SystemExit(main())
