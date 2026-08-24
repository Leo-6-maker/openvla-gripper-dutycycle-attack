#!/usr/bin/env python3
"""Aggregate and audit the Stage VI clean-only R3 reconstructions."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from gripper_attack.v5_r3_teacher import (  # noqa: E402
    canonicalize_fit670_episode,
    derive_episode_labels,
    validate_contact_row,
)

COUNTERS = {
    "protected_reads": 0,
    "eval160_reads": 0,
    "attack_rollouts": 0,
    "vis_pgd_attack_rollouts": 0,
}
FORBIDDEN_FIELDS = {
    "task_success", "terminal", "terminal_state", "reward", "outcome",
    "attack_result", "future", "future_frame", "future_label",
}
HEADS = ("physical_criticality", "k10_feasibility", "safe_release", "instability", "gripper_closing_state")
TRUTH_VALUES = ("TRUE", "FALSE", "UNKNOWN")


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"JSON_OBJECT_REQUIRED:{path}")
    return dict(value)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def git(*args: str) -> str:
    return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True).strip()


def forbidden_paths(value: Any, path: str = "") -> list[str]:
    found: list[str] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            child = f"{path}.{key}" if path else str(key)
            if str(key) in FORBIDDEN_FIELDS:
                found.append(child)
            found.extend(forbidden_paths(item, child))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found.extend(forbidden_paths(item, f"{path}[{index}]"))
    return found


def verify_sealed_root(root: Path) -> str:
    sums = root / "SHA256SUMS"
    sidecar = root / "SHA256SUMS.sha256"
    if not sums.is_file() or not sidecar.is_file():
        raise ValueError(f"RECONSTRUCTION_SEAL_MISSING:{root}")
    parts = sidecar.read_text(encoding="utf-8").strip().split()
    if parts != [sha256_file(sums), "SHA256SUMS"]:
        raise ValueError(f"RECONSTRUCTION_SEAL_INVALID:{root}")
    seen: set[Path] = set()
    for line in sums.read_text(encoding="utf-8").splitlines():
        expected, separator, relative = line.partition("  ")
        if not separator or not expected or not relative:
            raise ValueError(f"RECONSTRUCTION_SUM_LINE_INVALID:{root}")
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"RECONSTRUCTION_SUM_PATH_INVALID:{root}:{relative}")
        target = (root / path).resolve()
        if target in seen or not target.is_file() or root.resolve() not in target.parents:
            raise ValueError(f"RECONSTRUCTION_SUM_TARGET_INVALID:{root}:{relative}")
        if sha256_file(target) != expected:
            raise ValueError(f"RECONSTRUCTION_SUM_MISMATCH:{root}:{relative}")
        seen.add(target)
    return parts[0]


def exact_parent_keys(manifest: Mapping[str, Any]) -> list[str]:
    parents = manifest.get("parents")
    if not isinstance(parents, list):
        raise ValueError("EXACT_PLAN_PARENTS_MISSING")
    keys = [str(row["canonical_parent_key"]) for row in parents if isinstance(row, Mapping) and "canonical_parent_key" in row]
    if len(keys) != 40 or len(set(keys)) != 40:
        raise ValueError(f"EXACT_PLAN_PARENT_COUNT:{len(keys)}")
    return keys


def choose_full_roots(base: Path, expected: list[str]) -> tuple[dict[str, Path], list[dict[str, Any]]]:
    candidates: list[dict[str, Any]] = []
    full: dict[str, list[Path]] = {}
    for root in sorted(path for path in base.glob("STAGE_VI_M4_R3_RECONSTRUCTION_*") if path.is_dir()):
        validation_path = root / "R3_COVERAGE_VALIDATION.json"
        if not validation_path.is_file():
            continue
        validation = read_json(validation_path)
        identity = str(validation.get("canonical_parent_key") or "")
        status = str(validation.get("status") or "")
        candidates.append({"root": str(root), "canonical_parent_key": identity, "status": status, "rows": validation.get("rows_replayed")})
        if status == "PASS_FULL_CLEAN_R3_COVERAGE":
            full.setdefault(identity, []).append(root)
    missing = sorted(set(expected) - set(full))
    duplicate = sorted(identity for identity, roots in full.items() if len(roots) != 1)
    extra = sorted(set(full) - set(expected))
    if missing or duplicate or extra:
        raise ValueError(f"RECONSTRUCTION_PARENT_MAPPING_INVALID:missing={missing}:duplicate={duplicate}:extra={extra}")
    return {identity: roots[0] for identity, roots in full.items()}, candidates


def audit_parent(root: Path, identity: str, protocol: Mapping[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    root_seal = verify_sealed_root(root)
    validation = read_json(root / "R3_COVERAGE_VALIDATION.json")
    episode = read_json(root / "RECONSTRUCTED_FIT670_EPISODE.json")
    if validation.get("status") != "PASS_FULL_CLEAN_R3_COVERAGE" or validation.get("canonical_parent_key") != identity:
        raise ValueError(f"RECONSTRUCTION_VALIDATION_INVALID:{identity}")
    if any(validation.get(key) not in expected for key, expected in {
        "intervention_executed": (False,), "labels_generated": (0,), "outcomes_read": (False,), "protected_counters": (COUNTERS,),
    }.items()):
        raise ValueError(f"RECONSTRUCTION_BOUNDARY_INVALID:{identity}")
    if episode.get("schema") != "FIT670_EPISODE_V2" or episode.get("episode_id") != identity:
        raise ValueError(f"RECONSTRUCTION_EPISODE_ID_INVALID:{identity}")
    if any(episode.get(key) is not expected for key, expected in {
        "model_inference": False, "attack_enabled": False, "detector_loaded": False,
        "teacher_labels_generated": False, "outcomes_read": False,
    }.items()):
        raise ValueError(f"RECONSTRUCTION_EPISODE_BOUNDARY_INVALID:{identity}")
    forbidden = forbidden_paths(episode)
    if forbidden:
        raise ValueError(f"RECONSTRUCTION_FORBIDDEN_FIELDS:{identity}:{forbidden[:3]}")
    rows = canonicalize_fit670_episode(episode)
    for step, row in enumerate(rows):
        validate_contact_row(row, expected_step=step)
    if len(rows) != int(validation["rows_replayed"]) or len(rows) != int(validation["rows_available"]):
        raise ValueError(f"RECONSTRUCTION_ROW_COUNT_INVALID:{identity}")
    labels = derive_episode_labels(rows, protocol)
    if len(labels) != len(rows):
        raise ValueError(f"RECONSTRUCTION_LABEL_COUNT_INVALID:{identity}")
    for label in labels:
        if any(str(label["labels"][head]["value"]) not in TRUTH_VALUES for head in HEADS):
            raise ValueError(f"RECONSTRUCTION_LABEL_VALUE_INVALID:{identity}:{label['step']}")
    return {
        "canonical_parent_key": identity,
        "reconstruction_root": str(root),
        "reconstruction_root_sha256s_sha256": root_seal,
        "row_count": len(rows),
        "r3_rows_validated": int(validation["r3_rows_validated"]),
        "status": "PASS_CLEAN_ONLY_R3_RECONSTRUCTION_AUDIT",
    }, labels


def seal_output(root: Path) -> str:
    files = sorted(path for path in root.iterdir() if path.is_file() and path.name not in {"SHA256SUMS", "SHA256SUMS.sha256", "ROOT_SEAL.json"})
    (root / "SHA256SUMS").write_text("".join(f"{sha256_file(path)}  {path.name}\n" for path in files), encoding="utf-8")
    digest = sha256_file(root / "SHA256SUMS")
    (root / "SHA256SUMS.sha256").write_text(f"{digest}  SHA256SUMS\n", encoding="utf-8")
    write_json(root / "ROOT_SEAL.json", {
        "schema": "STAGE_VI_M4_R3_CLEAN_RECONSTRUCTION_AGGREGATE_ROOT_SEAL_V1",
        "status": "PASS_CLEAN_ONLY_R3_RECONSTRUCTION_AGGREGATE",
        "sha256sums_sha256": digest,
        "protected_counters": dict(COUNTERS),
        "eval160_status": "UNREAD",
    })
    return digest


def run(args: argparse.Namespace) -> int:
    source_commit, source_tree = git("rev-parse", "HEAD"), git("rev-parse", "HEAD^{tree}")
    if source_commit != args.source_commit or source_tree != args.source_tree:
        raise ValueError("SOURCE_COMMIT_OR_TREE_MISMATCH")
    exact_root = args.exact_plan_root.resolve()
    manifest_path = exact_root / "EXACT_PROBE_AND_SNAPSHOT_MANIFEST.json"
    manifest = read_json(manifest_path)
    expected = exact_parent_keys(manifest)
    selected, candidates = choose_full_roots(args.reconstruction_base.resolve(), expected)
    protocol = read_json(args.protocol_path.resolve())
    entries: list[dict[str, Any]] = []
    all_labels: list[dict[str, Any]] = []
    label_counts = {head: {value: 0 for value in TRUTH_VALUES} for head in HEADS}
    for identity in expected:
        entry, labels = audit_parent(selected[identity], identity, protocol)
        entries.append(entry)
        for label in labels:
            output = {"canonical_parent_key": identity, "step": int(label["step"]), "labels": label["labels"], "right_censored": bool(label.get("right_censored")), "evidence_fields": label.get("evidence_fields", {})}
            all_labels.append(output)
            for head in HEADS:
                label_counts[head][str(label["labels"][head]["value"])] += 1
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    write_json(output / "PARENT_INDEX.json", {
        "schema": "STAGE_VI_M4_R3_CLEAN_RECONSTRUCTION_PARENT_INDEX_V1",
        "status": "PASS_EXACT_PLAN_40_UNIQUE_FULL_CLEAN_ROOTS",
        "exact_plan_manifest": str(manifest_path),
        "exact_plan_manifest_sha256": sha256_file(manifest_path),
        "parents": entries,
        "candidate_roots_audited": candidates,
    })
    with (output / "M4_R3_TEACHER_LABELS_DIAGNOSTIC.jsonl").open("w", encoding="utf-8") as handle:
        for row in all_labels:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n")
    write_json(output / "AGGREGATE_AUDIT.json", {
        "schema": "STAGE_VI_M4_R3_CLEAN_RECONSTRUCTION_AGGREGATE_AUDIT_V1",
        "status": "PASS_CLEAN_ONLY_R3_RECONSTRUCTION_AGGREGATE",
        "parent_count": len(entries),
        "total_r3_rows": len(all_labels),
        "label_counts": label_counts,
        "intervention_executed": False,
        "outcomes_read": False,
        "teacher_labels_are_diagnostic_only": True,
        "not_formal_m4_consumable_labels": True,
        "protected_counters": dict(COUNTERS),
        "eval160_status": "UNREAD",
    })
    write_json(output / "PROVENANCE.json", {
        "schema": "STAGE_VI_M4_R3_CLEAN_RECONSTRUCTION_AGGREGATE_PROVENANCE_V1",
        "status": "PASS_DIAGNOSTIC_ONLY",
        "source_commit": source_commit,
        "source_tree": source_tree,
        "source_status": git("status", "--porcelain"),
        "diagnostic_code_sha256": sha256_file(Path(__file__)),
        "exact_plan_manifest": str(manifest_path),
        "exact_plan_manifest_sha256": sha256_file(manifest_path),
        "protocol_path": str(args.protocol_path.resolve()),
        "protocol_sha256": sha256_file(args.protocol_path.resolve()),
        "reconstruction_base": str(args.reconstruction_base.resolve()),
        "candidate_root_count": len(candidates),
        "selected_parent_count": len(entries),
        "intervention_executed": False,
        "outcomes_read": False,
        "teacher_labels_generated": True,
        "teacher_labels_use": "DIAGNOSTIC_JOIN_ONLY_NO_STUDENT_TRAINING_NO_FORMAL_M4_CONSUMPTION",
        "protected_counters": dict(COUNTERS),
        "eval160_status": "UNREAD",
    })
    digest = seal_output(output)
    print(json.dumps({"status": "PASS", "parent_count": len(entries), "r3_rows": len(all_labels), "root": str(output), "root_seal": digest}, sort_keys=True))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reconstruction-base", type=Path, required=True)
    parser.add_argument("--exact-plan-root", type=Path, required=True)
    parser.add_argument("--protocol-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--source-tree", required=True)
    args = parser.parse_args(argv)
    try:
        return run(args)
    except Exception as exc:
        print(json.dumps({"status": "FAIL", "reason": f"{type(exc).__name__}:{exc}"}, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
