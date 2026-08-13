#!/usr/bin/env python3
"""Read-only closure audit for a parent stopped before its first branch."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


COUNTERS = {
    "protected_reads": 0,
    "eval160_reads": 0,
    "attack_rollouts": 0,
    "vis_pgd_attack_rollouts": 0,
}
FORBIDDEN_SCIENCE_ARTIFACTS = {
    "M4_COUNTERFACTUAL_BRANCHES_V1.jsonl",
    "M4_TREATMENT_OBSERVATIONS_V1.jsonl",
    "M4_V_PHYS_LABELS_V1.jsonl",
    "PARENT_RESULT.json",
    "M4_INDEPENDENT_AUDIT.json",
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"JSON_OBJECT_REQUIRED:{path}")
    return dict(value)


def audit(parent_root: Path, output_root: Path, parent_key: str, auditor_source_commit: str = "") -> dict[str, Any]:
    parent_root = parent_root.resolve()
    output_root = output_root.resolve()
    if not parent_root.is_dir():
        raise FileNotFoundError(parent_root)
    if output_root.exists():
        raise FileExistsError(f"OUTPUT_ROOT_EXISTS:{output_root}")
    gate = parent_root / "gate"
    status_path = gate / "PARENT_STATUS.json"
    release_path = gate / "RESOURCE_RELEASE.json"
    if not status_path.is_file() or not release_path.is_file():
        raise FileNotFoundError("PARENT_STATUS_OR_RESOURCE_RELEASE_MISSING")
    status = _load(status_path)
    release = _load(release_path)
    errors: list[str] = []
    if status.get("schema") != "STAGE_V_M4_FORMAL_PARENT_STATUS_V2":
        errors.append("PARENT_STATUS_SCHEMA")
    if status.get("canonical_parent_key") != parent_key:
        errors.append("PARENT_IDENTITY")
    if status.get("status") != "HOLD_FORMAL_M4_STRUCTURAL_FAILURE":
        errors.append("PARENT_STATUS_NOT_STRUCTURAL_HOLD")
    for field in ("intervention_started", "intervention_executed", "m4_outcomes_materialized", "v_phys_generated", "outcomes_read"):
        if status.get(field) is not False:
            errors.append(f"PARENT_{field.upper()}")
    if status.get("protected_counters") != COUNTERS:
        errors.append("PARENT_PROTECTED_COUNTERS")
    if release.get("schema") != "STAGE_V_M4_FORMAL_RESOURCE_RELEASE_V1" or release.get("status") != "PASS" or release.get("release_ok") is not True:
        errors.append("RESOURCE_RELEASE")
    if release.get("outcomes_read") is not False or release.get("protected_counters") != COUNTERS:
        errors.append("RELEASE_BOUNDARY")

    science = parent_root / "science"
    present = {path.name for path in science.iterdir()} if science.is_dir() else set()
    forbidden_present = sorted(present & FORBIDDEN_SCIENCE_ARTIFACTS)
    if forbidden_present:
        errors.append("SCIENCE_ARTIFACTS_PRESENT")

    input_hashes = {str(path.relative_to(parent_root).as_posix()): _sha(path) for path in (status_path, release_path)}
    report = {
        "schema": "STAGE_V_M4_PREBRANCH_ABORT_CLOSURE_V1",
        "status": "PASS_PREBRANCH_ABORT_CLOSURE" if not errors else "HOLD_PREBRANCH_CLOSURE_INVALID",
        "sealed": True,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "parent_key": parent_key,
        "auditor_source_commit": auditor_source_commit,
        "parent_root": str(parent_root),
        "input_sha256": input_hashes,
        "parent_status_sha256": _sha(status_path),
        "resource_release_sha256": _sha(release_path),
        "branch_records": 0,
        "primary_window_steps": 0,
        "forced_open_steps": 0,
        "treatment_receipts": 0,
        "binary_consumable_label_count": 0,
        "v_phys_generated": False,
        "intervention_executed": False,
        "outcomes_read": False,
        "outcomes_read_uncertain": status.get("outcomes_read_uncertain") is True,
        "protected_counters": dict(COUNTERS),
        "forbidden_science_artifacts_present": forbidden_present,
        "errors": sorted(set(errors)),
    }
    output_root.mkdir(parents=True, exist_ok=False)
    report_path = output_root / "PREBRANCH_ABORT_CLOSURE.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    sums_path = output_root / "SHA256SUMS"
    sums_path.write_text(f"{_sha(report_path)}  {report_path.name}\n", encoding="utf-8")
    (output_root / "SHA256SUMS.sha256").write_text(f"{_sha(sums_path)}  SHA256SUMS\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--parent-key", required=True)
    parser.add_argument("--auditor-source-commit", required=True)
    args = parser.parse_args(argv)
    try:
        report = audit(args.parent_root, args.output_root, args.parent_key, args.auditor_source_commit)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(json.dumps({"status": "HOLD_PREBRANCH_CLOSURE_ERROR", "error": f"{type(exc).__name__}:{exc}"}, sort_keys=True))
        return 2
    print(json.dumps({key: report[key] for key in ("status", "branch_records", "primary_window_steps", "forced_open_steps", "treatment_receipts", "binary_consumable_label_count", "errors")}, sort_keys=True))
    return 0 if report["status"] == "PASS_PREBRANCH_ABORT_CLOSURE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
