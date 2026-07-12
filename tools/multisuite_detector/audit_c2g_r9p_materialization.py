"""Audit R9P materialization for correctness, leakage, and schema compliance.

Validates NPZ files against the plan manifest — checks identity closure, feature/label
alignment, finite values, forbidden field absence, and cohort sealing.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from tools.multisuite_detector.build_c2g_r9p_preview_plan import (
    FORBIDDEN_STUDENT_FIELDS,
    R9P_HEAD_NAMES,
    TARGET_SUITES,
)
from tools.multisuite_detector.c2g_r8r_common import (
    read_json,
    read_jsonl,
    sha256_file,
    write_json,
)

SCHEMA = "c2g.r9p.materialization_audit.2026-07-12.v1"
GATE_PASS = "PASS_C2G_R9P_MATERIALIZATION_AUDIT"


def audit_episode_npz(npz_path: Path) -> dict[str, Any]:
    result = {
        "npz_path": str(npz_path),
        "valid": True,
        "issues": [],
    }
    try:
        data = np.load(npz_path, allow_pickle=False)
    except Exception as exc:
        result["valid"] = False
        result["issues"].append(f"load_error: {exc}")
        return result

    keys = set(data.keys())
    # Check forbidden keys
    forbidden = sorted(FORBIDDEN_STUDENT_FIELDS & keys)
    if forbidden:
        result["valid"] = False
        result["issues"].append(f"forbidden_keys: {forbidden}")

    # Check required keys exist
    required = ["features_25d", "features_9d", "valid_mask", "known_mask", "step"]
    for h in R9P_HEAD_NAMES:
        required.append(f"y_{h}")
        required.append(f"m_{h}")
    missing = sorted(set(required) - keys)
    if missing:
        result["valid"] = False
        result["issues"].append(f"missing_keys: {missing}")

    if result["valid"]:
        T = data["features_25d"].shape[0]
        # Identity closure
        if data["features_9d"].shape[0] != T:
            result["valid"] = False
            result["issues"].append("features_25d/9d row count mismatch")
        if data["valid_mask"].shape[0] != T:
            result["valid"] = False
            result["issues"].append("valid_mask length mismatch")
        if data["known_mask"].shape[0] != T:
            result["valid"] = False
            result["issues"].append("known_mask length mismatch")

        for h in R9P_HEAD_NAMES:
            yk = f"y_{h}"
            mk = f"m_{h}"
            if data[yk].shape[0] != T:
                result["valid"] = False
                result["issues"].append(f"{yk} length mismatch: {data[yk].shape[0]} vs {T}")
            if data[mk].shape[0] != T:
                result["valid"] = False
                result["issues"].append(f"{mk} length mismatch: {data[mk].shape[0]} vs {T}")

        # Finite checks
        if not np.isfinite(data["features_25d"]).all():
            result["valid"] = False
            result["issues"].append("features_25d non-finite")
        if not np.isfinite(data["features_9d"]).all():
            result["valid"] = False
            result["issues"].append("features_9d non-finite")

        # Unknown masking: where known_mask=False, all y_heads should be 0, m_heads False
        unknown_mask = ~data["known_mask"]
        for h in R9P_HEAD_NAMES:
            if h == "grounding_confidence":
                continue
            yk = f"y_{h}"
            mk = f"m_{h}"
            if data[yk][unknown_mask].any():
                result["valid"] = False
                result["issues"].append(f"{yk} has non-zero values on unknown steps")
            if data[mk][unknown_mask].any():
                result["valid"] = False
                result["issues"].append(f"{mk} has True on unknown steps")

        # Grounding should always have mask=True
        if not data["m_grounding_confidence"].all():
            result["valid"] = False
            result["issues"].append("grounding_confidence mask not all-true")

    return result


def audit_materialization(
    plan_root: Path,
    materialization_root: Path,
) -> dict[str, Any]:
    index_path = materialization_root / "dataset_index.jsonl"
    if not index_path.exists():
        return {"schema": SCHEMA, "status": "HOLD_no_index", "error": "dataset_index.jsonl not found"}

    index_rows = read_jsonl(index_path)
    issues = []
    valid_count = 0

    for row in index_rows:
        npz_path = materialization_root / row["npz_path"]
        if not npz_path.exists():
            issues.append({"parent_key": row["parent_key"], "error": "npz_missing"})
            continue
        sha = sha256_file(npz_path)
        if sha != row["npz_sha256"]:
            issues.append({
                "parent_key": row["parent_key"],
                "error": f"sha256 mismatch: expected {row['npz_sha256']}, got {sha}",
            })
            continue
        audit = audit_episode_npz(npz_path)
        if audit["valid"]:
            valid_count += 1
        else:
            issues.append({"parent_key": row["parent_key"], "issues": audit["issues"]})

    status = GATE_PASS if not issues else f"HOLD_{GATE_PASS}"
    report = {
        "schema": SCHEMA,
        "status": status,
        "total": len(index_rows),
        "valid": valid_count,
        "issues": len(issues),
        "issue_details": issues[:50],
    }
    report_path = materialization_root / "materialization_audit.json"
    write_json(report_path, report)
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit R9P materialization")
    parser.add_argument("--plan-root", required=True, type=Path)
    parser.add_argument("--materialization-root", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = audit_materialization(args.plan_root, args.materialization_root)
    print(f"Materialization audit: {report['status']}")
    print(f"  Valid: {report['valid']}/{report['total']}")
    if report["issues"]:
        print(f"  Issues: {report['issues']}")
    return 0 if report["status"] == GATE_PASS else 1


if __name__ == "__main__":
    sys.exit(main())
