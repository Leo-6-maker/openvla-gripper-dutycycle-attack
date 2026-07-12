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
    R9P_HEAD_NAMES,
    TARGET_SUITES,
)
from tools.multisuite_detector.materialize_c2g_r9p_ogs1500 import (
    select_smoke_episodes,
)

FORBIDDEN_STUDENT_KEYS = frozenset({
    "object_pose", "target_pose", "object_target_distance",
    "contact_pairs", "teacher_phase", "teacher_reason_code",
    "resolved_target_objects", "resolved_target_manipulable_entities",
    "attack_outcome", "post_intervention_state",
    "clean_final_success", "late_success_in_extended_source",
    "uses_privileged_sim_state", "uses_attack_outcome",
    "uses_future_student_input",
})
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
    forbidden = sorted(FORBIDDEN_STUDENT_KEYS & keys)
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
    output_root: Path,
    *,
    smoke: bool = False,
) -> dict[str, Any]:
    index_path = materialization_root / "dataset_index.jsonl"
    plan_manifest_path = plan_root / "r9p_preview_episode_manifest.jsonl"
    smoke_manifest_path = materialization_root / "smoke_selection_manifest.jsonl"

    if not index_path.exists():
        return {"schema": SCHEMA, "status": "HOLD_no_index", "error": "dataset_index.jsonl not found"}

    index_rows = read_jsonl(index_path)

    if smoke:
        if not plan_manifest_path.exists():
            return {"schema": SCHEMA, "status": "HOLD_no_plan",
                    "error": "plan manifest not found — needed to recompute smoke selection"}
        plan_rows = read_jsonl(plan_manifest_path)
        # Independently recompute smoke selection from plan manifest
        recomputed = select_smoke_episodes(plan_rows)
        recomputed_keys = {r["parent_key"] for r in recomputed}
        # Verify 8/8/8 per-suite count
        suite_counts = {}
        for r in recomputed:
            suite_counts[r["suite"]] = suite_counts.get(r["suite"], 0) + 1
        for s in TARGET_SUITES:
            if suite_counts.get(s, 0) != 8:
                return {"schema": SCHEMA, "status": "HOLD_smoke_count",
                        "error": f"recomputed smoke selection: {s}={suite_counts.get(s, 0)}, expected 8"}
        if len(recomputed) != 24:
            return {"schema": SCHEMA, "status": "HOLD_smoke_count",
                    "error": f"recomputed smoke selection: {len(recomputed)} total, expected 24"}

        if not smoke_manifest_path.exists():
            return {"schema": SCHEMA, "status": "HOLD_no_smoke_manifest",
                    "error": "smoke_selection_manifest.jsonl not found"}
        materializer_selection = read_jsonl(smoke_manifest_path)
        materializer_keys = {r["parent_key"] for r in materializer_selection}

        # Verify recomputed == materializer manifest
        if recomputed_keys != materializer_keys:
            extra = sorted(materializer_keys - recomputed_keys)
            missing = sorted(recomputed_keys - materializer_keys)
            return {"schema": SCHEMA, "status": "HOLD_smoke_mismatch",
                    "error": f"smoke selection mismatch: extra_in_manifest={len(extra)}, missing={len(missing)}"}

        reference_rows = recomputed
        expected_count = 24
    else:
        if not plan_manifest_path.exists():
            return {"schema": SCHEMA, "status": "HOLD_no_plan", "error": "plan manifest not found"}
        reference_rows = read_jsonl(plan_manifest_path)
        expected_count = len(reference_rows)

    # Closure: reference identities == index identities == actual NPZ set
    ref_keys = {r["parent_key"] for r in reference_rows}
    index_keys = {r["parent_key"] for r in index_rows}

    missing_from_index = sorted(ref_keys - index_keys)
    extra_in_index = sorted(index_keys - ref_keys)
    closure_issues = []
    if missing_from_index:
        closure_issues.append(f"in reference but not index: {len(missing_from_index)} episodes")
    if extra_in_index:
        closure_issues.append(f"in index but not reference: {len(extra_in_index)} episodes")
    if len(index_rows) != expected_count:
        closure_issues.append(
            f"count mismatch: expected={expected_count}, index={len(index_rows)}"
        )

    # Verify split distribution
    ref_splits = {}
    for r in reference_rows:
        ref_splits[r["parent_key"]] = r.get("preview_split", "")
    index_splits = {}
    for r in index_rows:
        index_splits[r["parent_key"]] = r.get("preview_split", "")

    split_mismatches = []
    for key in ref_keys & index_keys:
        if ref_splits.get(key) != index_splits.get(key):
            split_mismatches.append(key)

    issues = []
    valid_count = 0
    npz_keys = set()

    for row in index_rows:
        npz_path = materialization_root / row["npz_path"]
        npz_keys.add(str(npz_path))
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

    # Check for extra NPZ files not in index
    episodes_dir = materialization_root / "episodes"
    actual_npz_files = set()
    if episodes_dir.is_dir():
        actual_npz_files = {str(p) for p in episodes_dir.rglob("*.npz")}
    extra_npz = sorted(actual_npz_files - npz_keys)
    missing_npz = sorted(npz_keys - actual_npz_files)

    all_ok = (
        not closure_issues
        and not split_mismatches
        and not issues
        and not extra_npz
        and not missing_npz
    )
    status = GATE_PASS if all_ok else f"HOLD_{GATE_PASS}"

    if output_root.exists():
        raise FileExistsError(f"audit output root already exists: {output_root}")
    output_root.mkdir(parents=True)
    report = {
        "schema": SCHEMA,
        "status": status,
        "smoke": smoke,
        "reference_episodes": len(reference_rows),
        "index_episodes": len(index_rows),
        "valid_npz": valid_count,
        "total_npz": len(index_rows),
        "closure_issues": closure_issues,
        "split_mismatches": len(split_mismatches),
        "npz_issues": len(issues),
        "extra_npz_files": len(extra_npz),
        "missing_npz_files": len(missing_npz),
        "issue_details": issues[:50],
    }
    report_path = output_root / "materialization_audit.json"
    write_json(report_path, report)
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit R9P materialization")
    parser.add_argument("--plan-root", required=True, type=Path)
    parser.add_argument("--materialization-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--smoke", action="store_true", help="Audit smoke (24-ep) materialization")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = audit_materialization(
        args.plan_root, args.materialization_root, args.output_root, smoke=args.smoke)
    print(f"Materialization audit: {report['status']}")
    print(f"  Valid NPZ: {report.get('valid_npz', 0)}/{report.get('total_npz', 0)}")
    closure = report.get("closure_issues", [])
    if closure:
        print(f"  Closure issues: {closure}")
    npz_issues = report.get("npz_issues", 0)
    if npz_issues:
        print(f"  NPZ issues: {npz_issues}")
    return 0 if report["status"] == GATE_PASS else 1


if __name__ == "__main__":
    sys.exit(main())
