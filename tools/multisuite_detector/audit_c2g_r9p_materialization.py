"""Independent R9P materialization audit.

The audit intentionally does not import the materializer's selector or source
projection.  It verifies the plan and materialization checksum closures first,
then compares the complete ordered selection/index/fileset and validates every
NPZ against the student-only schema.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from tools.multisuite_detector.build_c2g_r9p_preview_plan import R9P_HEAD_NAMES, TARGET_SUITES
from tools.multisuite_detector.c2g_r8r_common import read_json, read_jsonl, sha256_file, write_json

SMOKE_SALT = "C2G_R9P_SMOKE"
SMOKE_PER_SUITE = 8
SCHEMA = "c2g.r9p.materialization_audit.2026-07-12.v2"
GATE_PASS = "PASS_C2G_R9P_MATERIALIZATION_AUDIT"

FORBIDDEN_STUDENT_KEYS = frozenset({
    "object_pose", "target_pose", "object_target_distance", "contact_pairs",
    "teacher_phase", "teacher_reason_code", "resolved_target_objects",
    "resolved_target_manipulable_entities", "attack_outcome",
    "post_intervention_state", "clean_final_success", "late_success_in_extended_source",
    "uses_privileged_sim_state", "uses_attack_outcome", "uses_future_student_input",
})


def _safe_relative_path(value: str) -> str:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"unsafe relative path: {value}")
    result = path.as_posix()
    if not result or result.startswith("/"):
        raise ValueError(f"unsafe relative path: {value}")
    return result


def _verify_checksum_closure(root: Path) -> dict[str, Any]:
    sums_path = root / "SHA256SUMS"
    sidecar_path = root / "SHA256SUMS.sha256"
    if not sums_path.is_file() or not sidecar_path.is_file():
        raise FileNotFoundError(f"checksum closure missing in {root}")
    tokens = sidecar_path.read_text(encoding="utf-8").strip().split()
    if not tokens or tokens[0] != sha256_file(sums_path):
        raise ValueError(f"SHA256SUMS sidecar mismatch in {root}")
    listed: list[str] = []
    for line in sums_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split("  ", 1)
        if len(parts) != 2 or len(parts[0]) != 64:
            raise ValueError(f"malformed checksum line: {line!r}")
        rel = _safe_relative_path(parts[1])
        if rel in listed:
            raise ValueError(f"duplicate checksum path: {rel}")
        listed.append(rel)
        path = root / rel
        if not path.is_file() or sha256_file(path) != parts[0]:
            raise ValueError(f"checksum mismatch or missing file: {rel}")
    actual = sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file())
    expected = sorted(set(listed) | {"SHA256SUMS", "SHA256SUMS.sha256"})
    if actual != expected:
        raise ValueError(
            f"fileset mismatch in {root}: extra={sorted(set(actual)-set(expected))}, "
            f"missing={sorted(set(expected)-set(actual))}"
        )
    return {"fileset": actual, "sha256sums_sha256": sha256_file(sums_path)}


def _smoke_rank(parent_key: str) -> int:
    return int.from_bytes(
        hashlib.sha256(f"{SMOKE_SALT}|{parent_key}".encode("utf-8")).digest(), "big"
    )


def _select_smoke_independent(plan_rows: list[dict]) -> list[dict]:
    selected: list[dict] = []
    for suite in TARGET_SUITES:
        rows = [r for r in plan_rows if r.get("suite") == suite]
        ranked = sorted(rows, key=lambda r: (_smoke_rank(r["parent_key"]), r["parent_key"]))
        for row in ranked[:SMOKE_PER_SUITE]:
            selected.append({
                **row,
                "selection_salt": SMOKE_SALT,
                "selection_rank": _smoke_rank(row["parent_key"]),
            })
    return selected


def _selection_record(row: dict, *, smoke: bool) -> tuple:
    fields = (
        "parent_key", "suite", "task_index", "state_id", "cohort",
        "preview_split", "task_language", "metadata_path",
    )
    values = tuple(row.get(k) for k in fields)
    if smoke:
        values += (row.get("selection_salt"), int(row.get("selection_rank", -1)))
    return values


def audit_episode_npz(npz_path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {"npz_path": str(npz_path), "valid": True, "issues": []}
    try:
        data = np.load(npz_path, allow_pickle=False)
    except Exception as exc:
        return {**result, "valid": False, "issues": [f"load_error: {exc}"]}

    keys = set(data.keys())
    forbidden = sorted(FORBIDDEN_STUDENT_KEYS & keys)
    if forbidden:
        result["valid"] = False
        result["issues"].append(f"forbidden_keys: {forbidden}")
    required = {"features_25d", "features_9d", "valid_mask", "known_mask", "step"}
    required |= {f"y_{h}" for h in R9P_HEAD_NAMES}
    required |= {f"m_{h}" for h in R9P_HEAD_NAMES}
    missing = sorted(required - keys)
    if missing:
        result["valid"] = False
        result["issues"].append(f"missing_keys: {missing}")
        return result

    f25 = data["features_25d"]
    f9 = data["features_9d"]
    if f25.ndim != 2 or f25.shape[1:] != (25,) or f25.dtype != np.float32:
        result["valid"] = False
        result["issues"].append(f"features_25d_schema: shape={f25.shape}, dtype={f25.dtype}")
    if f9.ndim != 2 or f9.shape[1:] != (9,) or f9.dtype != np.float32:
        result["valid"] = False
        result["issues"].append(f"features_9d_schema: shape={f9.shape}, dtype={f9.dtype}")
    T = f25.shape[0] if f25.ndim else -1
    if f9.ndim != 2 or f9.shape[0] != T:
        result["valid"] = False
        result["issues"].append("feature_row_count_mismatch")
    if not np.isfinite(f25).all() or not np.isfinite(f9).all():
        result["valid"] = False
        result["issues"].append("feature_nonfinite")

    for name in ("valid_mask", "known_mask"):
        arr = data[name]
        if arr.shape != (T,) or arr.dtype != np.bool_:
            result["valid"] = False
            result["issues"].append(f"{name}_schema: shape={arr.shape}, dtype={arr.dtype}")
    step = data["step"]
    if step.shape != (T,) or not np.issubdtype(step.dtype, np.integer) or not np.array_equal(step, np.arange(T)):
        result["valid"] = False
        result["issues"].append("step_not_integer_arange")

    unknown = ~data["known_mask"]
    for h in R9P_HEAD_NAMES:
        y = data[f"y_{h}"]
        m = data[f"m_{h}"]
        if y.shape != (T,) or y.dtype != np.float32:
            result["valid"] = False
            result["issues"].append(f"y_{h}_schema")
        if m.shape != (T,) or m.dtype != np.bool_:
            result["valid"] = False
            result["issues"].append(f"m_{h}_schema")
        if not np.isfinite(y).all():
            result["valid"] = False
            result["issues"].append(f"y_{h}_nonfinite")
        if h == "grounding_confidence":
            if np.any((y < 0.0) | (y > 1.0)):
                result["valid"] = False
                result["issues"].append("grounding_confidence_out_of_range")
        else:
            if np.any((y != 0.0) & (y != 1.0)):
                result["valid"] = False
                result["issues"].append(f"y_{h}_not_binary")
            if np.any(y[unknown] != 0.0) or np.any(m[unknown]):
                result["valid"] = False
                result["issues"].append(f"{h}_unknown_not_masked")
    if not np.all(data["m_grounding_confidence"]):
        result["valid"] = False
        result["issues"].append("grounding_confidence_mask_not_frozen_all_true")
    result["n_steps"] = int(T)
    return result


def _source_identity_check(plan: dict, row: dict) -> list[str]:
    issues: list[str] = []
    provenance = plan.get("source_provenance", {})
    root_key = {"libero_spatial": "spatial_root", "libero_object": "object_root", "libero_goal": "goal_root"}.get(row.get("suite"))
    if not root_key or not row.get("metadata_path"):
        return ["missing_source_binding_fields"]
    root = Path(provenance.get(root_key, ""))
    try:
        rel = _safe_relative_path(row["metadata_path"])
        meta_path = (root / rel).resolve()
        if root.resolve() not in meta_path.parents or not meta_path.is_file():
            return ["metadata_path_missing_or_outside"]
        ep_dir = meta_path.parent
        checks = {
            "metadata_sha256": meta_path,
            "step_records_sha256": ep_dir / "step_records_prefix.jsonl",
            "teacher_labels_sha256": ep_dir / "teacher_v2_labels.jsonl",
        }
        if row.get("source_binding_path"):
            checks["source_binding_sha256"] = ep_dir / row["source_binding_path"]
        for key, path in checks.items():
            if not path.is_file() or row.get(key) != sha256_file(path):
                issues.append(f"source_hash_mismatch:{key}")
        meta = read_json(meta_path)
        for key in ("suite", "task_index", "state_id", "parent_key", "cohort", "split", "task_language"):
            if meta.get(key) != row.get(key if key != "split" else "split", meta.get(key)):
                # The plan has preview_split, while source metadata has split=train.
                if key == "split" and meta.get(key) == "train":
                    continue
                issues.append(f"metadata_identity_mismatch:{key}")
    except Exception as exc:
        issues.append(f"source_check_error:{exc}")
    return issues


def audit_materialization(
    plan_root: Path,
    materialization_root: Path,
    output_root: Path,
    *,
    smoke: bool = False,
) -> dict[str, Any]:
    if output_root.exists():
        return {"schema": SCHEMA, "status": "HOLD_output_root_exists", "error": str(output_root)}
    try:
        plan_closure = _verify_checksum_closure(plan_root)
        materialization_closure = _verify_checksum_closure(materialization_root)
        plan = read_json(plan_root / "r9p_preview_plan.json")
        if plan.get("status") != "PASS_C2G_R9P_PLAN":
            raise ValueError(f"plan status is {plan.get('status')}")
        plan_rows = read_jsonl(plan_root / "r9p_preview_episode_manifest.jsonl")
        index_rows = read_jsonl(materialization_root / "dataset_index.jsonl")
    except Exception as exc:
        return {"schema": SCHEMA, "status": "HOLD_provenance_or_checksum", "error": str(exc)}

    if smoke:
        reference_rows = _select_smoke_independent(plan_rows)
        smoke_path = materialization_root / "smoke_selection_manifest.jsonl"
        if not smoke_path.is_file():
            return {"schema": SCHEMA, "status": "HOLD_no_smoke_manifest", "error": str(smoke_path)}
        materializer_rows = read_jsonl(smoke_path)
        if [_selection_record(r, smoke=True) for r in materializer_rows] != [_selection_record(r, smoke=True) for r in reference_rows]:
            return {"schema": SCHEMA, "status": "HOLD_smoke_selection_mismatch", "error": "ordered selection records differ"}
        expected_count = 24
    else:
        reference_rows = plan_rows
        expected_count = 900

    closure_issues: list[str] = []
    if len(reference_rows) != expected_count:
        closure_issues.append(f"reference_count={len(reference_rows)} expected={expected_count}")
    if len(index_rows) != expected_count:
        closure_issues.append(f"index_count={len(index_rows)} expected={expected_count}")
    if [_selection_record(r, smoke=False) for r in index_rows] != [_selection_record(r, smoke=False) for r in reference_rows]:
        closure_issues.append("ordered index identity/source records differ from plan")

    expected_npz = set()
    npz_results: list[dict[str, Any]] = []
    for row in index_rows:
        try:
            rel = _safe_relative_path(row.get("npz_path", ""))
            path = (materialization_root / rel).resolve()
            if materialization_root.resolve() not in path.parents or path.suffix != ".npz":
                raise ValueError("npz path outside materialization root or wrong suffix")
            expected_npz.add(rel)
            npz_results.append(audit_episode_npz(path))
            if row.get("npz_sha256") != sha256_file(path):
                closure_issues.append(f"npz_hash_mismatch:{row.get('parent_key')}")
            closure_issues.extend(_source_identity_check(plan, row))
        except Exception as exc:
            closure_issues.append(f"npz_path_error:{row.get('parent_key')}:{exc}")

    actual_npz = sorted(
        p.relative_to(materialization_root).as_posix()
        for p in (materialization_root / "episodes").rglob("*.npz")
        if p.is_file()
    ) if (materialization_root / "episodes").is_dir() else []
    if set(actual_npz) != expected_npz:
        closure_issues.append("actual NPZ files differ from dataset index")

    if smoke:
        suite_counts = {s: sum(1 for r in index_rows if r.get("suite") == s) for s in TARGET_SUITES}
        if suite_counts != {s: 8 for s in TARGET_SUITES}:
            closure_issues.append(f"smoke_suite_counts={suite_counts}")
    else:
        split_counts = {s: sum(1 for r in index_rows if r.get("preview_split") == s) for s in ("FIT", "CAL", "CHECK")}
        if split_counts != {"FIT": 720, "CAL": 90, "CHECK": 90}:
            closure_issues.append(f"split_counts={split_counts}")

    invalid = [r for r in npz_results if not r.get("valid")]
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "status": GATE_PASS if not closure_issues and not invalid else "HOLD_C2G_R9P_MATERIALIZATION_AUDIT",
        "smoke": smoke,
        "expected_episodes": expected_count,
        "total_npz": len(npz_results),
        "valid_npz": len(npz_results) - len(invalid),
        "invalid_npz": len(invalid),
        "closure_issues": closure_issues,
        "npz_issues": sum(len(r.get("issues", [])) for r in npz_results),
        "npz_issue_details": invalid[:50],
        "plan_sha256s_sha256": plan_closure["sha256sums_sha256"],
        "materialization_sha256s_sha256": materialization_closure["sha256sums_sha256"],
        "nontrain_cohorts_read": 0,
    }
    output_root.mkdir(parents=True)
    report_path = output_root / "materialization_audit.json"
    write_json(report_path, report)
    report_sha = sha256_file(report_path)
    (output_root / "materialization_audit.json.sha256").write_text(
        f"{report_sha}  materialization_audit.json\n", encoding="utf-8"
    )
    sums_path = output_root / "SHA256SUMS"
    sums_path.write_text(
        f"{report_sha}  materialization_audit.json\n"
        f"{sha256_file(output_root / 'materialization_audit.json.sha256')}  materialization_audit.json.sha256\n",
        encoding="utf-8",
    )
    sums_sha = sha256_file(sums_path)
    (output_root / "SHA256SUMS.sha256").write_text(f"{sums_sha}  SHA256SUMS\n", encoding="utf-8")
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit R9P materialization")
    parser.add_argument("--plan-root", required=True, type=Path)
    parser.add_argument("--materialization-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = audit_materialization(args.plan_root, args.materialization_root, args.output_root, smoke=args.smoke)
    print(f"Materialization audit: {report['status']}")
    print(f"  Valid NPZ: {report.get('valid_npz', 0)}/{report.get('total_npz', 0)}")
    if report.get("closure_issues"):
        print(f"  Closure issues: {report['closure_issues']}")
    return 0 if report["status"] == GATE_PASS else 1


if __name__ == "__main__":
    sys.exit(main())
