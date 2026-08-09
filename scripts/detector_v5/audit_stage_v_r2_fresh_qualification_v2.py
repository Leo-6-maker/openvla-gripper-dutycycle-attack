"""Independently audit fresh parent-atomic clean qualification."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

try:
    from .audit_stage_v_r2_control_qualification_v2 import _pair, engineering_valid
    from .stage_v_dynamic_common import atomic_write_json, normalize_parent, sha256_file, utc_now
except ImportError:  # direct execution on the server
    from audit_stage_v_r2_control_qualification_v2 import _pair, engineering_valid
    from stage_v_dynamic_common import atomic_write_json, normalize_parent, sha256_file, utc_now


SUITES = ("libero_10", "libero_goal", "libero_object", "libero_spatial")
BOUNDARIES = ("eval160_reads", "protected_eval_reads", "vis_pgd_attack_rollouts", "attack_rollouts")


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"expected JSON object: {path}")
    return dict(value)


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def audit(args: argparse.Namespace) -> dict[str, Any]:
    root = args.run_root.resolve()
    protocol = _json(args.protocol.resolve())
    manifest = _json(args.candidate_manifest.resolve())
    report = _json((root / "Q2_CONTROL_QUALIFICATION_REPORT.json"))
    rows_value = _json((root / "Q2_CONTROL_QUALIFICATION_ROWS.json"))
    rows = [normalize_parent(row) for row in rows_value.get("rows", []) if isinstance(row, Mapping)]
    errors: list[str] = []
    expected_runtime_environment = {str(key): str(value) for key, value in protocol.get("runtime_environment", {}).items()}
    if protocol.get("schema") != "STAGE_V_R2_FRESH_QUALIFICATION_PROTOCOL_V2" or protocol.get("status") != "FROZEN_THROUGHPUT_PARENT_ATOMIC":
        errors.append("PROTOCOL_NOT_FROZEN")
    if manifest.get("schema") != "STAGE_V_R2_QUALIFICATION_CANDIDATE_MANIFEST_V1" or manifest.get("status") != "FROZEN":
        errors.append("CANDIDATE_MANIFEST_NOT_FROZEN")
    if report.get("protocol_sha256") != sha256_file(args.protocol):
        errors.append("REPORT_PROTOCOL_SHA256_MISMATCH")
    if report.get("candidate_manifest_sha256") != sha256_file(args.candidate_manifest):
        errors.append("REPORT_CANDIDATE_MANIFEST_SHA256_MISMATCH")
    if report.get("source_commit") != args.source_commit or report.get("source_tree") != args.source_tree:
        errors.append("REPORT_SOURCE_MISMATCH")
    if report.get("queue_state") != "COMPLETE" or report.get("parent_atomic") is not True:
        errors.append("RUN_NOT_COMPLETE_PARENT_ATOMIC")
    if report.get("gpus") != report.get("eligible_gpu_ids") or any(int(gpu) < 0 or int(gpu) > 7 for gpu in report.get("gpus", [])):
        errors.append("RESOURCE_GPU_SET_INVALID")
    if any(int(report.get(field, -1)) != 0 for field in BOUNDARIES):
        errors.append("REPORT_BOUNDARY_NONZERO")
    expected_rows = {str(row["canonical_parent_key"]): normalize_parent(row) for row in manifest.get("selected_parents", []) if isinstance(row, Mapping)}
    if len(rows) != len(expected_rows) or set(expected_rows) != {str(row.get("canonical_parent_key")) for row in rows}:
        errors.append("ROW_MANIFEST_IDENTITY_MISMATCH")
    valid_count = 0
    qualified_by_suite = {suite: [] for suite in SUITES}
    engineering_invalid = 0
    protected_totals = {field: 0 for field in BOUNDARIES}
    for row in rows:
        key = str(row.get("canonical_parent_key"))
        candidate = expected_rows.get(key)
        if candidate is None:
            errors.append(f"ROW_NOT_IN_MANIFEST:{key}")
            continue
        expected_rank = hashlib.sha256(f"{protocol['salt']}::{key}".encode()).hexdigest()
        if row.get("qualification_rank_sha256") != expected_rank:
            errors.append(f"RANK_MISMATCH:{key}")
        replicate_dirs = row.get("replicate_output_dirs") if isinstance(row.get("replicate_output_dirs"), Mapping) else {}
        actual: dict[str, dict[str, Any]] = {}
        valid: dict[str, bool] = {}
        parent_dirs: set[Path] = set()
        for replicate in ("A", "B"):
            output = Path(str(replicate_dirs.get(replicate, ""))).resolve()
            parent_dir = output.parent
            parent_dirs.add(parent_dir)
            if not _inside(output, root / "qualification") or not output.is_dir():
                errors.append(f"OUTPUT_OUTSIDE_ROOT:{key}:{replicate}")
            control = output / "CONTROL_RESULT.json"
            actual[replicate] = _json(control) if control.is_file() else {"status": "FAIL", "exit_code": 1}
            if actual[replicate].get("runtime_environment") != expected_runtime_environment:
                errors.append(f"{key}:{replicate}:RUNTIME_ENVIRONMENT_MISMATCH")
            stored = (row.get("replicates") or {}).get(replicate) if isinstance(row.get("replicates"), Mapping) else {}
            process_exit = (row.get("replicate_exit_codes") or {}).get(replicate) if isinstance(row.get("replicate_exit_codes"), Mapping) else stored.get("process_exit_code") if isinstance(stored, Mapping) else 1
            valid[replicate], hard_errors = engineering_valid(candidate, actual[replicate], process_exit, args.source_commit, args.source_tree)
            errors.extend(f"{key}:{replicate}:{item}" for item in hard_errors)
            for field in BOUNDARIES:
                protected_totals[field] += int(actual[replicate].get(field, 0) or 0)
        if len(parent_dirs) != 1:
            errors.append(f"PARENT_ATOMIC_GPU_ROOT_MISMATCH:{key}")
        parent_dir = next(iter(parent_dirs)) if parent_dirs else root
        pre = _json(parent_dir / "PRE_JOB_RESOURCE_RECEIPT.json") if (parent_dir / "PRE_JOB_RESOURCE_RECEIPT.json").is_file() else {}
        post = _json(parent_dir / "POST_JOB_RESOURCE_RECEIPT.json") if (parent_dir / "POST_JOB_RESOURCE_RECEIPT.json").is_file() else {}
        child_gpus = {actual[replicate].get("worker_gpu") for replicate in ("A", "B")}
        if len(child_gpus) != 1 or (pre and next(iter(child_gpus)) != pre.get("gpu_id")):
            errors.append(f"PARENT_CHILD_GPU_MISMATCH:{key}")
        if not pre or not post or pre.get("gpu_id") != post.get("gpu_id"):
            errors.append(f"PARENT_RESOURCE_AFFINITY_INVALID:{key}")
        for field in BOUNDARIES:
            protected_totals[field] += int(_json(parent_dir / "PARENT_RESULT.json").get(field, 0) or 0) if (parent_dir / "PARENT_RESULT.json").is_file() else 0
        pair_ok, classification, pair_errors = _pair(candidate, actual, valid)
        if row.get("qualified") is not pair_ok or row.get("classification") != classification:
            errors.append(f"ROW_DECISION_MISMATCH:{key}")
        if pair_ok:
            valid_count += 1
            qualified_by_suite[str(candidate["suite"])].append(dict(candidate))
        if not valid["A"] or not valid["B"]:
            engineering_invalid += 1
        errors.extend(f"{key}:{item}" for item in pair_errors)
    selected = {suite: qualified_by_suite[suite][:int(protocol["target_per_suite"])] for suite in SUITES}
    if any(len(selected[suite]) < int(protocol["target_per_suite"]) for suite in SUITES):
        errors.append("SUITE_QUOTA_UNDERFILLED")
    if any(value != 0 for value in protected_totals.values()):
        errors.append("PROTECTED_BOUNDARY_NONZERO")
    verdict = "PASS" if not errors and report.get("status") == "PASS" and engineering_invalid == 0 else "FAIL"
    payload = {
        "schema": "STAGE_V_R2_FRESH_QUALIFICATION_INDEPENDENT_AUDIT_V2",
        "verdict": verdict, "status": "PASS_CLASSIFIED" if verdict == "PASS" else "FAIL_CLOSED",
        "source_commit": args.source_commit, "source_tree": args.source_tree,
        "runtime_environment": expected_runtime_environment,
        "protocol_sha256": sha256_file(args.protocol), "candidate_manifest_sha256": sha256_file(args.candidate_manifest),
        "evaluated_rows": len(rows), "qualified_rows": valid_count,
        "qualified_by_suite": {suite: len(qualified_by_suite[suite]) for suite in SUITES},
        "selected_by_suite": {suite: [row["canonical_parent_key"] for row in selected[suite]] for suite in SUITES},
        "gpus": report.get("gpus", []), "worker_count": len(report.get("gpus", [])),
        "parent_atomic": True, "maximum_project_workers_per_gpu": 1,
        "foreign_workload_allowed": True, "engineering_invalid_parent_count": engineering_invalid,
        "protected_boundaries": protected_totals, "old_artifacts_reused": False,
        "source_artifacts_modified": False, "errors": sorted(set(errors)), "audited_utc": utc_now(),
    }
    atomic_write_json(root / "FRESH_QUALIFICATION_INDEPENDENT_AUDIT.json", payload)
    if verdict == "PASS":
        selected_rows = [
            {**row, "selection_role": "fresh_qualification_clean_control_parent", "qualification_mode": "FRESH_CLEAN_AB_REPLAY", "source_artifact_read": False, "old_artifacts_reused": False}
            for suite in SUITES for row in selected[suite]
        ]
        formal = {
            "schema": "STAGE_V_FORMAL_PARENT_MANIFEST_V1", "status": "FROZEN", "source_commit": args.source_commit, "source_tree": args.source_tree,
            "candidate_manifest_sha256": sha256_file(args.candidate_manifest), "fresh_qualification_audit_sha256": sha256_file(root / "FRESH_QUALIFICATION_INDEPENDENT_AUDIT.json"),
            "selected_parents": selected_rows, "selected_count": len(selected_rows), "selected_by_suite": {suite: len(selected[suite]) for suite in SUITES},
            "old_artifacts_reused": False, "source_artifacts_modified": False, **{field: 0 for field in BOUNDARIES}, "generated_utc": utc_now(),
        }
        atomic_write_json(root / "STAGE_V_FORMAL_PARENT_MANIFEST_V1.json", formal)
        (root / "STAGE_V_FORMAL_PARENT_MANIFEST_V1.sha256").write_text(f"{sha256_file(root / 'STAGE_V_FORMAL_PARENT_MANIFEST_V1.json')}  STAGE_V_FORMAL_PARENT_MANIFEST_V1.json\n", encoding="utf-8")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--source-tree", required=True)
    args = parser.parse_args(argv)
    payload = audit(args)
    print(json.dumps({"schema": payload["schema"], "verdict": payload["verdict"], "evaluated_rows": payload["evaluated_rows"], "qualified_rows": payload["qualified_rows"], "errors": payload["errors"]}, sort_keys=True))
    return 0 if payload["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
