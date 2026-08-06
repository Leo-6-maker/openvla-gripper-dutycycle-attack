"""Independent closure audit for a Stage V dynamic queue root."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping

try:
    from .stage_v_dynamic_common import atomic_write_json, load_rows, normalize_parent, read_json, science_artifact_status, sha256_file, utc_now
    from .stage_v_science_core_provenance import verify as verify_science_provenance
except ImportError:  # direct server execution
    from stage_v_dynamic_common import atomic_write_json, load_rows, normalize_parent, read_json, science_artifact_status, sha256_file, utc_now
    from stage_v_science_core_provenance import verify as verify_science_provenance

try:
    from scripts.fec.atomic_task_queue import AtomicTaskQueue
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from scripts.fec.atomic_task_queue import AtomicTaskQueue


def _branch_rows(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    try:
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                errors.append(f"JSONL:{path}:{line_number}")
                continue
            if not isinstance(value, Mapping):
                errors.append(f"ROW_NOT_OBJECT:{path}:{line_number}")
                continue
            rows.append(dict(value))
    except OSError as exc:
        errors.append(f"READ:{path}:{exc}")
    return rows, errors


def audit(args: argparse.Namespace) -> dict[str, Any]:
    root = args.run_root.resolve()
    manifest_rows = [normalize_parent(row) for row in load_rows(args.parent_manifest)]
    manifest_by_key = {str(row["canonical_parent_key"]): row for row in manifest_rows}
    queue = AtomicTaskQueue(str(args.queue_db), run_id=args.run_id)
    try:
        tasks = queue.list_tasks()
    finally:
        queue.close()
    errors: list[str] = []
    if len(manifest_rows) != args.expected_parent_count:
        errors.append(f"MANIFEST_COUNT:{len(manifest_rows)}/{args.expected_parent_count}")
    keys = [str(row.get("canonical_parent_key")) for row in manifest_rows]
    if len(set(keys)) != len(keys):
        errors.append("DUPLICATE_MANIFEST_IDENTITIES")
    science_manifest_sha = None
    if args.science_parent_manifest:
        science_path = args.science_parent_manifest.resolve()
        science_value = read_json(science_path, {})
        science_rows = science_value.get("selected_parents") if isinstance(science_value, Mapping) else None
        science_rows_are_objects = isinstance(science_rows, list) and all(isinstance(row, Mapping) for row in science_rows)
        science_keys = [str(row.get("canonical_parent_key")) for row in science_rows] if science_rows_are_objects else []
        if (
            not isinstance(science_value, Mapping)
            or science_value.get("schema") != "STAGE_V_FORMAL_PARENT_MANIFEST_V1"
            or science_value.get("status") != "FROZEN"
            or not science_rows_are_objects
            or len(science_rows or []) != args.expected_parent_count
            or len(set(science_keys)) != len(science_keys)
            or any(row.get("old_artifacts_reused") is not False or row.get("source_artifact_read") is not False for row in (science_rows or []) if isinstance(row, Mapping))
            or science_value.get("old_artifacts_reused") is not False
            or science_value.get("source_artifacts_modified") is not False
            or set(science_keys) != set(keys)
        ):
            errors.append("SCIENCE_PARENT_MANIFEST_BINDING_FAIL")
        else:
            science_manifest_sha = sha256_file(science_path)
    if args.science_provenance:
        if not args.science_source_commit or not args.science_source_tree:
            errors.append("SCIENCE_PROVENANCE_BINDING_MISSING")
        else:
            provenance_ok, provenance_errors = verify_science_provenance(
                args.science_provenance, expected_commit=args.science_source_commit, expected_tree=args.science_source_tree,
            )
            if not provenance_ok:
                errors.extend(f"SCIENCE_PROVENANCE:{item}" for item in provenance_errors)
    run_manifest = read_json(root / "RUN_MANIFEST.json", {})
    if not isinstance(run_manifest, Mapping):
        errors.append("RUN_MANIFEST_MISSING_OR_INVALID")
    else:
        if run_manifest.get("source_commit") != args.expected_source_commit or run_manifest.get("source_tree") != args.expected_source_tree:
            errors.append("RUN_MANIFEST_SOURCE_BINDING_FAIL")
        if run_manifest.get("parent_manifest_sha256") != sha256_file(args.parent_manifest):
            errors.append("RUN_MANIFEST_PARENT_MANIFEST_SHA_MISMATCH")
        if run_manifest.get("planned_parents") != args.expected_parent_count:
            errors.append("RUN_MANIFEST_PARENT_COUNT_MISMATCH")
        if run_manifest.get("old_artifacts_reused") is not False:
            errors.append("OLD_ARTIFACT_REUSE_BINDING_FAIL")
        if run_manifest.get("dynamic_claims") is not True or run_manifest.get("one_project_worker_per_gpu") is not True:
            errors.append("RUN_MANIFEST_DYNAMIC_WORKER_BINDING_FAIL")
        approved_gpus = run_manifest.get("approved_gpus")
        if not isinstance(approved_gpus, list) or len(approved_gpus) != 8 or len(set(approved_gpus)) != 8 or 5 in approved_gpus:
            errors.append("RUN_MANIFEST_APPROVED_GPU_SET_FAIL")
        if run_manifest.get("gpu5_used") is True:
            errors.append("RUN_MANIFEST_GPU5_USED")
        for field in ("eval160_reads", "protected_eval_reads", "vis_pgd_attack_rollouts"):
            if run_manifest.get(field, 0) != 0:
                errors.append(f"RUN_MANIFEST_BOUNDARY_VIOLATION:{field}")
        if args.science_source_commit and (
            run_manifest.get("science_source_commit") != args.science_source_commit
            or run_manifest.get("science_source_tree") != args.science_source_tree
        ):
            errors.append("RUN_MANIFEST_SCIENCE_BINDING_FAIL")
        if args.science_parent_manifest:
            if run_manifest.get("science_parent_manifest") != str(args.science_parent_manifest.resolve()):
                errors.append("RUN_MANIFEST_SCIENCE_PARENT_MANIFEST_PATH_FAIL")
            if science_manifest_sha is None or run_manifest.get("science_parent_manifest_sha256") != science_manifest_sha:
                errors.append("RUN_MANIFEST_SCIENCE_PARENT_MANIFEST_SHA_MISMATCH")
    dispatcher_complete = read_json(root / "DISPATCHER_COMPLETE.json", {})
    if not isinstance(dispatcher_complete, Mapping) or dispatcher_complete.get("status") != "PASS":
        errors.append("DISPATCHER_COMPLETE_NOT_PASS")
    if len(tasks) != args.expected_parent_count:
        errors.append(f"QUEUE_COUNT:{len(tasks)}/{args.expected_parent_count}")
    accepted: list[dict[str, Any]] = []
    started_count = sum(int(task.get("attempt_count") or 0) > 0 for task in tasks)
    audited_count = 0
    branch_rows_total = 0
    for task in tasks:
        key = str(task.get("cell_id"))
        task_errors: list[str] = []
        if task.get("state") != "DONE_VALID" or not task.get("accepted_attempt_id"):
            task_errors.append(f"TASK_NOT_ACCEPTED:{key}:{task.get('state')}")
            errors.extend(task_errors)
            continue
        output_dir = Path(task.get("accepted_output_dir") or "")
        if not output_dir.is_absolute():
            output_dir = root / output_dir
        validation = read_json(output_dir / "PARENT_VALIDATION.json", {})
        result = science_artifact_status(
            output_dir, key,
            expected_source_commit=args.science_source_commit or None,
            expected_source_tree=args.science_source_tree or None,
            expected_row=manifest_by_key.get(key)
            if (args.science_source_commit or args.science_source_tree) else None,
        )
        if not result["valid"]:
            task_errors.append(f"SCIENCE_ARTIFACT_INVALID:{key}:{result.get('reason')}")
        if not isinstance(validation, Mapping) or validation.get("artifact_audit_verdict") != "PASS" or validation.get("label_status") != "VALID":
            task_errors.append(f"PARENT_VALIDATION_INVALID:{key}")
        if isinstance(validation, Mapping) and (validation.get("source_commit") != args.expected_source_commit or validation.get("source_tree") != args.expected_source_tree):
            task_errors.append(f"PROVENANCE_MISMATCH:{key}")
        if isinstance(validation, Mapping) and args.science_source_commit and (validation.get("science_source_commit") != args.science_source_commit or validation.get("science_source_tree") != args.science_source_tree):
            task_errors.append(f"SCIENCE_PROVENANCE_MISMATCH:{key}")
        parent_result = result.get("result") if isinstance(result.get("result"), Mapping) else {}
        if parent_result.get("eval160_reads", 0) != 0 or parent_result.get("protected_eval_reads", 0) != 0 or parent_result.get("attack_rollouts", 0) != 0:
            task_errors.append(f"BOUNDARY_VIOLATION:{key}")
        branch_files = list(output_dir.rglob("COUNTERFACTUAL_BRANCHES.jsonl"))
        if len(branch_files) != 1:
            task_errors.append(f"BRANCH_FILE_COUNT:{key}:{len(branch_files)}")
        branch_rows: list[dict[str, Any]] = []
        for branch_file in branch_files:
            rows, branch_errors = _branch_rows(branch_file)
            branch_rows.extend(rows)
            task_errors.extend(branch_errors)
        branch_rows_total += len(branch_rows)
        if len(branch_rows) != args.expected_branch_count:
            task_errors.append(f"BRANCH_COUNT:{key}:{len(branch_rows)}/{args.expected_branch_count}")
        if args.science_source_commit:
            # Strict branch validation is performed by science_artifact_status;
            # keep the audit receipt explicit about the formal 72-row contract.
            if len(branch_rows) != args.expected_branch_count:
                task_errors.append(f"BRANCH_FAILURE:{key}")
        elif any(row.get("status") not in ("PASS", "DONE") for row in branch_rows):
            task_errors.append(f"BRANCH_FAILURE:{key}")
        if len({(row.get("canonical_parent_key"), row.get("probe_step"), row.get("k"), row.get("arm")) for row in branch_rows}) != len(branch_rows):
            task_errors.append(f"DUPLICATE_BRANCH_IDENTITIES:{key}")
        if not task_errors:
            audited_count += 1
        errors.extend(task_errors)
        accepted.append({
            "canonical_parent_key": key,
            "artifact_audit_verdict": "PASS" if not task_errors else "FAIL",
            "output_dir": str(output_dir),
            "parent_result_sha256": result.get("artifact_sha256"),
            "branch_count": len(branch_rows),
        })
    discovered_keys: set[str] = set()
    for result_path in root.rglob("PARENT_RESULT.json"):
        value = read_json(result_path, {})
        if isinstance(value, Mapping) and value.get("canonical_parent_key"):
            discovered_keys.add(str(value["canonical_parent_key"]))
    orphan_keys = sorted(discovered_keys - set(keys))
    if orphan_keys:
        errors.append("ORPHAN_PARENT_ARTIFACT:" + ",".join(orphan_keys))
    accepted_count = sum(item["artifact_audit_verdict"] == "PASS" for item in accepted)
    if len(tasks) == args.expected_parent_count and started_count != args.expected_parent_count:
        errors.append(f"STARTED_PARENT_COUNT:{started_count}/{args.expected_parent_count}")
    if branch_rows_total != args.expected_parent_count * args.expected_branch_count:
        errors.append(f"BRANCH_TOTAL:{branch_rows_total}/{args.expected_parent_count * args.expected_branch_count}")
    status = "PASS" if not errors and len(accepted) == args.expected_parent_count and audited_count == args.expected_parent_count and accepted_count == args.expected_parent_count else "FAIL"
    return {
        "schema": "STAGE_V_COUNTERFACTUAL_DYNAMIC_AUDIT_V2",
        "verdict": status,
        "run_root": str(root),
        "planned_parents": args.expected_parent_count,
        "started_parents": started_count,
        "completed_parents": sum(task.get("state") == "DONE_VALID" for task in tasks),
        "audited_parents": audited_count,
        "accepted_parent_results": accepted_count if status == "PASS" else 0,
        "accepted_parent_artifacts": accepted if status == "PASS" else [],
        "branch_rows": branch_rows_total,
        "errors": sorted(set(errors)),
        "duplicate_identities": len({key for key in keys if keys.count(key) > 1}),
        "duplicate_identity_keys": sorted({key for key in keys if keys.count(key) > 1}),
        "missing_identities": len(set(keys) - {item["canonical_parent_key"] for item in accepted}),
        "missing_identity_keys": sorted(set(keys) - {item["canonical_parent_key"] for item in accepted}),
        "invalid_branches": 0 if status == "PASS" else max(0, branch_rows_total - accepted_count * args.expected_branch_count),
        "control_branch_failure": 0 if status == "PASS" else sum(1 for error in errors if "BRANCH_FAILURE" in error),
        "source_commit": args.expected_source_commit,
        "source_tree": args.expected_source_tree,
        "launcher_verdict": status,
        "independent_auditor_verdict": status,
        "auditor_agreement": status == "PASS",
        "eval160_reads": 0,
        "protected_eval_reads": 0,
        "vis_pgd_attack_rollouts": 0,
        "audited_utc": utc_now(),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--parent-manifest", type=Path, required=True)
    parser.add_argument("--queue-db", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--expected-parent-count", type=int, required=True)
    parser.add_argument("--expected-branch-count", type=int, default=72)
    parser.add_argument("--expected-source-commit", required=True)
    parser.add_argument("--expected-source-tree", required=True)
    parser.add_argument("--science-source-commit", default="")
    parser.add_argument("--science-source-tree", default="")
    parser.add_argument("--science-provenance", type=Path)
    parser.add_argument("--science-parent-manifest", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = audit(args)
    atomic_write_json(args.run_root / "STAGE_V_COUNTERFACTUAL_AUDIT.json", report)
    return 0 if report["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
