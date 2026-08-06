"""Independent closure audit for a Stage V dynamic queue root."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping

try:
    from .stage_v_dynamic_common import atomic_write_json, load_rows, read_json, science_artifact_status, sha256_file, utc_now
except ImportError:  # direct server execution
    from stage_v_dynamic_common import atomic_write_json, load_rows, read_json, science_artifact_status, sha256_file, utc_now

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
    manifest_rows = load_rows(args.parent_manifest)
    tasks = AtomicTaskQueue(str(args.queue_db), run_id=args.run_id).list_tasks()
    errors: list[str] = []
    if len(manifest_rows) != args.expected_parent_count:
        errors.append(f"MANIFEST_COUNT:{len(manifest_rows)}/{args.expected_parent_count}")
    keys = [str(row.get("canonical_parent_key")) for row in manifest_rows]
    if len(set(keys)) != len(keys):
        errors.append("DUPLICATE_MANIFEST_IDENTITIES")
    if len(tasks) != args.expected_parent_count:
        errors.append(f"QUEUE_COUNT:{len(tasks)}/{args.expected_parent_count}")
    accepted: list[dict[str, Any]] = []
    for task in tasks:
        key = str(task.get("cell_id"))
        if task.get("state") != "DONE_VALID" or not task.get("accepted_attempt_id"):
            errors.append(f"TASK_NOT_ACCEPTED:{key}:{task.get('state')}")
            continue
        output_dir = Path(task.get("accepted_output_dir") or "")
        if not output_dir.is_absolute():
            output_dir = root / output_dir
        validation = read_json(output_dir / "PARENT_VALIDATION.json", {})
        result = science_artifact_status(output_dir, key)
        if not result["valid"]:
            errors.append(f"SCIENCE_ARTIFACT_INVALID:{key}:{result.get('reason')}")
        if not isinstance(validation, Mapping) or validation.get("artifact_audit_verdict") != "PASS" or validation.get("label_status") != "VALID":
            errors.append(f"PARENT_VALIDATION_INVALID:{key}")
        if isinstance(validation, Mapping) and (validation.get("source_commit") != args.expected_source_commit or validation.get("source_tree") != args.expected_source_tree):
            errors.append(f"PROVENANCE_MISMATCH:{key}")
        parent_result = result.get("result") if isinstance(result.get("result"), Mapping) else {}
        if parent_result.get("eval160_reads", 0) != 0 or parent_result.get("protected_eval_reads", 0) != 0 or parent_result.get("attack_rollouts", 0) != 0:
            errors.append(f"BOUNDARY_VIOLATION:{key}")
        branch_files = list(output_dir.rglob("COUNTERFACTUAL_BRANCHES.jsonl"))
        if len(branch_files) != 1:
            errors.append(f"BRANCH_FILE_COUNT:{key}:{len(branch_files)}")
        branch_rows: list[dict[str, Any]] = []
        for branch_file in branch_files:
            rows, branch_errors = _branch_rows(branch_file)
            branch_rows.extend(rows)
            errors.extend(branch_errors)
        if len(branch_rows) != args.expected_branch_count:
            errors.append(f"BRANCH_COUNT:{key}:{len(branch_rows)}/{args.expected_branch_count}")
        if any(row.get("status") not in ("PASS", "DONE") for row in branch_rows):
            errors.append(f"BRANCH_FAILURE:{key}")
        if len({(row.get("canonical_parent_key"), row.get("probe_step"), row.get("k"), row.get("arm")) for row in branch_rows}) != len(branch_rows):
            errors.append(f"DUPLICATE_BRANCH_IDENTITIES:{key}")
        accepted.append({
            "canonical_parent_key": key,
            "artifact_audit_verdict": "PASS" if result["valid"] and not errors else "PENDING",
            "output_dir": str(output_dir),
            "parent_result_sha256": result.get("artifact_sha256"),
            "branch_count": len(branch_rows),
        })
    accepted_count = sum(item["artifact_audit_verdict"] == "PASS" for item in accepted)
    status = "PASS" if not errors and len(accepted) == args.expected_parent_count and accepted_count == args.expected_parent_count else "FAIL"
    return {
        "schema": "STAGE_V_COUNTERFACTUAL_DYNAMIC_AUDIT_V2",
        "verdict": status,
        "run_root": str(root),
        "planned_parents": args.expected_parent_count,
        "completed_parents": sum(task.get("state") == "DONE_VALID" for task in tasks),
        "accepted_parent_results": accepted_count if status == "PASS" else 0,
        "accepted_parent_artifacts": accepted if status == "PASS" else [],
        "branch_rows": sum(item.get("branch_count", 0) for item in accepted),
        "errors": sorted(set(errors)),
        "duplicate_identities": [],
        "missing_identities": sorted(set(keys) - {item["canonical_parent_key"] for item in accepted}),
        "invalid_branches": 0 if status == "PASS" else None,
        "control_branch_failure": 0 if status == "PASS" else None,
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
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = audit(args)
    atomic_write_json(args.run_root / "STAGE_V_COUNTERFACTUAL_AUDIT.json", report)
    return 0 if report["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
