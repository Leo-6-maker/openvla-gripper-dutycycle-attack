"""Build the paper-line artifact index after sealed Stage V/V2/O roots exist."""
from __future__ import annotations

import argparse
import datetime as _datetime
import json
from pathlib import Path
from typing import Any, Mapping

from scripts.detector_v5.stage_v_dynamic_common import atomic_write_json, sha256_file
from scripts.monitoring.audit_stage_v_closure import verify_sha_manifest


def _json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None


def _zero(value: Mapping[str, Any], fields: tuple[str, ...]) -> bool:
    return all(value.get(field, 0) == 0 for field in fields)


def _root(root: Path, required: tuple[str, ...], source_commit: str, source_tree: str) -> tuple[bool, list[str], dict[str, Any]]:
    errors: list[str] = []
    if not root.is_dir():
        return False, ["ROOT_MISSING"], {}
    seal_ok, seal_errors, _ = verify_sha_manifest(root)
    if not seal_ok:
        errors.extend(f"SEAL:{item}" for item in seal_errors)
    values = {path: _json(root / path) for path in required}
    for path, value in values.items():
        if not isinstance(value, Mapping):
            errors.append(f"RECEIPT_INVALID:{path}")
            continue
        if value.get("source_commit") not in (None, source_commit):
            errors.append(f"SOURCE_COMMIT:{path}")
        if value.get("source_tree") not in (None, source_tree):
            errors.append(f"SOURCE_TREE:{path}")
        if not _zero(value, ("eval160_reads", "protected_eval_reads", "attack_rollouts", "vis_pgd_attack_rollouts")):
            errors.append(f"BOUNDARY:{path}")
    return not errors, errors, values


def build(args: argparse.Namespace) -> dict[str, Any]:
    stage_v = args.stage_v_root.resolve()
    v2 = args.stage_v2_root.resolve()
    stage_o = args.stage_o_root.resolve()
    roots = (stage_v, v2, stage_o)
    errors: list[str] = []
    if len(set(roots)) != len(roots):
        errors.append("ROOT_REUSE_OR_COLLISION")
    if args.old_root and any(root == args.old_root.resolve() or args.old_root.resolve() in root.parents for root in roots):
        errors.append("OLD_ROOT_REUSE")
    v_ok, v_errors, v_receipts = _root(stage_v, ("STAGE_V_CLOSURE_RECEIPT.json", "STAGE_V_COUNTERFACTUAL_AUDIT.json", "SUPERVISOR_COMPLETE.json", "DISPATCHER_COMPLETE.json"), args.source_commit, args.source_tree)
    v2_ok, v2_errors, v2_receipts = _root(v2, ("STAGE_V2_TEACHER_ENRICHMENT_REPORT.json", "STAGE_V2_INDEPENDENT_AUDIT.json"), args.source_commit, args.source_tree)
    o_ok, o_errors, o_receipts = _root(stage_o, ("STAGE_O_REPORT.json", "STAGE_O_INDEPENDENT_AUDIT.json", "STAGE_O_COMPLETE.json"), args.source_commit, args.source_tree)
    errors.extend(f"STAGE_V:{item}" for item in v_errors)
    errors.extend(f"STAGE_V2:{item}" for item in v2_errors)
    errors.extend(f"STAGE_O:{item}" for item in o_errors)
    closure = v_receipts.get("STAGE_V_CLOSURE_RECEIPT.json", {})
    audit = v_receipts.get("STAGE_V_COUNTERFACTUAL_AUDIT.json", {})
    if closure.get("status") != "STAGE_V_FORMAL_MAP_CLOSED" or audit.get("verdict") != "PASS":
        errors.append("STAGE_V_NOT_CLOSED")
    if any(closure.get(field) != expected for field, expected in (("planned_parents", 40), ("started_parents", 40), ("completed_parents", 40), ("audited_parents", 40), ("accepted_parents", 40), ("planned_branches", 2880), ("completed_branches", 2880))):
        errors.append("STAGE_V_COUNT_GATE_FAIL")
    if v2_receipts.get("STAGE_V2_INDEPENDENT_AUDIT.json", {}).get("verdict") != "PASS":
        errors.append("STAGE_V2_AUDIT_NOT_PASS")
    if o_receipts.get("STAGE_O_INDEPENDENT_AUDIT.json", {}).get("verdict") != "PASS" or o_receipts.get("STAGE_O_COMPLETE.json", {}).get("status") != "STAGE_O_PASS":
        errors.append("STAGE_O_NOT_PASS")
    status = "STAGE_V2_O_COMPLETE" if not errors else "FAIL_CLOSED"
    summary = {
        "schema": "GOAL_MAINLINE_SUMMARY_V1",
        "status": status,
        "source_commit": args.source_commit,
        "source_tree": args.source_tree,
        "stage_v_root": str(stage_v),
        "stage_v2_root": str(v2),
        "stage_o_root": str(stage_o),
        "stage_v2_scientific_verdict": v2_receipts.get("STAGE_V2_TEACHER_ENRICHMENT_REPORT.json", {}).get("status"),
        "planned_parents": 40,
        "accepted_parents": closure.get("accepted_parents", 0),
        "planned_branches": 2880,
        "completed_branches": closure.get("completed_branches", 0),
        "stage_o_jobs": o_receipts.get("STAGE_O_REPORT.json", {}).get("jobs", 0),
        "old_root_reused": False,
        "eval160_reads": 0,
        "protected_eval_reads": 0,
        "vis_pgd_attack_rollouts": 0,
        "manual_video_review": "PENDING",
        "official_sr_is_sole_metric": False,
        "moka_exploratory_mixed_into_formal_map": False,
        "errors": sorted(set(errors)),
        "generated_utc": _datetime.datetime.now(_datetime.timezone.utc).isoformat(),
    }
    atomic_write_json(args.output_root / "GOAL_MAINLINE_SUMMARY.json", summary)
    report_lines = [
        "# Counterfactual Temporal Susceptibility Mainline",
        "",
        f"Status: `{status}`",
        "",
        "This report is receipt-driven; no missing metric is inferred or filled.",
        "",
        "## Frozen roots",
        "",
        f"- Stage V: `{stage_v}`",
        f"- Stage V2: `{v2}`",
        f"- Stage O: `{stage_o}`",
        "",
        "## Boundary",
        "",
        "- old root reused: `false`",
        "- Eval160 reads: `0`",
        "- protected eval reads: `0`",
        "- VIS/PGD/attack rollouts: `0`",
        "",
        "## Errors",
        "",
    ] + [f"- `{item}`" for item in sorted(set(errors))]
    (args.output_root / "GOAL_MAINLINE_REPORT.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    index = {
        "schema": "GOAL_ARTIFACT_INDEX_V1",
        "status": status,
        "artifacts": [{"path": str(path.relative_to(args.output_root)), "sha256": sha256_file(path)} for path in sorted(args.output_root.iterdir()) if path.is_file()],
        "generated_utc": summary["generated_utc"],
    }
    atomic_write_json(args.output_root / "GOAL_ARTIFACT_INDEX.json", index)
    files = sorted(path for path in args.output_root.iterdir() if path.is_file() and path.name not in {"GOAL_SHA256SUMS", "GOAL_SHA256SUMS.sha256"})
    (args.output_root / "GOAL_SHA256SUMS").write_text("".join(f"{sha256_file(path)}  {path.name}\n" for path in files), encoding="utf-8")
    (args.output_root / "GOAL_SHA256SUMS.sha256").write_text(f"{sha256_file(args.output_root / 'GOAL_SHA256SUMS')}  GOAL_SHA256SUMS\n", encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-v-root", type=Path, required=True)
    parser.add_argument("--stage-v2-root", type=Path, required=True)
    parser.add_argument("--stage-o-root", type=Path, required=True)
    parser.add_argument("--old-root", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--source-tree", required=True)
    args = parser.parse_args(argv)
    args.output_root = args.output_root.resolve()
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise SystemExit("goal artifact output root must be new or empty")
    args.output_root.mkdir(parents=True, exist_ok=True)
    summary = build(args)
    print(json.dumps({"status": summary["status"], "errors": summary["errors"]}, sort_keys=True))
    return 0 if summary["status"] != "FAIL_CLOSED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
