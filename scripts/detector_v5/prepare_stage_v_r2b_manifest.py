"""Prepare the pre-registered Stage V R2B extension, without launching it."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

try:
    from .stage_v_dynamic_common import atomic_write_json, load_rows, normalize_parent, sha256_file, utc_now
    from scripts.monitoring.audit_stage_v_closure import verify_sha_manifest
except ImportError:  # direct server execution
    from stage_v_dynamic_common import atomic_write_json, load_rows, normalize_parent, sha256_file, utc_now
    from scripts.monitoring.audit_stage_v_closure import verify_sha_manifest


SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
MIN_POSITIVE = 12
MIN_NEGATIVE = 12
MIN_SUITE_POSITIVE = 2
MIN_SUITE_NEGATIVE = 2


def _json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return default


def _bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        value = value.strip().lower()
        if value in {"1", "true", "yes", "positive", "present"}:
            return True
        if value in {"0", "false", "no", "negative", "absent"}:
            return False
    return None


def _parent_class(parent_dir: Path, result: Mapping[str, Any]) -> bool | None:
    for key in ("local_vulnerability", "local_vulnerable", "local_positive", "local_vulnerability_positive"):
        value = _bool(result.get(key))
        if value is not None:
            return value
    labels: list[bool] = []
    branch_file = parent_dir / "COUNTERFACTUAL_BRANCHES.jsonl"
    if branch_file.is_file():
        for line in branch_file.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, Mapping):
                continue
            comparison = row.get("comparison") if isinstance(row.get("comparison"), Mapping) else {}
            for value in (row.get("local_vulnerability"), row.get("local_vulnerable"), comparison.get("local_vulnerability"), comparison.get("local_vulnerable")):
                parsed = _bool(value)
                if parsed is not None:
                    labels.append(parsed)
                    break
    if True in labels:
        return True
    if False in labels:
        return False
    return None


def _r2a_classes(root: Path) -> tuple[dict[str, bool | None], list[str]]:
    values: dict[str, bool | None] = {}
    errors: list[str] = []
    for result_path in sorted(root.rglob("PARENT_RESULT.json")):
        if "MONITOR" in result_path.relative_to(root).parts:
            continue
        result = _json(result_path)
        if not isinstance(result, Mapping):
            errors.append(f"INVALID_PARENT_RESULT:{result_path}")
            continue
        key = str(result.get("canonical_parent_key", ""))
        if not key or key in values:
            errors.append(f"DUPLICATE_OR_EMPTY_PARENT:{key}")
            continue
        values[key] = _parent_class(result_path.parent, result)
    return values, errors


def _support(classes: Mapping[str, bool | None], suites: Mapping[str, str]) -> dict[str, Any]:
    positive = [key for key, value in classes.items() if value is True]
    negative = [key for key, value in classes.items() if value is False]
    by_suite = {
        suite: {
            "positive": sum(suites.get(key) == suite for key in positive),
            "negative": sum(suites.get(key) == suite for key in negative),
        }
        for suite in SUITES
    }
    evaluable = [suite for suite, item in by_suite.items() if item["positive"] >= MIN_SUITE_POSITIVE and item["negative"] >= MIN_SUITE_NEGATIVE]
    return {
        "local_vulnerable_positive_parents": len(positive),
        "local_vulnerable_negative_parents": len(negative),
        "unknown_parent_class_count": sum(value is None for value in classes.values()),
        "by_suite": by_suite,
        "evaluable_suite_count": len(evaluable),
        "gate": {
            "positive_ge_12": len(positive) >= MIN_POSITIVE,
            "negative_ge_12": len(negative) >= MIN_NEGATIVE,
            "three_suites_with_2_positive_and_2_negative": len(evaluable) >= 3,
        },
    }


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    root = args.r2a_root.resolve()
    closure = _json(root / "STAGE_V_CLOSURE_RECEIPT.json", {})
    audit = _json(root / "STAGE_V_COUNTERFACTUAL_AUDIT.json", {})
    errors: list[str] = []
    if closure.get("status") != "STAGE_V_FORMAL_MAP_CLOSED":
        errors.append("R2A_CLOSURE_NOT_PASS")
    if audit.get("verdict") != "PASS":
        errors.append("R2A_AUDIT_NOT_PASS")
    if int(closure.get("accepted_parents", -1)) != 40 or int(closure.get("completed_branches", -1)) != 2880:
        errors.append("R2A_CLOSURE_COUNTS_INVALID")
    seal_ok, seal_errors, _ = verify_sha_manifest(root)
    if not seal_ok:
        errors.extend(f"R2A_SEAL:{item}" for item in seal_errors)
    classes, class_errors = _r2a_classes(root)
    errors.extend(class_errors)
    r2a_manifest = set(str(row.get("canonical_parent_key")) for row in load_rows(args.r2a_manifest))
    candidate_rows = [normalize_parent(row) for row in load_rows(args.candidate_manifest)]
    candidate_by_key = {str(row["canonical_parent_key"]): row for row in candidate_rows}
    suites = {key: str(row.get("suite")) for key, row in candidate_by_key.items()}
    support = _support(classes, suites)
    support_pass = all(support["gate"].values()) and not errors
    selected: list[dict[str, Any]] = []
    if not support_pass:
        for suite in SUITES:
            candidates = [row for row in candidate_rows if str(row.get("suite")) == suite and str(row.get("canonical_parent_key")) not in r2a_manifest and row.get("old_artifacts_reused") is not True and row.get("qualified", True) is not False]
            candidates.sort(key=lambda row: (hashlib.sha256(f"{args.salt}::{row['canonical_parent_key']}".encode()).hexdigest(), str(row["canonical_parent_key"])))
            selected.extend(candidates[:args.parents_per_suite])
        if len(selected) != len(SUITES) * args.parents_per_suite:
            errors.append(f"R2B_NEXT_BATCH_INSUFFICIENT:{len(selected)}/{len(SUITES) * args.parents_per_suite}")
    status = "R2B_NOT_REQUIRED" if support_pass else "R2B_REQUIRED" if not errors else "R2B_PREPARATION_FAIL"
    decision = {
        "schema": "STAGE_V_R2B_PRE_REGISTERED_DECISION_V1",
        "status": status,
        "r2a_root": str(root),
        "r2a_audit_sha256": sha256_file(root / "STAGE_V_COUNTERFACTUAL_AUDIT.json") if (root / "STAGE_V_COUNTERFACTUAL_AUDIT.json").is_file() else None,
        "r2a_manifest_sha256": sha256_file(args.r2a_manifest),
        "candidate_manifest_sha256": sha256_file(args.candidate_manifest),
        "salt": args.salt,
        "support": support,
        "selected_count": len(selected),
        "selected_parents": selected,
        "errors": sorted(set(errors)),
        "eval160_reads": 0,
        "protected_eval_reads": 0,
        "vis_pgd_attack_rollouts": 0,
        "generated_utc": utc_now(),
    }
    atomic_write_json(args.output_root / "STAGE_V_R2B_DECISION.json", decision)
    if status == "R2B_REQUIRED":
        manifest = {
            "schema": "STAGE_V_R2B_PARENT_MANIFEST_V1",
            "status": "FROZEN_PRELAUNCH",
            "source_commit": args.source_commit,
            "source_tree": args.source_tree,
            "r2a_root": str(root),
            "r2a_manifest_sha256": sha256_file(args.r2a_manifest),
            "candidate_manifest_sha256": sha256_file(args.candidate_manifest),
            "salt": args.salt,
            "parents": selected,
            "selected_parents": selected,
            "selected_count": len(selected),
            "old_artifacts_reused": False,
            "generated_utc": utc_now(),
        }
        path = args.output_root / "STAGE_V_R2B_PARENT_MANIFEST.json"
        atomic_write_json(path, manifest)
        (args.output_root / "STAGE_V_R2B_PARENT_MANIFEST.sha256").write_text(f"{sha256_file(path)}  {path.name}\n", encoding="utf-8")
    return decision


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r2a-root", type=Path, required=True)
    parser.add_argument("--r2a-manifest", type=Path, required=True)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--source-tree", required=True)
    parser.add_argument("--salt", default="STAGE_V_R2B_NEXT_BATCH_20260807")
    parser.add_argument("--parents-per-suite", type=int, default=10)
    args = parser.parse_args(argv)
    args.output_root = args.output_root.resolve()
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise SystemExit("R2B output root must be new or empty")
    args.output_root.mkdir(parents=True, exist_ok=True)
    decision = prepare(args)
    print(json.dumps({"status": decision["status"], "errors": decision["errors"]}, sort_keys=True))
    return 0 if decision["status"] in {"R2B_NOT_REQUIRED", "R2B_REQUIRED"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
