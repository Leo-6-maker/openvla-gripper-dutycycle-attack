#!/usr/bin/env python3
"""Materialize the frozen post-HOLD V1.1 inventories from source evidence."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping


COUNTERS = {"protected_reads": 0, "eval160_reads": 0, "attack_rollouts": 0, "vis_pgd_attack_rollouts": 0}
SUITES = ("libero_10", "libero_goal", "libero_spatial")
TARGETS = {"libero_10": 1, "libero_goal": 3, "libero_spatial": 4}


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON_OBJECT_REQUIRED:{path}")
    return value


def _jsonl(path: Path) -> list[tuple[dict[str, Any], str]]:
    rows: list[tuple[dict[str, Any], str]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise ValueError(f"JSONL_OBJECT_REQUIRED:{path}")
        rows.append((value, hashlib.sha256((raw + "\n").encode()).hexdigest()))
    return rows


def _write(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)
    path.with_name(path.name + ".sha256").write_text(f"{_sha(path)}  {path.name}\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text("".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    os.replace(temporary, path)
    path.with_name(path.name + ".sha256").write_text(f"{_sha(path)}  {path.name}\n", encoding="utf-8")


def _pair_inventory(report_path: Path, report: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    if report.get("status") != "HOLD_FORMAL_M4_CORRIDOR_INSUFFICIENT" or report.get("sealed") is not True or report.get("immutable") is not True:
        raise ValueError("TERMINAL_REPORT_NOT_IMMUTABLY_SEALED")
    audit = report.get("independent_audit", {})
    if audit.get("protected_counters") != COUNTERS or audit.get("outcomes_read") is not False:
        raise ValueError("TERMINAL_REPORT_BOUNDARY_INVALID")
    rows = audit.get("receipt_rows")
    if not isinstance(rows, list) or len(rows) != 80:
        raise ValueError("TERMINAL_RECEIPT_COUNT_NOT_80")
    grouped: dict[str, dict[str, Any]] = {}
    for row in rows:
        key, replicate = str(row.get("canonical_parent_key", "")), str(row.get("replicate", ""))
        if not key or replicate not in {"A", "B"} or replicate in grouped.setdefault(key, {}):
            raise ValueError(f"TERMINAL_RECEIPT_KEY_INVALID:{key}:{replicate}")
        receipt = Path(str(row.get("path", "")))
        if not receipt.is_file() or _sha(receipt) != row.get("receipt_sha256"):
            raise ValueError(f"TERMINAL_RECEIPT_HASH_MISMATCH:{key}:{replicate}")
        grouped[key][replicate] = dict(row)
    if len(grouped) != 40 or any(set(pair) != {"A", "B"} for pair in grouped.values()):
        raise ValueError("TERMINAL_PARENT_PAIR_ACCOUNTING_INVALID")
    parents = []
    for key in sorted(grouped):
        pair = grouped[key]
        statuses = [str(pair[rep].get("status")) for rep in ("A", "B")]
        if statuses != ["PASS", "PASS"]:
            continue
        parents.append({
            "canonical_parent_key": key,
            "suite": key.split("/", 1)[0],
            "status_pair": "PASS/PASS",
            "A": {field: pair["A"].get(field) for field in ("path", "receipt_sha256", "trajectory_sha256", "reason")},
            "B": {field: pair["B"].get(field) for field in ("path", "receipt_sha256", "trajectory_sha256", "reason")},
        })
    if len(parents) != 32:
        raise ValueError(f"PREDECESSOR_PASS_PASS_COUNT_NOT_32:{len(parents)}")
    inventory = {
        "schema": "STAGE_V_M4_CORRIDOR_PREDECESSOR_PASS_PASS_INVENTORY_V1",
        "status": "PASS_EXACT_32_IMMUTABLE_CURRENT_SOURCE_PAIRS",
        "sealed": True,
        "source_report": str(report_path),
        "source_report_sha256": _sha(report_path),
        "science_commit": report["source_binding"]["science_commit"],
        "science_tree": report["source_binding"]["science_tree"],
        "parent_count": 32,
        "receipt_count": 64,
        "parents": parents,
        "outcomes_read": False,
        "protected_counters": dict(COUNTERS),
    }
    return inventory, grouped


def _taxonomy_function(science_root: Path, expected_sha: str) -> Any:
    source = science_root / "src/gripper_attack/stage_v_m3_5_physical_taxonomy.py"
    if _sha(source) != expected_sha:
        raise ValueError("TAXONOMY_IMPLEMENTATION_HASH_MISMATCH")
    spec = importlib.util.spec_from_file_location("frozen_stage_v_taxonomy", source)
    if spec is None or spec.loader is None:
        raise ValueError("TAXONOMY_IMPORT_SPEC_INVALID")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.taxonomy_eligibility_from_bddl


def _qualified_rows(paths: dict[str, Path], source_commit: str, source_tree: str) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    by_key: dict[str, dict[str, Any]] = {}
    bindings: dict[str, Any] = {}
    for suite, path in paths.items():
        bindings[suite] = {"path": str(path), "sha256": _sha(path)}
        for row, line_sha in _jsonl(path):
            key = str(row.get("canonical_parent_key", ""))
            if key in by_key:
                raise ValueError(f"DUPLICATE_V7_ROW:{key}")
            replicates = row.get("replicates", {})
            if row.get("qualified") is True:
                if row.get("errors") != [] or any(replicates.get(rep, {}).get("status") != "PASS" for rep in ("A", "B")):
                    raise ValueError(f"V7_QUALIFIED_ROW_INVALID:{key}")
                if any(replicates[rep].get("source_commit") != source_commit or replicates[rep].get("source_tree") != source_tree for rep in ("A", "B")):
                    raise ValueError(f"V7_SOURCE_BINDING_INVALID:{key}")
            by_key[key] = {**row, "source_rows_path": str(path), "source_rows_sha256": bindings[suite]["sha256"], "source_row_sha256": line_sha}
    return by_key, bindings


def _static_rows(
    *, v7_manifest: Mapping[str, Any], v7_rows: Mapping[str, Mapping[str, Any]], attempted: set[str],
    upstream_root: Path, taxonomy: Any, taxonomy_sha: str,
) -> list[dict[str, Any]]:
    sys.path.insert(0, str(upstream_root))
    from libero.libero import benchmark, get_libero_path  # type: ignore

    candidate_keys = {str(row.get("canonical_parent_key")) for row in v7_manifest.get("candidates", [])}
    selected = [row for row in v7_rows.values() if row.get("qualified") is True and str(row.get("canonical_parent_key")) in candidate_keys and str(row.get("canonical_parent_key")) not in attempted and str(row.get("suite")) in SUITES]
    selected.sort(key=lambda row: (SUITES.index(str(row["suite"])), str(row.get("qualification_rank_sha256", "")), str(row["canonical_parent_key"])))
    suites = {suite: benchmark.get_benchmark_dict()[suite]() for suite in SUITES}
    result: list[dict[str, Any]] = []
    for rank, row in enumerate(selected, 1):
        suite, task_index = str(row["suite"]), int(row["task_index"])
        task = suites[suite].get_task(task_index)
        bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        try:
            value = taxonomy(bddl)
            status = "SUPPORTED" if value.get("status") == "PASS" and value.get("eligible") is True else "UNSUPPORTED_TAXONOMY"
            reason = str(value.get("reason") or "SUPPORTED_IN_OR_ON_SOURCE_OBJECT")
        except (OSError, UnicodeError, ValueError) as exc:
            value = {"declared_object_ids": [], "target_object_ids": []}
            status, reason = "ABSTAIN_STATIC", f"{type(exc).__name__}:{exc}"
        result.append({
            "canonical_parent_key": row["canonical_parent_key"],
            "suite": suite,
            "task_index": task_index,
            "state_index": int(row["state_index"]),
            "static_audit_rank": rank,
            "qualification_rank_sha256": row.get("qualification_rank_sha256"),
            "v7_candidate_sha256": row.get("candidate_sha256"),
            "bddl_path": str(bddl),
            "bddl_sha256": _sha(bddl),
            "declared_objects": list(value.get("declared_object_ids", [])),
            "goal_source_objects": list(value.get("target_object_ids", [])),
            "status": status,
            "reason": reason,
            "taxonomy_implementation_sha256": taxonomy_sha,
            "v7_source_rows_path": row["source_rows_path"],
            "v7_source_rows_sha256": row["source_rows_sha256"],
            "v7_source_row_sha256": row["source_row_sha256"],
        })
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--terminal-report", type=Path, required=True)
    parser.add_argument("--invalid-hold", type=Path, required=True)
    parser.add_argument("--prior-candidate-manifest", type=Path, required=True)
    parser.add_argument("--v7-candidate-manifest", type=Path, required=True)
    parser.add_argument("--v7-rows", action="append", required=True, metavar="SUITE=PATH")
    parser.add_argument("--science-root", type=Path, required=True)
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--source-tree", required=True)
    parser.add_argument("--taxonomy-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, help="durable path recorded in generated cross-bindings")
    args = parser.parse_args(argv)

    paths = {name: Path(value).resolve() for name, value in (item.split("=", 1) for item in args.v7_rows)}
    if set(paths) != set(SUITES):
        raise ValueError("V7_ROWS_REQUIRE_EXACT_TARGET_SUITES")
    terminal_path, invalid_path = args.terminal_report.resolve(), args.invalid_hold.resolve()
    report, invalid = _load(terminal_path), _load(invalid_path)
    inventory, old_pairs = _pair_inventory(terminal_path, report)
    if invalid.get("status") != "HOLD_ENGINEERING_INVALID_PRELAUNCH_GOVERNANCE_INCOMPLETE" or invalid.get("sealed") is not True or invalid.get("consumable") is not False:
        raise ValueError("INVALID_PRELAUNCH_ROOT_NOT_SEALED")
    new_attempted = {str(row.get("canonical_parent_key")) for row in invalid.get("attempted_identities", [])}
    if len(new_attempted) != 3 or new_attempted & set(old_pairs):
        raise ValueError("INVALID_ROOT_ATTEMPT_ACCOUNTING_INVALID")
    attempts = []
    for key in sorted(old_pairs):
        pair = old_pairs[key]
        attempts.append({"canonical_parent_key": key, "suite": key.split("/", 1)[0], "origin": "V2_CURRENT_SOURCE_FINAL40", "status_pair": f"{pair['A'].get('status')}/{pair['B'].get('status')}", "A_receipt_sha256": pair["A"].get("receipt_sha256"), "B_receipt_sha256": pair["B"].get("receipt_sha256")})
    attempts.extend({"canonical_parent_key": row["canonical_parent_key"], "suite": row["suite"], "origin": "POST_HOLD_V1_ENGINEERING_INVALID", "status_pair": row["pair_status"]} for row in invalid["attempted_identities"])
    attempt_registry = {
        "schema": "STAGE_V_M4_CORRIDOR_ATTEMPT_REGISTRY_V1_1",
        "status": "FROZEN_EXACT_43",
        "terminal_report_sha256": _sha(terminal_path),
        "invalid_hold_sha256": _sha(invalid_path),
        "attempted_identity_count": len(attempts),
        "attempted_identities": attempts,
        "firewall": "EXCLUDE_FROM_FUTURE_CORRIDOR_AND_PRIMARY_TEACHER_STUDENT_FIT_CAL_CHECK_THRESHOLD_CHECKPOINT_FEATURE_MODEL_SELECTION",
        "protected_counters": dict(COUNTERS),
        "outcomes_read": False,
    }
    if len(attempts) != 43 or len({row["canonical_parent_key"] for row in attempts}) != 43:
        raise ValueError("ATTEMPT_REGISTRY_NOT_EXACT_43")

    v7_manifest_path = args.v7_candidate_manifest.resolve()
    v7_manifest = _load(v7_manifest_path)
    if v7_manifest.get("status") != "FROZEN" or v7_manifest.get("source_commit") != args.source_commit or v7_manifest.get("source_tree") != args.source_tree:
        raise ValueError("V7_CANDIDATE_MANIFEST_INVALID")
    qualified, row_bindings = _qualified_rows(paths, args.source_commit, args.source_tree)
    taxonomy = _taxonomy_function(args.science_root.resolve(), args.taxonomy_sha256)
    static_rows = _static_rows(v7_manifest=v7_manifest, v7_rows=qualified, attempted=set(old_pairs), upstream_root=args.upstream_root.resolve(), taxonomy=taxonomy, taxonomy_sha=args.taxonomy_sha256)
    counts = {status: sum(row["status"] == status for row in static_rows) for status in ("SUPPORTED", "UNSUPPORTED_TAXONOMY", "ABSTAIN_STATIC")}
    static_audit = {
        "schema": "STAGE_V_M4_POST_HOLD_STATIC_TAXONOMY_AUDIT_V1",
        "status": "PASS_STATIC_SUPPORTED_POOL" if counts["ABSTAIN_STATIC"] == 0 else "HOLD_STATIC_ABSTAIN",
        "rule": "taxonomy_eligibility_from_bddl applied uniformly to every V7-qualified, V2-corridor-unattempted candidate; no task or identity blacklist",
        "taxonomy_path": str(args.science_root.resolve() / "src/gripper_attack/stage_v_m3_5_physical_taxonomy.py"),
        "taxonomy_sha256": args.taxonomy_sha256,
        "v7_candidate_manifest_path": str(v7_manifest_path),
        "v7_candidate_manifest_sha256": _sha(v7_manifest_path),
        "v7_rows": row_bindings,
        "terminal_report_sha256": _sha(terminal_path),
        "candidate_count": len(static_rows),
        "counts": counts,
        "rows": static_rows,
        "outcomes_read": False,
        "protected_counters": dict(COUNTERS),
    }
    prior_path = args.prior_candidate_manifest.resolve()
    prior = _load(prior_path)
    prior_keys = [str(row["canonical_parent_key"]) for row in prior.get("parents", [])]
    supported = {str(row["canonical_parent_key"]): row for row in static_rows if row["status"] == "SUPPORTED"}
    if set(prior_keys) != set(supported) or len(prior_keys) != len(supported):
        raise ValueError("STATIC_SUPPORTED_SET_DIFFERS_FROM_PRE_RECEIPT_V1_FREEZE")
    remaining_by_suite = {suite: [] for suite in SUITES}
    for old in prior["parents"]:
        key, suite = str(old["canonical_parent_key"]), str(old["suite"])
        if key in new_attempted:
            continue
        source = qualified.get(key)
        if source is None or key not in supported:
            raise ValueError(f"V1_1_PARENT_SOURCE_INVALID:{key}")
        remaining_by_suite[suite].append({
            "canonical_parent_key": key,
            "suite": suite,
            "task_index": int(old["task_index"]),
            "state_index": int(old["state_index"]),
            "selection_rank": len(remaining_by_suite[suite]) + 1,
            "predecessor_v1_rank": int(old["selection_rank"]),
            "taxonomy_status": "SUPPORTED",
            "bddl_path": supported[key]["bddl_path"],
            "bddl_sha256": supported[key]["bddl_sha256"],
            "v7_candidate_sha256": source["candidate_sha256"],
            "qualification_rank_sha256": source["qualification_rank_sha256"],
        })
    if any(len(remaining_by_suite[suite]) < TARGETS[suite] for suite in SUITES):
        raise ValueError("REMAINING_STATIC_POOL_CANNOT_FILL_DEFICIT")
    parents = [row for suite in SUITES for row in remaining_by_suite[suite]]
    compact_rows = []
    for parent in parents:
        source = qualified[parent["canonical_parent_key"]]
        compact_rows.append({
            "schema": source.get("schema"),
            "canonical_parent_key": source["canonical_parent_key"],
            "suite": source["suite"],
            "task_index": source["task_index"],
            "state_index": source["state_index"],
            "qualified": True,
            "errors": [],
            "candidate_sha256": source["candidate_sha256"],
            "qualification_rank_sha256": source["qualification_rank_sha256"],
            "replicates": {rep: {"status": source["replicates"][rep]["status"], "source_commit": source["replicates"][rep]["source_commit"], "source_tree": source["replicates"][rep]["source_tree"]} for rep in ("A", "B")},
            "source_rows_path": source["source_rows_path"],
            "source_rows_sha256": source["source_rows_sha256"],
            "source_row_sha256": source["source_row_sha256"],
        })
    output = args.output_dir.resolve()
    artifact_root = (args.artifact_root or output).resolve()
    subset_path = output / "STAGE_V_M4_POST_HOLD_V7_QUALIFIED_ROWS_V1_1.jsonl"
    _write_jsonl(subset_path, compact_rows)
    manifest = {
        "schema": "STAGE_V_M4_CORRIDOR_RESERVE_PARENT_MANIFEST_V1",
        "version": "POST-32-OF-40-HOLD-V1.1",
        "status": "FROZEN",
        "selection_status": "FROZEN_OUTCOME_BLIND_PRE_ROLLOUT",
        "source_binding": {"science_commit": args.source_commit, "science_tree": args.source_tree},
        "inputs": {
            "pre_receipt_v1_manifest_path": str(prior_path),
            "pre_receipt_v1_manifest_sha256": _sha(prior_path),
            "static_taxonomy_audit_path": str(artifact_root / "STAGE_V_M4_POST_HOLD_STATIC_TAXONOMY_AUDIT_V1.json"),
            "attempt_registry_path": str(artifact_root / "STAGE_V_M4_CORRIDOR_ATTEMPT_REGISTRY_V1_1.json"),
            "v7_control_qualification_rows_path": str(artifact_root / subset_path.name),
            "v7_control_qualification_rows_sha256": _sha(subset_path),
        },
        "selection_rule": "Preserve V1 frozen per-suite order; remove every newly CORRIDOR_ATTEMPTED identity; admit only uniformly taxonomy-supported V7-qualified identities; no fresh fallback tranche.",
        "parents": parents,
        "candidate_count": len(parents),
        "counts_by_suite": {suite: len(remaining_by_suite[suite]) for suite in SUITES},
        "target_stable_by_suite": dict(TARGETS),
        "outcomes_read": False,
        "protected_counters": dict(COUNTERS),
    }
    _write(output / "STAGE_V_M4_CORRIDOR_PREDECESSOR_PASS_PASS_INVENTORY_V1.json", inventory)
    _write(output / "STAGE_V_M4_POST_HOLD_STATIC_TAXONOMY_AUDIT_V1.json", static_audit)
    _write(output / "STAGE_V_M4_CORRIDOR_ATTEMPT_REGISTRY_V1_1.json", attempt_registry)
    _write(output / "STAGE_V_M4_POST_HOLD_CANDIDATE_PARENT_MANIFEST_V1_1.json", manifest)
    print(json.dumps({"status": "PASS", "predecessor_pairs": 32, "static_counts": counts, "attempted": 43, "remaining": manifest["counts_by_suite"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
