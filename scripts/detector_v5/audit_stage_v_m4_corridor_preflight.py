#!/usr/bin/env python3
"""Static audit for the clean-only M4 corridor qualification."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[2]
COUNTERS = {"protected_reads": 0, "eval160_reads": 0, "attack_rollouts": 0, "vis_pgd_attack_rollouts": 0}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"JSON_OBJECT_REQUIRED:{path}")
    return value


def _calls(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return {node.func.id if isinstance(node.func, ast.Name) else getattr(node.func, "attr", "") for node in ast.walk(tree) if isinstance(node, ast.Call)}


def audit(protocol_path: Path, *, source_commit: str, source_tree: str) -> dict[str, Any]:
    protocol = _load(protocol_path)
    runner = REPO_ROOT / "scripts/detector_v5/run_stage_v_m4_corridor_preflight.py"
    text = runner.read_text(encoding="utf-8")
    source = protocol.get("source_binding", {})
    inputs = protocol.get("inputs", {})
    checks = {
        "protocol_schema": protocol.get("schema") == "STAGE_V_M4_CORRIDOR_QUALIFICATION_PROTOCOL_V1",
        "protocol_frozen_authorized": protocol.get("status") == "FROZEN_RUNTIME_AUTHORIZED" and protocol.get("runtime_authorized") is True,
        "source_binding": source.get("runtime_commit") == source_commit and source.get("runtime_tree") == source_tree,
        "candidate_count_exact": protocol.get("qualification", {}).get("candidate_parent_count") == 40,
        "replicates_exact": protocol.get("qualification", {}).get("clean_replicates") == ["A", "B"],
        "corridor_threshold_exact": protocol.get("qualification", {}).get("minimum_corridor_candidates") == 24,
        "outcome_blind": protocol.get("operation", {}).get("outcomes_read") is False,
        "no_intervention": protocol.get("operation", {}).get("intervention_executed") is False,
        "protected_counters_zero": protocol.get("protected_counters") == COUNTERS,
        "runner_has_probe_planner": "select_probe_steps" in text,
        "runner_has_no_branch_execution": "_run_branch(" not in text and "_pair_label(" not in text,
        "runner_has_no_open_literal": "OPEN_T" not in text and "forced_open" not in text,
        "runner_writes_clean_only_receipt": "M4_CORRIDOR_PREFLIGHT_V1" in text,
        "runner_records_registered_clean_failure": "OBJECT_TAXONOMY_BINDING_" in text and "CLEAN_FAILURE" in text,
        "runner_checks_frozen_split": "formal_parent_split_path" in text and "formal_parent_split_sha256" in text,
        "formal_split_bound": bool(inputs.get("formal_parent_split_path")) and bool(inputs.get("formal_parent_split_sha256")),
    }
    status = "PASS_STATIC_DESIGN_ONLY" if all(checks.values()) else "FAIL_STATIC_CONTRACT"
    return {
        "schema": "STAGE_V_M4_CORRIDOR_STATIC_AUDIT_V1",
        "status": status,
        "runtime_authorized": False,
        "protocol": str(protocol_path.resolve()),
        "protocol_sha256": _sha(protocol_path),
        "source_commit": source_commit,
        "source_tree": source_tree,
        "runner_sha256": _sha(runner),
        "checks": checks,
        "calls": sorted(_calls(runner)),
        "protected_counters": dict(COUNTERS),
        "next_action": "AUTHORIZE_AND_LAUNCH_CORRIDOR_PREFLIGHT" if status == "PASS_STATIC_DESIGN_ONLY" else "HOLD_AND_REPAIR",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--source-tree", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = audit(args.protocol.resolve(), source_commit=args.source_commit, source_tree=args.source_tree)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "output": str(args.output.resolve())}, sort_keys=True))
    return 0 if result["status"] == "PASS_STATIC_DESIGN_ONLY" else 2


if __name__ == "__main__":
    raise SystemExit(main())
