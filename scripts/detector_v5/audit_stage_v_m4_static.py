#!/usr/bin/env python3
"""CPU-only static audit for the formal M4 matched-action contract."""
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


def _function(tree: ast.AST, name: str) -> ast.AST:
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    raise ValueError(f"FUNCTION_MISSING:{name}")


def _called_names(node: ast.AST) -> set[str]:
    names: set[str] = set()
    for call in ast.walk(node):
        if isinstance(call, ast.Call):
            names.add(call.func.id if isinstance(call.func, ast.Name) else getattr(call.func, "attr", ""))
    return names


def audit(protocol_path: Path, *, source_commit: str, source_tree: str) -> dict[str, Any]:
    protocol = _load(protocol_path)
    runner_path = REPO_ROOT / "scripts/detector_v5/run_stage_v_m4_matched_parent.py"
    gate_a_path = REPO_ROOT / "scripts/detector_v5/run_stage_v_m3_5_v1_4_gate_a.py"
    gate_b_path = REPO_ROOT / "scripts/detector_v5/run_stage_v_m3_5_v1_4_gate_b.py"
    auditor_path = REPO_ROOT / "scripts/detector_v5/audit_stage_v_m4_matched_parent.py"
    snapshot_path = REPO_ROOT / "src/gripper_attack/stage_v_causal_observation_snapshot.py"
    runner_tree = ast.parse(runner_path.read_text(encoding="utf-8"), filename=str(runner_path))
    gate_a_tree = ast.parse(gate_a_path.read_text(encoding="utf-8"), filename=str(gate_a_path))
    gate_b_tree = ast.parse(gate_b_path.read_text(encoding="utf-8"), filename=str(gate_b_path))
    runner_text = runner_path.read_text(encoding="utf-8")
    gate_b_text = gate_b_path.read_text(encoding="utf-8")
    auditor_text = auditor_path.read_text(encoding="utf-8")
    snapshot_text = snapshot_path.read_text(encoding="utf-8")
    replay = _function(gate_a_tree, "_replay_canary")
    branch = _function(gate_b_tree, "_run_branch")
    replay_calls = _called_names(replay)
    branch_calls = _called_names(branch)
    matrix = protocol.get("matrix", {})
    operation = protocol.get("operation", {})
    source = protocol.get("source_binding", {})
    checks = {
        "protocol_schema": protocol.get("schema") == "STAGE_V_M4_MATCHED_ACTION_PROTOCOL_V1",
        "protocol_frozen_authorized": protocol.get("status") == "FROZEN_RUNTIME_AUTHORIZED" and protocol.get("runtime_authorized") is True,
        "source_binding": source.get("runtime_commit") == source_commit and source.get("runtime_tree") == source_tree,
        "matrix_exact": matrix == {
            "parents": 40, "probes_per_parent": 24, "repetitions": 1,
            "conditions": ["CONTROL", "T3", "T5", "T10"],
            "physical_executions_per_parent": 96, "treatment_labels_per_parent": 72,
            "physical_executions_total": 3840, "treatment_labels_total": 2880,
            "primary_dose": "T5", "secondary_doses": ["T3", "T10"],
            "dose_steps": {"T3": 3, "T5": 5, "T10": 10}, "h_phys": 10,
        },
        "primary_is_matched_replay": operation.get("matched_clean_action_replay") is True and operation.get("clean_reference_action_lineage") is True,
        "primary_has_no_native_policy_calls": operation.get("native_policy_calls_in_primary_window") == 0 and "predict_action" not in branch_calls,
        "primary_has_no_fresh_render_consumption": operation.get("fresh_render_primary_consumption") is False and operation.get("fresh_render_equality_gate_used") is False,
        "arm_isolation_contract": operation.get("arm_delta_linf_exact_zero") is True and operation.get("treatment_only_difference") == "GRIPPER_OPEN",
        "snapshot_canary_bound": {"load_snapshot", "capture_simulator_state", "capture_runtime_state"}.issubset(replay_calls),
        "branch_uses_matched_action": "matched_action" in branch_calls,
        "runner_emits_exact_counts": all(token in runner_text for token in ('"expected_physical_executions": 96', '"expected_treatment_labels": 72', "len(probes) != PROBE_COUNT")),
        "runner_emits_protected_counters": '"protected_counters": dict(COUNTERS)' in runner_text,
        "independent_auditor_is_producer_free": "run_stage_v_m4_matched_parent" not in auditor_text,
        "snapshot_module_has_exact_binding": all(token in snapshot_text for token in ("CAUSAL_PROBE_SNAPSHOT_V2", "assert_primary_observation_exact", "write_snapshot", "load_snapshot")),
        "protected_counters_zero": protocol.get("protected_counters") == COUNTERS,
    }
    status = "PASS_STATIC_DESIGN_ONLY" if all(checks.values()) else "FAIL_STATIC_CONTRACT"
    return {
        "schema": "STAGE_V_M4_STATIC_AUDIT_V1",
        "status": status,
        "runtime_authorized": False,
        "runtime_executed": False,
        "protocol": str(protocol_path.resolve()),
        "protocol_sha256": _sha(protocol_path),
        "source_commit": source_commit,
        "source_tree": source_tree,
        "runner_sha256": _sha(runner_path),
        "gate_a_sha256": _sha(gate_a_path),
        "gate_b_sha256": _sha(gate_b_path),
        "independent_auditor_sha256": _sha(auditor_path),
        "checks": checks,
        "primary_replay_calls": sorted(replay_calls),
        "primary_branch_calls": sorted(branch_calls),
        "protected_counters": dict(COUNTERS),
        "next_action": "AUTHORIZE_AND_LAUNCH_M4" if status == "PASS_STATIC_DESIGN_ONLY" else "HOLD_AND_REPAIR",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--source-tree", required=True)
    args = parser.parse_args(argv)
    result = audit(args.protocol.resolve(), source_commit=args.source_commit, source_tree=args.source_tree)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "output": str(args.output.resolve())}, sort_keys=True))
    return 0 if result["status"] == "PASS_STATIC_DESIGN_ONLY" else 2


if __name__ == "__main__":
    raise SystemExit(main())
