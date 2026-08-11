#!/usr/bin/env python3
"""CPU-only static audit for the prospective M3.5 V1.4 Gate-A path."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[2]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"JSON_OBJECT_REQUIRED:{path}")
    return value


def _function(tree: ast.AST, name: str) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    raise ValueError(f"FUNCTION_MISSING:{name}")


def _called_names(node: ast.AST) -> set[str]:
    names = set()
    for call in ast.walk(node):
        if isinstance(call, ast.Call):
            if isinstance(call.func, ast.Name):
                names.add(call.func.id)
            elif isinstance(call.func, ast.Attribute):
                names.add(call.func.attr)
    return names


def audit(protocol_path: Path, *, source_commit: str, source_tree: str) -> dict[str, Any]:
    runner_path = REPO_ROOT / "scripts/detector_v5/run_stage_v_m3_5_v1_4_gate_a.py"
    snapshot_path = REPO_ROOT / "src/gripper_attack/stage_v_causal_observation_snapshot.py"
    protocol = _load(protocol_path)
    runner_tree = ast.parse(runner_path.read_text(encoding="utf-8"), filename=str(runner_path))
    snapshot_tree = ast.parse(snapshot_path.read_text(encoding="utf-8"), filename=str(snapshot_path))
    replay = _function(runner_tree, "_replay_canary")
    required_runner_calls = {"load_snapshot", "restore_rng_state", "capture_simulator_state", "capture_runtime_state", "assert_exact"}
    required_snapshot_functions = {"write_snapshot", "load_snapshot", "capture_runtime_state", "capture_simulator_state", "restore_rng_state", "reference_action_window", "matched_action"}
    snapshot_names = {node.name for node in ast.walk(snapshot_tree) if isinstance(node, ast.FunctionDef)}
    primary_calls = _called_names(replay)
    checks = {
        "protocol_schema": protocol.get("schema") == "STAGE_V_M3_5_DIAGNOSTIC_PROTOCOL_V1_4_GATE_A",
        "protocol_is_draft_or_frozen": protocol.get("version") == "V1.4-GATE-A",
        "source_binding_is_not_predecessor": str(protocol.get("source_binding", {}).get("runtime_commit")) != "d104713027a82eeb858ba9036200d7ab010959cc",
        "runner_has_exact_replay_calls": required_runner_calls.issubset(primary_calls),
        "runner_primary_does_not_call_renderer": "get_libero_image" not in primary_calls,
        "snapshot_helpers_present": required_snapshot_functions.issubset(snapshot_names),
        "fresh_render_gate_literal_false": protocol.get("operation", {}).get("fresh_render_equality_gate_used") is False,
        "fresh_render_primary_hard_stop": protocol.get("operation", {}).get("fresh_render_primary_consumption") == "HARD_STOP",
        "protected_counters_zero": protocol.get("protected_counters") == {"protected_reads": 0, "eval160_reads": 0, "attack_rollouts": 0, "vis_pgd_attack_rollouts": 0},
    }
    status = "PASS_STATIC_DESIGN_ONLY" if all(checks.values()) else "FAIL_STATIC_CONTRACT"
    return {
        "schema": "STAGE_V_M3_5_V1_4_STATIC_AUDIT_V1",
        "status": status,
        "runtime_authorized": False,
        "runtime_executed": False,
        "source_commit": source_commit,
        "source_tree": source_tree,
        "runner_sha256": _sha256(runner_path),
        "snapshot_module_sha256": _sha256(snapshot_path),
        "checks": checks,
        "primary_replay_calls": sorted(primary_calls),
        "protected_counters": {"protected_reads": 0, "eval160_reads": 0, "attack_rollouts": 0, "vis_pgd_attack_rollouts": 0},
        "next_action": "KEEP_PROTOCOL_DRAFT_AND_DO_NOT_LAUNCH" if status == "PASS_STATIC_DESIGN_ONLY" else "HOLD_AND_REPAIR",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=REPO_ROOT / "configs/STAGE_V_M3_5_V1_4_GATE_A_PROTOCOL_DRAFT.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--source-tree", required=True)
    args = parser.parse_args(argv)
    result = audit(args.protocol.resolve(), source_commit=args.source_commit, source_tree=args.source_tree)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes((json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8"))
    print(json.dumps({"status": result["status"], "output": str(args.output.resolve())}, sort_keys=True))
    return 0 if result["status"] == "PASS_STATIC_DESIGN_ONLY" else 2


if __name__ == "__main__":
    raise SystemExit(main())
