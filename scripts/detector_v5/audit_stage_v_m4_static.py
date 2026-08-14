#!/usr/bin/env python3
"""CPU-only static audit for the formal M4 matched-action contract."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.detector_v5.stage_v_m4_governance import protocol_declares_corridor_gate  # noqa: E402

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


def _current_authority_binding_complete(protocol: Mapping[str, Any]) -> bool:
    inputs = protocol.get("inputs")
    if not isinstance(inputs, Mapping):
        return False
    files = (
        "v1_supersession_receipt",
        "formal_parent_manifest",
        "formal_parent_split",
        "exact_plan_manifest",
        "exact_plan_audit",
        "exact_plan_result",
        "primary_firewall_report",
        "teacher_student_freeze_report",
        "pre_m4_lock_report",
        "student_checkpoint",
        "student_thresholds",
        "feature_schema",
        "architecture_addendum",
    )
    complete = all(isinstance(inputs.get(f"{name}_{suffix}"), str) and inputs[f"{name}_{suffix}"] for name in files for suffix in ("path", "sha256")) and all(isinstance(inputs.get(f"{name}_root"), str) and inputs[f"{name}_root"] and isinstance(inputs.get(f"{name}_root_seal_sha256"), str) and inputs[f"{name}_root_seal_sha256"] for name in ("exact_plan", "primary_firewall", "teacher_student_freeze", "pre_m4_lock")) and isinstance(inputs.get("feature_order_sha256"), str) and bool(inputs["feature_order_sha256"])
    if protocol.get("successor_protocol") is True:
        complete = complete and all(isinstance(inputs.get(f"{name}_{suffix}"), str) and inputs[f"{name}_{suffix}"] for name in ("snapshot_rebind_receipt", "compatibility_q00_result", "compatibility_q00_audit", "compatibility_fleet_preflight", "compatibility_fleet_authority", "compatibility_fleet_result", "compatibility_runtime_provenance", "successor_runtime_provenance") for suffix in ("path", "sha256"))
        complete = complete and isinstance(inputs.get("compatibility_audit_root"), str) and bool(inputs["compatibility_audit_root"]) and isinstance(inputs.get("compatibility_audit_root_seal_sha256"), str) and bool(inputs["compatibility_audit_root_seal_sha256"])
    return complete


def _snapshot_inventory_sha256(manifest: Mapping[str, Any]) -> str:
    rows = []
    for raw in manifest.get("probe_authorities", []):
        if not isinstance(raw, Mapping):
            return ""
        row = {key: raw.get(key) for key in ("canonical_parent_key", "probe_id", "probe_step", "snapshot_path", "snapshot_manifest_sha256")}
        try:
            row["probe_step"] = int(row["probe_step"])
        except (TypeError, ValueError):
            return ""
        if any(not row[key] for key in row if key != "probe_step"):
            return ""
        rows.append(row)
    identities = {(row["canonical_parent_key"], row["probe_id"]) for row in rows}
    if len(rows) != 960 or len(identities) != 960:
        return ""
    payload = json.dumps(sorted(rows, key=lambda row: (row["canonical_parent_key"], row["probe_id"])), sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _successor_snapshot_rebind_complete(protocol: Mapping[str, Any], *, source_commit: str, source_tree: str) -> bool:
    if protocol.get("successor_protocol") is not True:
        return True
    inputs = protocol.get("inputs")
    source = protocol.get("source_binding")
    if not isinstance(inputs, Mapping) or not isinstance(source, Mapping):
        return False
    try:
        def bound(name: str) -> Path:
            path = Path(str(inputs[f"{name}_path"]))
            return path.resolve()

        bridge_path = bound("snapshot_rebind_receipt")
        exact_path = bound("exact_plan_manifest")
        bridge = _load(bridge_path)
        exact = _load(exact_path)
        if _sha(bridge_path) != inputs.get("snapshot_rebind_receipt_sha256") or _sha(exact_path) != inputs.get("exact_plan_manifest_sha256"):
            return False
        if bridge.get("schema") != "STAGE_V_M4_SNAPSHOT_REBIND_RECEIPT_V1" or bridge.get("status") != "PASS_SNAPSHOT_REBIND_AUTHORITY" or bridge.get("compatibility_only") is not True:
            return False
        if bridge.get("successor_runtime_source") != {"commit": source_commit, "tree": source_tree} or bridge.get("exact_plan_manifest_sha256") != inputs.get("exact_plan_manifest_sha256") or bridge.get("snapshot_inventory_sha256") != _snapshot_inventory_sha256(exact):
            return False
        if bridge.get("successor_runtime_file_sha256") != source.get("runtime_file_sha256") or bridge.get("execution_runtime_files_unchanged") is not True:
            return False
        for name in ("compatibility_q00_result", "compatibility_q00_audit", "compatibility_fleet_preflight", "compatibility_fleet_authority", "compatibility_fleet_result", "compatibility_runtime_provenance"):
            path = bound(name)
            if _sha(path) != inputs.get(f"{name}_sha256"):
                return False
        q00 = _load(bound("compatibility_q00_result"))
        fleet = _load(bound("compatibility_fleet_result"))
        return q00.get("status") == "PASS_ZERO_TREATMENT_COMPATIBILITY" and fleet.get("status") == "PASS_960_ZERO_TREATMENT_COMPATIBILITY" and fleet.get("probe_count") == 960 and fleet.get("runtime_diff_count") == 0 and fleet.get("protected_counters") == COUNTERS
    except (KeyError, OSError, TypeError, ValueError):
        return False


def audit(protocol_path: Path, *, source_commit: str, source_tree: str) -> dict[str, Any]:
    protocol = _load(protocol_path)
    is_v2 = protocol.get("schema") == "STAGE_V_M4_MATCHED_ACTION_PROTOCOL_V2"
    runner_path = REPO_ROOT / "scripts/detector_v5/run_stage_v_m4_matched_parent.py"
    formal_gate_path = REPO_ROOT / "scripts/detector_v5/run_stage_v_m4_formal_parent_with_resource_gate.py"
    scheduler_path = REPO_ROOT / "scripts/detector_v5/run_stage_v_m4_formal_scheduler.py"
    gate_a_path = REPO_ROOT / "scripts/detector_v5/run_stage_v_m3_5_v1_4_gate_a.py"
    gate_b_path = REPO_ROOT / "scripts/detector_v5/run_stage_v_m3_5_v1_4_gate_b.py"
    auditor_path = REPO_ROOT / "scripts/detector_v5/audit_stage_v_m4_matched_parent.py"
    governance_path = REPO_ROOT / "scripts/detector_v5/stage_v_m4_governance.py"
    authorization_issuer_path = REPO_ROOT / ("scripts/detector_v5/issue_stage_v_m4_runtime_authorization_v2.py" if is_v2 else "scripts/detector_v5/issue_stage_v_m4_runtime_authorization.py")
    snapshot_path = REPO_ROOT / "src/gripper_attack/stage_v_causal_observation_snapshot.py"
    runner_tree = ast.parse(runner_path.read_text(encoding="utf-8"), filename=str(runner_path))
    gate_a_tree = ast.parse(gate_a_path.read_text(encoding="utf-8"), filename=str(gate_a_path))
    gate_b_tree = ast.parse(gate_b_path.read_text(encoding="utf-8"), filename=str(gate_b_path))
    runner_text = runner_path.read_text(encoding="utf-8")
    formal_gate_text = formal_gate_path.read_text(encoding="utf-8")
    scheduler_text = scheduler_path.read_text(encoding="utf-8")
    gate_b_text = gate_b_path.read_text(encoding="utf-8")
    auditor_text = auditor_path.read_text(encoding="utf-8")
    governance_text = governance_path.read_text(encoding="utf-8")
    authorization_issuer_text = authorization_issuer_path.read_text(encoding="utf-8")
    snapshot_text = snapshot_path.read_text(encoding="utf-8")
    replay = _function(gate_a_tree, "_replay_canary")
    branch = _function(gate_b_tree, "_run_branch")
    replay_calls = _called_names(replay)
    branch_calls = _called_names(branch)
    matrix = protocol.get("matrix", {})
    operation = protocol.get("operation", {})
    source = protocol.get("source_binding", {})
    checks = {
        "protocol_schema": protocol.get("schema") == ("STAGE_V_M4_MATCHED_ACTION_PROTOCOL_V2" if is_v2 else "STAGE_V_M4_MATCHED_ACTION_PROTOCOL_V1"),
        "protocol_frozen_authorized": (protocol.get("status") == "FROZEN_PROSPECTIVE_NOT_AUTHORIZED" and protocol.get("runtime_authorized") is False) if is_v2 else (protocol.get("status") == "FROZEN_RUNTIME_AUTHORIZED" and protocol.get("runtime_authorized") is True),
        "formal_corridor_gate_bound": protocol_declares_corridor_gate(protocol) if not is_v2 else _current_authority_binding_complete(protocol),
        "current_authority_binding_complete": _current_authority_binding_complete(protocol) if is_v2 else True,
        "successor_snapshot_rebind": _successor_snapshot_rebind_complete(protocol, source_commit=source_commit, source_tree=source_tree),
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
        "runner_emits_exact_counts": all(token in runner_text for token in ('"expected_physical_executions": 96', '"expected_treatment_labels": 72', 'probe_count": len(probes)')),
        "runner_loads_exact_frozen_plan": all(token in runner_text for token in ('_load_exact_plan_authority', '--exact-plan-root', 'EXACT_FROZEN_PLAN_MANIFEST')),
        "runner_never_recomputes_probe_selection": 'select_probe_steps' not in runner_text and 'write_snapshot' not in runner_text and '"probe_selection_recomputed": False' in runner_text,
        "runner_emits_protected_counters": '"protected_counters": dict(COUNTERS)' in runner_text,
        "global_atomic_parent_scheduler": scheduler_path.is_file() and all(token in scheduler_text for token in ("_scheduler_lock", "_pending", "_eligible_gpus", "_write_progress", "rolling_replenishment")) and protocol.get("resource_contract", {}).get("atomic_global_queue") is True,
        "scheduler_claims_frozen_40_only": all(token in scheduler_text for token in ("_frozen_queue", '"parent_keys"', "parent_count")),
        "scheduler_dynamic_admission": all(token in scheduler_text for token in ("query_inventory", "admit_mode_b_or_c", "MIN_FREE_MEMORY_MIB", "_reservation_gpu_ids")),
        "scheduler_rolling_replenishment": all(token in scheduler_text for token in ("active", "process.poll", "assigned.discard", "subprocess.Popen")),
        "claim_identity_binding": all(token in formal_gate_text for token in ("physical_gpu_index", "gpu_uuid", "cuda_visible_devices", "worker_pid", "authority_sha256", "protocol_sha256", "runtime_provenance_sha256", "attempt_ordinal", "claim_timestamp")),
        "claim_atomic_exclusion": "path.open(\"x\"" in formal_gate_text,
        "scheduler_source_binding": source.get("runtime_file_sha256", {}).get("scripts/detector_v5/run_stage_v_m4_formal_scheduler.py") == _sha(scheduler_path),
        "independent_auditor_is_producer_free": "run_stage_v_m4_matched_parent" not in auditor_text,
        "runner_requires_formal_corridor_gate": "validate_formal_m4_corridor_gate" in runner_text if not is_v2 else "validate_formal_m4_v2_authority" in runner_text and "M4_PROTOCOL_V1_SUPERSEDED_CURRENT_MAINLINE" in governance_text and "validate_formal_m4_corridor_gate" not in runner_text,
        "authorization_issuer_requires_formal_corridor_gate": "validate_formal_m4_corridor_gate" in authorization_issuer_text if not is_v2 else "validate_formal_m4_v2_authority" in authorization_issuer_text and "STAGE_V_M4_RUNTIME_AUTHORIZATION_V2" in authorization_issuer_text,
        "snapshot_module_has_exact_binding": all(token in snapshot_text for token in ("CAUSAL_PROBE_SNAPSHOT_V2", "assert_primary_observation_exact", "write_snapshot", "load_snapshot")),
        "protected_counters_zero": protocol.get("protected_counters") == COUNTERS,
    }
    status = "PASS_STATIC_DESIGN_ONLY" if all(checks.values()) else "FAIL_STATIC_CONTRACT"
    return {
        "schema": "STAGE_V_M4_STATIC_AUDIT_V2" if is_v2 else "STAGE_V_M4_STATIC_AUDIT_V1",
        "status": status,
        "runtime_authorized": False,
        "runtime_executed": False,
        "protocol": str(protocol_path.resolve()),
        "protocol_sha256": _sha(protocol_path),
        "source_commit": source_commit,
        "source_tree": source_tree,
        "runner_sha256": _sha(runner_path),
        "formal_gate_sha256": _sha(formal_gate_path),
        "scheduler_sha256": _sha(scheduler_path),
        "gate_a_sha256": _sha(gate_a_path),
        "gate_b_sha256": _sha(gate_b_path),
        "independent_auditor_sha256": _sha(auditor_path),
        "authorization_issuer_sha256": _sha(authorization_issuer_path),
        "governance_sha256": _sha(governance_path),
        "checks": checks,
        "primary_replay_calls": sorted(replay_calls),
        "primary_branch_calls": sorted(branch_calls),
        "protected_counters": dict(COUNTERS),
        "next_action": "ISSUE_FORMAL_M4_V2_AUTHORIZATION" if is_v2 and status == "PASS_STATIC_DESIGN_ONLY" else "AUTHORIZE_AND_LAUNCH_M4" if status == "PASS_STATIC_DESIGN_ONLY" else "HOLD_AND_REPAIR",
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
