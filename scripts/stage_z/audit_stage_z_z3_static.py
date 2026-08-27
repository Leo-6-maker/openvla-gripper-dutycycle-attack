#!/usr/bin/env python3
"""Static Z3-A audit; no model, simulator, or server execution."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "configs/STAGE_Z_Z3_CROSS_MODEL_COMMAND_OPEN_PHYSICAL_MATRIX_PROTOCOL_V1.json"
OUT = ROOT / "reports/STAGE_Z_Z3_STATIC_SOURCE_AUDIT_V1.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def imported_names(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            names.append(node.module or "")
    return names


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=PROTOCOL)
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    protocol = load(args.protocol)
    require(protocol.get("status") == "STAGE_Z_Z3_SOURCE_AUTHORITY_FROZEN", "PROTOCOL_NOT_FROZEN")
    require(protocol["population"] == {"all_three_both_anchor_intersection_secondary_only": 24, "critical_anchor_missing": 10, "eligible_model_parent_pairs": 92, "fixed_matrix_branches": 460, "shared_identity_panel": 36, "structural_model_parent_abstentions": 6}, "POPULATION_BINDING")
    action = protocol["action_contract"]
    require(action["final_action_dim"] == 7 and action["arm_indices"] == [0, 1, 2, 3, 4, 5] and action["gripper_index"] == 6 and action["native_libero_open"] == -1.0, "ACTION_CONTRACT")
    require(action["raw_native_open"] == {"M0_OPENVLA": 1.0, "M1_OPENVLA_OFT": 1.0, "M2_PI05_LIBERO": -1.0}, "RAW_NATIVE_OPEN")
    require([row["name"] for row in protocol["five_arms"]] == ["CLEAN_BRANCH_CRITICAL", "COMMAND_OPEN_T3_CRITICAL", "COMMAND_OPEN_T5_CRITICAL", "COMMAND_OPEN_T10_CRITICAL", "COMMAND_OPEN_T5_NONCRITICAL_CONTROL"], "FIVE_ARMS")
    require([row["duration"] for row in protocol["five_arms"]] == [0, 3, 5, 10, 5], "DOSE_SCHEDULE")
    require(protocol["resource_contract"]["free_memory_mib_strictly_greater_than"] == 20480, "GPU_THRESHOLD")
    require(protocol["resource_contract"]["one_project_worker_per_gpu"] is True, "WORKER_LIMIT")
    checks: list[dict[str, Any]] = []
    for relative, spec in protocol["source_files"].items():
        path = ROOT / relative
        require(path.is_file(), f"SOURCE_MISSING:{relative}")
        actual = sha(path)
        require(actual == spec["sha256"], f"SOURCE_SHA:{relative}")
        checks.append({"path": relative, "sha256": actual, "bytes": path.stat().st_size})
    runtime = ["src/stage_z_preparation/z3_contract.py", "scripts/stage_z/run_stage_z_z3_sentinel.py", "scripts/stage_z/run_stage_z_z3_worker.py"]
    forbidden_import_fragments = ("pgd", "f1", "bridge", "eval160", "protected")
    for relative in runtime:
        names = imported_names(ROOT / relative)
        lowered = [name.lower() for name in names]
        require(not any(any(fragment in name for fragment in forbidden_import_fragments) for name in lowered), f"FORBIDDEN_IMPORT:{relative}")
    contract_text = (ROOT / "src/stage_z_preparation/z3_contract.py").read_text(encoding="utf-8")
    worker_text = (ROOT / "scripts/stage_z/run_stage_z_z3_worker.py").read_text(encoding="utf-8")
    sentinel_text = (ROOT / "scripts/stage_z/run_stage_z_z3_sentinel.py").read_text(encoding="utf-8")
    for token in ("command_open_action", "GRIPPER_INDEX", "NATIVE_OPEN_RAW", "MODEL_M2"):
        require(token in contract_text, f"CONTRACT_TOKEN:{token}")
    for token in ("load_openvla", "load_pi05", "predict_action"):
        require(token not in worker_text, f"WORKER_MODEL_EXECUTION:{token}")
    for token in ("telemetry_from_env", "_check_success", "get_task_success"):
        require(token not in sentinel_text, f"SENTINEL_OUTCOME_READ:{token}")
    for relative, expected in (("reports/STAGE_Z_Z2_TERMINAL_ROOT_SEAL_V2.json", "e37659a552bea7665fbfcc7a52e8fa8131e29aef6613a197a7087ab8d7cf4c6f"), ("reports/STAGE_Z_Z2R1_M2_CLEAN_REPAIR_ROOT_SEAL_V1.json", "2e98aba1826f0492dc6080767a00502b0372434191d74149d81177f97241e9f9")):
        require(sha(ROOT / relative) == expected, f"ROOT_HASH:{relative}")
    report = {"schema": "STAGE_Z_Z3_STATIC_SOURCE_AUDIT_V1", "status": "STAGE_Z_Z3_STATIC_SOURCE_AUDIT_PASS", "protocol_sha256": sha(args.protocol), "checks": checks, "forbidden_import_namespaces": "PASS", "claim_boundary": "Static/source/resource contract only; no model inference, simulator, OPEN intervention, physical endpoint, V_phys, or protected read.", "next_legal_action": "Z3_B_ENGINEERING_SENTINELS"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "sources": len(checks)}, sort_keys=True))


if __name__ == "__main__":
    main()
