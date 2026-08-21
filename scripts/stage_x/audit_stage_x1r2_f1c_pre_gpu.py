#!/usr/bin/env python3
"""Seal the CPU/static F1-C contract before any canary model call."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/STAGE_X_X1R2_F1C_METHOD_FREEZE_T5_CANARY_PROTOCOL_V3.json"
F1A3_ROOT = ROOT / "reports/STAGE_X_X1R2_F1A3_SOURCE_SPLIT_AND_POPULATION_FREEZE_V3_20260821/F1A3_ROOT_SEAL_V3.json"
CANARY = ROOT / "reports/STAGE_X_X1R2_F1A3_SOURCE_SPLIT_AND_POPULATION_FREEZE_V3_20260821/F1A3_C_CANARY_V3_LEDGER_V3.json"
F1B_DECISION = ROOT / "reports/STAGE_X_X1R2_F1B_DEV_RESULT_AGGREGATION_V3_20260821/F1B_DEV_DECISION_V3.json"
F1B_ROOT = ROOT / "reports/STAGE_X_X1R2_F1B_DEV_RESULT_AGGREGATION_V3_20260821/F1B_DEV_ROOT_SEAL_V3.json"
OUT = ROOT / "reports/STAGE_X_X1R2_F1C_METHOD_FREEZE_T5_CANARY_V3_20260821"
METHOD_SPEC = OUT / "F1C_METHOD_SPEC_V3.json"
AUDIT = OUT / "F1C_PRE_GPU_AUDIT_V3.json"
ROOT_SEAL = OUT / "F1C_ROOT_SEAL_V3.json"
ROOT_SIDECAR = OUT / "F1C_ROOT_SEAL_V3.sha256"

SUITES = ("libero_10", "libero_goal", "libero_object", "libero_spatial")
SOURCE_PATHS = (
    "configs/STAGE_X_X1R2_F1C_METHOD_FREEZE_T5_CANARY_PROTOCOL_V3.json",
    "configs/STAGE_X_X1R2_F1B_DEV_PROTOCOL_V3.json",
    "configs/STAGE_X_X1R_SUITE_MATCHED_VICTIM_CONTRACT_V1.json",
    "scripts/stage_x/run_stage_x1r2_f1c_t5_canary.py",
    "scripts/stage_x/run_stage_x1r2_f1b_dev.py",
    "scripts/stage_x/run_stage_x1r_primary_matrix.py",
    "src/gripper_attack/attack_adapter.py",
    "src/gripper_attack/execution_target.py",
    "src/gripper_attack/route_contract.py",
    "src/gripper_attack/failure_evidence.py",
    "reports/STAGE_X_X1R2_F1A3_SOURCE_SPLIT_AND_POPULATION_FREEZE_V3_20260821/F1A3_C_CANARY_V3_LEDGER_V3.json",
    "reports/STAGE_X_X1R2_F1A3_SOURCE_SPLIT_AND_POPULATION_FREEZE_V3_20260821/F1A3_ROOT_SEAL_V3.json",
    "reports/STAGE_X_X1R2_F1B_DEV_RESULT_AGGREGATION_V3_20260821/F1B_DEV_DECISION_V3.json",
    "reports/STAGE_X_X1R2_F1B_DEV_RESULT_AGGREGATION_V3_20260821/F1B_DEV_ROOT_SEAL_V3.json",
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True, stderr=subprocess.STDOUT).strip()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def main() -> int:
    protocol = json.loads(CONFIG.read_text(encoding="utf-8"))
    f1a3_root = json.loads(F1A3_ROOT.read_text(encoding="utf-8"))
    canary = json.loads(CANARY.read_text(encoding="utf-8"))
    f1b_decision = json.loads(F1B_DECISION.read_text(encoding="utf-8"))
    f1b_root = json.loads(F1B_ROOT.read_text(encoding="utf-8"))
    errors: list[str] = []
    if protocol.get("status") != "FROZEN_F1C_T5_CANARY_V3" or protocol.get("scientific_authority") is not False:
        errors.append("PROTOCOL_STATUS_INVALID")
    method = protocol.get("method", {})
    expected_method = {
        "method": "M1",
        "objective": "autoregressive_prefix_gripper_native_open_logratio_v4",
        "iterations": 10,
        "epsilon_processor_pixel_values": 0.03,
        "step_size": 0.003,
        "random_start": False,
        "candidate_policy": "STRICT_CANDIDATE_AUDIT_V1",
        "target_token_id_secondary": 31745,
        "target_execution_class": "NATIVE_OPEN",
        "exact_arm_dimensions": [0, 1, 2, 3, 4, 5],
        "direct_action_token_count": 7,
        "strict_route": True,
        "allow_fallback": False,
        "no_decode_reencode": True,
        "no_actuator_overwrite": True,
    }
    for key, value in expected_method.items():
        if method.get(key) != value:
            errors.append(f"METHOD_{key.upper()}_INVALID")
    if tuple(protocol.get("temporal_arms", [])) != ("none", "prev_delta"):
        errors.append("TEMPORAL_ARM_SET_INVALID")
    if protocol.get("execution", {}).get("attempted_steps") != 5:
        errors.append("T5_ATTEMPT_COUNT_INVALID")
    for relative in SOURCE_PATHS:
        if not (ROOT / relative).is_file():
            errors.append(f"SOURCE_MISSING:{relative}")
    if f1a3_root.get("status") != "PASS_F1A3_SOURCE_SPLIT_AND_POPULATION_FREEZE_V3" or f1a3_root.get("selected_hard_or_unresolved_count") != 0:
        errors.append("F1A3_ROOT_INVALID")
    if canary.get("status") != "PASS_F1A3_SOURCE_SPLIT_AND_POPULATION_FREEZE_V3" or canary.get("role") != "C_CANARY_V3" or len(canary.get("rows", [])) != 8:
        errors.append("CANARY_LEDGER_INVALID")
    if any(row.get("role") != "C_CANARY_V3" or row.get("permanent_exclusion") is not True or row.get("outcome_read") is not False for row in canary.get("rows", [])):
        errors.append("CANARY_FIREWALL_INVALID")
    counts = {suite: sum(row.get("suite") == suite for row in canary.get("rows", [])) for suite in SUITES}
    if counts != {suite: 2 for suite in SUITES}:
        errors.append(f"CANARY_SUITE_COUNTS_INVALID:{counts}")
    if f1b_decision.get("status") != "F1B_NEW_METHOD_SELECTED_FOR_F1C":
        errors.append("F1B_DECISION_STATUS_INVALID")
    if f1b_decision.get("selected_method", {}).get("method") != "M1" or f1b_decision.get("selected_method", {}).get("iterations") != 10:
        errors.append("F1B_SELECTED_METHOD_INVALID")
    if f1b_root.get("status") != "PASS_F1B_DEV_RESULT_AGGREGATION":
        errors.append("F1B_RESULT_ROOT_INVALID")
    adapter_text = (ROOT / "src/gripper_attack/attack_adapter.py").read_text(encoding="utf-8")
    runner_text = (ROOT / "scripts/stage_x/run_stage_x1r2_f1c_t5_canary.py").read_text(encoding="utf-8")
    for needle, text in (
        ("temporal_init", adapter_text),
        ("_prev_delta", adapter_text),
        ("STRICT_CANDIDATE_AUDIT_V1", adapter_text),
        ("reset_temporal_state", runner_text),
        ("PASS_F1C_STRICT_CANDIDATE", runner_text),
        ("EXECUTE_CLEAN_ACTION", runner_text),
        ("attacked_env_steps", runner_text),
    ):
        if needle not in text:
            errors.append(f"IMPLEMENTATION_BINDING_MISSING:{needle}")
    if git("diff", "--name-only", "HEAD", "--", "paper").splitlines():
        errors.append("PAPER_V1_WORKTREE_CHANGED")
    try:
        py_files = [str(ROOT / path) for path in SOURCE_PATHS if path.endswith(".py")]
        subprocess.check_call([sys.executable, "-m", "py_compile", *py_files], cwd=ROOT)
        test = subprocess.run(["pytest", "-q", "tests/stage_x/test_stage_x1r2_gripper_selective_contract.py", "tests/stage_x/test_stage_x1r2_q3r3_e1_failure_persistence.py"], cwd=ROOT, text=True, capture_output=True, check=False)
        tests = {"returncode": int(test.returncode), "stdout_tail": test.stdout[-4000:], "stderr_tail": test.stderr[-2000:]}
        if test.returncode != 0:
            errors.append("CPU_REGRESSION_FAILED")
    except Exception as exc:
        tests = {"returncode": None, "error": f"{type(exc).__name__}:{exc}"}
        errors.append("CPU_REGRESSION_EXECUTION_FAILED")
    source = {
        "commit": git("rev-parse", "HEAD"),
        "tree": git("show", "-s", "--format=%T", "HEAD"),
        "branch": git("branch", "--show-current"),
        "status_porcelain": git("status", "--porcelain"),
        "runtime_python": sys.executable,
        "raw_sha256": {path: sha(ROOT / path) for path in SOURCE_PATHS if (ROOT / path).is_file()},
        "git_blob_sha1": {path: git("rev-parse", f"HEAD:{path}") for path in SOURCE_PATHS if (ROOT / path).is_file()},
    }
    method_spec = {
        "schema": "STAGE_X1R2_F1C_METHOD_SPEC_V3",
        "status": "PASS_F1C_METHOD_SPEC_SEALED" if not errors else "HOLD_F1C_METHOD_SPEC",
        "protocol_sha256": sha(CONFIG),
        "f1a3_root_seal_sha256": sha(F1A3_ROOT),
        "canary_ledger_sha256": sha(CANARY),
        "f1b_decision_sha256": sha(F1B_DECISION),
        "f1b_result_root_sha256": sha(F1B_ROOT),
        "source": source,
        "method": method,
        "temporal_arms": protocol["temporal_arms"],
        "temporal_selection": protocol["temporal_selection"],
        "probe": protocol["probe"],
        "execution": protocol["execution"],
        "protected_boundary": protocol["protected_boundary"],
    }
    write_json(METHOD_SPEC, method_spec)
    audit = {
        "schema": "STAGE_X1R2_F1C_PRE_GPU_AUDIT_V3",
        "status": "PASS_F1C_PRE_GPU_STATIC_CONTRACT" if not errors else "HOLD_F1C_PRE_GPU_STATIC_CONTRACT",
        "errors": errors,
        "source": source,
        "method_spec_sha256": sha(METHOD_SPEC),
        "f1a3_root_seal_sha256": sha(F1A3_ROOT),
        "canary_ledger_sha256": sha(CANARY),
        "f1b_decision_sha256": sha(F1B_DECISION),
        "f1b_result_root_sha256": sha(F1B_ROOT),
        "paper_v1_worktree_changed": bool(git("diff", "--name-only", "HEAD", "--", "paper").splitlines()),
        "cpu_regression": tests,
        "gpu_calls": 0,
        "model_inference_calls": 0,
        "simulator_calls": 0,
        "pgd_calls": 0,
        "attacked_env_steps": 0,
        "physical_interventions": 0,
        "vphys_reads": 0,
        "protected_reads": 0,
        "eval160": "UNREAD",
        "protected_evaluation": "UNREAD",
        "bridge_runtime_reads": 0,
        "bridge_outcome_reads": 0,
    }
    write_json(AUDIT, audit)
    artifacts = {
        "configs/STAGE_X_X1R2_F1C_METHOD_FREEZE_T5_CANARY_PROTOCOL_V3.json": sha(CONFIG),
        "reports/STAGE_X_X1R2_F1A3_SOURCE_SPLIT_AND_POPULATION_FREEZE_V3_20260821/F1A3_ROOT_SEAL_V3.json": sha(F1A3_ROOT),
        "reports/STAGE_X_X1R2_F1A3_SOURCE_SPLIT_AND_POPULATION_FREEZE_V3_20260821/F1A3_C_CANARY_V3_LEDGER_V3.json": sha(CANARY),
        "reports/STAGE_X_X1R2_F1B_DEV_RESULT_AGGREGATION_V3_20260821/F1B_DEV_DECISION_V3.json": sha(F1B_DECISION),
        "reports/STAGE_X_X1R2_F1B_DEV_RESULT_AGGREGATION_V3_20260821/F1B_DEV_ROOT_SEAL_V3.json": sha(F1B_ROOT),
        "reports/STAGE_X_X1R2_F1C_METHOD_FREEZE_T5_CANARY_V3_20260821/F1C_METHOD_SPEC_V3.json": sha(METHOD_SPEC),
        "reports/STAGE_X_X1R2_F1C_METHOD_FREEZE_T5_CANARY_V3_20260821/F1C_PRE_GPU_AUDIT_V3.json": sha(AUDIT),
        **source["raw_sha256"],
    }
    seal = {
        "schema": "STAGE_X1R2_F1C_ROOT_SEAL_V3",
        "status": audit["status"],
        "artifact_hashes": dict(sorted(artifacts.items())),
        "source_commit": source["commit"],
        "source_tree": source["tree"],
        "protocol_sha256": sha(CONFIG),
        "method_spec_sha256": sha(METHOD_SPEC),
        "pre_gpu_audit_sha256": sha(AUDIT),
        "f1a3_root_seal_sha256": sha(F1A3_ROOT),
        "canary_ledger_sha256": sha(CANARY),
        "f1b_decision_sha256": sha(F1B_DECISION),
        "f1b_result_root_sha256": sha(F1B_ROOT),
        "protected_boundary": protocol["protected_boundary"],
        "seal_scope_excludes_sidecar": True,
    }
    write_json(ROOT_SEAL, seal)
    root_sha = sha(ROOT_SEAL)
    ROOT_SIDECAR.write_text(f"{root_sha}  {ROOT_SEAL.name}\n", encoding="utf-8", newline="\n")
    print(json.dumps({"status": audit["status"], "errors": errors, "output": str(OUT), "root_seal_sha256": root_sha}, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
