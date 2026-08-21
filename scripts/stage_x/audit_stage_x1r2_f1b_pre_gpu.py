#!/usr/bin/env python3
"""Seal the CPU/static F1-B method contract before any model call."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/STAGE_X_X1R2_F1B_DEV_PROTOCOL_V3.json"
F1A3_ROOT = ROOT / "reports/STAGE_X_X1R2_F1A3_SOURCE_SPLIT_AND_POPULATION_FREEZE_V3_20260821/F1A3_ROOT_SEAL_V3.json"
OUT = ROOT / "reports/STAGE_X_X1R2_F1B_DEV_METHOD_FREEZE_V3_20260821"
METHOD_SPEC = OUT / "F1B_METHOD_SPEC_V3.json"
AUDIT = OUT / "F1B_PRE_GPU_AUDIT_V3.json"
ROOT_SEAL = OUT / "F1B_ROOT_SEAL_V3.json"
ROOT_SIDECAR = OUT / "F1B_ROOT_SEAL_V3.sha256"

SOURCE_PATHS = (
    "configs/STAGE_X_X1R2_F1B_DEV_PROTOCOL_V3.json",
    "configs/STAGE_X_X1R2_GRIPPER_SELECTIVE_ATTACK_CONTRACT_V1.json",
    "scripts/stage_x/run_stage_x1r2_f1b_dev.py",
    "scripts/stage_x/run_stage_x1r_primary_matrix.py",
    "src/gripper_attack/attack_adapter.py",
    "src/gripper_attack/execution_target.py",
    "src/gripper_attack/route_contract.py",
    "src/gripper_attack/failure_evidence.py",
    "reports/STAGE_X_X1R2_F1A3_SOURCE_SPLIT_AND_POPULATION_FREEZE_V3_20260821/F1A3_DEV_V3_LEDGER_V3.json",
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True, stderr=subprocess.STDOUT).strip()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    protocol = json.loads(CONFIG.read_text(encoding="utf-8"))
    f1a3 = json.loads(F1A3_ROOT.read_text(encoding="utf-8"))
    errors: list[str] = []
    required_objectives = {
        "M0": "autoregressive_prefix_gripper_target_token_logratio_arm_v3",
        "M1": "autoregressive_prefix_gripper_native_open_logratio_v4",
        "M2": "autoregressive_prefix_gripper_native_open_logratio_arm_v5",
    }
    if protocol.get("status") != "FROZEN_F1B_DEV_V3":
        errors.append("PROTOCOL_STATUS_INVALID")
    if protocol.get("scientific_authority") is not False:
        errors.append("SCIENTIFIC_AUTHORITY_NOT_FALSE")
    if set(protocol.get("methods", {})) != set(required_objectives):
        errors.append("METHOD_SET_INVALID")
    for method, objective in required_objectives.items():
        if protocol.get("methods", {}).get(method, {}).get("objective") != objective:
            errors.append(f"{method}_OBJECTIVE_INVALID")
    frozen = protocol.get("frozen_attack", {})
    exact = {
        "epsilon_processor_pixel_values": 0.03,
        "random_start": False,
        "candidate_policy": "STRICT_CANDIDATE_AUDIT_V1",
        "target_token_id_secondary": 31745,
        "target_execution_class": "NATIVE_OPEN",
        "strict_route": True,
        "allow_fallback": False,
        "exact_arm_dimensions": [0, 1, 2, 3, 4, 5],
        "direct_action_token_count": 7,
    }
    for key, expected in exact.items():
        if frozen.get(key) != expected:
            errors.append(f"FROZEN_ATTACK_{key}_INVALID")
    if frozen.get("step_size_by_iterations") != {"5": 0.006, "10": 0.003}:
        errors.append("STEP_SCHEDULE_INVALID")
    if f1a3.get("status") != "PASS_F1A3_SOURCE_SPLIT_AND_POPULATION_FREEZE_V3":
        errors.append("F1A3_NOT_PASS")
    if f1a3.get("role_counts") != {suite: {"BRIDGE_V3": 5, "C_CANARY_V3": 2, "DEV_V3": 6} for suite in ("libero_10", "libero_goal", "libero_object", "libero_spatial")}:
        errors.append("F1A3_ROLE_COUNTS_INVALID")
    if f1a3.get("selected_hard_or_unresolved_count") != 0:
        errors.append("F1A3_SELECTED_HARD_OR_UNRESOLVED")
    for relative in SOURCE_PATHS:
        if not (ROOT / relative).is_file():
            errors.append(f"SOURCE_MISSING:{relative}")
    adapter_text = (ROOT / "src/gripper_attack/attack_adapter.py").read_text(encoding="utf-8")
    target_text = (ROOT / "src/gripper_attack/execution_target.py").read_text(encoding="utf-8")
    route_text = (ROOT / "src/gripper_attack/route_contract.py").read_text(encoding="utf-8")
    for needle, text in (
        ("_generated_prefix_native_open_logratio_loss_and_stats_cached", adapter_text),
        ("native_open_logratio_loss_and_stats", target_text),
        ("autoregressive_prefix_gripper_native_open_logratio_v4", adapter_text),
        ("autoregressive_prefix_gripper_native_open_logratio_arm_v5", adapter_text),
        ("autoregressive_prefix_gripper_native_open_logratio_v4", route_text),
        ("autoregressive_prefix_gripper_native_open_logratio_arm_v5", route_text),
    ):
        if needle not in text:
            errors.append(f"IMPLEMENTATION_BINDING_MISSING:{needle}")
    paper_changed = git("diff", "--name-only", "HEAD", "--", "paper").splitlines()
    if paper_changed:
        errors.append("PAPER_V1_WORKTREE_CHANGED")
    try:
        subprocess.check_call([sys.executable, "-m", "py_compile", *[str(ROOT / path) for path in SOURCE_PATHS if path.endswith(".py")]], cwd=ROOT)
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
        "raw_sha256": {path: sha(ROOT / path) for path in SOURCE_PATHS if (ROOT / path).is_file()},
        "git_blob_sha1": {path: git("rev-parse", f"HEAD:{path}") for path in SOURCE_PATHS if (ROOT / path).is_file()},
    }
    method_spec = {
        "schema": "STAGE_X1R2_F1B_METHOD_SPEC_V3",
        "status": "PASS_F1B_METHOD_SPEC_SEALED" if not errors else "HOLD_F1B_METHOD_SPEC",
        "protocol_sha256": sha(CONFIG),
        "f1a3_root_seal_sha256": sha(F1A3_ROOT),
        "source": source,
        "methods": protocol["methods"],
        "frozen_attack": protocol["frozen_attack"],
        "probe": protocol["probe"],
        "selection": protocol["selection"],
        "protected_boundary": protocol["protected_boundary"],
    }
    write_json(METHOD_SPEC, method_spec)
    audit = {
        "schema": "STAGE_X1R2_F1B_PRE_GPU_AUDIT_V3",
        "status": "PASS_F1B_PRE_GPU_STATIC_CONTRACT" if not errors else "HOLD_F1B_PRE_GPU_STATIC_CONTRACT",
        "errors": errors,
        "source": source,
        "method_spec_sha256": sha(METHOD_SPEC),
        "f1a3_root_seal_sha256": sha(F1A3_ROOT),
        "paper_v1_worktree_changed": bool(paper_changed),
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
        "configs/STAGE_X_X1R2_F1B_DEV_PROTOCOL_V3.json": sha(CONFIG),
        "reports/STAGE_X_X1R2_F1A3_SOURCE_SPLIT_AND_POPULATION_FREEZE_V3_20260821/F1A3_ROOT_SEAL_V3.json": sha(F1A3_ROOT),
        "reports/STAGE_X_X1R2_F1B_DEV_METHOD_FREEZE_V3_20260821/F1B_METHOD_SPEC_V3.json": sha(METHOD_SPEC),
        "reports/STAGE_X_X1R2_F1B_DEV_METHOD_FREEZE_V3_20260821/F1B_PRE_GPU_AUDIT_V3.json": sha(AUDIT),
        **{path: value for path, value in source["raw_sha256"].items()},
    }
    seal = {
        "schema": "STAGE_X1R2_F1B_ROOT_SEAL_V3",
        "status": audit["status"],
        "artifact_hashes": dict(sorted(artifacts.items())),
        "source_commit": source["commit"],
        "source_tree": source["tree"],
        "protocol_sha256": sha(CONFIG),
        "method_spec_sha256": sha(METHOD_SPEC),
        "pre_gpu_audit_sha256": sha(AUDIT),
        "f1a3_root_seal_sha256": sha(F1A3_ROOT),
        "protected_boundary": protocol["protected_boundary"],
        "seal_scope_excludes_sidecar": True,
    }
    write_json(ROOT_SEAL, seal)
    seal_sha = sha(ROOT_SEAL)
    ROOT_SIDECAR.write_text(f"{seal_sha}  {ROOT_SEAL.name}\n", encoding="utf-8")
    print(json.dumps({"status": audit["status"], "errors": errors, "output": str(OUT), "root_seal_sha256": seal_sha}, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
