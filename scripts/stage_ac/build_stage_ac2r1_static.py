#!/usr/bin/env python3
"""Freeze AC2R1's M1 manifest repair and three-canary pre-GPU gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
PI_COMMENT_ID = 5435208429
M1 = "M1_OPENVLA_OFT"
CANARY_PLAN = "reports/STAGE_AA_AA2R2_ENGINEERING_CANARY_PLAN_V1.json"
Z1_PROTOCOL = "configs/STAGE_Z_Z1_RUNTIME_PROTOCOL_V11.json"
M1_MANIFEST = "reports/STAGE_Z_Z0R2_M1_OFT_CHECKPOINT_MANIFESTS_V2.json"
RECONCILIATION = "reports/STAGE_AC_AC2R1_M1_MANIFEST_BYTE_AUTHORITY_RECONCILIATION_V1.json"
PROTOCOL = "configs/STAGE_AC_AC2R1_M1_MANIFEST_REQUALIFICATION_PROTOCOL_V1.json"
SOURCE = "reports/STAGE_AC_AC2R1_M1_RUNTIME_SOURCE_AUTHORITY_V1.json"
ROOT_SEAL = "reports/STAGE_AC_AC2R1_PRE_GPU_ROOT_SEAL_V1.json"
ROOT_SIDEcar = "reports/STAGE_AC_AC2R1_PRE_GPU_ROOT_SEAL_V1.sha256"
RUNNER = "scripts/stage_ac/run_stage_ac2r1_m1_canary.py"
RECONCILIATION_SCRIPT = "scripts/stage_ac/reconcile_m1_manifest_authority.py"
HELPER = "src/stage_ac/m1_manifest_authority.py"
RUNTIME_FILES = (
    "scripts/stage_aa/run_stage_aa2r2_engineering_canary.py",
    "scripts/stage_aa/run_stage_aa1_engineering_canary.py",
    "scripts/stage_z/run_stage_z_z1_runtime_canary.py",
    "src/gripper_attack/stage_v_m3_5_physical_taxonomy.py",
    "src/stage_aa/action_semantics_v2.py",
)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def write_json(path: Path, value: Any) -> dict[str, Any]:
    data = (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return {"path": path.relative_to(ROOT).as_posix(), "bytes": len(data), "sha256": sha256_bytes(data)}


def binding(relative: str) -> dict[str, Any]:
    path = ROOT / relative
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": relative, "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def canonical_binding(relative: str) -> dict[str, Any]:
    path = ROOT / relative
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        data = subprocess.check_output(["git", "show", f"HEAD:{relative}"])
    except subprocess.CalledProcessError:
        data = path.read_bytes()
    return {"path": relative, "bytes": len(data), "sha256": sha256_bytes(data)}


def git_value(*args: str) -> str:
    return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True).strip()


def build(*, write: bool) -> dict[str, Any]:
    plan = json.loads((ROOT / CANARY_PLAN).read_text(encoding="utf-8"))
    z1 = json.loads((ROOT / Z1_PROTOCOL).read_text(encoding="utf-8"))
    reconciliation = json.loads((ROOT / RECONCILIATION).read_text(encoding="utf-8"))
    if plan.get("status") != "STAGE_AA_AA2R2_ENGINEERING_CANARY_PLAN_FROZEN" or plan.get("cell_count") != 9:
        raise RuntimeError("AC2R1_CANARY_PLAN_NOT_FROZEN")
    if reconciliation.get("status") != "STAGE_AC_AC2R1_M1_MANIFEST_BYTE_RECONCILIATION_PASS":
        raise RuntimeError("AC2R1_M1_RECONCILIATION_NOT_PASS")
    canaries = [row for row in plan["canaries"] if row.get("model_family") == M1]
    if len(canaries) != 3 or any(row.get("permanent_exclusion") is not True or row.get("scientific_use") is not False for row in canaries):
        raise RuntimeError("AC2R1_M1_CANARY_EXCLUSION_FIREWALL_INVALID")

    code_commit = git_value("rev-parse", "HEAD")
    code_tree = git_value("rev-parse", "HEAD^{tree}")
    recon_binding = binding(RECONCILIATION)
    m1_manifest_binding = canonical_binding(M1_MANIFEST)
    z1_binding = canonical_binding(Z1_PROTOCOL)
    plan_binding = canonical_binding(CANARY_PLAN)
    runtime_files = [canonical_binding(RUNNER), canonical_binding(HELPER), canonical_binding(RECONCILIATION_SCRIPT), *(canonical_binding(path) for path in RUNTIME_FILES)]

    protocol_value = {
        "schema": "STAGE_AC_AC2R1_M1_MANIFEST_REQUALIFICATION_PROTOCOL_V1",
        "status": "STAGE_AC_AC2R1_PRE_GPU_REQUALIFICATION_AUTHORIZED",
        "gate": "STAGE_Z_AC2R1_M1_MANIFEST_BYTE_AUTHORITY_RECONCILIATION_AND_PRE_GPU_REQUALIFICATION",
        "authorization_pi_comment_id": PI_COMMENT_ID,
        "claim_boundary": "M1 manifest byte-authority repair and clean-only engineering canary qualification; no fresh AC2 parent exposure or treatment result.",
        "manifest_authority": {
            "historical_z1_sha256": reconciliation["historical_z1_authority"]["sha256"],
            "git_lf_sha256": reconciliation["git_runtime_representation"]["sha256"],
            "reconciliation": recon_binding,
            "unchanged_verifier": True,
            "runtime_rule": "Derive exact historical CRLF bytes only for the unchanged Z1 per-file verifier.",
        },
        "canary_cells": [
            {
                "cell_id": row["cell_id"],
                "canonical_parent_key": row["canonical_parent_key"],
                "suite": row["suite"],
                "task_idx": row["task_idx"],
                "state_id": row["state_id"],
                "seed": row["seed"],
                "permanent_exclusion": True,
                "scientific_use": False,
            }
            for row in canaries
        ],
        "clean_only": True,
        "model_inference_allowed": True,
        "env_step_allowed": True,
        "open_intervention_allowed": False,
        "attack_or_pgd_allowed": False,
        "physical_endpoint_read_allowed": False,
        "v_phys_read_allowed": False,
        "task_success_read_allowed": False,
        "protected_or_eval160_allowed": False,
        "scientific_parent_exposure": 0,
        "replacement_or_top_up": False,
        "source_bindings": {
            "canary_plan": plan_binding,
            "z1_protocol": z1_binding,
            "m1_manifest": m1_manifest_binding,
            "m1_reconciliation": recon_binding,
        },
        "next_legal_action": "RUN_ONLY_M1_THREE_PERMANENTLY_EXCLUDED_CANARY_CLEAN_REQUALIFICATION",
    }
    protocol_path = ROOT / PROTOCOL
    if write:
        protocol_binding = write_json(protocol_path, protocol_value)
    else:
        protocol_binding = binding(PROTOCOL)

    source_value = {
        "schema": "STAGE_AC_AC2R1_M1_RUNTIME_SOURCE_AUTHORITY_V1",
        "status": "STAGE_AC_AC2R1_M1_RUNTIME_SOURCE_AUTHORITY_FROZEN",
        "gate": protocol_value["gate"],
        "authorization_pi_comment_id": PI_COMMENT_ID,
        "claim_boundary": protocol_value["claim_boundary"],
        "git_binding": {
            "repository": "Leo-6-maker/openvla-gripper-dutycycle-attack",
            "branch": git_value("branch", "--show-current"),
            "commit": code_commit,
            "tree": code_tree,
        },
        "runtime_environment": {
            "python": "/mnt/sdc/dty_user/openvla_attack/envs/openvla-official-a800/bin/python",
            "environment": "/mnt/sdc/dty_user/openvla_attack/envs/openvla-official-a800",
            "gpu_admission": "free_memory_mib > 20480 immediately before each worker launch",
            "max_project_workers_per_gpu": 1,
            "foreign_processes": "allowed and recorded; never touched",
        },
        "model_authority": z1["model_families"][M1],
        "runtime_files": runtime_files,
        "input_authorities": {
            "protocol": protocol_binding,
            "canary_plan": plan_binding,
            "z1_protocol": z1_binding,
            "m1_manifest": m1_manifest_binding,
            "m1_reconciliation": recon_binding,
        },
        "manifest_authority": protocol_value["manifest_authority"],
        "scientific_firewall": {
            "model_inference_calls": 0,
            "env_step_calls": 0,
            "open_intervention_steps": 0,
            "pgd_calls": 0,
            "physical_endpoint_reads": 0,
            "v_phys_reads": 0,
            "protected_reads": 0,
            "scientific_parent_exposure": 0,
        },
        "next_legal_action": "EXECUTE_ONLY_THE_THREE_FROZEN_M1_CANARY_CELLS",
    }
    source_path = ROOT / SOURCE
    if write:
        source_binding = write_json(source_path, source_value)
    else:
        source_binding = binding(SOURCE)

    root_payload = {
        "schema": "STAGE_AC_AC2R1_PRE_GPU_ROOT_SEAL_V1",
        "status": "STAGE_AC_AC2R1_PRE_GPU_STATIC_REQUALIFICATION_PASS",
        "gate": protocol_value["gate"],
        "authorization_pi_comment_id": PI_COMMENT_ID,
        "git_binding": source_value["git_binding"],
        "canary_count": len(canaries),
        "canary_cells": protocol_value["canary_cells"],
        "source_bindings": {
            "protocol": protocol_binding,
            "source_authority": source_binding,
            "canary_plan": plan_binding,
            "z1_protocol": z1_binding,
            "m1_manifest": m1_manifest_binding,
            "m1_reconciliation": recon_binding,
        },
        "scientific_firewall": {
            "model_inference": 0,
            "env_step": 0,
            "gpu_worker": 0,
            "open_intervention": 0,
            "pgd": 0,
            "physical_endpoint_read": 0,
            "v_phys": 0,
            "protected_or_eval160": 0,
            "scientific_parent_exposure": 0,
        },
        "next_legal_action": "GPU_AC2R1_M1_CANARIES_ONLY_AFTER_THIS_SEAL_IS_COMMITTED_AND_VERIFIED",
    }
    root = {**root_payload, "root_payload_sha256": sha256_bytes(canonical_json(root_payload))}
    root_path = ROOT / ROOT_SEAL
    if write:
        root_binding = write_json(root_path, root)
        (ROOT / ROOT_SIDEcar).write_bytes(f"{root_binding['sha256']}  {Path(ROOT_SEAL).name}\n".encode("ascii"))
    else:
        root_binding = binding(ROOT_SEAL)
    return {"status": root["status"], "canary_count": len(canaries), "root_payload_sha256": root["root_payload_sha256"], "outputs": {"protocol": protocol_binding, "source_authority": source_binding, "root": root_binding, "reconciliation": recon_binding}}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    print(json.dumps(build(write=args.write), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
