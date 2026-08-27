#!/usr/bin/env python3
"""Build the append-only Z3 source authority after the code revision is fixed."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "configs/STAGE_Z_Z3_CROSS_MODEL_COMMAND_OPEN_PHYSICAL_MATRIX_PROTOCOL_V1.json"
Z1 = ROOT / "configs/STAGE_Z_Z1_RUNTIME_PROTOCOL_V11.json"
Z1_ROOT = ROOT / "reports/STAGE_Z_Z1_TERMINAL_ROOT_SEAL_V1.json"
Z2_ROOT = ROOT / "reports/STAGE_Z_Z2_TERMINAL_ROOT_SEAL_V2.json"
Z2R1_ROOT = ROOT / "reports/STAGE_Z_Z2R1_M2_CLEAN_REPAIR_ROOT_SEAL_V1.json"
ELIGIBILITY = ROOT / "reports/STAGE_Z_Z3_ELIGIBILITY_RECONCILIATION_V1.json"
PANEL = ROOT / "reports/STAGE_Z_Z0R1_SHARED_36_IDENTITY_PANEL_V1.json"
SOURCE_PATHS = (
    "scripts/stage_z/run_stage_z_z1_runtime_canary.py",
    "scripts/stage_z/run_stage_z_z2_clean_reference.py",
    "scripts/stage_z/run_stage_z_z3_sentinel.py",
    "scripts/stage_z/run_stage_z_z3_worker.py",
    "scripts/stage_z/build_stage_z_z3_source_authority.py",
    "scripts/stage_z/audit_stage_z_z3_static.py",
    "src/stage_z_preparation/z3_contract.py",
    "src/stage_z_preparation/contract.py",
    "src/stage_z_preparation/adapters.py",
    "src/stage_z_preparation/matrix.py",
    "src/gripper_attack/stage_v_m3_5_physical_taxonomy.py",
    "scripts/detector_v5/audit_stage_v_m3_5_v1_4_gate_b.py",
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def artifact(path: Path) -> dict[str, Any]:
    return {"path": path.relative_to(ROOT).as_posix(), "bytes": path.stat().st_size, "sha256": sha(path)}


def git(*args: str) -> str:
    return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True).strip()


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-commit", default=None)
    parser.add_argument("--source-tree", default=None)
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    source_commit = args.source_commit or git("rev-parse", "HEAD")
    source_tree = args.source_tree or git("rev-parse", "HEAD^{tree}")
    for path in (Z1, Z1_ROOT, Z2_ROOT, Z2R1_ROOT, ELIGIBILITY, PANEL):
        if not path.is_file():
            raise RuntimeError(f"AUTHORITY_ARTIFACT_MISSING:{path}")
    source_files = {}
    for relative in SOURCE_PATHS:
        path = ROOT / relative
        if not path.is_file():
            raise RuntimeError(f"Z3_SOURCE_MISSING:{relative}")
        source_files[relative] = artifact(path)
    z1 = load(Z1)
    eligibility = load(ELIGIBILITY)
    if sha(Z2_ROOT) != "e37659a552bea7665fbfcc7a52e8fa8131e29aef6613a197a7087ab8d7cf4c6f":
        raise RuntimeError("Z2_ROOT_SHA256")
    if sha(Z2R1_ROOT) != "2e98aba1826f0492dc6080767a00502b0372434191d74149d81177f97241e9f9":
        raise RuntimeError("Z2R1_ROOT_SHA256")
    if eligibility.get("arithmetic", {}).get("fixed_matrix_branches") != 460:
        raise RuntimeError("ELIGIBILITY_NOT_460")
    protocol = {
        "schema": "STAGE_Z_Z3_CROSS_MODEL_COMMAND_OPEN_PHYSICAL_MATRIX_PROTOCOL_V1",
        "status": "STAGE_Z_Z3_SOURCE_AUTHORITY_FROZEN",
        "gate": "STAGE_Z_Z3_CROSS_MODEL_COMMAND_OPEN_PHYSICAL_MATRIX_V1",
        "authorization": {
            "source": "PI_INDEPENDENT_AUDIT_Z2R1_ACCEPTED_Z3_AUTHORIZATION_ATTACHMENT_8d20f947-f10b-4f94-b1f8-79a9a9f75532",
            "supersedes": "STAGE_Z_Z2R1_M2_PHASE_ACTION_SEMANTICS_RECONCILIATION_AND_CLEAN_ANCHOR_REPAIR_V1",
            "scientific_estimand_unchanged": True,
        },
        "source_revision": {"repository": "Leo-6-maker/openvla-gripper-dutycycle-attack", "branch": git("branch", "--show-current"), "commit": source_commit, "tree": source_tree},
        "historical_roots": {
            "z1": artifact(Z1_ROOT),
            "z2": artifact(Z2_ROOT),
            "z2r1": artifact(Z2R1_ROOT),
            "panel": artifact(PANEL),
            "eligibility": artifact(ELIGIBILITY),
        },
        "z1_runtime_authority": {
            "protocol_path": Z1.relative_to(ROOT).as_posix(),
            "protocol_sha256": sha(Z1),
            "model_families": z1["model_families"],
            "environment": z1["environment"],
            "runtime_dependency_binding": z1["runtime_dependency_binding"],
            "common_libero_authority": {
                "checkout": z1["environment"]["common_libero_checkout"],
                "commit": z1["environment"]["common_libero_commit"],
                "tree": z1["environment"]["common_libero_tree"],
            },
        },
        "population": {
            "shared_identity_panel": 36,
            "eligible_model_parent_pairs": 92,
            "critical_anchor_missing": 10,
            "structural_model_parent_abstentions": 6,
            "fixed_matrix_branches": 460,
            "all_three_both_anchor_intersection_secondary_only": 24,
        },
        "action_contract": {
            "final_action_dim": 7,
            "arm_indices": [0, 1, 2, 3, 4, 5],
            "gripper_index": 6,
            "native_libero_open": -1.0,
            "raw_native_open": {"M0_OPENVLA": 1.0, "M1_OPENVLA_OFT": 1.0, "M2_PI05_LIBERO": -1.0},
            "arm_delta_linf_max": 1e-7,
            "intervention": "replace_final_gripper_coordinate_only",
            "no_decode_reencode_fallback": True,
        },
        "model_boundaries": {
            "M0_OPENVLA": {"fresh_boundary": "FRESH_PER_STEP", "chunk_length": 1},
            "M1_OPENVLA_OFT": {"fresh_boundary": "FRESH_OFT_ACTION_QUEUE", "chunk_length": 8, "consume_steps": 8},
            "M2_PI05_LIBERO": {"fresh_boundary": "FRESH_PI05_REPLAN", "replan_steps": 5, "action_horizon": 10},
        },
        "five_arms": [
            {"name": "CLEAN_BRANCH_CRITICAL", "duration": 0, "anchor_class": "CRITICAL", "intervention": False},
            {"name": "COMMAND_OPEN_T3_CRITICAL", "duration": 3, "anchor_class": "CRITICAL", "intervention": True},
            {"name": "COMMAND_OPEN_T5_CRITICAL", "duration": 5, "anchor_class": "CRITICAL", "intervention": True},
            {"name": "COMMAND_OPEN_T10_CRITICAL", "duration": 10, "anchor_class": "CRITICAL", "intervention": True},
            {"name": "COMMAND_OPEN_T5_NONCRITICAL_CONTROL", "duration": 5, "anchor_class": "NONCRITICAL", "intervention": True},
        ],
        "physical_contract": {
            "h_phys": 10,
            "x0_endpoint_source": "scripts/detector_v5/audit_stage_v_m3_5_v1_4_gate_b.py",
            "drop_z_m": 0.02,
            "release_displacement_m": 0.01,
            "consecutive_contact_loss_steps": 2,
            "primary_unit": "model_parent",
            "official_task_success": "secondary_only",
        },
        "branch_contract": {
            "clean_action_source": "sealed_Z2_clean_anchor_action_rows",
            "scientific_branch_model_inference": False,
            "fresh_env_reinstantiate": True,
            "state_restore_exact_required": True,
            "rng": {"env_seed": 0, "dummy_wait_steps": 10, "manual_audit_salt": "STAGE_Z_Z3_MANUAL_AUDIT_V1_20260823"},
            "retry_consumed_branch": False,
            "atomic_receipt_before_each_step": True,
        },
        "manual_audit": {"selection": "deterministic hash, up to 2 eligible model-parent pairs per model x suite", "max_model_parent_pairs": 24, "max_videos": 120, "outcome_blind": True, "human_review_required": True},
        "storage": {"filesystem": "/mnt/sdc", "min_free_margin_bytes": 5 * 1024**3, "compact_receipts_telemetry_max_bytes": 1 * 1024**3, "manual_videos_max_bytes": 3 * 1024**3, "overhead_max_bytes": 512 * 1024**2, "no_all_branch_videos": True, "m1_sequential_by_suite": True},
        "resource_contract": {"free_memory_mib_strictly_greater_than": 20480, "one_project_worker_per_gpu": True, "foreign_processes": "record_only_never_touch"},
        "forbidden_scope": {"pgd": 0, "protected_reads": 0, "eval160_reads": 0, "f1": False, "bridge": False, "new_model": False, "new_identity": False, "replacement_or_top_up": False},
        "source_files": source_files,
        "next_legal_action": "Z3_STATIC_SOURCE_AUDIT",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": protocol["status"], "source_commit": source_commit, "source_tree": source_tree, "sha256": sha(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
