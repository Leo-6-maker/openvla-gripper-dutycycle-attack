#!/usr/bin/env python3
"""Freeze the AC2R2 engineering repair and same-manifest clean recovery."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import build_stage_ac2_static as base


ROOT = base.ROOT
OLD_PROTOCOL = ROOT / "configs/STAGE_AC_AC2_CLEAN_SCREEN_PROTOCOL_V1.json"
OLD_SOURCE = ROOT / "reports/STAGE_AC_AC2_RUNTIME_SOURCE_AUTHORITY_V1.json"
LAUNCH_MANIFEST = ROOT / "reports/STAGE_AC_AC2_CLEAN_SCREEN_LAUNCH_MANIFEST_V1.json"
AC2R1_ROOT = ROOT / "reports/STAGE_AC_AC2R1_M1_CANARY_ROOT_SEAL_V1.json"
PRIOR_FAILURES = ROOT / "reports/STAGE_AC_AC2R2_PRIOR_FAILURE_RECEIPTS_V1.json"
PROTOCOL = ROOT / "configs/STAGE_AC_AC2R2_CLEAN_SCREEN_REPAIR_PROTOCOL_V1.json"
SOURCE = ROOT / "reports/STAGE_AC_AC2R2_RUNTIME_SOURCE_AUTHORITY_V1.json"
ROOT_SEAL = ROOT / "reports/STAGE_AC_AC2R2_PRE_RESUME_ROOT_SEAL_V1.json"
ROOT_SIDECAR = ROOT / "reports/STAGE_AC_AC2R2_PRE_RESUME_ROOT_SEAL_V1.sha256"
PI_COMMENT_ID = 5435208429
GATE = "STAGE_AC_AC2R2_ENGINEERING_RUNTIME_REPAIR_AND_SAME_MANIFEST_CLEAN_RECOVERY"
DENOMINATOR_SALT = "STAGE_AC_AC2_MODEL_SPECIFIC_DENOMINATOR_V1_20260827"


def canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def write_json(path: Path, value: dict[str, Any]) -> dict[str, Any]:
    data = (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return {"path": path.relative_to(ROOT).as_posix(), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def artifact(path: Path) -> dict[str, Any]:
    return {"path": path.relative_to(ROOT).as_posix(), "bytes": path.stat().st_size, "sha256": base.sha256_file(path)}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def build(*, write: bool) -> dict[str, Any]:
    old_protocol = json.loads(OLD_PROTOCOL.read_text(encoding="utf-8"))
    old_source = json.loads(OLD_SOURCE.read_text(encoding="utf-8"))
    manifest = json.loads(LAUNCH_MANIFEST.read_text(encoding="utf-8"))
    canary_root = json.loads(AC2R1_ROOT.read_text(encoding="utf-8"))
    prior = json.loads(PRIOR_FAILURES.read_text(encoding="utf-8"))
    require(old_protocol.get("status") == "STAGE_AC_AC2_CLEAN_SCREEN_PROTOCOL_AUTHORIZED_PRE_EXPOSURE", "AC2R2_OLD_PROTOCOL_INVALID")
    require(old_source.get("status") == "STAGE_AC_AC2_RUNTIME_SOURCE_AUTHORITY_FROZEN", "AC2R2_OLD_SOURCE_INVALID")
    require(manifest.get("cell_count") == 720 and len(manifest.get("cells", [])) == 720, "AC2R2_MANIFEST_INVALID")
    require(canary_root.get("status") == "STAGE_AC_AC2R1_M1_CANARY_QUALIFICATION_PASS", "AC2R2_CANARY_PREREQUISITE_INVALID")
    require(prior.get("status") == "STAGE_AC_AC2_V1_ENGINEERING_FAILURES_PRESERVED", "AC2R2_PRIOR_FAILURES_INVALID")
    require(prior.get("receipt_count") == 8, "AC2R2_PRIOR_FAILURE_COUNT_INVALID")

    code_commit = base.git_value("rev-parse", "HEAD")
    code_tree = base.git_value("rev-parse", "HEAD^{tree}")
    old_protocol_binding = base.canonical_binding(OLD_PROTOCOL.relative_to(ROOT).as_posix())
    old_source_binding = base.canonical_binding(OLD_SOURCE.relative_to(ROOT).as_posix())
    launch_binding = base.canonical_binding(LAUNCH_MANIFEST.relative_to(ROOT).as_posix())
    canary_root_binding = base.canonical_binding(AC2R1_ROOT.relative_to(ROOT).as_posix())
    prior_binding = artifact(PRIOR_FAILURES)
    runtime_files = [base.canonical_binding(str(row["path"])) for row in old_source["runtime_files"]]

    protocol = copy.deepcopy(old_protocol)
    protocol.update({
        "schema": "STAGE_AC_AC2R2_CLEAN_SCREEN_REPAIR_PROTOCOL_V1",
        "status": "STAGE_AC_AC2R2_CLEAN_SCREEN_REPAIR_PROTOCOL_AUTHORIZED_PRE_RESUME",
        "gate": GATE,
        "authorization_pi_comment_id": PI_COMMENT_ID,
        "claim_boundary": "Versioned AC2 clean-only engineering recovery after preserved V1 runtime failures; same 720-cell manifest, no treatment or scientific outcome claim.",
        "repair_amendment": {
            "prior_protocol": old_protocol_binding,
            "prior_source_authority": old_source_binding,
            "prior_failure_receipts": prior_binding,
            "failure_count": prior["receipt_count"],
            "same_cell_recovery_only": True,
            "replacement_or_top_up": False,
            "scientific_definition_changed": False,
            "repairs": [
                "restore the two frozen deterministic candidate-selection salts in the protocol payload",
                "apply the already-official PI05 elementwise clip at the AC2 final-action boundary before validation and delivery",
            ],
        },
        "next_legal_action": "EXECUTE_AC2R2_SAME_720_CELL_MANIFEST_CLEAN_ONLY",
    })
    protocol["eligibility"] = dict(protocol["eligibility"])
    protocol["eligibility"].update({
        "critical_selection_salt": f"{DENOMINATOR_SALT}|CRITICAL",
        "noncritical_selection_salt": f"{DENOMINATOR_SALT}|NONCRITICAL",
    })
    protocol["model_runtime"] = dict(protocol["model_runtime"])
    protocol["model_runtime"]["m2_final_action_adapter"] = "elementwise clip[-1,+1] before AC2 semantics validation and LIBERO delivery; raw action remains preserved"
    protocol["source_bindings"] = {
        "prior_ac2_protocol": old_protocol_binding,
        "prior_ac2_source_authority": old_source_binding,
        "launch_manifest": launch_binding,
        "ac2r1_canary_root": canary_root_binding,
        "prior_failure_receipts": prior_binding,
    }
    protocol_binding = write_json(PROTOCOL, protocol) if write else artifact(PROTOCOL)

    source = copy.deepcopy(old_source)
    source.update({
        "schema": "STAGE_AC_AC2R2_RUNTIME_SOURCE_AUTHORITY_V1",
        "status": "STAGE_AC_AC2R2_RUNTIME_SOURCE_AUTHORITY_FROZEN",
        "gate": GATE,
        "authorization_pi_comment_id": PI_COMMENT_ID,
        "claim_boundary": protocol["claim_boundary"],
        "git_binding": {
            "repository": "Leo-6-maker/openvla-gripper-dutycycle-attack",
            "branch": base.git_value("branch", "--show-current"),
            "commit": code_commit,
            "tree": code_tree,
        },
        "runtime_files": runtime_files,
        "input_authorities": {
            "protocol": protocol_binding,
            "launch_manifest": launch_binding,
            "ac2r1_canary_root": canary_root_binding,
            "prior_failure_receipts": prior_binding,
            "z1_protocol": base.canonical_binding("configs/STAGE_Z_Z1_RUNTIME_PROTOCOL_V11.json"),
        },
        "repair_amendment": protocol["repair_amendment"],
        "next_legal_action": "EXECUTE_ONLY_THE_SAME_FROZEN_720_CELL_MANIFEST",
    })
    source_binding = write_json(SOURCE, source) if write else artifact(SOURCE)

    root_payload = {
        "schema": "STAGE_AC_AC2R2_PRE_RESUME_ROOT_SEAL_V1",
        "status": "STAGE_AC_AC2R2_PRE_RESUME_STATIC_REPAIR_PASS",
        "gate": GATE,
        "authorization_pi_comment_id": PI_COMMENT_ID,
        "git_binding": source["git_binding"],
        "same_frozen_manifest": artifact(LAUNCH_MANIFEST),
        "cell_count": 720,
        "prior_ac2_v1_failures": prior_binding,
        "ac2r1_canary_root": canary_root_binding,
        "source_bindings": {
            "protocol": protocol_binding,
            "source_authority": source_binding,
            "launch_manifest": launch_binding,
            "ac2r1_canary_root": canary_root_binding,
            "prior_failure_receipts": prior_binding,
        },
        "scientific_firewall": {
            "new_model_inference_calls": 0,
            "new_env_step_calls": 0,
            "open_intervention_steps": 0,
            "pgd_calls": 0,
            "physical_endpoint_reads": 0,
            "v_phys_reads": 0,
            "protected_reads": 0,
            "new_identity": 0,
            "replacement_or_top_up": False,
        },
        "recovery_contract": {
            "prior_receipts_preserved": True,
            "same_manifest_only": True,
            "same_seed_and_parent_bindings": True,
            "no_scientific_redenomination_before_720_cells": True,
        },
        "next_legal_action": "GPU_AC2R2_CLEAN_ONLY_AFTER_THIS_SEAL_IS_COMMITTED_AND_VERIFIED",
    }
    root = {**root_payload, "root_payload_sha256": hashlib.sha256(canonical(root_payload)).hexdigest()}
    root_binding = write_json(ROOT_SEAL, root) if write else artifact(ROOT_SEAL)
    if write:
        ROOT_SIDECAR.write_bytes(f"{root_binding['sha256']}  {ROOT_SEAL.name}\n".encode("ascii"))
    return {"status": root["status"], "cell_count": 720, "root_payload_sha256": root["root_payload_sha256"], "outputs": {"protocol": protocol_binding, "source": source_binding, "root": root_binding, "prior": prior_binding}}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    print(json.dumps(build(write=args.write), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
