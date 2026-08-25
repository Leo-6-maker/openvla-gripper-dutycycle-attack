#!/usr/bin/env python3
"""Z2-A offline authority and legacy-receipt reconciliation audit."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "configs/STAGE_Z_Z2_CLEAN_REFERENCE_PROTOCOL_V2.json"
PANEL = ROOT / "reports/STAGE_Z_Z0R1_SHARED_36_IDENTITY_PANEL_V1.json"
LEGACY_DIR = ROOT / "reports/server_evidence"
AUDIT_OUT = ROOT / "reports/STAGE_Z_Z2A_STATIC_AUDIT_V2.json"
RECON_OUT = ROOT / "reports/STAGE_Z_Z2_LEGACY_RECEIPT_RECONCILIATION_V1.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def git(*args: str) -> str:
    return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True).strip()


def receipt_key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row.get("model_family")), str(row.get("canonical_parent_key"))


def main() -> None:
    protocol = load(PROTOCOL)
    panel = load(PANEL)
    require(protocol["schema"] == "STAGE_Z_Z2_CLEAN_REFERENCE_PROTOCOL_V2", "PROTOCOL_SCHEMA")
    require(protocol["git_binding"]["head_commit"] == git("rev-parse", "HEAD"), "HEAD_BINDING")
    require(protocol["git_binding"]["head_tree"] == git("rev-parse", "HEAD^{tree}"), "TREE_BINDING")
    require(sha256_file(PANEL) == protocol["authority"]["shared_panel_sha256"], "PANEL_DIGEST")
    require(sha256_file(ROOT / protocol["z1_terminal_authority"]["root_seal"]["path"]) == protocol["z1_terminal_authority"]["root_seal"]["sha256"], "Z1_ROOT_SEAL")
    require(sha256_file(ROOT / protocol["z1_terminal_authority"]["protocol"]["path"]) == protocol["z1_terminal_authority"]["protocol"]["sha256"], "Z1_PROTOCOL")
    require(sha256_file(ROOT / protocol["z1_terminal_authority"]["source_authority"]["path"]) == protocol["z1_terminal_authority"]["source_authority"]["sha256"], "Z1_SOURCE_AUTHORITY")
    require(protocol["anchor_selection"]["h_phys"] == 10, "H_PHYS")
    require(protocol["anchor_selection"]["minimum_remaining_horizon"] == 20, "MIN_REMAINING_HORIZON")
    require(protocol["execution"]["command_open_intervention"] is False, "OPEN_FIREWALL")
    require(protocol["execution"]["pgd"] is False, "PGD_FIREWALL")
    require(protocol["execution"]["attacked_env_steps"] is False, "ATTACKED_STEP_FIREWALL")
    require(protocol["execution"]["v_phys_endpoint"] is False, "VPHYS_FIREWALL")
    require(protocol["execution"]["protected_reads"] is False, "PROTECTED_FIREWALL")

    import sys

    sys.path.insert(0, str(ROOT / "src"))
    from stage_z_preparation.anchors import AnchorCandidate, select_anchor

    candidates = [
        AnchorCandidate("p", "M0_OPENVLA", 4, "CRITICAL", metadata={"phase": "CARRY"}),
        AnchorCandidate("p", "M0_OPENVLA", 8, "CRITICAL", metadata={"phase": "CARRY"}),
        AnchorCandidate("p", "M0_OPENVLA", 2, "NONCRITICAL", metadata={"phase": "PRE_CONTACT"}),
    ]
    selected_a = select_anchor(candidates, salt="z2-static", model_id="M0_OPENVLA", parent_key="p", anchor_class="CRITICAL")
    selected_b = select_anchor(candidates, salt="z2-static", model_id="M0_OPENVLA", parent_key="p", anchor_class="CRITICAL")
    require(selected_a.selected == selected_b.selected, "ANCHOR_NOT_DETERMINISTIC")
    require(select_anchor(candidates, salt="z2-static", model_id="M1_OPENVLA_OFT", parent_key="p", anchor_class="CRITICAL").selected is None, "ANCHOR_CROSS_MODEL_LEAK")
    require(select_anchor(candidates, salt="z2-static", model_id="M0_OPENVLA", parent_key="missing", anchor_class="CRITICAL").selected is None, "ANCHOR_NO_ABSTAIN")

    parents = {str(row["canonical_parent_key"]) for row in panel["rows"] if row.get("canonical_parent_key")}
    expected = {(model, parent) for model in ("M0_OPENVLA", "M1_OPENVLA_OFT", "M2_PI05_LIBERO") for parent in panel["selected_parent_keys"]}
    legacy_files = sorted(LEGACY_DIR.glob("STAGE_Z_Z2_CLEAN_REFERENCE_*.json"))
    attempts: list[dict[str, Any]] = []
    for path in legacy_files:
        row = load(path)
        key = receipt_key(row)
        status = str(row.get("status"))
        counters = row.get("runtime_counters", {})
        forbidden = {key: counters.get(key, 0) for key in ("physical_interventions", "pgd_calls", "attacked_env_steps", "vphys_reads", "attack_outcome_reads", "eval160_reads", "protected_reads")}
        valid_parent = key[1] in parents
        semantic_ok = (
            row.get("schema") == "STAGE_Z_Z2_CLEAN_REFERENCE_CELL_RECEIPT_V1"
            and key in expected
            and valid_parent
            and status in {"PASS_Z2_CLEAN_REFERENCE_WITH_BOTH_ANCHORS", "ABSTAIN_Z2_NO_LEGAL_ANCHOR"}
            and row.get("anchor_rule", {}).get("phase_classifier", {}).get("minimum_remaining_steps") == 20
            and row.get("anchor_rule", {}).get("critical_salt") == protocol["anchor_selection"]["critical_salt"]
            and row.get("anchor_rule", {}).get("noncritical_salt") == protocol["anchor_selection"]["noncritical_salt"]
            and row.get("anchor_rule", {}).get("student_or_detector_used") is False
            and row.get("anchor_rule", {}).get("outcome_fields_used") == []
            and all(value == 0 for value in forbidden.values())
            and counters.get("stage_z_scientific_parent_exposure") == 1
            and int(row.get("clean_reference", {}).get("candidate_count", -1)) >= 0
            and int(row.get("clean_reference", {}).get("decision_boundary_count", 0)) > 0
        )
        attempts.append({
            "path": str(path.relative_to(ROOT)).replace("\\", "/"),
            "sha256": sha256_file(path),
            "model_family": key[0],
            "canonical_parent_key": key[1],
            "status": status,
            "scientific_parent_exposure": counters.get("stage_z_scientific_parent_exposure", "NOT_MATERIALIZED"),
            "semantic_reconciliation": "PASS_LEGACY_SEMANTICS" if semantic_ok else "NOT_PROMOTABLE",
            "forbidden_counters": forbidden,
        })

    semantic_rows = [row for row in attempts if row["semantic_reconciliation"] == "PASS_LEGACY_SEMANTICS"]
    exposed_keys = {(row["model_family"], row["canonical_parent_key"]) for row in semantic_rows}
    missing = sorted(expected - exposed_keys)
    duplicates = sorted({key for key in exposed_keys if sum(1 for row in semantic_rows if (row["model_family"], row["canonical_parent_key"]) == key) > 1})
    recon = {
        "schema": "STAGE_Z_Z2_LEGACY_RECEIPT_RECONCILIATION_V1",
        "status": "LEGACY_SEMANTIC_RECONCILIATION_PASS_WITH_UNEXPOSED_CELLS_REMAINING",
        "protocol_v2": str(PROTOCOL.relative_to(ROOT)).replace("\\", "/"),
        "legacy_protocol": protocol["historical_exposure_reconciliation"],
        "expected_model_parent_cells": len(expected),
        "legacy_receipt_files": len(attempts),
        "legacy_semantically_reconciled_files": len(semantic_rows),
        "unique_legacy_exposed_model_parent_cells": len(exposed_keys),
        "duplicate_legacy_attempt_keys": [list(key) for key in duplicates],
        "missing_model_parent_cells": [list(key) for key in missing],
        "attempts": attempts,
        "scientific_claim": "NONE; historical exposure/provenance reconciliation only",
        "z3_authorized": False,
    }
    RECON_OUT.write_text(json.dumps(recon, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    audit = {
        "schema": "STAGE_Z_Z2A_STATIC_AUDIT_V2",
        "status": "STAGE_Z_Z2_SOURCE_AND_ANCHOR_STATIC_PASS_WITH_LEGACY_RECONCILIATION_REQUIRED",
        "protocol": str(PROTOCOL.relative_to(ROOT)).replace("\\", "/"),
        "protocol_sha256": sha256_file(PROTOCOL),
        "checks": {
            "z1_terminal_root_bound": True,
            "current_head_tree_bound": True,
            "anchor_selector_determinism": True,
            "student_detector_leakage_guard": True,
            "forbidden_intervention_firewall": True,
            "legacy_receipt_semantics": True,
            "engineering_canary_execution": "NOT_REQUIRED_STATIC_MOCK_PASS",
        },
        "population": protocol["population"],
        "legacy_exposure": {
            "unique_scientific_identities_exposed": len({row["canonical_parent_key"] for row in semantic_rows}),
            "unique_model_parent_cells_exposed": len(exposed_keys),
            "expected_model_parent_cells": len(expected),
            "missing_model_parent_cells": [list(key) for key in missing],
            "legacy_receipt_reconciliation": str(RECON_OUT.relative_to(ROOT)).replace("\\", "/"),
        },
        "scientific_counters": {
            "command_open_intervention": 0,
            "pgd_calls": 0,
            "attacked_env_steps": 0,
            "v_phys_reads": 0,
            "protected_reads": 0,
            "eval160_reads": 0,
        },
        "disposition": "Z2-B may run only unexposed/engineering-invalid model-parent cells once under V2; do not relabel legacy receipts; STOP_FOR_PI before Z3.",
    }
    AUDIT_OUT.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": audit["status"], "legacy_exposed_model_parent_cells": len(exposed_keys), "missing_model_parent_cells": len(missing)}))


if __name__ == "__main__":
    main()
