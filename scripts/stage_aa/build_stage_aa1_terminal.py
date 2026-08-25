#!/usr/bin/env python3
"""Seal the AA1 engineering-only receipts without interpreting them scientifically."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


EXPECTED = {
    ("M0_OPENVLA", "libero_10/task_04/state_20"),
    ("M0_OPENVLA", "libero_object/task_02/state_42"),
    ("M0_OPENVLA", "libero_spatial/task_05/state_34"),
    ("M1_OPENVLA_OFT", "libero_10/task_04/state_20"),
    ("M1_OPENVLA_OFT", "libero_object/task_02/state_42"),
    ("M1_OPENVLA_OFT", "libero_spatial/task_05/state_34"),
    ("M2_PI05_LIBERO", "libero_10/task_04/state_20"),
    ("M2_PI05_LIBERO", "libero_object/task_02/state_42"),
    ("M2_PI05_LIBERO", "libero_spatial/task_05/state_34"),
}
PASS = "PASS_AA1_PASSIVE_PIPELINE_NO_LEGAL_ENGINEERING_POINT"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def file_row(root: Path, path: Path) -> dict[str, Any]:
    return {"path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size, "sha256": digest(path)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    receipt_dir = root / "reports/server_evidence/STAGE_AA_AA1"
    out_dir = root / "reports"

    latest_paths = sorted(receipt_dir.glob("AA1_REPAIR_V3_*.json"))
    if len(latest_paths) != len(EXPECTED):
        raise SystemExit(f"expected 9 V3 receipts, found {len(latest_paths)}")
    latest: dict[tuple[str, str], tuple[Path, dict[str, Any]]] = {}
    for path in latest_paths:
        receipt = json.loads(path.read_text(encoding="utf-8"))
        key = (receipt.get("model_family"), receipt.get("canonical_parent_key"))
        if key in latest or key not in EXPECTED:
            raise SystemExit(f"unexpected or duplicate V3 cell: {key}")
        latest[key] = (path, receipt)
        if receipt.get("status") != PASS:
            raise SystemExit(f"V3 cell is not passive PASS: {path.name}:{receipt.get('status')}")
        if receipt.get("branch_count") != 0 or receipt.get("scientific_use") is not False:
            raise SystemExit(f"scientific or branch exposure in {path.name}")
        counters = receipt.get("runtime_counters", {})
        for field in ("open_intervention_steps", "pgd_calls", "attacked_env_steps", "v_phys_reads", "protected_reads", "eval160_reads", "task_success_reads", "scientific_parent_exposure", "aa2_exposure"):
            if counters.get(field, 0) != 0:
                raise SystemExit(f"forbidden counter {field} in {path.name}")
    if set(latest) != EXPECTED:
        raise SystemExit(f"V3 cell coverage mismatch: {sorted(set(latest) ^ EXPECTED)}")

    receipt_rows = []
    total_inference = total_steps = total_telemetry = 0
    critical_anchor_cells = 0
    noncritical_anchor_cells = 0
    complete_engineering_point_cells = 0
    for key in sorted(latest):
        path, receipt = latest[key]
        counters = receipt["runtime_counters"]
        selected = receipt.get("selected_anchors", {})
        has_critical = selected.get("critical") is not None
        has_noncritical = selected.get("noncritical") is not None
        critical_anchor_cells += int(has_critical)
        noncritical_anchor_cells += int(has_noncritical)
        complete_engineering_point_cells += int(has_critical and has_noncritical)
        total_inference += int(counters.get("model_inference_calls", 0))
        total_steps += int(counters.get("env_step_calls", 0))
        total_telemetry += int(counters.get("physical_telemetry_reads", 0))
        receipt_rows.append({
            "model_family": key[0],
            "canonical_parent_key": key[1],
            "status": receipt["status"],
            "branch_count": receipt.get("branch_count", 0),
            "selected_anchors": receipt.get("selected_anchors", {}),
            "runtime_counters": counters,
            "artifact": file_row(root, path),
        })

    index = {
        "schema": "STAGE_AA_AA1_ENGINEERING_CANARY_RECEIPT_INDEX_V1",
        "status": "STAGE_AA_AA1_ENGINEERING_CANARY_RECEIPT_INDEX_FROZEN",
        "gate": "STAGE_AA_AA1_ENGINEERING_RUNTIME_QUALIFICATION_V1",
        "authorization_pi_comment_id": 5413730571,
        "source_authority": "reports/STAGE_AA_AA1_RUNTIME_SOURCE_AUTHORITY_V3.json",
        "latest_v3_cells": receipt_rows,
        "historical_receipts_retained": [file_row(root, path) for path in sorted(receipt_dir.glob("AA1_*.json"))],
        "coverage": {
            "expected_cells": 9,
            "latest_v3_cells": 9,
            "legal_critical_anchor_cells": critical_anchor_cells,
            "legal_noncritical_anchor_cells": noncritical_anchor_cells,
            "complete_engineering_point_cells": complete_engineering_point_cells,
            "branch_cells": 0,
            "branch_count": 0,
            "expected_branches_if_legal": 45,
        },
        "runtime_totals": {
            "model_inference_calls": total_inference,
            "env_step_calls": total_steps,
            "physical_telemetry_reads": total_telemetry,
        },
        "scientific_firewall": {
            "aa2_scientific_candidates_exposed": 0,
            "stage_z_identity_exposure": 0,
            "open_intervention_steps": 0,
            "pgd_calls": 0,
            "attacked_env_steps": 0,
            "v_phys_reads": 0,
            "protected_reads": 0,
            "eval160_reads": 0,
            "task_success_reads": 0,
            "paper_promotion": False,
        },
        "next_legal_action": "STOP_FOR_PI",
    }
    index_path = out_dir / "STAGE_AA_AA1_ENGINEERING_CANARY_RECEIPT_INDEX_V1.json"
    terminal_path = out_dir / "STAGE_AA_AA1_ENGINEERING_CANARY_TERMINAL_V1.json"
    root_path = out_dir / "STAGE_AA_AA1_ROOT_SEAL_V1.json"
    write_json(index_path, index)

    terminal = {
        "schema": "STAGE_AA_AA1_ENGINEERING_CANARY_TERMINAL_V1",
        "status": "STAGE_AA_AA1_ENGINEERING_RUNTIME_QUALIFICATION_HOLD_STOP_FOR_PI",
        "gate": "STAGE_AA_AA1_ENGINEERING_RUNTIME_QUALIFICATION_V1",
        "authorization_pi_comment_id": 5413730571,
        "source_authority": "reports/STAGE_AA_AA1_RUNTIME_SOURCE_AUTHORITY_V3.json",
        "receipt_index": file_row(root, index_path),
        "result": {
            "cells_expected": 9,
            "cells_completed_passive_clean": 9,
            "cells_with_legal_critical_anchor": critical_anchor_cells,
            "cells_with_legal_noncritical_anchor": noncritical_anchor_cells,
            "cells_with_complete_engineering_point": complete_engineering_point_cells,
            "branches_executed": 0,
            "branches_expected_if_legal": 45,
            "model_inference_calls": total_inference,
            "env_step_calls": total_steps,
            "physical_telemetry_reads": total_telemetry,
        },
        "engineering_history": {
            "m2_initial_allocator_failure_retained": True,
            "m2_allocator_repair_retry_retained": True,
            "v3_noncritical_contract_repair_applied": True,
            "same_nine_canaries_only": True,
            "foreign_gpu_processes_touched": False,
        },
        "interpretation": "Runtime clean eligibility pipeline completed for all nine permanently excluded canary cells. One cell produced a noncritical-only anchor, but no cell produced the paired critical+noncritical engineering point required for the five-branch qualification. This is an engineering qualification HOLD, not a scientific negative and not evidence about AA2 vulnerability.",
        "scientific_firewall": index["scientific_firewall"],
        "aa2_authorized": False,
        "next_legal_action": "STOP_FOR_PI",
    }
    write_json(terminal_path, terminal)

    seal_payload = {
        "schema": "STAGE_AA_AA1_ROOT_SEAL_V1",
        "status": terminal["status"],
        "gate": terminal["gate"],
        "source_authority": file_row(root, root / "reports/STAGE_AA_AA1_RUNTIME_SOURCE_AUTHORITY_V3.json"),
        "receipt_index": file_row(root, index_path),
        "terminal": file_row(root, terminal_path),
        "latest_v3_receipts": [row["artifact"] for row in receipt_rows],
        "historical_receipt_count": len(index["historical_receipts_retained"]),
        "scientific_firewall": index["scientific_firewall"],
    }
    seal_payload["root_payload_sha256"] = hashlib.sha256(canonical(seal_payload)).hexdigest()
    seal_payload["next_legal_action"] = "STOP_FOR_PI"
    write_json(root_path, seal_payload)
    print(json.dumps({"status": terminal["status"], "root_seal": digest(root_path), "root_payload_sha256": seal_payload["root_payload_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
