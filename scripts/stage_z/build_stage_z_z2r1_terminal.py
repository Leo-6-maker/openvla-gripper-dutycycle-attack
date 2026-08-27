#!/usr/bin/env python3
"""Build the append-only Z2R1 M2 clean-repair evidence package."""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "reports/STAGE_Z_Z2R1_M2_CLEAN_REPAIR_MANIFEST_V1.json"
PANEL = ROOT / "reports/STAGE_Z_Z0R1_SHARED_36_IDENTITY_PANEL_V1.json"
CANARY = ROOT / "reports/server_evidence/STAGE_Z_Z2R1_M2_ENGINEERING_CANARY_libero_10_PRIMARY.json"
EVIDENCE = ROOT / "reports/server_evidence"
OUT_LEDGER = ROOT / "reports/STAGE_Z_Z2R1_M2_CLEAN_REPAIR_RECEIPT_LEDGER_V1.json"
OUT_SYNTHESIS = ROOT / "reports/STAGE_Z_Z2R1_M2_CLEAN_REPAIR_TERMINAL_SYNTHESIS_V1.json"
OUT_SEAL = ROOT / "reports/STAGE_Z_Z2R1_M2_CLEAN_REPAIR_ROOT_SEAL_V1.json"
OUT_SIDECAR = ROOT / "reports/STAGE_Z_Z2R1_M2_CLEAN_REPAIR_ROOT_SEAL_V1.sha256"

RUNNER = ROOT / "scripts/stage_z/run_stage_z_z2_clean_reference.py"
ADAPTER = ROOT / "src/stage_z_preparation/action_semantics.py"
HISTORICAL_CLASSIFIER = ROOT / "src/gripper_attack/stage_v_m3_5_phase_classifier.py"
R1_PROTOCOL = ROOT / "configs/STAGE_Z_Z2R1_M2_ACTION_SEMANTICS_PROTOCOL_V1.json"
DISCREPANCY = ROOT / "reports/STAGE_Z_Z2R1_M2_ACTION_SEMANTICS_DISCREPANCY_V1.json"
STATIC_PARITY = ROOT / "reports/STAGE_Z_Z2R1_M2_ACTION_SEMANTICS_STATIC_PARITY_V1.json"
STORAGE = ROOT / "reports/STAGE_Z_Z2R1_STORAGE_PREFLIGHT_V1.json"
HISTORICAL_ROOT = ROOT / "reports/STAGE_Z_Z2_TERMINAL_ROOT_SEAL_V2.json"

VALID_STATUSES = {
    "PASS_Z2_CLEAN_REFERENCE_WITH_BOTH_ANCHORS",
    "PASS_Z2_CLEAN_REFERENCE_WITH_NONCRITICAL_ONLY",
    "ABSTAIN_NO_LEGAL_ANCHOR",
}
FORBIDDEN = (
    "attack_outcome_reads",
    "attacked_env_steps",
    "eval160_reads",
    "pgd_calls",
    "physical_interventions",
    "protected_reads",
    "vphys_reads",
)
HORIZONS = {"libero_10": 520, "libero_goal": 300, "libero_object": 280, "libero_spatial": 220}


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def git(*args: str) -> str:
    return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True).strip()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def artifact(path: Path) -> dict[str, Any]:
    return {"path": rel(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def compact_anchor(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    return {key: value.get(key) for key in ("status", "anchor_class", "step", "phase", "rank_digest", "state_sha256")}


def availability(receipt: dict[str, Any]) -> str:
    selected = receipt.get("selected_anchors") or {}
    critical = selected.get("critical") is not None
    noncritical = selected.get("noncritical") is not None
    if critical and noncritical:
        return "BOTH_ANCHORS_AVAILABLE"
    if critical:
        return "CRITICAL_ONLY_NONCRITICAL_CONTROL_MISSING"
    if noncritical:
        return "NONCRITICAL_ONLY_CRITICAL_ANCHOR_MISSING"
    return "NO_LEGAL_ANCHOR"


def validate_receipt(path: Path, receipt: dict[str, Any], expected: dict[str, Any]) -> dict[str, Any]:
    parent = expected["canonical_parent_key"]
    require(receipt.get("schema") == "STAGE_Z_Z2R1_CLEAN_REFERENCE_CELL_RECEIPT_V1", f"SCHEMA:{parent}")
    require(receipt.get("canonical_parent_key") == parent, f"PARENT:{parent}")
    require(receipt.get("model_family") == "M2_PI05_LIBERO", f"MODEL:{parent}")
    require(receipt.get("suite") == expected["suite"], f"SUITE:{parent}")
    require(receipt.get("population") == "scientific", f"POPULATION:{parent}")
    require(receipt.get("scientific_claim") == "Z2_CLEAN_REFERENCE_AND_ANCHOR_ONLY", f"CLAIM:{parent}")
    require(receipt.get("status") in VALID_STATUSES, f"STATUS:{parent}")

    clean = receipt.get("clean_reference") or {}
    semantics = clean.get("action_semantics") or {}
    horizon = int(clean.get("horizon", 0))
    require(horizon == HORIZONS[expected["suite"]], f"HORIZON:{parent}")
    require(semantics.get("rule") == "PI05_CLIP_RAW_TO_LIBERO_V1", f"SEMANTICS_RULE:{parent}")
    require(int(semantics.get("checks", -1)) == horizon, f"SEMANTICS_CHECKS:{parent}")
    require(int(semantics.get("accepted", -1)) == horizon, f"SEMANTICS_ACCEPTED:{parent}")
    require(int(semantics.get("invalid", -1)) == 0, f"SEMANTICS_INVALID:{parent}")

    counters = receipt.get("runtime_counters") or {}
    require(int(counters.get("stage_z_scientific_parent_exposure", 0)) == 1, f"EXPOSURE:{parent}")
    require(int(counters.get("model_inference_calls", 0)) > 0, f"INFERENCE:{parent}")
    require(int(counters.get("env_step_calls", 0)) > 0, f"ENV_STEP:{parent}")
    require(all(int(counters.get(name, 0)) == 0 for name in FORBIDDEN), f"FORBIDDEN:{parent}")

    selected = receipt.get("selected_anchors") or {}
    for label in ("critical", "noncritical"):
        anchor = selected.get(label)
        require(anchor is not None, f"ANCHOR_{label}:{parent}")
        rows = anchor.get("action_rows") or []
        require(rows, f"ACTION_ROWS_{label}:{parent}")
        for row in rows:
            raw = row.get("raw_action")
            env = row.get("env_action")
            require(isinstance(raw, list) and isinstance(env, list) and len(raw) == 7 and len(env) == 7, f"ACTION_7D:{parent}")
            require(all(math.isfinite(float(value)) for value in raw + env), f"ACTION_FINITE:{parent}")

    return {
        "receipt_path": rel(path),
        "receipt_sha256": sha256_file(path),
        "canonical_parent_key": parent,
        "suite": expected["suite"],
        "task_idx": expected["task_idx"],
        "state_id": expected["state_id"],
        "status": receipt["status"],
        "anchor_availability": availability(receipt),
        "horizon": horizon,
        "candidate_count": int(clean.get("candidate_count", 0)),
        "decision_boundary_count": int(clean.get("decision_boundary_count", 0)),
        "action_semantics": {
            "rule": semantics["rule"],
            "checks": int(semantics["checks"]),
            "accepted": int(semantics["accepted"]),
            "invalid": int(semantics["invalid"]),
        },
        "runtime_counters": {name: int(counters.get(name, 0)) for name in ("model_inference_calls", "env_step_calls", "anchor_telemetry_reads", "stage_z_scientific_parent_exposure", *FORBIDDEN)},
        "critical_anchor": compact_anchor(selected.get("critical")),
        "noncritical_anchor": compact_anchor(selected.get("noncritical")),
    }


def main() -> None:
    manifest = load(MANIFEST)
    require(manifest.get("status") == "STAGE_Z_Z2R1_M2_CLEAN_REPAIR_34_CELL_SCOPE_FROZEN", "MANIFEST_STATUS")
    expected_rows = {row["canonical_parent_key"]: row for row in manifest["cells"]}
    require(len(expected_rows) == 34, "EXPECTED_34_CELLS")
    require(set(manifest["scope_contract"]["structural_abstentions_not_loaded"]) == {"libero_goal/task_00/state_21", "libero_goal/task_07/state_30"}, "ABSTENTION_SCOPE")

    paths = sorted(EVIDENCE.glob("STAGE_Z_Z2R1_M2_CLEAN_REPAIR_*.json"))
    require(len(paths) == 34, f"RECEIPT_COUNT:{len(paths)}")
    receipts = [load(path) for path in paths]
    by_key = {receipt["canonical_parent_key"]: (path, receipt) for path, receipt in zip(paths, receipts)}
    require(len(by_key) == 34, "RECEIPT_DUPLICATE")
    require(set(by_key) == set(expected_rows), "RECEIPT_SCOPE")
    cells = [validate_receipt(by_key[key][0], by_key[key][1], expected_rows[key]) for key in sorted(expected_rows)]

    canary = load(CANARY)
    require(canary.get("status") == "PASS_Z2_CLEAN_REFERENCE_WITH_BOTH_ANCHORS", "CANARY_STATUS")
    require(canary.get("scientific_claim") == "NONE_ENGINEERING_ONLY", "CANARY_CLAIM")
    canary_counters = canary.get("runtime_counters") or {}
    require(all(int(canary_counters.get(name, 0)) == 0 for name in FORBIDDEN), "CANARY_FORBIDDEN")
    require(int(canary_counters.get("stage_z_scientific_parent_exposure", 0)) == 0, "CANARY_EXPOSURE")

    status_counts = dict(sorted(Counter(cell["status"] for cell in cells).items()))
    availability_counts = dict(sorted(Counter(cell["anchor_availability"] for cell in cells).items()))
    suite_summary = {}
    for suite in sorted({cell["suite"] for cell in cells}):
        rows = [cell for cell in cells if cell["suite"] == suite]
        suite_summary[suite] = {"cells": len(rows), "both_anchors": sum(row["anchor_availability"] == "BOTH_ANCHORS_AVAILABLE" for row in rows), "status_counts": dict(sorted(Counter(row["status"] for row in rows).items()))}

    forbidden_totals = {name: sum(cell["runtime_counters"][name] for cell in cells) for name in FORBIDDEN}
    authority = {
        "repository": "Leo-6-maker/openvla-gripper-dutycycle-attack",
        "branch": git("branch", "--show-current"),
        "source_head_before_terminal_seal": git("rev-parse", "HEAD"),
        "source_tree_before_terminal_seal": git("rev-parse", "HEAD^{tree}"),
        "working_tree_status_at_generation": git("status", "--porcelain=v1").splitlines(),
        "historical_z2_root_sha256": sha256_file(HISTORICAL_ROOT),
        "historical_phase_classifier_sha256": sha256_file(HISTORICAL_CLASSIFIER),
        "runner": artifact(RUNNER),
        "action_semantics_adapter": artifact(ADAPTER),
        "r1_protocol": artifact(R1_PROTOCOL),
        "panel": artifact(PANEL),
        "manifest": artifact(MANIFEST),
        "discrepancy": artifact(DISCREPANCY),
        "static_parity": artifact(STATIC_PARITY),
        "storage_preflight": artifact(STORAGE),
    }

    ledger = {
        "schema": "STAGE_Z_Z2R1_M2_CLEAN_REPAIR_RECEIPT_LEDGER_V1",
        "status": "STAGE_Z_Z2R1_M2_CLEAN_REPAIR_RECEIPTS_SEALED",
        "gate": "STAGE_Z_Z2R1_M2_PHASE_ACTION_SEMANTICS_RECONCILIATION_AND_CLEAN_ANCHOR_REPAIR_V1",
        "authorization_comment_id": 5405256740,
        "claim_boundary": "M2 clean action-semantics repair and anchor feasibility only; no physical OPEN susceptibility or attack efficacy claim.",
        "historical_z2_preserved": True,
        "scope": {"model_family": "M2_PI05_LIBERO", "cells": 34, "new_identity": False, "top_up": False, "m0_rerun": False, "m1_rerun": False, "structural_abstentions_not_loaded": manifest["scope_contract"]["structural_abstentions_not_loaded"]},
        "authority": authority,
        "engineering_canary": artifact(CANARY),
        "cells": cells,
        "summary": {"status_counts": status_counts, "anchor_availability": availability_counts, "suite_summary": suite_summary, "forbidden_counter_totals": forbidden_totals},
        "next_legal_action": "STOP_FOR_PI",
        "z3_authorized": False,
    }
    write_json(OUT_LEDGER, ledger)

    synthesis = {
        "schema": "STAGE_Z_Z2R1_M2_CLEAN_REPAIR_TERMINAL_SYNTHESIS_V1",
        "status": "SEALED_STAGE_Z_Z2R1_M2_CLEAN_REPAIR_STOP_FOR_PI",
        "gate": ledger["gate"],
        "authorization_comment_id": 5405256740,
        "claim_boundary": ledger["claim_boundary"],
        "historical_conclusion": "The prior M2 0/36 anchor result is not promoted as scientific negative evidence because the historical classifier used OpenVLA gripper semantics.",
        "repaired_conclusion": "Under the Stage-Z-only PI05 clip/raw-to-LIBERO adapter, all 34 clean-exposed M2 cells have both critical and noncritical anchors.",
        "matrix": {"panel_identities": 36, "clean_exposed_identities": 34, "model_parent_cells": 34, "both_anchors": availability_counts.get("BOTH_ANCHORS_AVAILABLE", 0), "noncritical_only": availability_counts.get("NONCRITICAL_ONLY_CRITICAL_ANCHOR_MISSING", 0), "no_legal_anchor": availability_counts.get("NO_LEGAL_ANCHOR", 0), "suite_summary": suite_summary},
        "execution_counters": {"model_inference_calls": sum(cell["runtime_counters"]["model_inference_calls"] for cell in cells), "env_step_calls": sum(cell["runtime_counters"]["env_step_calls"] for cell in cells), "anchor_telemetry_reads": sum(cell["runtime_counters"]["anchor_telemetry_reads"] for cell in cells), "stage_z_scientific_parent_exposure": sum(cell["runtime_counters"]["stage_z_scientific_parent_exposure"] for cell in cells), **forbidden_totals},
        "forbidden_execution_totals": forbidden_totals,
        "authority": authority,
        "artifact_outputs": {"receipt_ledger": rel(OUT_LEDGER)},
        "next_legal_action": "STOP_FOR_PI",
        "z3_authorized": False,
    }
    write_json(OUT_SYNTHESIS, synthesis)

    seal = {
        "schema": "STAGE_Z_Z2R1_M2_CLEAN_REPAIR_ROOT_SEAL_V1",
        "status": "SEALED_STAGE_Z_Z2R1_M2_CLEAN_REPAIR_STOP_FOR_PI",
        "claim_boundary": ledger["claim_boundary"],
        "historical_z2_root_preserved": {"path": rel(HISTORICAL_ROOT), "sha256": sha256_file(HISTORICAL_ROOT)},
        "artifact_manifest": [artifact(path) for path in (R1_PROTOCOL, DISCREPANCY, STATIC_PARITY, STORAGE, MANIFEST, CANARY, OUT_LEDGER, OUT_SYNTHESIS, RUNNER, ADAPTER, HISTORICAL_CLASSIFIER)],
        "git_binding": {"branch": authority["branch"], "head_before_root_seal": authority["source_head_before_terminal_seal"], "tree_before_root_seal": authority["source_tree_before_terminal_seal"]},
        "execution_scope": {"m2_clean_cells": 34, "structural_abstentions_not_loaded": manifest["scope_contract"]["structural_abstentions_not_loaded"], "attack": False, "physical_intervention": False, "vphys": False, "protected": False, "eval160": False},
        "next_legal_action": "STOP_FOR_PI",
        "z3_authorized": False,
    }
    write_json(OUT_SEAL, seal)
    OUT_SIDECAR.write_text(f"{sha256_file(OUT_SEAL)}  {OUT_SEAL.name}\n", encoding="utf-8")
    print(json.dumps({"status": seal["status"], "cells": len(cells), "both_anchors": availability_counts.get("BOTH_ANCHORS_AVAILABLE", 0), "root_sha256": sha256_file(OUT_SEAL)}, sort_keys=True))


if __name__ == "__main__":
    main()
