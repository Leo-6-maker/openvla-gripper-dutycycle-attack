#!/usr/bin/env python3
"""Build the append-only Z2 clean-reference and anchor-freeze package."""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_V3 = ROOT / "configs/STAGE_Z_Z2_CLEAN_REFERENCE_PROTOCOL_V3.json"
PROTOCOL_V2 = ROOT / "configs/STAGE_Z_Z2_CLEAN_REFERENCE_PROTOCOL_V2.json"
PANEL = ROOT / "reports/STAGE_Z_Z0R1_SHARED_36_IDENTITY_PANEL_V1.json"
STATIC_V2 = ROOT / "reports/STAGE_Z_Z2A_STATIC_AUDIT_V2.json"
STATIC_V3 = ROOT / "reports/STAGE_Z_Z2A_V3_STATIC_AUDIT.json"
LEGACY_RECON = ROOT / "reports/STAGE_Z_Z2_LEGACY_RECEIPT_RECONCILIATION_V1.json"
EVIDENCE = ROOT / "reports/server_evidence"
OUT_EXPOSURE = ROOT / "reports/STAGE_Z_Z2_SCIENTIFIC_EXPOSURE_LEDGER_V2.json"
OUT_ANCHORS = ROOT / "reports/STAGE_Z_Z2_ANCHOR_LEDGER_V2.json"
OUT_AVAILABILITY = ROOT / "reports/STAGE_Z_Z2_ANCHOR_AVAILABILITY_V2.json"
OUT_SYNTHESIS = ROOT / "reports/STAGE_Z_Z2_TERMINAL_SYNTHESIS_V2.json"
OUT_SEAL = ROOT / "reports/STAGE_Z_Z2_TERMINAL_ROOT_SEAL_V2.json"
OUT_SIDECAR = ROOT / "reports/STAGE_Z_Z2_TERMINAL_ROOT_SEAL_V2.sha256"

MODELS = ("M0_OPENVLA", "M1_OPENVLA_OFT", "M2_PI05_LIBERO")
VALID_STATUSES = {"PASS_Z2_CLEAN_REFERENCE_WITH_BOTH_ANCHORS", "ABSTAIN_Z2_NO_LEGAL_ANCHOR"}
FORBIDDEN_COUNTERS = (
    "physical_interventions",
    "pgd_calls",
    "attacked_env_steps",
    "vphys_reads",
    "attack_outcome_reads",
    "eval160_reads",
    "protected_reads",
)


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
    return {
        key: value.get(key)
        for key in ("status", "anchor_class", "step", "phase", "rank_digest", "state_sha256")
        if key in value
    }


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


def counter_snapshot(receipt: dict[str, Any]) -> dict[str, int]:
    counters = receipt.get("runtime_counters") or {}
    return {key: int(counters.get(key, 0)) for key in FORBIDDEN_COUNTERS}


def receipt_summary(path: Path, receipt: dict[str, Any], provenance: str, attempt_chain: list[str] | None = None) -> dict[str, Any]:
    counters = receipt.get("runtime_counters") or {}
    clean = receipt.get("clean_reference") or {}
    selected = receipt.get("selected_anchors") or {}
    return {
        "receipt_path": rel(path),
        "receipt_sha256": sha256_file(path),
        "receipt_schema": receipt.get("schema"),
        "provenance": provenance,
        "status": receipt.get("status"),
        "model_family": receipt.get("model_family"),
        "suite": receipt.get("suite"),
        "canonical_parent_key": receipt.get("canonical_parent_key"),
        "scientific_parent_exposure": int(counters.get("stage_z_scientific_parent_exposure", 0)),
        "model_inference_calls": int(counters.get("model_inference_calls", 0)),
        "env_step_calls": int(counters.get("env_step_calls", 0)),
        "anchor_telemetry_reads": int(counters.get("anchor_telemetry_reads", 0)),
        "candidate_count": int(clean.get("candidate_count", 0)),
        "decision_boundary_count": int(clean.get("decision_boundary_count", 0)),
        "anchor_availability": availability(receipt),
        "critical_anchor": compact_anchor(selected.get("critical")),
        "noncritical_control": compact_anchor(selected.get("noncritical")),
        "forbidden_counters": counter_snapshot(receipt),
        "abstention_reason": clean.get("abstention_reason"),
        "object_binding_status": (receipt.get("object_binding") or {}).get("status"),
        "object_binding_reason": (receipt.get("object_binding") or {}).get("reason"),
        "attempt_chain": attempt_chain or [rel(path)],
    }


def validate_receipt(receipt: dict[str, Any], *, selected: bool, protocol_v2: dict[str, Any]) -> None:
    counters = receipt.get("runtime_counters") or {}
    require(receipt.get("schema") == "STAGE_Z_Z2_CLEAN_REFERENCE_CELL_RECEIPT_V1", "RECEIPT_SCHEMA")
    require(all(int(counters.get(key, 0)) == 0 for key in FORBIDDEN_COUNTERS), "FORBIDDEN_COUNTER_NONZERO")
    if not selected:
        require(receipt.get("status") == "ENGINEERING_INVALID_Z2_CLEAN_REFERENCE", "UNSELECTED_NOT_ENGINEERING_INVALID")
        return
    require(receipt.get("status") in VALID_STATUSES, "SELECTED_STATUS_INVALID")
    rule = receipt.get("anchor_rule") or {}
    require(rule.get("student_or_detector_used") is False, "STUDENT_LEAK")
    require(rule.get("outcome_fields_used") == [], "OUTCOME_LEAK")
    require(rule.get("critical_salt") == protocol_v2["anchor_selection"]["critical_salt"], "CRITICAL_SALT")
    require(rule.get("noncritical_salt") == protocol_v2["anchor_selection"]["noncritical_salt"], "NONCRITICAL_SALT")
    exposure = int(counters.get("stage_z_scientific_parent_exposure", 0))
    if exposure == 0:
        require(int(counters.get("model_inference_calls", 0)) == 0, "ABSTAIN_INFERENCE_NONZERO")
        require(int(counters.get("env_step_calls", 0)) == 0, "ABSTAIN_ENV_STEP_NONZERO")
        require((receipt.get("object_binding") or {}).get("status") == "INELIGIBLE", "ZERO_EXPOSURE_NOT_INELIGIBLE")
    else:
        require(exposure == 1, "EXPOSURE_NOT_BINARY")
        require(int(counters.get("model_inference_calls", 0)) > 0, "EXPOSED_NO_INFERENCE")
        require(int(counters.get("env_step_calls", 0)) > 0, "EXPOSED_NO_ENV_STEP")


def main() -> None:
    protocol_v3 = load(PROTOCOL_V3)
    protocol_v2 = load(PROTOCOL_V2)
    panel = load(PANEL)
    static_v2 = load(STATIC_V2)
    static_v3 = load(STATIC_V3)
    legacy_recon = load(LEGACY_RECON)
    expected_parents = set(panel["selected_parent_keys"])
    expected = {(model, parent) for model in MODELS for parent in expected_parents}
    missing = {tuple(item) for item in legacy_recon["missing_model_parent_cells"]}
    require(len(expected) == 108, "EXPECTED_108_MODEL_PARENT_CELLS")
    require(legacy_recon["unique_legacy_exposed_model_parent_cells"] == 98, "LEGACY_98_BINDING")
    require(len(missing) == 10, "EXPECTED_10_MISSING_CELLS")

    legacy_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for row in legacy_recon["attempts"]:
        if row.get("semantic_reconciliation") != "PASS_LEGACY_SEMANTICS":
            continue
        key = (str(row["model_family"]), str(row["canonical_parent_key"]))
        require(key not in legacy_by_key, f"LEGACY_DUPLICATE:{key}")
        path = ROOT / str(row["path"])
        require(path.is_file(), f"LEGACY_RECEIPT_MISSING:{path}")
        require(sha256_file(path) == row["sha256"], f"LEGACY_RECEIPT_SHA:{path}")
        receipt = load(path)
        require(key in expected, f"LEGACY_KEY_OUTSIDE_PANEL:{key}")
        validate_receipt(receipt, selected=True, protocol_v2=protocol_v2)
        legacy_by_key[key] = {"path": path, "receipt": receipt, "attempt_chain": [rel(path)]}
    require(len(legacy_by_key) == 98, "LEGACY_UNIQUE_COUNT")

    v3_attempts: list[dict[str, Any]] = []
    attempts_by_key: defaultdict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for path in sorted(EVIDENCE.glob("STAGE_Z_Z2_V3*.json")):
        receipt = load(path)
        key = (str(receipt.get("model_family")), str(receipt.get("canonical_parent_key")))
        require(key in missing, f"V3_KEY_NOT_AUTHORIZED:{key}")
        selected = receipt.get("status") in VALID_STATUSES
        validate_receipt(receipt, selected=selected, protocol_v2=protocol_v2)
        error = receipt.get("error") or {}
        row = {
            "receipt_path": rel(path),
            "receipt_sha256": sha256_file(path),
            "model_family": key[0],
            "canonical_parent_key": key[1],
            "status": receipt.get("status"),
            "selected_for_model_parent": selected,
            "scientific_parent_exposure": int((receipt.get("runtime_counters") or {}).get("stage_z_scientific_parent_exposure", 0)),
            "error_type": error.get("type"),
            "error_message": error.get("message"),
        }
        v3_attempts.append(row)
        attempts_by_key[key].append({"path": path, "receipt": receipt, "row": row})

    v3_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for key in sorted(missing):
        attempts = attempts_by_key[key]
        valid = [item for item in attempts if item["row"]["selected_for_model_parent"]]
        require(len(valid) == 1, f"V3_SELECTED_ATTEMPT_COUNT:{key}:{len(valid)}")
        for item in attempts:
            if item not in valid:
                require(item["receipt"].get("status") == "ENGINEERING_INVALID_Z2_CLEAN_REFERENCE", f"V3_UNSELECTED_STATUS:{key}")
        selected = valid[0]
        v3_by_key[key] = {
            "path": selected["path"],
            "receipt": selected["receipt"],
            "attempt_chain": [item["row"]["receipt_path"] for item in attempts],
        }
    require(set(v3_by_key) == missing, "V3_MISSING_CELL_COVERAGE")

    cells: list[dict[str, Any]] = []
    for model, parent in sorted(expected):
        key = (model, parent)
        if key in legacy_by_key:
            item = legacy_by_key[key]
            provenance = "LEGACY_V1_SEMANTICALLY_RECONCILED_READ_ONLY"
        else:
            require(key in v3_by_key, f"MODEL_PARENT_UNCOVERED:{key}")
            item = v3_by_key[key]
            provenance = "V3_NEW_CLEAN_REFERENCE_EXECUTION"
        summary = receipt_summary(item["path"], item["receipt"], provenance, item["attempt_chain"])
        summary["model_parent_key"] = f"{model}|{parent}"
        cells.append(summary)

    by_identity: dict[str, dict[str, Any]] = {}
    for parent in sorted(expected_parents):
        rows = [row for row in cells if row["canonical_parent_key"] == parent]
        exposed = any(row["scientific_parent_exposure"] == 1 for row in rows)
        by_identity[parent] = {
            "canonical_parent_key": parent,
            "suite": parent.split("/", 1)[0],
            "model_parent_cells": len(rows),
            "scientific_parent_exposure": int(exposed),
            "exposing_models": [row["model_family"] for row in rows if row["scientific_parent_exposure"] == 1],
        }

    status_counts = Counter(str(row["status"]) for row in cells)
    availability_counts = Counter(str(row["anchor_availability"]) for row in cells)
    provenance_counts = Counter(str(row["provenance"]) for row in cells)
    suite_summary: dict[str, Any] = {}
    model_summary: dict[str, Any] = {}
    for grouping, output in (("suite", suite_summary), ("model_family", model_summary)):
        for value in sorted({str(row[grouping]) for row in cells}):
            rows = [row for row in cells if row[grouping] == value]
            output[value] = {
                "model_parent_cells": len(rows),
                "pass_both_anchors": sum(row["status"] == "PASS_Z2_CLEAN_REFERENCE_WITH_BOTH_ANCHORS" for row in rows),
                "abstain_no_legal_anchor": sum(row["status"] == "ABSTAIN_Z2_NO_LEGAL_ANCHOR" for row in rows),
                "scientific_parent_exposed": sum(row["scientific_parent_exposure"] == 1 for row in rows),
                "anchor_availability": dict(sorted(Counter(row["anchor_availability"] for row in rows).items())),
            }

    common_binding = {
        "repository": "Leo-6-maker/openvla-gripper-dutycycle-attack",
        "branch": git("branch", "--show-current"),
        "source_head_before_terminal_seal": git("rev-parse", "HEAD"),
        "source_tree_before_terminal_seal": git("rev-parse", "HEAD^{tree}"),
        "working_tree_status_at_generation": git("status", "--porcelain=v1").splitlines(),
        "protocol_v3": artifact(PROTOCOL_V3),
        "protocol_v2": artifact(PROTOCOL_V2),
        "panel": artifact(PANEL),
        "static_audit_v2": artifact(STATIC_V2),
        "static_audit_v3": artifact(STATIC_V3),
        "legacy_reconciliation": artifact(LEGACY_RECON),
        "runner": artifact(ROOT / "scripts/stage_z/run_stage_z_z2_clean_reference.py"),
        "anchor_selector": artifact(ROOT / "src/stage_z_preparation/anchors.py"),
        "historical_static_audit_status": static_v2.get("status"),
        "v3_static_audit_status": static_v3.get("status"),
    }

    forbidden_totals = Counter()
    for row in cells:
        for name, value in row["forbidden_counters"].items():
            forbidden_totals[name] += value
    require(all(value == 0 for value in forbidden_totals.values()), "TERMINAL_FORBIDDEN_NONZERO")
    total_inference = sum(row["model_inference_calls"] for row in cells)
    total_steps = sum(row["env_step_calls"] for row in cells)
    total_telemetry = sum(row["anchor_telemetry_reads"] for row in cells)
    identity_exposed = sum(row["scientific_parent_exposure"] for row in by_identity.values())
    model_parent_exposed = sum(row["scientific_parent_exposure"] for row in cells)

    exposure_ledger = {
        "schema": "STAGE_Z_Z2_SCIENTIFIC_EXPOSURE_LEDGER_V2",
        "status": "STAGE_Z_Z2_EXPOSURE_LEDGER_SEALED_STOP_FOR_PI",
        "claim_boundary": "Clean references and anchor availability only; no physical mechanism or attack efficacy claim.",
        "authority": common_binding,
        "population": {
            "shared_fresh_identities": len(expected_parents),
            "suite_denominators": dict(protocol_v2["population"]["suite_denominators"]),
            "model_families": list(MODELS),
            "expected_model_parent_cells": len(expected),
        },
        "coverage": {
            "covered_model_parent_cells": len(cells),
            "legacy_reconciled_cells": provenance_counts["LEGACY_V1_SEMANTICALLY_RECONCILED_READ_ONLY"],
            "v3_selected_cells": provenance_counts["V3_NEW_CLEAN_REFERENCE_EXECUTION"],
            "v3_raw_attempts": len(v3_attempts),
            "v3_engineering_invalid_attempts": sum(not row["selected_for_model_parent"] for row in v3_attempts),
            "status_counts": dict(sorted(status_counts.items())),
            "provenance_counts": dict(sorted(provenance_counts.items())),
        },
        "identity_exposure": {
            "scientific_parent_exposed": identity_exposed,
            "scientific_parent_unexposed": len(expected_parents) - identity_exposed,
            "denominator": len(expected_parents),
            "entries": list(by_identity.values()),
        },
        "model_parent_exposure": {
            "scientific_parent_exposed": model_parent_exposed,
            "scientific_parent_unexposed": len(cells) - model_parent_exposed,
            "denominator": len(cells),
            "entries": cells,
        },
        "v3_attempts": v3_attempts,
        "forbidden_execution_totals": dict(sorted(forbidden_totals.items())),
        "z3_authorized": False,
        "terminal_action": "STOP_FOR_PI",
    }

    anchor_ledger = {
        "schema": "STAGE_Z_Z2_ANCHOR_LEDGER_V2",
        "status": "STAGE_Z_Z2_ANCHOR_LEDGER_SEALED_STOP_FOR_PI",
        "claim_boundary": "Outcome-blind clean anchor snapshots only; no OPEN intervention, attacked rollout, V_phys, or success label.",
        "authority": common_binding,
        "selection_rule": protocol_v2["anchor_selection"],
        "entries": cells,
        "summary": {
            "model_parent_cells": len(cells),
            "anchor_availability": dict(sorted(availability_counts.items())),
            "suite_summary": suite_summary,
            "model_summary": model_summary,
        },
        "z3_authorized": False,
        "terminal_action": "STOP_FOR_PI",
    }

    availability_report = {
        "schema": "STAGE_Z_Z2_ANCHOR_AVAILABILITY_V2",
        "status": "STAGE_Z_Z2_ANCHOR_AVAILABILITY_AUDIT_COMPLETE_STOP_FOR_PI",
        "claim_boundary": "Descriptive clean-anchor feasibility; abstention is not a scientific negative.",
        "authority": common_binding,
        "classification_definitions": {
            "BOTH_ANCHORS_AVAILABLE": "critical anchor and noncritical control anchor are both selected",
            "CRITICAL_ONLY_NONCRITICAL_CONTROL_MISSING": "critical anchor selected but no noncritical control anchor",
            "NONCRITICAL_ONLY_CRITICAL_ANCHOR_MISSING": "noncritical control selected but no critical anchor",
            "NO_LEGAL_ANCHOR": "neither anchor is legally selected; includes pre-inference fixture ineligibility abstention",
        },
        "counts": {
            "model_parent_cells": len(cells),
            "by_availability": dict(sorted(availability_counts.items())),
            "by_status": dict(sorted(status_counts.items())),
            "by_suite": suite_summary,
            "by_model_family": model_summary,
        },
        "no_legal_anchor_reasons": dict(sorted(Counter(row["abstention_reason"] for row in cells if row["abstention_reason"]).items())),
        "entries": [
            {
                "model_parent_key": row["model_parent_key"],
                "model_family": row["model_family"],
                "canonical_parent_key": row["canonical_parent_key"],
                "status": row["status"],
                "provenance": row["provenance"],
                "anchor_availability": row["anchor_availability"],
                "critical_anchor": row["critical_anchor"],
                "noncritical_control": row["noncritical_control"],
                "abstention_reason": row["abstention_reason"],
                "scientific_parent_exposure": row["scientific_parent_exposure"],
            }
            for row in cells
        ],
        "z3_authorized": False,
        "terminal_action": "STOP_FOR_PI",
    }

    synthesis = {
        "schema": "STAGE_Z_Z2_TERMINAL_SYNTHESIS_V2",
        "status": "STAGE_Z_Z2_CLEAN_REFERENCE_AND_ANCHOR_FREEZE_COMPLETE_STOP_FOR_PI",
        "gate": "STAGE_Z_Z2_CLEAN_REFERENCE_AND_ANCHOR_FREEZE_V1",
        "authorization_source": "PR_139_PI_COMMENT_5404535968",
        "claim_boundary": "Z2 establishes clean-reference/anchor feasibility bookkeeping only. It does not test cross-model physical OPEN susceptibility.",
        "authority": common_binding,
        "matrix": {
            "expected_model_parent_cells": len(expected),
            "covered_model_parent_cells": len(cells),
            "identity_exposure": f"{identity_exposed}/{len(expected_parents)}",
            "model_parent_exposure": f"{model_parent_exposed}/{len(cells)}",
            "status_counts": dict(sorted(status_counts.items())),
            "anchor_availability": dict(sorted(availability_counts.items())),
            "suite_summary": suite_summary,
            "model_summary": model_summary,
        },
        "execution_counters": {
            "clean_model_inference_calls": total_inference,
            "clean_env_step_calls": total_steps,
            "clean_anchor_telemetry_reads": total_telemetry,
            "physical_interventions": 0,
            "pgd_calls": 0,
            "attacked_env_steps": 0,
            "vphys_reads": 0,
            "attack_outcome_reads": 0,
            "eval160_reads": 0,
            "protected_reads": 0,
            "scientific_parent_exposure": model_parent_exposed,
        },
        "scientific_interpretation": {
            "z1_runtime_action_interface_portability": "prior Z1 engineering result remains separate and accepted",
            "cross_model_physical_mechanism_tested": False,
            "cross_model_visual_pgd_tested": False,
            "clean_anchor_abstentions_are_scientific_negatives": False,
        },
        "artifact_outputs": {
            "exposure_ledger": rel(OUT_EXPOSURE),
            "anchor_ledger": rel(OUT_ANCHORS),
            "anchor_availability": rel(OUT_AVAILABILITY),
        },
        "forbidden_execution_totals": dict(sorted(forbidden_totals.items())),
        "z3_authorized": False,
        "next_legal_action": "STOP_FOR_PI",
    }

    write_json(OUT_EXPOSURE, exposure_ledger)
    write_json(OUT_ANCHORS, anchor_ledger)
    write_json(OUT_AVAILABILITY, availability_report)
    write_json(OUT_SYNTHESIS, synthesis)

    seal_entries = [
        artifact(PROTOCOL_V3), artifact(PROTOCOL_V2), artifact(PANEL), artifact(STATIC_V2), artifact(STATIC_V3), artifact(LEGACY_RECON),
        artifact(ROOT / "scripts/stage_z/run_stage_z_z2_clean_reference.py"), artifact(ROOT / "src/stage_z_preparation/anchors.py"),
        artifact(OUT_EXPOSURE), artifact(OUT_ANCHORS), artifact(OUT_AVAILABILITY), artifact(OUT_SYNTHESIS),
    ]
    seal_entries.extend({"path": row["receipt_path"], "bytes": (ROOT / row["receipt_path"]).stat().st_size, "sha256": row["receipt_sha256"]} for row in v3_attempts)
    seal_entries = sorted(seal_entries, key=lambda row: row["path"])
    seal = {
        "schema": "STAGE_Z_Z2_TERMINAL_ROOT_SEAL_V2",
        "status": "SEALED_STAGE_Z_Z2_CLEAN_REFERENCE_AND_ANCHOR_FREEZE_STOP_FOR_PI",
        "claim_boundary": synthesis["claim_boundary"],
        "git_binding": {
            "repository": common_binding["repository"],
            "branch": common_binding["branch"],
            "head_before_root_seal": common_binding["source_head_before_terminal_seal"],
            "tree_before_root_seal": common_binding["source_tree_before_terminal_seal"],
        },
        "artifact_manifest": {
            "terminal_synthesis": artifact(OUT_SYNTHESIS),
            "entries": seal_entries,
        },
        "population": {
            "shared_fresh_identities": len(expected_parents),
            "model_parent_cells": len(cells),
            "scientific_parent_exposure": f"{identity_exposed}/{len(expected_parents)} identities; {model_parent_exposed}/{len(cells)} model-parent cells",
        },
        "counters": synthesis["execution_counters"],
        "scientific_rollout_started": False,
        "z3_authorized": False,
        "next_legal_action": "STOP_FOR_PI",
    }
    write_json(OUT_SEAL, seal)
    OUT_SIDECAR.write_text(f"{sha256_file(OUT_SEAL)}  {OUT_SEAL.name}\n", encoding="utf-8")
    print(json.dumps({
        "status": synthesis["status"],
        "model_parent_cells": len(cells),
        "identity_exposure": f"{identity_exposed}/{len(expected_parents)}",
        "model_parent_exposure": f"{model_parent_exposed}/{len(cells)}",
        "status_counts": dict(sorted(status_counts.items())),
        "anchor_availability": dict(sorted(availability_counts.items())),
        "root_seal_sha256": sha256_file(OUT_SEAL),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
