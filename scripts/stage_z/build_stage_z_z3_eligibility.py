#!/usr/bin/env python3
"""Reconcile the frozen Z2 and Z2R1 clean-anchor populations for Z3."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
Z2_ROOT = ROOT / "reports/STAGE_Z_Z2_TERMINAL_ROOT_SEAL_V2.json"
Z2_AVAILABILITY = ROOT / "reports/STAGE_Z_Z2_ANCHOR_AVAILABILITY_V2.json"
Z2_LEDGER = ROOT / "reports/STAGE_Z_Z2_ANCHOR_LEDGER_V2.json"
Z2R1_ROOT = ROOT / "reports/STAGE_Z_Z2R1_M2_CLEAN_REPAIR_ROOT_SEAL_V1.json"
Z2R1_SYNTHESIS = ROOT / "reports/STAGE_Z_Z2R1_M2_CLEAN_REPAIR_TERMINAL_SYNTHESIS_V1.json"
Z2R1_LEDGER = ROOT / "reports/STAGE_Z_Z2R1_M2_CLEAN_REPAIR_RECEIPT_LEDGER_V1.json"
PANEL = ROOT / "reports/STAGE_Z_Z0R1_SHARED_36_IDENTITY_PANEL_V1.json"
OUT = ROOT / "reports/STAGE_Z_Z3_ELIGIBILITY_RECONCILIATION_V1.json"

EXPECTED_Z2_ROOT = "e37659a552bea7665fbfcc7a52e8fa8131e29aef6613a197a7087ab8d7cf4c6f"
EXPECTED_Z2R1_ROOT = "2e98aba1826f0492dc6080767a00502b0372434191d74149d81177f97241e9f9"
MODELS = ("M0_OPENVLA", "M1_OPENVLA_OFT", "M2_PI05_LIBERO")


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def artifact(path: Path) -> dict[str, Any]:
    return {"path": rel(path), "bytes": path.stat().st_size, "sha256": sha(path)}


def root_artifact(root: dict[str, Any], path: str) -> dict[str, Any]:
    manifest = root.get("artifact_manifest", [])
    entries = manifest.get("entries", []) if isinstance(manifest, dict) else manifest
    match = [row for row in entries if isinstance(row, dict) and row.get("path") == path]
    require(len(match) == 1, f"ROOT_ARTIFACT_MISSING:{path}")
    return match[0]


def anchor(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    return {key: value.get(key) for key in ("anchor_class", "phase", "step", "state_sha256", "rank_digest", "status")}


def normalize(model: str, row: dict[str, Any], source: str) -> dict[str, Any]:
    parent = str(row.get("canonical_parent_key"))
    return {
        "model_family": model,
        "canonical_parent_key": parent,
        "model_parent_key": f"{model}|{parent}",
        "suite": str(row.get("suite") or parent.split("/", 1)[0]),
        "task_idx": int(row.get("task_idx", int(parent.split("/task_", 1)[1].split("/", 1)[0]))),
        "state_id": int(row.get("state_id", int(parent.rsplit("_", 1)[1]))),
        "anchor_availability": str(row.get("anchor_availability")),
        "status": str(row.get("status")),
        "receipt_path": row.get("receipt_path"),
        "receipt_sha256": row.get("receipt_sha256"),
        "critical_anchor": anchor(row.get("critical_anchor")),
        "noncritical_anchor": anchor(row.get("noncritical_anchor") or row.get("noncritical_control")),
        "scientific_parent_exposure": int(row.get("scientific_parent_exposure", 1)),
        "source": source,
    }


def main() -> None:
    require(sha(Z2_ROOT) == EXPECTED_Z2_ROOT, "Z2_ROOT_SHA256")
    require(sha(Z2R1_ROOT) == EXPECTED_Z2R1_ROOT, "Z2R1_ROOT_SHA256")
    z2_root = load(Z2_ROOT)
    z2r1_root = load(Z2R1_ROOT)
    z2_availability = load(Z2_AVAILABILITY)
    z2_ledger = load(Z2_LEDGER)
    z2r1_synthesis = load(Z2R1_SYNTHESIS)
    z2r1_ledger = load(Z2R1_LEDGER)
    panel = load(PANEL)

    require(z2_root.get("status") == "SEALED_STAGE_Z_Z2_CLEAN_REFERENCE_AND_ANCHOR_FREEZE_STOP_FOR_PI", "Z2_STATUS")
    require(z2r1_root.get("status") == "SEALED_STAGE_Z_Z2R1_M2_CLEAN_REPAIR_STOP_FOR_PI", "Z2R1_STATUS")
    require(root_artifact(z2_root, rel(Z2_LEDGER))["sha256"] == sha(Z2_LEDGER), "Z2_LEDGER_SEAL")
    require(root_artifact(z2_root, rel(Z2_AVAILABILITY))["sha256"] == sha(Z2_AVAILABILITY), "Z2_AVAILABILITY_SEAL")
    require(root_artifact(z2r1_root, rel(Z2R1_LEDGER))["sha256"] == sha(Z2R1_LEDGER), "Z2R1_LEDGER_SEAL")
    require(z2r1_synthesis.get("matrix", {}).get("both_anchors") == 34, "Z2R1_BOTH_COUNT")
    require(len(panel.get("selected_parent_keys", [])) == 36, "PANEL_36")
    require(len(z2_ledger.get("entries", [])) == 108, "Z2_108_ENTRIES")
    require(len(z2r1_ledger.get("cells", [])) == 34, "Z2R1_34_ENTRIES")

    rows: list[dict[str, Any]] = []
    structural_m2 = {"libero_goal/task_00/state_21", "libero_goal/task_07/state_30"}
    for row in z2_ledger["entries"]:
        model = str(row.get("model_family"))
        if model in ("M0_OPENVLA", "M1_OPENVLA_OFT") or (model == "M2_PI05_LIBERO" and row.get("canonical_parent_key") in structural_m2):
            rows.append(normalize(model, row, "Z2_V2_SEALED_CLEAN_ANCHOR_LEDGER"))
    for row in z2r1_ledger["cells"]:
        rows.append(normalize("M2_PI05_LIBERO", row, "Z2R1_M2_SEALED_CLEAN_REPAIR_LEDGER"))
    require(len(rows) == 108, "COMBINED_108_ROWS")
    keys = [(row["model_family"], row["canonical_parent_key"]) for row in rows]
    require(len(set(keys)) == 108, "DUPLICATE_MODEL_PARENT")
    require(set(model for model, _ in keys) == set(MODELS), "MODEL_SET")

    by_model = {model: [row for row in rows if row["model_family"] == model] for model in MODELS}
    expected_counts = {
        "M0_OPENVLA": {"BOTH_ANCHORS_AVAILABLE": 32, "NONCRITICAL_ONLY_CRITICAL_ANCHOR_MISSING": 2, "NO_LEGAL_ANCHOR": 2},
        "M1_OPENVLA_OFT": {"BOTH_ANCHORS_AVAILABLE": 26, "NONCRITICAL_ONLY_CRITICAL_ANCHOR_MISSING": 8, "NO_LEGAL_ANCHOR": 2},
        "M2_PI05_LIBERO": {"BOTH_ANCHORS_AVAILABLE": 34, "NO_LEGAL_ANCHOR": 2},
    }
    for model, expected in expected_counts.items():
        actual = dict(Counter(row["anchor_availability"] for row in by_model[model]))
        require(actual == expected, f"MODEL_COUNT:{model}:{actual}")
        for row in by_model[model]:
            if row["anchor_availability"] == "BOTH_ANCHORS_AVAILABLE":
                require(row["critical_anchor"] is not None and row["noncritical_anchor"] is not None, f"ANCHOR_HASHES:{row['model_parent_key']}")
                require(row["critical_anchor"].get("step") is not None and row["critical_anchor"].get("state_sha256"), f"CRITICAL_HASH:{row['model_parent_key']}")
                require(row["noncritical_anchor"].get("step") is not None and row["noncritical_anchor"].get("state_sha256"), f"NONCRITICAL_HASH:{row['model_parent_key']}")

    eligible = [row for row in rows if row["anchor_availability"] == "BOTH_ANCHORS_AVAILABLE"]
    critical_missing = [row for row in rows if row["anchor_availability"] == "NONCRITICAL_ONLY_CRITICAL_ANCHOR_MISSING"]
    structural = [row for row in rows if row["anchor_availability"] == "NO_LEGAL_ANCHOR"]
    require(len(eligible) == 92 and len(critical_missing) == 10 and len(structural) == 6, "92_10_6_ARITHMETIC")

    both_by_model = {model: {row["canonical_parent_key"] for row in by_model[model] if row["anchor_availability"] == "BOTH_ANCHORS_AVAILABLE"} for model in MODELS}
    intersection_keys = sorted(set.intersection(*(both_by_model[model] for model in MODELS)))
    require(len(intersection_keys) == 24, "INTERSECTION_24")
    intersection = []
    for parent in intersection_keys:
        intersection.append({
            "canonical_parent_key": parent,
            "suite": parent.split("/", 1)[0],
            "model_parent_rows": [next(row for row in eligible if row["model_family"] == model and row["canonical_parent_key"] == parent) for model in MODELS],
        })

    output = {
        "schema": "STAGE_Z_Z3_ELIGIBILITY_RECONCILIATION_V1",
        "status": "STAGE_Z_Z3_ELIGIBILITY_RECONCILIATION_PASS",
        "authorization_comment_id": None,
        "authorization_source": "PI_INDEPENDENT_AUDIT_Z2R1_ACCEPTED_Z3_AUTHORIZATION_ATTACHMENT_8d20f947-f10b-4f94-b1f8-79a9a9f75532",
        "claim_boundary": "Eligibility and anchor bookkeeping only; no OPEN intervention, attacked env.step, physical endpoint, V_phys, outcome, or protected read.",
        "historical_roots": {"z2_v2": artifact(Z2_ROOT), "z2r1": artifact(Z2R1_ROOT)},
        "source_artifacts": {"panel": artifact(PANEL), "z2_availability": artifact(Z2_AVAILABILITY), "z2_ledger": artifact(Z2_LEDGER), "z2r1_synthesis": artifact(Z2R1_SYNTHESIS), "z2r1_ledger": artifact(Z2R1_LEDGER)},
        "arithmetic": {"model_parent_pairs": 108, "complete_five_arm_eligible": 92, "critical_anchor_missing": 10, "structural_model_parent_abstentions": 6, "fixed_matrix_branches": 460},
        "model_counts": {model: dict(Counter(row["anchor_availability"] for row in by_model[model])) for model in MODELS},
        "primary_eligible_model_parent_pairs": eligible,
        "fixed_incomplete_model_parent_pairs": {"critical_anchor_missing": critical_missing, "structural_abstentions": structural},
        "all_three_both_anchor_intersection": {"count": len(intersection), "use": "SECONDARY_PAIRED_CROSS_MODEL_SUBSET_ONLY", "rows": intersection},
        "selection_contract": {"reranking": False, "replacement": False, "top_up": False, "outcome_inputs": [], "primary_analysis_keeps_all_eligible_model_parent_pairs": True},
        "forbidden_counters": {"open_interventions": 0, "attacked_env_steps": 0, "v_phys_reads": 0, "attack_outcome_reads": 0, "protected_reads": 0, "eval160_reads": 0, "pgd_calls": 0},
        "next_legal_action": "Z3_STATIC_SOURCE_FREEZE",
    }
    OUT.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": output["status"], "eligible": len(eligible), "critical_missing": len(critical_missing), "structural": len(structural), "intersection": len(intersection)}, sort_keys=True))


if __name__ == "__main__":
    main()
