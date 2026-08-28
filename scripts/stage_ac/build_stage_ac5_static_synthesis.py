#!/usr/bin/env python3
"""Build the read-only AC5 synthesis from sealed AC2-AC4 artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports/STAGE_AC_AC5_STATIC_SYNTHESIS_V1"
SOURCES = [
    "reports/STAGE_AC_AC2R3_MODEL_SPECIFIC_DENOMINATOR_LEDGER_V1.json",
    "reports/STAGE_AC_AC2R3_EVIDENCE_INTEGRITY_AUDIT_V1.json",
    "reports/STAGE_AC_AC3_G2_BRANCH_RECEIPT_INDEX_V1.json",
    "reports/STAGE_AC_AC3_G2R1_G3R1_CENSORING_AWARE_STATISTICS_V2/STAGE_AC_AC3_G2R1_G3R1_CENSORING_AWARE_STATISTICS_V2.json",
    "reports/STAGE_AC_AC3_G2R1_G3R1_CENSORING_AWARE_STATISTICS_V2/STAGE_AC_AC3_G2R1_G3R1_ROOT_SEAL_V2.json",
    "reports/STAGE_AC_AC4_NEUTRAL_BLIND_MANIFEST_V1.json",
    "reports/STAGE_AC_AC4_NEUTRAL_BLIND_PACKAGE_SEAL_V1.json",
    "reports/STAGE_AC_AC4_AI_SECONDARY_BLINDED_LABELS_V1.json",
    "reports/STAGE_AC_AC4_AI_SECONDARY_LABEL_SEAL_V1.json",
    "reports/STAGE_AC_AC4_AI_SECONDARY_RECONCILIATION_V1/STAGE_AC_AC4_AI_SECONDARY_RECONCILIATION_V1.json",
    "reports/STAGE_AC_AC4_AI_SECONDARY_RECONCILIATION_V1/STAGE_AC_AC4_AI_SECONDARY_ROOT_SEAL_V1.json",
]


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    paths = [ROOT / item for item in SOURCES]
    if any(not path.is_file() for path in paths):
        missing = [str(path) for path in paths if not path.is_file()]
        raise SystemExit("AC5_SOURCE_MISSING:" + ",".join(missing))
    ac2 = load(paths[0])
    ac2_integrity = load(paths[1])
    g2_index = load(paths[2])
    g3 = load(paths[3])
    g3_root = load(paths[4])
    ac4_manifest = load(paths[5])
    ac4_package_seal = load(paths[6])
    labels = load(paths[7])
    label_seal = load(paths[8])
    ac4 = load(paths[9])
    ac4_root = load(paths[10])

    denominator = ac2["denominator"]
    expected_models = ["M0_OPENVLA", "M1_OPENVLA_OFT", "M2_PI05_LIBERO"]
    if set(denominator) != set(expected_models):
        raise SystemExit("AC5_MODEL_SET")
    if ac2_integrity["counts"]["manifest_cells"] != 720 or ac2_integrity["counts"]["accepted_receipts"] != 720:
        raise SystemExit("AC5_AC2_CENSUS")
    if len(g2_index.get("rows", [])) != 384:
        raise SystemExit("AC5_G2_BRANCH_COUNT")
    if g3["status"] != "STAGE_AC_AC3_G2R1_G3R1_CENSORING_AWARE_STATISTICS_V2_COMPLETE_CONTINUE_TO_AC4":
        raise SystemExit("AC5_G3_STATUS")
    if ac4["status"] != "STAGE_AC_AC4_AI_SECONDARY_RECONCILIATION_COMPLETE_CONTINUE_TO_AC5":
        raise SystemExit("AC5_AC4_STATUS")
    if label_seal.get("human_review_gate_satisfied") is not False or label_seal.get("labels_sealed_before_unblind") is not True:
        raise SystemExit("AC5_LABEL_GOVERNANCE")

    model_summary = {}
    for model in expected_models:
        d = denominator[model]
        s = g3["model_summary"][model]
        model_summary[model] = {
            "ac2_eligible_count": d["eligible_count"],
            "ac2_frozen_parent_count": len(d["frozen_primary_parent_keys"]),
            "ac2_frozen_h0_count": d["frozen_h0_count"],
            "ac2_frozen_hc_count": d["frozen_hc_count"],
            "g3_complete_t3_t10_pair_count": s["complete_t3_t10_pair_count"],
            "g3_complete_t3_t5_t10_triplet_count": s["complete_t3_t5_t10_triplet_count"],
            "g3_triplet_patterns": s["complete_triplet_monotone_patterns"],
            "g3_nonmonotone_triplet_patterns": s["complete_triplet_nonmonotone_patterns"],
            "g3_observed_conditional_rates_t3_t5_t10": s["dose_response_diagnostic"]["observed_conditional_rates_t3_t5_t10"],
            "g3_fixed_32_lower_rates": s["dose_response_diagnostic"]["fixed_32_lower_rates"],
            "g3_fixed_32_all_unknown_upper_rates": s["dose_response_diagnostic"]["fixed_32_all_unknown_upper_rates"],
            "g3_holm_adjusted_p_value": s["paired_exact_test"]["holm_adjusted_p_value_across_models"],
            "g3_strong_model_support_gate": s["strong_model_support_gate"],
        }

    report = {
        "schema": "STAGE_AC_AC5_STATIC_SYNTHESIS_V1",
        "status": "STAGE_AC_AC5_STATIC_SYNTHESIS_COMPLETE_STOP_FOR_PI",
        "gate": "STAGE_AC_AC3_AC4_AC5_TREATMENT_NAIVE_MULTI_MODEL_PHYSICAL_REPLICATION_PROGRAM_V1",
        "execution_scope": "sealed-byte static synthesis only; no new model inference, simulator step, intervention, PGD, or protected read",
        "ac2_clean_census": {
            "cells": 720,
            "parents": 240,
            "models": 3,
            "accepted_receipts": 720,
            "model_specific_eligible_counts": {model: denominator[model]["eligible_count"] for model in expected_models},
            "frozen_parent_counts": {model: len(denominator[model]["frozen_primary_parent_keys"]) for model in expected_models},
        },
        "ac3_censoring_aware_statistics": {
            "branch_terminals": 384,
            "authoritative_treatment_rows": g3["counts"]["authoritative_treatment_rows"],
            "complete_treatment_rows": g3["counts"]["complete_treatment_rows"],
            "unknown_treatment_rows": g3["counts"]["unknown_treatment_rows"],
            "true_horizon_unknown": g3["counts"]["true_horizon_unknown"],
            "action_semantics_unknown": g3["counts"]["action_semantics_unknown"],
            "model_summary": model_summary,
            "interpretation": "All three model families show complete-case monotone T3/T5/T10 patterns and positive one-sided paired diagnostics, but fixed-denominator unknown branches prevent the strong all-32 endpoint-identifiable classification.",
        },
        "ac4_endpoint_validity": {
            "frozen_slots": ac4["counts"]["frozen_slots"],
            "present_videos": ac4["counts"]["present_videos"],
            "missing_frozen_videos": ac4["counts"]["missing_frozen_videos"],
            "ai_secondary_label_distribution": ac4["label_distribution"],
            "human_review_gate_satisfied": False,
            "automatic_labels_rewritten": 0,
            "interpretation": "The AI-only blinded audit is endpoint-validity evidence, not preregistered human review; ambiguous/not-identifiable visual evidence remains substantial and automatic endpoint labels are unchanged.",
        },
        "claim_boundary": {
            "supported": [
                "AC2 clean-only model-specific denominators are frozen at 32/32/32 from the complete 720-cell census.",
                "AC3 contains censoring-aware physical branch evidence with dose-ordered complete-case patterns in all three model families.",
                "The stronger pristine cross-model replication claim is limited by 12 unknown treatment branches and endpoint-validity evidence that is AI-secondary only.",
            ],
            "not_supported": [
                "pristine all-32 endpoint-identifiable 3-of-3 replication",
                "robustness or immunity of M1/M2",
                "human-review completion",
                "Paper V2 scientific promotion",
            ],
            "safe_synthesis": "Censoring-aware, endpoint-limited physical evidence is consistent with a shared dose-ordered signal across the three qualified policy families, while the stronger preregistered cross-model replication claim remains unestablished.",
        },
        "scientific_firewall": {
            "new_model_inference": 0,
            "new_env_step": 0,
            "new_open_intervention": 0,
            "new_pgd": 0,
            "new_protected_reads": 0,
            "denominator_changed": 0,
            "automatic_labels_rewritten": 0,
            "replacement_or_top_up": 0,
            "paper_promotion": 0,
        },
        "source_authority": {str(path.relative_to(ROOT)).replace("\\", "/"): {"bytes": path.stat().st_size, "sha256": sha(path)} for path in paths},
        "root_inputs": {
            "g3_root_seal_sha256": sha(paths[4]),
            "ac4_root_seal_sha256": sha(paths[10]),
            "ac4_package_seal_sha256": sha(paths[6]),
            "label_seal_sha256": sha(paths[8]),
            "package_manifest_status": ac4_manifest["status"],
            "package_seal_status": ac4_package_seal["status"],
            "ac4_root_status": ac4_root["status"],
            "g3_root_status": g3_root["status"],
            "labels_status": labels["status"],
        },
        "next_legal_action": "STOP_FOR_PI",
    }
    OUT.mkdir(parents=True, exist_ok=False)
    report_path = OUT / "STAGE_AC_AC5_STATIC_SYNTHESIS_V1.json"
    write(report_path, report)
    payload = {
        "schema": report["schema"],
        "status": report["status"],
        "report_bytes": report_path.stat().st_size,
        "report_sha256": sha(report_path),
        "source_count": len(paths),
        "scientific_firewall": report["scientific_firewall"],
    }
    root = {
        "schema": "STAGE_AC_AC5_ROOT_SEAL_V1",
        "status": report["status"],
        "root_payload": payload,
        "root_payload_sha256": hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "artifacts": {"report": {"path": report_path.name, "bytes": report_path.stat().st_size, "sha256": sha(report_path)}},
        "source_authority": report["source_authority"],
        "claim_boundary": report["claim_boundary"],
        "next_legal_action": "STOP_FOR_PI",
    }
    write(OUT / "STAGE_AC_AC5_ROOT_SEAL_V1.json", root)
    print(json.dumps({"status": report["status"], "report_bytes": report_path.stat().st_size, "report_sha256": sha(report_path), "root_sha256": sha(OUT / "STAGE_AC_AC5_ROOT_SEAL_V1.json")}, sort_keys=True))


if __name__ == "__main__":
    main()
