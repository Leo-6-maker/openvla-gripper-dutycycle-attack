#!/usr/bin/env python3
"""Seal the final read-only AC program terminal from existing evidence."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports/STAGE_AC_AC5_TREATMENT_NAIVE_MULTI_MODEL_PROGRAM_COMPLETE_STOP_FOR_PI_V1"
STATUS = "STAGE_AC_AC5_TREATMENT_NAIVE_MULTI_MODEL_PROGRAM_COMPLETE_STOP_FOR_PI"
SOURCES = [
    "reports/STAGE_X_X1R2_F1T_TERMINAL_SYNTHESIS_V1.json",
    "reports/STAGE_Z_Z4_STATIC_CROSS_MODEL_SYNTHESIS_V1.json",
    "reports/STAGE_Z_Z4_CLAIM_BOUNDARY_V1.json",
    "reports/STAGE_AA_AA0_HISTORICAL_ENDPOINT_DENOMINATOR_AUDIT_V1.json",
    "reports/STAGE_AA_AA2_ELIGIBILITY_CONSTRUCT_VALIDITY_AUDIT_V1.json",
    "reports/STAGE_AA_AA2R2_PHASE_B_V2_CENSUS_TERMINAL_V1.json",
    "reports/STAGE_AC_AC2R3_MODEL_SPECIFIC_DENOMINATOR_LEDGER_V1.json",
    "reports/STAGE_AC_AC2R3_EVIDENCE_INTEGRITY_AUDIT_V1.json",
    "reports/STAGE_AC_AC3_G2_BRANCH_RECEIPT_INDEX_V1.json",
    "reports/STAGE_AC_AC3_G2R1_B0R1_ACTION_ONLY_AUDIT_V1.json",
    "reports/STAGE_AC_AC3_G2R1_B0R1_ROOT_SEAL_V1.json",
    "reports/STAGE_AC_AC3_G2R1_B1_TARGET_INFERENCE_ONLY_V1/AC3-65bcfd948a45dd0be9ac_INFERENCE_ONLY.json",
    "reports/STAGE_AC_AC3_G2R1_B1R1_TARGET_INFERENCE_ONLY_V1/AC3-65bcfd948a45dd0be9ac_INFERENCE_ONLY.json",
    "reports/STAGE_AC_AC3_G2R1_B1R1_TARGET_INFERENCE_ONLY_V1/STAGE_AC_AC3_G2R1_B1R1_ROOT_SEAL_V1.json",
    "reports/STAGE_AC_AC3_G2R1_B1R1_STRUCTURAL_RESEAL_V1/STAGE_AC_AC3_G2R1_B1R1_STRUCTURAL_RESEAL_V1.json",
    "reports/STAGE_AC_AC3_G2R1_G3R1_CENSORING_AWARE_STATISTICS_V2/STAGE_AC_AC3_G2R1_G3R1_CENSORING_AWARE_STATISTICS_V2.json",
    "reports/STAGE_AC_AC3_G2R1_G3R1_CENSORING_AWARE_STATISTICS_V2/STAGE_AC_AC3_G2R1_G3R1_ROOT_SEAL_V2.json",
    "reports/STAGE_AC_AC4_NEUTRAL_BLIND_MANIFEST_V1.json",
    "reports/STAGE_AC_AC4_NEUTRAL_BLIND_PACKAGE_SEAL_V1.json",
    "reports/STAGE_AC_AC4_AI_SECONDARY_LABEL_SEAL_V1.json",
    "reports/STAGE_AC_AC4_AI_SECONDARY_RECONCILIATION_V1/STAGE_AC_AC4_AI_SECONDARY_RECONCILIATION_V1.json",
    "reports/STAGE_AC_AC4_AI_SECONDARY_RECONCILIATION_V1/STAGE_AC_AC4_AI_SECONDARY_ROOT_SEAL_V1.json",
    "reports/STAGE_AC_AC5_STATIC_SYNTHESIS_V1/STAGE_AC_AC5_STATIC_SYNTHESIS_V1.json",
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


def git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def main() -> None:
    paths = [ROOT / item for item in SOURCES]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise SystemExit("AC5_FINAL_SOURCE_MISSING:" + ",".join(missing))
    x = load(paths[0])
    z = load(paths[1])
    z_claim = load(paths[2])
    aa_audit = load(paths[3])
    aa_construct = load(paths[4])
    aa_terminal = load(paths[5])
    ac2 = load(paths[6])
    ac2_integrity = load(paths[7])
    g2 = load(paths[8])
    b0 = load(paths[9])
    b0_root = load(paths[10])
    b1_old = load(paths[11])
    b1r1 = load(paths[12])
    b1r1_root = load(paths[13])
    structural = load(paths[14])
    g3 = load(paths[15])
    g3_root = load(paths[16])
    ac4_manifest = load(paths[17])
    ac4_package = load(paths[18])
    ac4_labels = load(paths[19])
    ac4 = load(paths[20])
    ac4_root = load(paths[21])
    prior_ac5 = load(paths[22])

    if ac2_integrity["counts"]["manifest_cells"] != 720 or ac2_integrity["counts"]["accepted_receipts"] != 720:
        raise SystemExit("AC5_FINAL_AC2_CENSUS")
    if len(g2.get("rows", [])) != 384 or structural["counts"]["authoritative_branches"] != 384:
        raise SystemExit("AC5_FINAL_G2_COVERAGE")
    if b1r1.get("branch_id") != "AC3-65bcfd948a45dd0be9ac" or b1r1.get("seed_bound_before_inference") is not True:
        raise SystemExit("AC5_FINAL_B1R1_SEED_BINDING")
    if b1r1["reconciliation"]["v1_rejected"] != 0 or b1r1["reconciliation"]["v2_rejected"] != 0:
        raise SystemExit("AC5_FINAL_B1R1_PATH")
    if g3["counts"]["complete_treatment_rows"] != 276 or g3["counts"]["unknown_treatment_rows"] != 12:
        raise SystemExit("AC5_FINAL_G3_COUNTS")
    if ac4["reviewer"]["human_review_gate_satisfied"] is not False or ac4["scientific_firewall"]["automatic_labels_rewritten"] != 0:
        raise SystemExit("AC5_FINAL_AC4_GOVERNANCE")

    source_authority = {
        str(path.relative_to(ROOT)).replace("\\", "/"): {"bytes": path.stat().st_size, "sha256": sha(path)}
        for path in paths
    }
    report = {
        "schema": "STAGE_AC_AC5_TREATMENT_NAIVE_MULTI_MODEL_PROGRAM_COMPLETE_V1",
        "status": STATUS,
        "gate": "STAGE_AC_AC3_G2R1_B1R1_SEED_BOUND_TARGET_FORENSIC_CONDITIONAL_RECOVERY_AND_CENSORING_AWARE_COMPLETION_V1",
        "repository": {"git_head_at_seal": git_head(), "branch": subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip()},
        "historical_layers": {
            "stage_x": {
                "artifact": "reports/STAGE_X_X1R2_F1T_TERMINAL_SYNTHESIS_V1.json",
                "status": x.get("status"),
                "safe_summary": "OpenVLA-specific visual/token execution evidence remains factorized: single-step strict realizability is not sustained selective delivery or physical efficacy.",
            },
            "stage_z": {
                "artifact": "reports/STAGE_Z_Z4_STATIC_CROSS_MODEL_SYNTHESIS_V1.json",
                "status": z.get("status"),
                "classification": z.get("classification"),
                "safe_summary": "Stage Z established shared engineering/action-interface portability but did not establish historical OpenVLA OPEN-mechanism generalization; AI-only endpoint-validity evidence remains diagnostic.",
                "claim_boundary_status": z_claim.get("status"),
            },
            "stage_aa": {
                "artifacts": [
                    "reports/STAGE_AA_AA0_HISTORICAL_ENDPOINT_DENOMINATOR_AUDIT_V1.json",
                    "reports/STAGE_AA_AA2_ELIGIBILITY_CONSTRUCT_VALIDITY_AUDIT_V1.json",
                    "reports/STAGE_AA_AA2R2_PHASE_B_V2_CENSUS_TERMINAL_V1.json",
                ],
                "statuses": [aa_audit.get("status"), aa_construct.get("status"), aa_terminal.get("status")],
                "safe_summary": "Stage AA remains historical common-parent/measurement-limited evidence; its capacity-limited denominator is not reopened or used as a multi-model physical result.",
            },
            "stage_ac": {
                "ac2_clean_census": {"cells": 720, "eligible_counts": {model: ac2["denominator"][model]["eligible_count"] for model in sorted(ac2["denominator"])}, "frozen_parents": {model: len(ac2["denominator"][model]["frozen_primary_parent_keys"]) for model in sorted(ac2["denominator"]) }},
                "ac3_physical": {"branch_terminals": 384, "complete_treatment_rows": g3["counts"]["complete_treatment_rows"], "unknown_treatment_rows": g3["counts"]["unknown_treatment_rows"], "true_horizon_unknown": g3["counts"]["true_horizon_unknown"], "action_semantics_unknown": g3["counts"]["action_semantics_unknown"]},
                "ac4_review": {"frozen_slots": ac4["counts"]["frozen_slots"], "present_videos": ac4["counts"]["present_videos"], "missing_slots": ac4["counts"]["missing_frozen_videos"], "reviewer_type": ac4["reviewer"]["type"], "human_review_gate_satisfied": False},
            },
        },
        "b0_b1_governance_disclosure": {
            "contaminated_b0": {"status": "DISCARDED_OUTCOME_FIREWALL_CONTAMINATION", "scientific_authority": False, "source_comment": "PR #140 issue comment 5447732443", "replacement": "B0R1 fresh action-only audit"},
            "noncompliant_b1": {"artifact": "reports/STAGE_AC_AC3_G2R1_B1_TARGET_INFERENCE_ONLY_V1/AC3-65bcfd948a45dd0be9ac_INFERENCE_ONLY.json", "status": b1_old.get("status"), "scientific_authority": False, "reason": "seed was not explicitly rebound before this inference-only attempt"},
            "b0r1": {"artifact": "reports/STAGE_AC_AC3_G2R1_B0R1_ACTION_ONLY_AUDIT_V1.json", "status": b0.get("status"), "root_status": b0_root.get("status"), "outcome_firewall": "PASS"},
            "b1r1": {"artifact": "reports/STAGE_AC_AC3_G2R1_B1R1_TARGET_INFERENCE_ONLY_V1/AC3-65bcfd948a45dd0be9ac_INFERENCE_ONLY.json", "status": b1r1.get("status"), "seed": 348544072, "seed_bound_before_inference": True, "model_inference_calls": 1, "env_step": 0, "open_intervention": 0, "physical_reads": 0, "v1_rejected": 0, "v2_rejected": 0, "classification": "ACTION_SEMANTICS_FAILURE_UNRESOLVED_UNKNOWN_BRANCH", "physical_recovery": "NOT_RUN"},
        },
        "scientific_synthesis": {
            "supported": "The treatment-naive AC program produced censoring-aware physical evidence consistent with a dose-ordered signal in all three qualified policy families, while preserving unknown branches and endpoint-validity limitations.",
            "not_established": ["pristine all-32 endpoint-identifiable 3-of-3 replication", "robustness or immunity of M1/M2", "human manual-review completion", "cross-model visual PGD transfer", "automatic endpoint relabeling"],
            "unknown_policy": "The 11 true horizon-censored branches and the one unresolved M1 action-semantics branch remain UNKNOWN; no success/failure imputation or denominator deletion was performed.",
            "paper_promotion": "NOT_AUTHORIZED",
        },
        "source_authority": source_authority,
        "source_roots": {"b1r1_root_status": b1r1_root.get("status"), "structural_status": structural.get("status"), "g3_root_status": g3_root.get("status"), "ac4_root_status": ac4_root.get("status"), "prior_ac5_status": prior_ac5.get("status"), "ac4_manifest_status": ac4_manifest.get("status"), "ac4_package_status": ac4_package.get("status"), "ac4_label_status": ac4_labels.get("status")},
        "scientific_firewall": {"new_model_inference_after_b1r1": 0, "new_env_step_after_b1r1": 0, "new_open_intervention_after_b1r1": 0, "new_pgd": 0, "new_protected_reads": 0, "denominator_changed": 0, "replacement_or_top_up": 0, "automatic_labels_rewritten": 0, "paper_promotion": 0},
        "next_legal_action": "STOP_FOR_PI",
    }
    OUT.mkdir(parents=True, exist_ok=False)
    report_path = OUT / "STAGE_AC_AC5_TREATMENT_NAIVE_MULTI_MODEL_PROGRAM_COMPLETE_V1.json"
    write(report_path, report)
    payload = {"schema": report["schema"], "status": STATUS, "report_bytes": report_path.stat().st_size, "report_sha256": sha(report_path), "source_count": len(paths), "scientific_firewall": report["scientific_firewall"]}
    root = {"schema": "STAGE_AC_AC5_TREATMENT_NAIVE_MULTI_MODEL_PROGRAM_ROOT_SEAL_V1", "status": STATUS, "root_payload": payload, "root_payload_sha256": hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest(), "artifacts": {"report": {"path": report_path.name, "bytes": report_path.stat().st_size, "sha256": sha(report_path)}}, "source_authority": source_authority, "next_legal_action": "STOP_FOR_PI"}
    write(OUT / "STAGE_AC_AC5_TREATMENT_NAIVE_MULTI_MODEL_PROGRAM_ROOT_SEAL_V1.json", root)
    print(json.dumps({"status": STATUS, "report_bytes": report_path.stat().st_size, "report_sha256": sha(report_path), "root_sha256": sha(OUT / "STAGE_AC_AC5_TREATMENT_NAIVE_MULTI_MODEL_PROGRAM_ROOT_SEAL_V1.json")}, sort_keys=True))


if __name__ == "__main__":
    main()
