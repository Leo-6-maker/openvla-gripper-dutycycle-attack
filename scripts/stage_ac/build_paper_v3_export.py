#!/usr/bin/env python3
"""Build the paper-safe, static Stage AC export and closeout seal."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EXPORT_DIR = ROOT / "reports/STAGE_AC_PAPER_V3_EXPORT_V1"
EXPORT_PATH = EXPORT_DIR / "STAGE_AC_PAPER_V3_EVIDENCE_EXPORT_V1.json"
ROOT_PATH = EXPORT_DIR / "STAGE_AC_PAPER_V3_EXPORT_ROOT_SEAL_V1.json"
CLOSEOUT_PATH = EXPORT_DIR / "STAGE_AC_CODE_AND_PAPER_EXPORT_CLOSEOUT_V1.json"

MODEL_ORDER = ("M0_OPENVLA", "M1_OPENVLA_OFT", "M2_PI05_LIBERO")
DOSES = (3, 5, 10)
PATTERNS = ("000", "001", "011", "111")


SOURCE_SPECS = {
    "protocol": (
        "configs/STAGE_AC_AC3_AC4_AC5_PROGRAM_PROTOCOL_V1.json",
        "frozen Stage AC protocol",
    ),
    "ac0_construct_terminal": (
        "reports/STAGE_AC_AC0_CONSTRUCT_VALIDATION_TERMINAL_V1.json",
        "consumed-only construct validation",
    ),
    "ac1r2_population": (
        "reports/STAGE_AC_AC1R2_TREATMENT_NAIVE_POPULATION_V1.json",
        "treatment-naive population authority",
    ),
    "ac2_launch_manifest": (
        "reports/STAGE_AC_AC2_CLEAN_SCREEN_LAUNCH_MANIFEST_V1.json",
        "720-cell clean-only launch manifest",
    ),
    "ac2r3_denominator": (
        "reports/STAGE_AC_AC2R3_MODEL_SPECIFIC_DENOMINATOR_LEDGER_V1.json",
        "model-specific clean denominator ledger",
    ),
    "ac2r3_evidence_integrity": (
        "reports/STAGE_AC_AC2R3_EVIDENCE_INTEGRITY_AUDIT_V1.json",
        "clean evidence integrity audit",
    ),
    "ac2r3_telemetry_parity": (
        "reports/STAGE_AC_AC2R3_TELEMETRY_SEMANTIC_PARITY_AUDIT_V1.json",
        "telemetry semantic parity audit",
    ),
    "ac2r3_root": (
        "reports/STAGE_AC_AC2R3_ROOT_SEAL_V1.json",
        "AC2R3 root seal",
    ),
    "ac3_terminal": (
        "reports/STAGE_AC_AC3_G2_TERMINAL_V1.json",
        "384-branch physical execution terminal",
    ),
    "ac3_branch_index": (
        "reports/STAGE_AC_AC3_G2_BRANCH_RECEIPT_INDEX_V1.json",
        "AC3 branch receipt index",
    ),
    "g3r1_statistics": (
        "reports/STAGE_AC_AC3_G2R1_G3R1_CENSORING_AWARE_STATISTICS_V2/STAGE_AC_AC3_G2R1_G3R1_CENSORING_AWARE_STATISTICS_V2.json",
        "censoring-aware dose statistics",
    ),
    "g3r1_root": (
        "reports/STAGE_AC_AC3_G2R1_G3R1_CENSORING_AWARE_STATISTICS_V2/STAGE_AC_AC3_G2R1_G3R1_ROOT_SEAL_V2.json",
        "G3R1 root seal",
    ),
    "ac4_sample": (
        "reports/STAGE_AC_AC4_BLIND_AUDIT_SAMPLE_V1.json",
        "frozen neutral endpoint-audit sample",
    ),
    "ac4_manifest": (
        "reports/STAGE_AC_AC4_NEUTRAL_BLIND_MANIFEST_V1.json",
        "neutral blind manifest",
    ),
    "ac4_package_seal": (
        "reports/STAGE_AC_AC4_NEUTRAL_BLIND_PACKAGE_SEAL_V1.json",
        "neutral blind package seal",
    ),
    "ac4_single_ai_labels": (
        "reports/STAGE_AC_AC4_AI_SECONDARY_BLINDED_LABELS_V1.json",
        "historical single-AI secondary labels",
    ),
    "ac4_single_ai_label_seal": (
        "reports/STAGE_AC_AC4_AI_SECONDARY_LABEL_SEAL_V1.json",
        "historical single-AI label seal",
    ),
    "ac4_single_ai_reconciliation": (
        "reports/STAGE_AC_AC4_AI_SECONDARY_RECONCILIATION_V1/STAGE_AC_AC4_AI_SECONDARY_RECONCILIATION_V1.json",
        "historical single-AI reconciliation",
    ),
    "ac4_single_ai_root": (
        "reports/STAGE_AC_AC4_AI_SECONDARY_RECONCILIATION_V1/STAGE_AC_AC4_AI_SECONDARY_ROOT_SEAL_V1.json",
        "historical single-AI root seal",
    ),
    "ac4_panel_input": (
        "reports/STAGE_AC_AC4_THREE_AGENT_BLINDED_AI_ADJUDICATION_INPUT_V1.txt",
        "three-agent blinded input",
    ),
    "ac4_panel_labels": (
        "reports/STAGE_AC_AC4_THREE_AGENT_BLINDED_AI_ADJUDICATION_LABELS_V1.json",
        "three-agent exact sealed labels",
    ),
    "ac4_panel_label_seal": (
        "reports/STAGE_AC_AC4_THREE_AGENT_BLINDED_AI_ADJUDICATION_LABEL_SEAL_V1.json",
        "three-agent label seal",
    ),
    "ac4_panel_reconciliation": (
        "reports/STAGE_AC_AC4_THREE_AGENT_AI_ADJUDICATION_RECONCILIATION_V1/STAGE_AC_AC4_THREE_AGENT_AI_ADJUDICATION_RECONCILIATION_V1.json",
        "three-agent unblind reconciliation",
    ),
    "ac4_panel_root": (
        "reports/STAGE_AC_AC4_THREE_AGENT_AI_ADJUDICATION_RECONCILIATION_V1/STAGE_AC_AC4_THREE_AGENT_AI_ADJUDICATION_ROOT_SEAL_V1.json",
        "three-agent root seal",
    ),
    "ac5_static_synthesis": (
        "reports/STAGE_AC_AC5_STATIC_SYNTHESIS_V1/STAGE_AC_AC5_STATIC_SYNTHESIS_V1.json",
        "AC5 static synthesis",
    ),
    "ac5_terminal": (
        "reports/STAGE_AC_AC5_TREATMENT_NAIVE_MULTI_MODEL_PROGRAM_COMPLETE_STOP_FOR_PI_V1/STAGE_AC_AC5_TREATMENT_NAIVE_MULTI_MODEL_PROGRAM_COMPLETE_V1.json",
        "AC5 final program terminal",
    ),
    "ac5_root": (
        "reports/STAGE_AC_AC5_TREATMENT_NAIVE_MULTI_MODEL_PROGRAM_COMPLETE_STOP_FOR_PI_V1/STAGE_AC_AC5_TREATMENT_NAIVE_MULTI_MODEL_PROGRAM_ROOT_SEAL_V1.json",
        "AC5 root seal",
    ),
}


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n").encode("utf-8")


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(f"PAPER_V3_EXPORT_FAIL:{message}")


def git_value(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def file_ref(path: Path, role: str) -> dict[str, object]:
    require(path.is_file(), f"missing_source:{rel(path)}")
    payload = path.read_bytes()
    return {
        "path": rel(path),
        "bytes": len(payload),
        "sha256": sha256_bytes(payload),
        "role": role,
    }


def load_json(path: Path) -> dict:
    require(path.is_file(), f"missing_json:{rel(path)}")
    return json.loads(path.read_text(encoding="utf-8"))


def zero_firewall(firewall: dict) -> bool:
    return all(value == 0 for value in firewall.values())


def validate_ac5_root(ac5_root: dict) -> dict[str, dict[str, object]]:
    require(ac5_root["status"] == "STAGE_AC_AC5_TREATMENT_NAIVE_MULTI_MODEL_PROGRAM_COMPLETE_STOP_FOR_PI", "ac5_root_status")
    records = {}
    failures = []
    for source_path, expected in ac5_root["source_authority"].items():
        path = ROOT / source_path.replace("\\", "/")
        actual = file_ref(path, "AC5 root source authority")
        records[source_path] = actual
        if actual["bytes"] != expected["bytes"] or actual["sha256"] != expected["sha256"]:
            failures.append(source_path)
    require(not failures, "ac5_root_source_mismatch:" + ",".join(failures))
    return records


def build_model_export(g3r1: dict) -> dict[str, dict]:
    result = {}
    for model in MODEL_ORDER:
        summary = g3r1["model_summary"][model]
        dose_rows = []
        for dose in DOSES:
            row = summary["by_dose"][str(dose)]
            dose_rows.append(
                {
                    "dose": dose,
                    "condition": row["condition"],
                    "events": row["events"],
                    "non_events": row["non_events"],
                    "unknown_parent_count": row["unknown_parent_count"],
                    "complete_parent_count": row["complete_parent_count"],
                    "fixed_parent_count": row["fixed_parent_count"],
                    "observed_conditional_rate": row["conditional_rate_observed"],
                    "fixed_32_lower_rate": row["p_lower_fixed_32"],
                    "fixed_32_upper_rate_all_unknown": row["p_upper_all_unknown_fixed_32"],
                    "conditional_95pct_ci": row["exact_binomial_95pct_ci_conditional_observed"],
                }
            )
        patterns = {pattern: summary["complete_t3_t5_t10_patterns"].get(pattern, 0) for pattern in PATTERNS}
        diagnostic = summary["dose_response_diagnostic"]
        result[model] = {
            "fixed_parent_count": summary["fixed_parent_count"],
            "ac2_pair_floor_24_pass": summary["pair_floor_24_pass"],
            "ac2_triplet_floor_24_pass": summary["triplet_floor_24_pass"],
            "complete_t3_t10_pair_count": summary["complete_t3_t10_pair_count"],
            "complete_t3_t5_t10_triplet_count": summary["complete_t3_t5_t10_triplet_count"],
            "dose_rows": dose_rows,
            "triplet_patterns": patterns,
            "nonmonotone_triplet_patterns": summary["complete_triplet_nonmonotone_patterns"],
            "observed_conditional_monotone": diagnostic["observed_conditional_monotone"],
            "observed_t10_minus_t3": diagnostic["t10_minus_t3_observed_conditional"],
            "fixed_32_t10_minus_t3_interval_all_unknown": diagnostic["t10_minus_t3_all_unknown_interval"],
            "paired_exact_test": {
                "test": summary["paired_exact_test"]["test"],
                "p_value": summary["paired_exact_test"]["p_value"],
                "holm_adjusted_p_value": summary["paired_exact_test"]["holm_adjusted_p_value_across_models"],
                "discordant_total": summary["paired_exact_test"]["discordant_total"],
            },
            "strong_model_support_gate": summary["strong_model_support_gate"],
        }
    return result


def build_panel_export(panel: dict) -> dict:
    rows = panel["joined_rows"]
    require(len(rows) == 91, "panel_joined_rows")
    by_model: dict[str, list] = defaultdict(list)
    overall_agent = {}
    for row in rows:
        by_model[row["model_family"]].append(row)
    for agent_key in ("agent_a_label", "agent_b_label", "agent_c_label"):
        overall_agent[agent_key.upper()] = dict(sorted(Counter(row[agent_key] for row in rows).items()))

    model_summary = {}
    confusion = {}
    for model in MODEL_ORDER:
        model_rows = by_model[model]
        majority = Counter(row["consensus_or_majority_label"] or "NO_CONSENSUS" for row in model_rows)
        auto = Counter(row["automatic_v_phys_label"] for row in model_rows)
        cross = defaultdict(Counter)
        for row in model_rows:
            cross[row["automatic_v_phys_label"]][row["consensus_or_majority_label"] or "NO_CONSENSUS"] += 1
        model_summary[model] = {
            "rows": len(model_rows),
            "majority_or_consensus_labels": dict(sorted(majority.items())),
            "automatic_v_phys_labels": dict(sorted(auto.items())),
        }
        confusion[model] = {auto_label: dict(sorted(counts.items())) for auto_label, counts in sorted(cross.items())}

    agreement = panel["agreement"]
    governance = panel["reviewer_governance"]
    firewall = panel["scientific_firewall"]
    require(governance["human_review_gate_satisfied"] is False, "human_gate_must_remain_false")
    require(governance["formal_human_review_claim"] is False, "human_claim_must_remain_false")
    require(governance["agent_sessions_mapping_exposure"] is False, "panel_mapping_exposure")
    require(zero_firewall(firewall), "panel_firewall")
    return {
        "frozen_slots": panel["counts"]["frozen_slots"],
        "present_videos": panel["counts"]["present_videos"],
        "missing_frozen_videos": panel["counts"]["missing_frozen_videos"],
        "rows_per_agent": panel["counts"]["label_rows_per_agent"],
        "agent_count": governance["agent_count"],
        "agent_marginal_distributions": agreement["agent_marginal_distributions"],
        "consensus_or_majority_distribution": agreement["consensus_or_majority_distribution"],
        "agreement_classes": {
            "unanimous_3_of_3": agreement["unanimous_3_of_3"],
            "majority_2_of_3": agreement["majority_2_of_3"],
            "disagreement_1_1_1": agreement["disagreement_1_1_1"],
            "fleiss_kappa_nominal": agreement["fleiss_kappa_nominal"]["fleiss_kappa"],
            "mean_pairwise_agreement_rate": agreement["mean_pairwise_agreement_rate"],
        },
        "model_summary": model_summary,
        "automatic_v_phys_to_video_majority": confusion,
        "governance": {
            "reviewer_type": governance["reviewer_type"],
            "human_review_gate_satisfied": governance["human_review_gate_satisfied"],
            "formal_human_review_claim": governance["formal_human_review_claim"],
            "labels_sealed_before_unblind": governance["labels_sealed_before_unblind"],
            "agent_sessions_mapping_exposure": governance["agent_sessions_mapping_exposure"],
            "orchestrator_prior_mapping_exposure_before_this_panel": governance["orchestrator_prior_mapping_exposure_before_this_panel"],
        },
        "scientific_firewall": firewall,
        "overall_agent_label_counts": overall_agent,
    }


def build_export() -> tuple[dict, dict[str, dict[str, object]]]:
    source_head = git_value("rev-parse", "HEAD")
    source_tree = git_value("rev-parse", "HEAD^{tree}")
    remote = git_value("remote", "get-url", "origin")

    loaded = {key: load_json(ROOT / spec[0]) for key, spec in SOURCE_SPECS.items() if spec[0].endswith(".json")}
    ac5_root_sources = validate_ac5_root(loaded["ac5_root"])
    source_records = {
        key: file_ref(ROOT / path, role)
        for key, (path, role) in SOURCE_SPECS.items()
    }
    source_records.update({f"ac5_root_source_{index:02d}": value for index, value in enumerate(ac5_root_sources.values())})

    ac5 = loaded["ac5_terminal"]
    ac5_static = loaded["ac5_static_synthesis"]
    g3r1 = loaded["g3r1_statistics"]
    panel = loaded["ac4_panel_reconciliation"]
    require(ac5["status"] == "STAGE_AC_AC5_TREATMENT_NAIVE_MULTI_MODEL_PROGRAM_COMPLETE_STOP_FOR_PI", "ac5_status")
    require(ac5["next_legal_action"] == "STOP_FOR_PI", "ac5_next_action")
    require(zero_firewall(ac5["scientific_firewall"]), "ac5_firewall")
    require(g3r1["status"] == "STAGE_AC_AC3_G2R1_G3R1_CENSORING_AWARE_STATISTICS_V2_COMPLETE_CONTINUE_TO_AC4", "g3r1_status")
    require(g3r1["counts"] == {
        "action_semantics_unknown": 1,
        "authoritative_treatment_rows": 288,
        "complete_treatment_rows": 276,
        "fixed_parents_per_model": 32,
        "models": 3,
        "new_execution": 0,
        "true_horizon_unknown": 11,
        "unknown_treatment_rows": 12,
    }, "g3r1_counts")

    export = {
        "schema": "STAGE_AC_PAPER_V3_EVIDENCE_EXPORT_V1",
        "status": "STAGE_AC_PAPER_V3_EXPORT_READY_STOP_FOR_PI",
        "source_repository": {
            "repository": "Leo-6-maker/openvla-gripper-dutycycle-attack",
            "remote": remote,
            "branch": git_value("branch", "--show-current"),
            "source_head_at_export": source_head,
            "source_tree_at_export": source_tree,
        },
        "program_terminal": {
            "stage_ac_status": ac5["status"],
            "next_legal_action": ac5["next_legal_action"],
            "paper_promotion": ac5["scientific_synthesis"]["paper_promotion"],
        },
        "ac2_clean_census": ac5_static["ac2_clean_census"],
        "ac3_physical_statistics": {
            "branch_terminals": ac5_static["ac3_censoring_aware_statistics"]["branch_terminals"],
            "authoritative_treatment_rows": g3r1["counts"]["authoritative_treatment_rows"],
            "complete_treatment_rows": g3r1["counts"]["complete_treatment_rows"],
            "fixed_parents_per_model": g3r1["counts"]["fixed_parents_per_model"],
            "true_horizon_unknown": g3r1["counts"]["true_horizon_unknown"],
            "action_semantics_unknown": g3r1["counts"]["action_semantics_unknown"],
            "unknown_treatment_rows": g3r1["counts"]["unknown_treatment_rows"],
            "models": build_model_export(g3r1),
        },
        "ac4_endpoint_observability": build_panel_export(panel),
        "historical_program_layers": {
            "stage_aa": ac5["historical_layers"]["stage_aa"],
            "stage_x": ac5["historical_layers"]["stage_x"],
            "stage_z": ac5["historical_layers"]["stage_z"],
        },
        "source_artifacts": source_records,
        "source_authority_validation": {
            "ac5_root_source_count": len(ac5_root_sources),
            "ac5_root_source_mismatches": 0,
            "all_listed_sources_present_and_hashed": True,
        },
        "claim_contract": {
            "headline": "Three VLA policy families support a dose-ordered command-level gripper-OPEN physical mechanism, while timing, visual selective realizability, sustained delivery, and endpoint observability remain independent evidence layers.",
            "supported": [
                "In the treatment-naive AC program, all three qualified policy families show observed conditional T3/T5/T10 rates that are dose-monotone under the sealed automatic physical endpoint.",
                "The complete-case triplet patterns are monotone in each model family, with UNKNOWN branches retained in the fixed denominator.",
                "The three-agent endpoint audit is supplemental AI-only evidence and exposes limited video-only observability relative to privileged automatic telemetry.",
            ],
            "not_established": [
                "pristine all-32 endpoint-identifiable 3-of-3 replication",
                "cross-model visual PGD transfer or visual physical efficacy",
                "human manual-review completion",
                "robustness, immunity, or a vulnerability ranking between model families",
            ],
            "prohibited_claims": [
                "all VLAs are vulnerable",
                "PGD transferred to three models",
                "3/3 pristine replication",
                "human review confirmed endpoints",
                "OFT is more vulnerable than OpenVLA",
            ],
        },
        "scientific_firewall": {
            "new_model_inference": 0,
            "new_env_step": 0,
            "new_open_intervention": 0,
            "new_pgd": 0,
            "new_physical_outcome_reads": 0,
            "new_protected_reads": 0,
            "automatic_labels_rewritten": 0,
            "denominator_changed": 0,
            "replacement_or_top_up": 0,
        },
        "next_legal_action": "STOP_FOR_PI",
    }
    return export, source_records


def write_outputs() -> None:
    export, source_records = build_export()
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    export_bytes = json_bytes(export)
    EXPORT_PATH.write_bytes(export_bytes)

    root_payload = {
        "schema": "STAGE_AC_PAPER_V3_EXPORT_ROOT_SEAL_V1",
        "status": "STAGE_AC_CODE_AND_PAPER_EXPORT_CLOSEOUT_COMPLETE_STOP_FOR_PI",
        "source_head_at_export": export["source_repository"]["source_head_at_export"],
        "source_tree_at_export": export["source_repository"]["source_tree_at_export"],
        "export": {
            "path": rel(EXPORT_PATH),
            "bytes": len(export_bytes),
            "sha256": sha256_bytes(export_bytes),
        },
        "source_artifact_count": len(source_records),
        "scientific_firewall": export["scientific_firewall"],
        "next_legal_action": "STOP_FOR_PI",
    }
    root = {
        "schema": "STAGE_AC_PAPER_V3_EXPORT_ROOT_SEAL_V1",
        "status": "STAGE_AC_CODE_AND_PAPER_EXPORT_CLOSEOUT_COMPLETE_STOP_FOR_PI",
        "root_payload": root_payload,
        "root_payload_sha256": sha256_bytes(json_bytes(root_payload)),
        "artifacts": {"export": root_payload["export"]},
        "source_authority": source_records,
        "next_legal_action": "STOP_FOR_PI",
    }
    root_bytes = json_bytes(root)
    ROOT_PATH.write_bytes(root_bytes)
    closeout = {
        "schema": "STAGE_AC_CODE_AND_PAPER_EXPORT_CLOSEOUT_V1",
        "status": "STAGE_AC_CODE_AND_PAPER_EXPORT_CLOSEOUT_COMPLETE_STOP_FOR_PI",
        "source_repository": export["source_repository"],
        "export_artifact": {
            "path": rel(EXPORT_PATH),
            "bytes": len(export_bytes),
            "sha256": sha256_bytes(export_bytes),
        },
        "root_seal": {
            "path": rel(ROOT_PATH),
            "bytes": len(root_bytes),
            "sha256": sha256_bytes(root_bytes),
        },
        "claim_boundary": export["claim_contract"],
        "scientific_firewall": export["scientific_firewall"],
        "next_legal_action": "STOP_FOR_PI",
    }
    CLOSEOUT_PATH.write_bytes(json_bytes(closeout))
    print(
        "STAGE_AC_CODE_AND_PAPER_EXPORT_CLOSEOUT_COMPLETE_STOP_FOR_PI "
        f"export_sha256={sha256_bytes(export_bytes)} root_sha256={sha256_bytes(root_bytes)}"
    )


def check_outputs() -> None:
    require(EXPORT_PATH.is_file(), "missing_export")
    require(ROOT_PATH.is_file(), "missing_root")
    require(CLOSEOUT_PATH.is_file(), "missing_closeout")
    export = load_json(EXPORT_PATH)
    root = load_json(ROOT_PATH)
    closeout = load_json(CLOSEOUT_PATH)
    require(export["status"] == "STAGE_AC_PAPER_V3_EXPORT_READY_STOP_FOR_PI", "export_status")
    require(root["status"] == "STAGE_AC_CODE_AND_PAPER_EXPORT_CLOSEOUT_COMPLETE_STOP_FOR_PI", "root_status")
    require(closeout["status"] == "STAGE_AC_CODE_AND_PAPER_EXPORT_CLOSEOUT_COMPLETE_STOP_FOR_PI", "closeout_status")
    export_bytes = EXPORT_PATH.read_bytes()
    root_bytes = ROOT_PATH.read_bytes()
    require(root["artifacts"]["export"]["bytes"] == len(export_bytes), "export_bytes")
    require(root["artifacts"]["export"]["sha256"] == sha256_bytes(export_bytes), "export_sha")
    require(closeout["export_artifact"]["sha256"] == sha256_bytes(export_bytes), "closeout_export_sha")
    require(closeout["root_seal"]["sha256"] == sha256_bytes(root_bytes), "closeout_root_sha")
    for key, record in export["source_artifacts"].items():
        path = ROOT / record["path"]
        current = file_ref(path, record["role"])
        require(current["bytes"] == record["bytes"], f"source_bytes:{key}")
        require(current["sha256"] == record["sha256"], f"source_sha:{key}")
    require(zero_firewall(export["scientific_firewall"]), "export_firewall")
    print(
        "STAGE_AC_CODE_AND_PAPER_EXPORT_CLOSEOUT_CHECK_PASS "
        f"export_sha256={sha256_bytes(export_bytes)} root_sha256={sha256_bytes(root_bytes)}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    check_outputs() if args.check else write_outputs()


if __name__ == "__main__":
    main()
