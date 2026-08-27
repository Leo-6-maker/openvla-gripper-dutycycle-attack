#!/usr/bin/env python3
"""CPU-only static reconciliation for the sealed seven-agent Z3DH panel."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


PANEL_LABELS = "reports/server_evidence/STAGE_Z_Z3DH_AI_SUBAGENT_PANEL_V1/STAGE_Z_Z3DH_AI_SUBAGENT_PANEL_LABELS_V1.json"
PANEL_SEAL = "reports/server_evidence/STAGE_Z_Z3DH_AI_SUBAGENT_PANEL_V1/STAGE_Z_Z3DH_AI_SUBAGENT_PANEL_SEAL_V1.json"
PRIOR_RECON = "reports/server_evidence/STAGE_Z_Z3D_AI_SECONDARY_RECONCILIATION_V1R1/STAGE_Z_Z3D_AI_SECONDARY_UNBLIND_RECONCILIATION_V1.json"
PRIOR_ROOT = "reports/server_evidence/STAGE_Z_Z3D_AI_SECONDARY_RECONCILIATION_V1R1/STAGE_Z_Z3D_AI_SECONDARY_ROOT_SEAL_V1.json"
Z3C_ROOT = "reports/STAGE_Z_Z3C_ROOT_SEAL_V1.json"
Z3C_INDEX = "reports/STAGE_Z_Z3C_BRANCH_RECEIPT_INDEX_V1.json"
Z3C_TERMINAL = "reports/STAGE_Z_Z3C_TERMINAL_SYNTHESIS_V1.json"
SCRIPT_REL = "scripts/stage_z/reconcile_stage_z_z3dh_ai_subagent_panel.py"

PANEL_LABELS_OUT = "reports/STAGE_Z_Z3DH_AI_SUBAGENT_PANEL_UNBLIND_RECONCILIATION_V1.json"
CROSS_AUDIT_OUT = "reports/STAGE_Z_Z3DH_AI_PANEL_CROSS_AUDIT_V1.json"
Z4_SYNTHESIS_OUT = "reports/STAGE_Z_Z4_STATIC_CROSS_MODEL_SYNTHESIS_V1.json"
Z4_CLAIM_OUT = "reports/STAGE_Z_Z4_CLAIM_BOUNDARY_V1.json"
Z4_DEVIATION_OUT = "reports/STAGE_Z_Z4_PROTOCOL_DEVIATION_V1.json"
Z4_ROOT_OUT = "reports/STAGE_Z_Z4_ROOT_SEAL_V1.json"
Z4_SIDECAR_OUT = "reports/STAGE_Z_Z4_ROOT_SEAL_V1.sha256"

EXPECTED_PANEL = (66752, "aec7b751380603b0d384c011b175dd083203fa4672249f1a90c849273cc42cb1")
EXPECTED_SEAL = (1241, "2111adebc5f76ce1944373f21daf7ee7af9ad97cb2e6a08fa69f2f9d08bcc83c")
PANEL_LABEL_VOCABULARY = {
    "STABLE_GRASP",
    "PREMATURE_APERTURE",
    "CONTACT_LOSS",
    "PREMATURE_RELEASE_OR_DROP",
    "OBJECT_DISPLACEMENT",
    "AMBIGUOUS_OR_OCCLUDED",
    "NOT_IDENTIFIABLE",
}
PRIMARY_LABELS = {"V_PHYS", "NO_PHYSICAL_VULNERABILITY"}
PANEL_EVENT_LABELS = {"PREMATURE_APERTURE", "CONTACT_LOSS", "PREMATURE_RELEASE_OR_DROP"}
PANEL_BOOL_FIELDS = (
    "stable_grasp_maintained",
    "premature_aperture",
    "slip_or_contact_loss",
    "premature_release_or_drop",
    "object_displacement_consistent_with_loss",
)


def load(root: Path, relative: str) -> Any:
    return json.loads((root / relative).read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(root: Path, relative: str, value: Any) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def rel_artifact(root: Path, relative: str) -> dict[str, Any]:
    path = root / relative
    return {"path": relative, "bytes": path.stat().st_size, "sha256": sha256(path)}


def normalized(value: Any) -> str:
    return "<NULL>" if value is None else str(value)


def counts(values: Iterable[Any]) -> dict[str, int]:
    return dict(sorted(Counter(normalized(value) for value in values).items()))


def rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def crosstab(rows: list[dict[str, Any]], left: str, right: str) -> dict[str, dict[str, int]]:
    table: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        table[normalized(row[left])][normalized(row[right])] += 1
    return {key: dict(sorted(value.items())) for key, value in sorted(table.items())}


def grouped_panel_counts(rows: list[dict[str, Any]], keys: tuple[str, ...]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(normalized(row[key]) for key in keys)].append(row)
    result = []
    for group_key, group_rows in sorted(groups.items()):
        result.append({
            "group": dict(zip(keys, group_key)),
            "row_count": len(group_rows),
            "panel_label_counts": counts(row["panel_label"] for row in group_rows),
            "primary_ai_label_counts": counts(row["primary_ai_label"] for row in group_rows),
        })
    return result


def reviewer_confidence_type(value: Any) -> str:
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, (int, float)):
        return "number"
    return type(value).__name__


def same_parent_diagnostics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_model[row["model_family"]].append(row)
    output: dict[str, Any] = {}
    for model, model_rows in sorted(by_model.items()):
        all_parents = {row["canonical_parent_key"] for row in model_rows}
        dose_parent_labels: dict[int, dict[str, dict[str, str]]] = defaultdict(dict)
        duplicate_keys: list[str] = []
        for row in model_rows:
            if row["role"] != "CRITICAL_OPEN_PRIMARY" or row["dose"] not in (3, 5, 10):
                continue
            label = row["auto_v_phys_label"]
            if label not in PRIMARY_LABELS:
                continue
            parent = row["canonical_parent_key"]
            key = f"{parent}|{row['dose']}"
            if parent in dose_parent_labels[row["dose"]]:
                duplicate_keys.append(key)
            dose_parent_labels[row["dose"]][parent] = label
        valid_counts = {str(dose): len(dose_parent_labels[dose]) for dose in (3, 5, 10)}
        complete = set(dose_parent_labels[3]) & set(dose_parent_labels[5]) & set(dose_parent_labels[10])
        complete_rows = []
        pattern_counts: Counter[str] = Counter()
        for parent in sorted(complete):
            labels = [dose_parent_labels[dose][parent] for dose in (3, 5, 10)]
            pattern = "".join("1" if label == "V_PHYS" else "0" for label in labels)
            pattern_counts[pattern] += 1
            first = next(row for row in model_rows if row["canonical_parent_key"] == parent)
            complete_rows.append({
                "canonical_parent_key": parent,
                "suite": first["suite"],
                "labels_3_5_10": labels,
                "pattern_3_5_10": pattern,
            })
        output[model] = {
            "total_parents": len(all_parents),
            "dose_valid_parent_counts": valid_counts,
            "complete_all_dose_parents": len(complete),
            "complete_all_dose_pattern_counts": dict(sorted(pattern_counts.items())),
            "complete_all_dose_rows": complete_rows,
            "duplicate_parent_dose_keys": sorted(duplicate_keys),
        }
    return output


def git_binding(root: Path) -> dict[str, str | None]:
    def value(*args: str) -> str | None:
        try:
            return subprocess.check_output(["git", *args], cwd=root, text=True, stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            return None
    return {"generation_head": value("rev-parse", "HEAD"), "generation_tree": value("rev-parse", "HEAD^{tree}")}


def main(root: Path) -> None:
    failures: list[str] = []
    panel_path = root / PANEL_LABELS
    seal_path = root / PANEL_SEAL
    for name, path, expected in (("panel_labels", panel_path, EXPECTED_PANEL), ("panel_seal", seal_path, EXPECTED_SEAL)):
        if not path.is_file():
            failures.append(f"MISSING:{name}:{path}")
            continue
        actual = (path.stat().st_size, sha256(path))
        if actual != expected:
            failures.append(f"EXACT_BYTE_MISMATCH:{name}:{actual!r}!={expected!r}")
    panel = load(root, PANEL_LABELS)
    seal = load(root, PANEL_SEAL)
    prior = load(root, PRIOR_RECON)
    prior_root = load(root, PRIOR_ROOT)
    z3c_root = load(root, Z3C_ROOT)

    panel_rows = panel.get("rows", [])
    prior_rows = prior.get("rows", [])
    if panel.get("status") != "AI_SUBAGENT_PANEL_COMPLETE":
        failures.append(f"PANEL_STATUS:{panel.get('status')}")
    if panel.get("row_count") != 120 or len(panel_rows) != 120:
        failures.append("PANEL_ROW_COUNT")
    if panel.get("unique_video_id_count") != 120 or len({row.get("video_id") for row in panel_rows}) != 120:
        failures.append("PANEL_UNIQUE_IDS")
    if prior.get("status") != "STAGE_Z_Z3D_AI_SECONDARY_RECONCILIATION_COMPLETE_STOP_FOR_PI":
        failures.append(f"PRIOR_RECON_STATUS:{prior.get('status')}")
    if len(prior_rows) != 120 or len({row.get("blinded_video_id") for row in prior_rows}) != 120:
        failures.append("PRIOR_ROW_INTEGRITY")
    if seal.get("labels_artifact_sha256") != EXPECTED_PANEL[1] or seal.get("labels_artifact_bytes") != EXPECTED_PANEL[0]:
        failures.append("PANEL_SEAL_LABEL_BINDING")
    if seal.get("human_review_gate_satisfied") is not False:
        failures.append("PANEL_HUMAN_GATE_NOT_FALSE")
    for field in ("hidden_mapping_read", "automatic_labels_read", "telemetry_read", "scientific_outcomes_read", "unblind_performed"):
        if seal.get(field) is not False:
            failures.append(f"PANEL_SEAL_FIREWALL:{field}")
    actual_panel_counts = counts(row.get("primary_label") for row in panel_rows)
    declared_panel_counts = {str(k): int(v) for k, v in panel.get("label_counts", {}).items()}
    if actual_panel_counts != declared_panel_counts:
        failures.append("PANEL_LABEL_COUNTS")
    if any(row.get("primary_label") not in PANEL_LABEL_VOCABULARY for row in panel_rows):
        failures.append("PANEL_LABEL_VOCABULARY")
    if any(not all(field in row and isinstance(row[field], bool) for field in PANEL_BOOL_FIELDS) for row in panel_rows):
        failures.append("PANEL_BOOLEAN_SCHEMA")

    prior_by_video = {row["blinded_video_id"]: row for row in prior_rows}
    joined: list[dict[str, Any]] = []
    for panel_row in sorted(panel_rows, key=lambda row: row["video_id"]):
        video_id = panel_row["video_id"]
        prior_row = prior_by_video.get(video_id)
        if prior_row is None:
            failures.append(f"MISSING_PRIOR_VIDEO:{video_id}")
            continue
        joined.append({
            "video_id": video_id,
            "panel": {
                "primary_label": panel_row["primary_label"],
                **{field: panel_row[field] for field in PANEL_BOOL_FIELDS},
                "reviewer_confidence": panel_row.get("reviewer_confidence"),
                "reviewer_confidence_type": reviewer_confidence_type(panel_row.get("reviewer_confidence")),
                "blinded_note": panel_row.get("blinded_note"),
                "source_shard": panel_row.get("source_shard"),
                "source_agent_id": panel_row.get("source_agent_id"),
            },
            "primary_ai": {
                "label": prior_row.get("ai_label"),
                "reviewer_confidence": prior_row.get("reviewer_confidence"),
                "blinded_note": prior_row.get("blinded_note"),
            },
            "mapping": {
                key: prior_row.get(key)
                for key in ("source_blinded_video_id", "manual_audit_id", "branch_id", "model_family", "suite", "canonical_parent_key", "arm", "role", "dose", "duration")
            },
            "automatic": {
                key: prior_row.get(key)
                for key in ("auto_physical_class", "auto_v_phys_label", "auto_valid_primary", "auto_abstain_primary")
            },
            "telemetry": prior_row.get("telemetry"),
        })
    joined_ids = {row["video_id"] for row in joined}
    if joined_ids != set(prior_by_video):
        failures.append("JOIN_ID_SET")
    if failures:
        raise SystemExit(json.dumps({"status": "FAIL_STATIC_VALIDATION", "failures": failures}, indent=2))

    flat = []
    for row in joined:
        flat.append({
            "panel_label": row["panel"]["primary_label"],
            "primary_ai_label": row["primary_ai"]["label"],
            **row["mapping"],
            **row["automatic"],
        })
    panel_distribution = {
        "overall": counts(row["panel_label"] for row in flat),
        "by_model": grouped_panel_counts(flat, ("model_family",)),
        "by_suite": grouped_panel_counts(flat, ("suite",)),
        "by_role": grouped_panel_counts(flat, ("role",)),
        "by_arm": grouped_panel_counts(flat, ("arm",)),
        "by_model_arm_dose": grouped_panel_counts(flat, ("model_family", "arm", "dose")),
        "by_model_suite_role_dose": grouped_panel_counts(flat, ("model_family", "suite", "role", "dose")),
    }
    panel_counts = panel_distribution["overall"]
    primary_panel_agreement = sum(row["panel_label"] == row["primary_ai_label"] for row in flat)
    primary_ai_vs_panel = {
        "row_count": len(flat),
        "exact_label_agreement_count": primary_panel_agreement,
        "exact_label_agreement_rate": rate(primary_panel_agreement, len(flat)),
        "confusion_matrix_primary_ai_to_panel": crosstab(flat, "primary_ai_label", "panel_label"),
        "label_vocabularies": {
            "primary_ai": sorted({row["primary_ai_label"] for row in flat}),
            "panel": sorted(PANEL_LABEL_VOCABULARY),
        },
        "interpretation": "This is primary-AI versus assigned seven-agent panel agreement; the seven shards are disjoint, so inter-rater agreement among panel agents is not estimable.",
    }
    auto_vs_panel = {
        "auto_physical_class_to_panel": crosstab(flat, "auto_physical_class", "panel_label"),
        "auto_v_phys_label_to_panel": crosstab(flat, "auto_v_phys_label", "panel_label"),
        "gripper_contact_loss_panel_counts": counts(
            row["panel_label"] for row in flat if row["auto_physical_class"] == "GRIPPER_CONTACT_LOSS"
        ),
        "gripper_contact_loss_rows": sum(row["auto_physical_class"] == "GRIPPER_CONTACT_LOSS" for row in flat),
        "gripper_contact_loss_not_identifiable_or_ambiguous": sum(
            row["auto_physical_class"] == "GRIPPER_CONTACT_LOSS" and row["panel_label"] in {"NOT_IDENTIFIABLE", "AMBIGUOUS_OR_OCCLUDED"}
            for row in flat
        ),
        "gripper_contact_loss_stable_grasp": sum(
            row["auto_physical_class"] == "GRIPPER_CONTACT_LOSS" and row["panel_label"] == "STABLE_GRASP"
            for row in flat
        ),
        "interpretation_guardrail": "Panel labels diagnose endpoint visibility/construct validity only; automatic abstentions and V_phys labels are not relabeled.",
    }
    same_parent_audit_subset = same_parent_diagnostics(flat)
    prior_same_parent = prior.get("static_diagnostics", {}).get("same_parent_t3_t5_t10")
    prior_static_preserved = isinstance(prior_same_parent, dict) and set(prior_same_parent) == {"M0_OPENVLA", "M1_OPENVLA_OFT", "M2_PI05_LIBERO"}
    panel_confidence_types = counts(row["panel"]["reviewer_confidence_type"] for row in joined)
    panel_event_count = sum(row["panel"]["primary_label"] in PANEL_EVENT_LABELS for row in joined)
    panel_nid_ambiguous = sum(row["panel"]["primary_label"] in {"NOT_IDENTIFIABLE", "AMBIGUOUS_OR_OCCLUDED"} for row in joined)
    firewall = {
        "new_model_inference": 0,
        "new_env_step": 0,
        "new_open_intervention": 0,
        "pgd": 0,
        "protected_reads": 0,
        "eval160_reads": 0,
        "new_identities": 0,
        "branch_reexecution": 0,
        "v_phys_relabeling": 0,
        "human_review_gate_satisfied": False,
        "human_review_requirement_waived_for_exploratory_closeout_by_PI": True,
    }
    authority = {
        "panel_labels": rel_artifact(root, PANEL_LABELS),
        "panel_seal": rel_artifact(root, PANEL_SEAL),
        "prior_ai_unblind_reconciliation": rel_artifact(root, PRIOR_RECON),
        "prior_ai_root_seal": rel_artifact(root, PRIOR_ROOT),
        "z3c_root_seal": rel_artifact(root, Z3C_ROOT),
        "z3c_branch_index": rel_artifact(root, Z3C_INDEX),
        "z3c_terminal": rel_artifact(root, Z3C_TERMINAL),
    }
    reconciliation = {
        "schema": "STAGE_Z_Z3DH_AI_SUBAGENT_PANEL_UNBLIND_RECONCILIATION_V1",
        "status": "STAGE_Z_Z3DH_AI_SUBAGENT_PANEL_RECONCILIATION_COMPLETE",
        "authority": authority,
        "ingestion": {
            "exact_panel_bytes_verified": True,
            "exact_panel_sha256": EXPECTED_PANEL[1],
            "exact_panel_bytes": EXPECTED_PANEL[0],
            "exact_seal_bytes_verified": True,
            "exact_seal_sha256": EXPECTED_SEAL[1],
            "exact_seal_bytes": EXPECTED_SEAL[0],
            "labels_sealed_before_unblind": True,
            "unblind_source": "prior sealed AI-secondary reconciliation rows",
        },
        "reviewer_panel": {
            "reviewer_class": "AI_SUBAGENT_PANEL",
            "shard_count": panel.get("shard_count"),
            "shard_row_counts": panel.get("shard_row_counts"),
            "row_count": len(joined),
            "disjoint_shards": True,
            "inter_rater_agreement_estimable": False,
            "confidence_encoding_preserved": True,
            "confidence_type_counts": panel_confidence_types,
        },
        "panel_distribution": panel_distribution,
        "panel_summary": {
            "not_identifiable_or_ambiguous_count": panel_nid_ambiguous,
            "not_identifiable_or_ambiguous_rate": rate(panel_nid_ambiguous, len(joined)),
            "direct_contact_release_label_count": panel_event_count,
            "direct_contact_release_label_rate": rate(panel_event_count, len(joined)),
            "stable_grasp_count": panel_counts.get("STABLE_GRASP", 0),
            "object_displacement_count": panel_counts.get("OBJECT_DISPLACEMENT", 0),
        },
        "primary_ai_vs_panel": primary_ai_vs_panel,
        "automatic_vs_panel": auto_vs_panel,
        "same_parent_t3_t5_t10": {
            "authoritative_full_z3c_static_diagnostic": prior_same_parent,
            "audit_subset_recompute": same_parent_audit_subset,
            "prior_static_diagnostic_preserved": prior_static_preserved,
            "comparison_note": "The 120 audit videos are a selected presentation subset; their parent counts cannot be compared to the full 92-parent Z3-C matrix. The sealed full-matrix diagnostic is preserved as authoritative.",
        },
        "telemetry_by_model_role_dose": prior.get("static_diagnostics", {}).get("telemetry_by_model_role_dose"),
        "joined_rows": joined,
        "scientific_firewall": firewall,
        "claim_boundary": "Static AI-only endpoint-validity reconciliation; no human-review completion, no endpoint relabeling, no new execution, no Paper V2 promotion.",
        "next_legal_action": "GENERATE_STATIC_Z4_CLOSEOUT_THEN_STOP_FOR_PI",
    }
    write_json(root, PANEL_LABELS_OUT, reconciliation)

    cross_audit = {
        "schema": "STAGE_Z_Z3DH_AI_PANEL_CROSS_AUDIT_V1",
        "status": "STAGE_Z_Z3DH_AI_PANEL_CROSS_AUDIT_COMPLETE",
        "source_reconciliation": rel_artifact(root, PANEL_LABELS_OUT),
        "primary_ai_vs_assigned_panel": primary_ai_vs_panel,
        "assigned_panel_distribution": panel_distribution,
        "automatic_endpoint_vs_assigned_panel": auto_vs_panel,
        "same_parent_dose_diagnostics": reconciliation["same_parent_t3_t5_t10"],
        "panel_design_guardrail": {
            "seven_shards_are_disjoint": True,
            "panel_inter_rater_agreement": "NOT_ESTIMABLE",
            "panel_is_not_human_review": True,
            "heterogeneous_confidence_encoding_preserved": True,
        },
        "interpretation": {
            "endpoint_validity_signal": "The panel supplies exploratory visual-validity evidence and does not repair the primary endpoint.",
            "automatic_contact_loss_guardrail": "Automatic GRIPPER_CONTACT_LOSS remains an automatic label; panel disagreement is diagnostic only.",
            "cross_model_guardrail": "The observed denominator attrition and panel indeterminacy do not establish M1/M2 robustness or immunity.",
        },
        "scientific_firewall": firewall,
        "next_legal_action": "STATIC_Z4_CLOSEOUT_ONLY",
    }
    write_json(root, CROSS_AUDIT_OUT, cross_audit)

    source_refs = {
        "panel_reconciliation": rel_artifact(root, PANEL_LABELS_OUT),
        "panel_cross_audit": rel_artifact(root, CROSS_AUDIT_OUT),
        "z3c_root_seal": authority["z3c_root_seal"],
        "prior_ai_root_seal": authority["prior_ai_root_seal"],
    }
    z4_synthesis = {
        "schema": "STAGE_Z_Z4_STATIC_CROSS_MODEL_SYNTHESIS_V1",
        "status": "STAGE_Z_Z4_STATIC_CLOSEOUT_COMPLETE_STOP_FOR_PI",
        "classification": "STAGE_Z_CROSS_MODEL_GENERALIZATION_NOT_ESTABLISHED_ENDPOINT_VALIDITY_LIMITED_AI_ONLY",
        "source_refs": source_refs,
        "evidence_summary": {
            "z1_engineering_runtime_qualification": {"M0": "4/4", "M1": "4/4", "M2": "4/4", "scientific_parent_exposure": 0},
            "z3c_fixed_physical_matrix": {"branches": 460, "pass_receipts": 460, "new_model_inference": 0, "new_env_step": 0, "protected_reads": 0},
            "z3dh_ai_only_panel": {
                "rows": len(joined),
                "not_identifiable_or_ambiguous": panel_nid_ambiguous,
                "not_identifiable_or_ambiguous_rate": rate(panel_nid_ambiguous, len(joined)),
                "direct_contact_release_labels": panel_event_count,
                "distribution": panel_counts,
                "human_review_gate_satisfied": False,
                "pi_waiver_for_exploratory_closeout": True,
            },
            "same_parent_complete_dose_patterns": prior_same_parent,
            "same_parent_audit_subset": same_parent_audit_subset,
        },
        "claim_safe_findings": [
            "Z1 established engineering portability of the shared experiment/action-interface framework across M0, M1, and M2; it was not a scientific vulnerability result.",
            "Z3-C completed the frozen 460-branch command-OPEN physical matrix with sealed execution provenance, but the resulting effective primary endpoint denominators are model- and phase-dependent.",
            "The seven-agent panel provides AI-only exploratory endpoint-validity evidence; 91/120 rows were NOT_IDENTIFIABLE or AMBIGUOUS_OR_OCCLUDED and 6/120 received direct contact/release labels.",
            "Complete same-parent T3/T5/T10 patterns remain M0 000 x3, M1 000 x2, and M2 001/100/111; these patterns are preserved from sealed automatic endpoints and are not repaired by the panel.",
            "The evidence does not establish cross-model generalization of the historical OpenVLA OPEN mechanism, and it does not establish M1 or M2 robustness/immunity.",
        ],
        "not_claimed": [
            "M1 or M2 robust/immune to command-OPEN physical intervention",
            "a human manual-audit completion",
            "a cross-model visual PGD transfer result",
            "a relabeled V_phys denominator",
            "a causal impossibility claim about the OPEN mechanism",
        ],
        "scientific_firewall": firewall,
        "next_legal_action": "STOP_FOR_PI_NO_Z4_PROMOTION_NO_PAPER_PROMOTION_NO_BRIDGE_NO_NEW_SIMULATOR_EXECUTION",
    }
    write_json(root, Z4_SYNTHESIS_OUT, z4_synthesis)

    claim_boundary = {
        "schema": "STAGE_Z_Z4_CLAIM_BOUNDARY_V1",
        "status": "SEALED_FOR_PI_REVIEW",
        "classification": z4_synthesis["classification"],
        "allowed_claims": z4_synthesis["claim_safe_findings"],
        "prohibited_claims": z4_synthesis["not_claimed"],
        "primary_endpoint_rule": "Automatic V_phys and CONTROL_*_ABSTAIN labels remain frozen; AI panel labels are diagnostic-only and cannot alter denominators.",
        "reviewer_status": {
            "human_review_gate_satisfied": False,
            "human_review_requirement_waived_for_exploratory_closeout_by_PI": True,
            "formal_label_source": "AI-only exploratory evidence; not human review",
        },
        "promotion_status": {"z4_static_closeout": "COMPLETE", "paper_v2_promotion": "NOT_AUTHORIZED"},
        "source": source_refs,
    }
    write_json(root, Z4_CLAIM_OUT, claim_boundary)

    protocol_deviation = {
        "schema": "STAGE_Z_Z4_PROTOCOL_DEVIATION_V1",
        "status": "PI_WAIVED_FOR_EXPLORATORY_AI_ONLY_CLOSEOUT",
        "original_protocol": {
            "human_review_required": True,
            "human_review_gate_satisfied": False,
            "primary_manual_audit_max_videos": 120,
        },
        "pi_amendment": {
            "human_review_requirement_waived_for_exploratory_closeout_by_PI": True,
            "allowed_evidence": ["single-AI R1 secondary audit", "seven-agent AI subagent coverage panel"],
            "formal_human_review_claim": False,
            "paper_reporting_requirement": "Identify this as AI-only exploratory endpoint-validity evidence.",
        },
        "deviation_reason": "No human reviewer was supplied; the PI explicitly authorized an exploratory AI-only closeout while preserving the unsatisfied human gate.",
        "preserved_invariants": [
            "No new model inference, simulator/env.step, OPEN intervention, PGD, protected/Eval160 read, BRIDGE, F1, identity, replacement, or top-up.",
            "No primary automatic endpoint or V_phys/abstention relabeling.",
            "Panel labels were ingested from exact sealed bytes; aggregate reconstruction was not used.",
            "Labels were sealed before unblind reconciliation.",
        ],
        "source": source_refs,
    }
    write_json(root, Z4_DEVIATION_OUT, protocol_deviation)

    manifest = [
        rel_artifact(root, PANEL_LABELS),
        rel_artifact(root, PANEL_SEAL),
        rel_artifact(root, PRIOR_RECON),
        rel_artifact(root, PRIOR_ROOT),
        rel_artifact(root, Z3C_ROOT),
        rel_artifact(root, Z3C_INDEX),
        rel_artifact(root, Z3C_TERMINAL),
        rel_artifact(root, PANEL_LABELS_OUT),
        rel_artifact(root, CROSS_AUDIT_OUT),
        rel_artifact(root, Z4_SYNTHESIS_OUT),
        rel_artifact(root, Z4_CLAIM_OUT),
        rel_artifact(root, Z4_DEVIATION_OUT),
        rel_artifact(root, SCRIPT_REL),
    ]
    root_seal = {
        "schema": "STAGE_Z_Z4_ROOT_SEAL_V1",
        "status": "STAGE_Z_Z4_STATIC_CLOSEOUT_COMPLETE_STOP_FOR_PI",
        "classification": z4_synthesis["classification"],
        "git_binding": git_binding(root),
        "artifact_manifest": {"entries": manifest, "root_seal_excludes_self": True},
        "reviewer_governance": {
            "human_review_gate_satisfied": False,
            "human_review_requirement_waived_for_exploratory_closeout_by_PI": True,
            "ai_only_exploratory": True,
        },
        "scientific_firewall": firewall,
        "claim_boundary": rel_artifact(root, Z4_CLAIM_OUT),
        "protocol_deviation": rel_artifact(root, Z4_DEVIATION_OUT),
        "next_legal_action": "STOP_FOR_PI_NO_PAPER_PROMOTION_NO_BRIDGE_NO_NEW_SIMULATOR_EXECUTION",
        "validation_failures": [],
    }
    write_json(root, Z4_ROOT_OUT, root_seal)
    root_hash = sha256(root / Z4_ROOT_OUT)
    (root / Z4_SIDECAR_OUT).write_text(f"{root_hash}  {Path(Z4_ROOT_OUT).name}\n", encoding="utf-8")
    print(json.dumps({
        "status": root_seal["status"],
        "classification": root_seal["classification"],
        "panel_counts": panel_counts,
        "panel_primary_ai_agreement": primary_ai_vs_panel["exact_label_agreement_count"],
        "panel_primary_ai_agreement_rate": primary_ai_vs_panel["exact_label_agreement_rate"],
        "prior_static_diagnostic_preserved": prior_static_preserved,
        "root_sha256": root_hash,
    }, sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    main(args.root.resolve())
