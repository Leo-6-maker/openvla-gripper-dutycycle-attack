#!/usr/bin/env python3
"""Build the read-only Stage-AA0 protocol, audit, capacity inventory, and seal."""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
AA_DATE = "20260826"
AA1_CANARY_SALT = f"STAGE_AA_AA1_ENGINEERING_CANARY_V1_{AA_DATE}"

Z0_PANEL = "reports/STAGE_Z_Z0_SHARED_40_IDENTITY_PANEL_V1.json"
Z0R1_PANEL = "reports/STAGE_Z_Z0R1_SHARED_36_IDENTITY_PANEL_V1.json"
Z0R1_AMENDMENT = "reports/STAGE_Z_Z0R1_STRUCTURAL_MISSINGNESS_AMENDMENT_V1.json"
Z2_AVAILABILITY = "reports/STAGE_Z_Z2_ANCHOR_AVAILABILITY_V2.json"
Z2_TERMINAL = "reports/STAGE_Z_Z2_TERMINAL_SYNTHESIS_V2.json"
Z3_PROTOCOL = "configs/STAGE_Z_Z3_CROSS_MODEL_COMMAND_OPEN_PHYSICAL_MATRIX_PROTOCOL_V2.json"
Z3_ELIGIBILITY = "reports/STAGE_Z_Z3_ELIGIBILITY_RECONCILIATION_V1.json"
Z3_TERMINAL = "reports/STAGE_Z_Z3C_TERMINAL_SYNTHESIS_V1.json"
Z3_ROOT = "reports/STAGE_Z_Z3C_ROOT_SEAL_V1.json"
Z3_CONTRACT = "src/stage_z_preparation/z3_contract.py"
Z4_CROSS_AUDIT = "reports/STAGE_Z_Z3DH_AI_PANEL_CROSS_AUDIT_V1.json"
Z4_ROOT = "reports/STAGE_Z_Z4_ROOT_SEAL_V1.json"
Z4_SYNTHESIS = "reports/STAGE_Z_Z4_STATIC_CROSS_MODEL_SYNTHESIS_V1.json"

AUDIT_OUT = "reports/STAGE_AA_AA0_HISTORICAL_ENDPOINT_DENOMINATOR_AUDIT_V1.json"
CAPACITY_OUT = "reports/STAGE_AA_AA0_FRESH_CAPACITY_INVENTORY_V1.json"
PROTOCOL_OUT = "configs/STAGE_AA_AA0_PROSPECTIVE_PROTOCOL_V1.json"
ROOT_OUT = "reports/STAGE_AA_AA0_ROOT_SEAL_V1.json"
SIDECAR_OUT = "reports/STAGE_AA_AA0_ROOT_SEAL_V1.sha256"
SCRIPT_REL = "scripts/stage_aa/build_stage_aa0_static.py"

MODELS = ("M0_OPENVLA", "M1_OPENVLA_OFT", "M2_PI05_LIBERO")
DOSES = (3, 5, 10)
PANEL_MISSING = {
    "libero_goal/task_01",
    "libero_goal/task_04",
    "libero_goal/task_06",
    "libero_goal/task_09",
}


def load(relative: str) -> Any:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact(relative: str) -> dict[str, Any]:
    path = ROOT / relative
    return {"path": relative, "bytes": path.stat().st_size, "sha256": sha256(path)}


def write_json(relative: str, value: Any) -> None:
    path = ROOT / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sorted_counts(values: Iterable[Any]) -> dict[str, int]:
    return dict(sorted(Counter("<NULL>" if value is None else str(value) for value in values).items()))


def git_binding() -> dict[str, str | None]:
    def value(*args: str) -> str | None:
        try:
            return subprocess.check_output(["git", *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            return None
    return {"generation_head": value("rev-parse", "HEAD"), "generation_tree": value("rev-parse", "HEAD^{tree}")}


def require(condition: bool, message: str, failures: list[str]) -> None:
    if not condition:
        failures.append(message)


def model_dose_summary(parent_rows: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for model in MODELS:
        rows = [row for row in parent_rows if row.get("model_family") == model]
        dose = {}
        for value in DOSES:
            labels = [row.get("critical", {}).get(str(value), {}).get("label") for row in rows]
            physical = [row.get("critical", {}).get(str(value), {}).get("physical_class") for row in rows]
            valid = sum(label in {"V_PHYS", "NO_PHYSICAL_VULNERABILITY"} for label in labels)
            dose[str(value)] = {
                "parent_count": len(rows),
                "automatic_label_counts": sorted_counts(labels),
                "automatic_physical_class_counts": sorted_counts(physical),
                "valid_primary_parent_count": valid,
                "post_hoc_abstention_count": len(rows) - valid,
            }
        clean = sorted_counts(row.get("clean_physical_class") for row in rows)
        output[model] = {
            "parent_count": len(rows),
            "clean_physical_class_counts": clean,
            "noncritical_t5_physical_class_counts": sorted_counts(
                row.get("noncritical_t5_control", {}).get("physical_class") for row in rows
            ),
            "dose": dose,
        }
    return output


def model_suite_summary(parent_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in parent_rows:
        groups[(str(row.get("model_family")), str(row.get("suite")))].append(row)
    output = []
    for (model, suite), rows in sorted(groups.items()):
        output.append({
            "model_family": model,
            "suite": suite,
            "parent_count": len(rows),
            "clean_physical_class_counts": sorted_counts(row.get("clean_physical_class") for row in rows),
            "critical_label_counts_by_dose": {
                str(dose): sorted_counts(row.get("critical", {}).get(str(dose), {}).get("label") for row in rows)
                for dose in DOSES
            },
            "noncritical_t5_physical_class_counts": sorted_counts(
                row.get("noncritical_t5_control", {}).get("physical_class") for row in rows
            ),
        })
    return output


def flatten_fresh_candidates(panel: dict[str, Any]) -> list[dict[str, Any]]:
    candidates = []
    for row in panel["rows"]:
        for candidate in row.get("ranked_fresh_candidates", []):
            key = str(candidate["canonical_parent_key"])
            suite, task, state = key.split("/")
            candidates.append({
                "canonical_parent_key": key,
                "suite": suite,
                "task": task,
                "state": state,
                "rank_sha256": candidate["rank_sha256"],
                "source_task_idx": row["task_idx"],
            })
    return sorted(candidates, key=lambda row: row["canonical_parent_key"])


def select_canaries(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_suite: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        by_suite[candidate["suite"]].append(candidate)
    chosen = []
    for suite in sorted(by_suite):
        ranked = sorted(
            by_suite[suite],
            key=lambda row: hashlib.sha256(
                f"{AA1_CANARY_SALT}|{row['canonical_parent_key']}".encode()
            ).hexdigest(),
        )
        row = dict(ranked[0])
        row["selection_rank_sha256"] = hashlib.sha256(
            f"{AA1_CANARY_SALT}|{row['canonical_parent_key']}".encode()
        ).hexdigest()
        chosen.append(row)
    return chosen


def build_audit() -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    failures: list[str] = []
    z3 = load(Z3_TERMINAL)
    z3_root = load(Z3_ROOT)
    z3_contract_text = (ROOT / Z3_CONTRACT).read_text(encoding="utf-8")
    parent_rows = z3["physical_results"]["parent_rows"]
    require(len(parent_rows) == 92, "Z3_PARENT_ROWS_NOT_92", failures)
    require(z3.get("status") == "PASS_Z3C_FIXED_MATRIX_COMPLETE", "Z3_TERMINAL_STATUS", failures)
    require(z3_root.get("status") == "PASS_Z3C_FIXED_MATRIX_COMPLETE", "Z3_ROOT_STATUS", failures)
    require("count >= 2" in z3_contract_text, "CONTACT_LOSS_STREAK_RULE_NOT_FOUND", failures)
    require("post_object_gripper_contact" in z3_contract_text, "CONTACT_FIELD_NOT_FOUND", failures)

    z4_cross = load(Z4_CROSS_AUDIT)
    z4_root = load(Z4_ROOT)
    z4_auto = z4_cross.get("automatic_endpoint_vs_assigned_panel", {})
    ai_contact = z4_auto.get("gripper_contact_loss_panel_counts", {})
    audit = {
        "schema": "STAGE_AA_AA0_HISTORICAL_ENDPOINT_DENOMINATOR_AUDIT_V1",
        "status": "STAGE_AA_AA0_HISTORICAL_STATIC_AUDIT_COMPLETE",
        "claim_boundary": "Historical static audit only; no new inference, env.step, OPEN intervention, scientific outcome, or protected read.",
        "source_bindings": {
            name: artifact(path)
            for name, path in {
                "z3_protocol": Z3_PROTOCOL,
                "z3_eligibility": Z3_ELIGIBILITY,
                "z3_terminal": Z3_TERMINAL,
                "z3_root": Z3_ROOT,
                "z3_contract": Z3_CONTRACT,
                "z4_cross_audit": Z4_CROSS_AUDIT,
                "z4_root": Z4_ROOT,
                "z4_synthesis": Z4_SYNTHESIS,
            }.items()
        },
        "historical_z3_execution": {
            "fixed_branches": z3_root["execution"]["expected_branches"],
            "pass_receipts": z3_root["execution"]["pass_receipts"],
            "model_parent_count": z3_root["execution"]["model_parent_count"],
            "new_model_inference_in_audit": 0,
            "new_env_step_in_audit": 0,
        },
        "old_endpoint_contract": {
            "source": Z3_CONTRACT,
            "clean_contact_loss_rule": "two consecutive rows where post_object_gripper_contact is not True",
            "requires_prior_established_grasp": False,
            "requires_clean_reference_stability": False,
            "contact_false_alone_can_start_streak": True,
            "invalid_and_control_abstentions_are_post_hoc_labels": True,
            "new_aa_rule": "A contact=False observation is never sufficient by itself; AA requires a clean-established grasp, clean continuation stability, and a post-OPEN physical consequence.",
        },
        "full_matrix_clean_class_counts": sorted_counts(row.get("clean_physical_class") for row in parent_rows),
        "full_matrix_model_summary": model_dose_summary(parent_rows),
        "full_matrix_model_suite_summary": model_suite_summary(parent_rows),
        "construct_validity_evidence": {
            "automatic_gripper_contact_loss_rows_in_blinded_audit": int(z4_auto.get("gripper_contact_loss_rows", 0)),
            "panel_labels_for_automatic_gripper_contact_loss": ai_contact,
            "panel_nid_or_ambiguous_for_automatic_gripper_contact_loss": int(z4_auto.get("gripper_contact_loss_not_identifiable_or_ambiguous", 0)),
            "panel_stable_grasp_for_automatic_gripper_contact_loss": int(z4_auto.get("gripper_contact_loss_stable_grasp", 0)),
            "human_review_gate_satisfied": z4_root.get("reviewer_governance", {}).get("human_review_gate_satisfied"),
            "interpretation": "AI-only exploratory diagnostic evidence; it does not relabel the primary endpoint and does not satisfy a human-review gate.",
        },
        "same_parent_patterns_preserved_from_z4": load(Z4_SYNTHESIS)["evidence_summary"]["same_parent_complete_dose_patterns"],
        "historical_conclusion": {
            "denominator_attrition_is_observed": True,
            "old_endpoint_construct_validity_is_ambiguous": True,
            "m1_m2_robustness_not_established": True,
            "stage_z_remains_immutable": True,
        },
        "validation_failures": failures,
    }
    return audit, z3_root, failures


def build_capacity() -> tuple[dict[str, Any], list[dict[str, Any]], list[str]]:
    failures: list[str] = []
    panel = load(Z0_PANEL)
    panel_r1 = load(Z0R1_PANEL)
    amendment = load(Z0R1_AMENDMENT)
    pre_stage_z_candidates = flatten_fresh_candidates(panel)
    stage_z_keys = set(panel_r1["selected_parent_keys"])
    candidates = [row for row in pre_stage_z_candidates if row["canonical_parent_key"] not in stage_z_keys]
    keys = [row["canonical_parent_key"] for row in candidates]
    require(len(pre_stage_z_candidates) == 147, "PRE_STAGE_Z_CANDIDATE_COUNT_NOT_147", failures)
    require(len(stage_z_keys) == 36, "STAGE_Z_SELECTED_IDENTITY_COUNT_NOT_36", failures)
    require(len(candidates) == 111, "AA_FRESH_CANDIDATE_COUNT_NOT_111", failures)
    require(len(set(keys)) == len(keys), "FRESH_CANDIDATE_DUPLICATES", failures)
    require(panel["population_accounting"]["remaining_fresh_after_union"] == 147, "PANEL_REMAINING_CAPACITY_BINDING", failures)
    require(panel["population_accounting"]["consumed_union"] == 63, "PANEL_CONSUMED_UNION_BINDING", failures)
    require(set(panel_r1["selected_parent_keys"]).isdisjoint(keys), "AA_POOL_OVERLAPS_Z0R1_36", failures)
    suite_all = Counter(row["suite"] for row in candidates)
    canaries = select_canaries(candidates)
    canary_keys = {row["canonical_parent_key"] for row in canaries}
    require(len(canaries) == len(suite_all), "AA1_CANARY_COUNT", failures)
    require(not canary_keys.intersection(panel_r1["selected_parent_keys"]), "AA1_CANARY_OVERLAPS_Z0R1", failures)
    analysis_pool = [row for row in candidates if row["canonical_parent_key"] not in canary_keys]
    suite_analysis = Counter(row["suite"] for row in analysis_pool)
    inventory = {
        "schema": "STAGE_AA_AA0_FRESH_CAPACITY_INVENTORY_V1",
        "status": "STAGE_AA_AA0_FRESH_CAPACITY_INVENTORY_SEALED",
        "claim_boundary": "Static inventory only; no identity has been run by AA and no identity is promoted as science before AA2 freeze.",
        "source_bindings": {
            "z0_panel": artifact(Z0_PANEL),
            "z0r1_panel": artifact(Z0R1_PANEL),
            "z0r1_amendment": artifact(Z0R1_AMENDMENT),
        },
        "population_accounting_from_sealed_z0_panel": panel["population_accounting"],
        "structural_missing_task_cells": sorted(PANEL_MISSING),
        "pre_stage_z_candidate_inventory": pre_stage_z_candidates,
        "stage_z_exposed_identity_count": len(stage_z_keys),
        "full_fresh_inventory": candidates,
        "aa1_engineering_canary_reservation": {
            "selection_salt": AA1_CANARY_SALT,
            "count": len(canaries),
            "selection_rule": "one minimum sha256(selection_salt|canonical_parent_key) candidate per suite",
            "run_scope": "AA1 engineering-only canaries across the three model families; no AA2 analysis use",
            "permanent_exclusion": True,
            "replacement_or_top_up": False,
            "reserved_rows": canaries,
        },
        "analysis_pool_after_aa1_reservation": {
            "count": len(analysis_pool),
            "by_suite": dict(sorted(suite_analysis.items())),
            "capacity_before_clean_eligibility": True,
            "keys": [row["canonical_parent_key"] for row in analysis_pool],
        },
        "capacity_plan": {
            "common_primary_target_n": 32,
            "common_primary_floor_n": 24,
            "capacity_after_canary_reservation": len(analysis_pool),
            "capacity_sufficient_for_target_by_identity_count": len(analysis_pool) >= 32,
            "eligibility_not_yet_measured": True,
            "selection_is_not_an_outcome_result": True,
            "capacity_limit_rule": "If the clean-only common intersection is below 24, report CAPACITY_LIMIT and do not replace, top-up, or reuse Stage-Z identities.",
        },
        "suite_capacity_before_and_after_canaries": {
            suite: {"before_canaries": suite_all[suite], "after_canaries": suite_analysis[suite]}
            for suite in sorted(suite_all)
        },
        "validation_failures": failures,
    }
    return inventory, analysis_pool, failures


def build_protocol(audit: dict[str, Any], inventory: dict[str, Any]) -> dict[str, Any]:
    source = {
        "aa0_audit": artifact(AUDIT_OUT),
        "aa0_capacity": artifact(CAPACITY_OUT),
        "stage_z4_root": artifact(Z4_ROOT),
        "stage_z0r1_amendment": artifact(Z0R1_AMENDMENT),
    }
    return {
        "schema": "STAGE_AA_AA0_PROSPECTIVE_PROTOCOL_V1",
        "status": "STAGE_AA_AA0_PROSPECTIVE_PROTOCOL_FROZEN_STOP_FOR_PI",
        "gate": "STAGE_AA_AA0_ENDPOINT_REDESIGN_AND_FRESH_CAPACITY_FREEZE",
        "authorization": {
            "pi_comment_id": 5413317545,
            "pi_long_range_goal": "Prospective phase-aware fixed-denominator command-level OPEN physical-mechanism replication across M0/M1/M2.",
            "aa1_to_aa5_authorized": False,
            "aa0_only": True,
        },
        "immutable_stage_z": {
            "stage_z4_status": "STAGE_Z_CROSS_MODEL_GENERALIZATION_NOT_ESTABLISHED_ENDPOINT_VALIDITY_LIMITED_AI_ONLY",
            "stage_z_artifacts_immutable": True,
            "stage_z_primary_endpoint_not_rewritten": True,
            "stage_z_identity_reuse_forbidden": True,
        },
        "source_bindings": source,
        "estimand": {
            "primary_unit": "model_family x frozen_common_parent",
            "primary_outcome": "AA_V_PHYS",
            "definition": "Among a clean-only frozen common parent set with an established grasp and a 20-step clean-stable continuation, compare the paired command-OPEN T3/T5/T10 premature-grasp-loss rate for each model family.",
            "critical_distinction": "The denominator is frozen before any OPEN treatment; treatment endpoint, V_phys, control status, or video label cannot remove a parent.",
            "secondary_outcomes": ["aperture_excess", "contact_loss_with_physical_consequence", "premature_release", "object_drop", "object_displacement", "official_task_success_secondary"],
        },
        "population_and_denominator": {
            "inventory_artifact": CAPACITY_OUT,
            "aa1_canary_reservation_is_excluded": True,
            "common_primary_target_n": 32,
            "common_primary_floor_n": 24,
            "clean_screen_all_remaining_inventory_before_freeze": True,
            "common_intersection_rule": "Run the same clean-only eligibility rule for all three model families; primary eligible parents are the intersection of the three model-specific eligible sets.",
            "freeze_rule": {
                "if_common_eligible_at_least_32": "select exactly 32 by minimum sha256(AA2_SELECTION_SALT|canonical_parent_key)",
                "if_common_eligible_24_to_31": "freeze all common eligible parents; report exact N",
                "if_common_eligible_below_24": "CAPACITY_LIMIT; no AA3 physical matrix and no replacement/top-up",
                "selection_inputs": ["canonical_parent_key", "clean eligibility only", "AA2 selection salt"],
                "forbidden_selection_inputs": ["OPEN outcome", "V_phys", "task success", "manual label", "dose response", "model-specific treatment result"],
            },
            "fixed_denominator_invariant": "Once frozen, N is unchanged across T3/T5/T10. Invalid branches are reported separately and cannot trigger post-hoc parent deletion.",
        },
        "aa1_engineering_canaries": {
            "count": inventory["aa1_engineering_canary_reservation"]["count"],
            "source_inventory": CAPACITY_OUT,
            "exact_reserved_rows": inventory["aa1_engineering_canary_reservation"]["reserved_rows"],
            "model_families": list(MODELS),
            "scientific_use": False,
            "permanent_exclusion_from_aa2_aa5": True,
        },
        "aa2_clean_screen_and_anchor_freeze": {
            "clean_only": True,
            "model_inference_allowed_at_aa2": True,
            "open_intervention_allowed_at_aa2": False,
            "attack_or_pgd_allowed_at_aa2": False,
            "anchor_selection_is_outcome_blind": True,
            "clean_trajectory_requirements": {
                "complete_trajectory": True,
                "telemetry_valid_for_eligibility_window": True,
                "object_identity_and_baseline_valid": True,
                "minimum_remaining_horizon_steps": 20,
                "stable_grasp_window_steps": 3,
                "gripper_contact_true_for_window": True,
                "object_support_contact_false_for_window": True,
                "object_eef_distance_max_m": 0.12,
                "lift_delta_min_m": 0.015,
                "carry_relative_distance_max_m": 0.04,
                "clean_continuation_steps_after_anchor": 20,
                "clean_continuation_must_remain_grasp_stable": True,
                "clean_terminal_release_in_continuation": "ineligible; not a treatment failure",
            },
            "anchor_semantics": {
                "critical": "stable carry / established grasp only",
                "noncritical_control": "pre-contact candidate with no established grasp, selected separately and never used to define the critical denominator",
                "model_action_adapter": "M0/M1 official [0,1] gripper mapping; M2 official raw continuous clip-to-LIBERO mapping",
                "deterministic_selection": "minimum hash rank after eligibility, with fixed AA2 salt",
            },
        },
        "aa3_fixed_complete_dose_matrix": {
            "arms": [
                {"name": "CLEAN_REFERENCE", "open_steps": 0, "anchor": "critical"},
                {"name": "OPEN_T3_CRITICAL", "open_steps": 3, "anchor": "critical"},
                {"name": "OPEN_T5_CRITICAL", "open_steps": 5, "anchor": "critical"},
                {"name": "OPEN_T10_CRITICAL", "open_steps": 10, "anchor": "critical"},
                {"name": "OPEN_T5_NONCRITICAL_CONTROL", "open_steps": 5, "anchor": "noncritical_control"},
            ],
            "same_parent_across_doses": True,
            "same_anchor_across_critical_doses": True,
            "arm_preservation": "command-level OPEN changes only final gripper action coordinate; arm coordinates remain exact within frozen tolerance",
            "branch_validity": {
                "required": ["exact_state_replay", "exact_clean_action_reference", "exact_open_delivery", "telemetry_valid", "horizon_complete"],
                "invalid_branch_policy": "retain in fixed denominator, report separately, and block promotion if any primary branch is invalid",
            },
        },
        "aa_endpoint_contract": {
            "contact_false_is_not_contact_loss": True,
            "labels": ["AA_V_PHYS", "AA_NO_PREMATURE_LOSS", "AA_ENDPOINT_AMBIGUOUS", "AA_BRANCH_INVALID", "AA_NONCAUSAL_TERMINAL_RELEASE"],
            "primary_aa_v_phys": {
                "clean_grasp_precondition": "frozen AA2 established grasp plus clean 20-step stable continuation",
                "open_delivery_precondition": "exact native OPEN delivery at the frozen critical anchor for requested T3/T5/T10 steps",
                "loss_transition": "treatment has contact=True immediately before the event, then contact=False for at least two consecutive valid frames",
                "physical_consequence": "during that loss run, support contact remains false and either object z decreases by at least 0.02 m or object displacement versus the matched clean branch is at least 0.01 m",
                "clean_counterfactual_guard": "matched clean branch must remain grasp-stable over the same physical horizon; otherwise label AA_NONCAUSAL_TERMINAL_RELEASE or AA_ENDPOINT_AMBIGUOUS, never AA_V_PHYS",
                "window": "from first OPEN delivery through T_dose + 10 physical horizon steps",
            },
            "noncritical_control": "descriptive phase-control arm; contact=False alone is never a positive endpoint",
            "no_endpoint_relabeling": "AA labels do not rewrite Stage-Z V_phys, CONTROL_*_ABSTAIN, or historical labels",
        },
        "aa4_secondary_blinded_endpoint_audit": {
            "outcome_blind": True,
            "maximum_videos": 120,
            "uniform_presentation_required": True,
            "mapping_hidden_until_labels_sealed": True,
            "reviewer_type": "must be explicitly recorded; AI cannot silently satisfy a human-review requirement",
            "role": "construct-validity/interpretation only; never denominator repair or V_phys relabeling",
        },
        "statistical_promotion_criteria": {
            "alpha": 0.05,
            "multiple_testing": "Holm correction across the three model-specific primary T10-versus-T3 paired tests",
            "primary_test": "one-sided exact paired McNemar test for AA_V_PHYS(T10) > AA_V_PHYS(T3)",
            "minimum_effect": "absolute paired rate difference p_T10 - p_T3 >= 0.20",
            "dose_monotonicity": "p_T3 <= p_T5 <= p_T10 at the frozen common denominator",
            "phase_specific_secondary": "T5 critical rate exceeds T5 noncritical-control rate by at least 0.20 with valid control coverage; otherwise phase-specific promotion is not established",
            "replicated_3_of_3": "N>=24, 100% primary branch validity, endpoint audit not failed, and all three model families satisfy the primary dose test and dose monotonicity",
            "partial_model_dependent": "same validity requirements, but only one or two model families satisfy the primary criteria",
            "generalization_not_established": "same validity requirements, but zero model families satisfy the primary criteria",
            "measurement_or_capacity_limited": "N<24, any primary branch invalid, or endpoint audit/construct validity fails; do not convert this to a scientific negative",
            "secondary_reporting": ["exact 000/001/011/111 dose patterns", "non-monotone patterns", "Wilson or exact binomial intervals", "aperture/contact/displacement telemetry", "task success as secondary only"],
        },
        "forbidden_until_new_pi_authorization": [
            "AA1-AA5 execution",
            "new model inference",
            "new env.step",
            "new OPEN intervention",
            "PGD",
            "Stage-Z identity reuse",
            "replacement or top-up",
            "protected/Eval160",
            "BRIDGE/F1",
            "Paper V2 promotion",
        ],
        "scientific_firewall": {
            "new_model_inference": 0,
            "new_env_step": 0,
            "new_open_intervention": 0,
            "pgd": 0,
            "protected_reads": 0,
            "eval160_reads": 0,
            "fresh_science_exposure": 0,
            "stage_z_mutation": 0,
        },
        "next_legal_action": "STOP_FOR_PI",
    }


def main() -> None:
    audit, z3_root, audit_failures = build_audit()
    inventory, _analysis_pool, capacity_failures = build_capacity()
    failures = audit_failures + capacity_failures
    audit["validation_failures"] = failures
    inventory["validation_failures"] = failures
    if failures:
        raise SystemExit(json.dumps({"status": "FAIL_AA0_STATIC_VALIDATION", "failures": failures}, indent=2))
    write_json(AUDIT_OUT, audit)
    write_json(CAPACITY_OUT, inventory)
    protocol = build_protocol(audit, inventory)
    write_json(PROTOCOL_OUT, protocol)

    manifest_paths = [
        Z0_PANEL, Z0R1_PANEL, Z0R1_AMENDMENT, Z2_AVAILABILITY, Z2_TERMINAL,
        Z3_PROTOCOL, Z3_ELIGIBILITY, Z3_TERMINAL, Z3_ROOT, Z3_CONTRACT,
        Z4_CROSS_AUDIT, Z4_ROOT, Z4_SYNTHESIS, AUDIT_OUT, CAPACITY_OUT,
        PROTOCOL_OUT, SCRIPT_REL,
    ]
    root_seal = {
        "schema": "STAGE_AA_AA0_ROOT_SEAL_V1",
        "status": "STAGE_AA_AA0_PROSPECTIVE_PROTOCOL_FROZEN_STOP_FOR_PI",
        "gate": "STAGE_AA_AA0_ENDPOINT_REDESIGN_AND_FRESH_CAPACITY_FREEZE",
        "git_binding": git_binding(),
        "artifact_manifest": {
            "entries": [artifact(path) for path in manifest_paths],
            "root_seal_excludes_self": True,
        },
        "population": {
            "fresh_inventory_count": inventory["analysis_pool_after_aa1_reservation"]["count"],
            "aa1_reserved_canaries": inventory["aa1_engineering_canary_reservation"]["count"],
            "common_primary_target_n": protocol["population_and_denominator"]["common_primary_target_n"],
            "common_primary_floor_n": protocol["population_and_denominator"]["common_primary_floor_n"],
            "structural_missing_task_cells": inventory["structural_missing_task_cells"],
        },
        "historical_audit": {
            "z3_parent_count": len(load(Z3_TERMINAL)["physical_results"]["parent_rows"]),
            "old_clean_gripper_contact_loss_count": audit["full_matrix_clean_class_counts"].get("GRIPPER_CONTACT_LOSS", 0),
            "old_clean_total": sum(audit["full_matrix_clean_class_counts"].values()),
            "stage_z_immutable": True,
        },
        "scientific_firewall": protocol["scientific_firewall"],
        "validation_failures": [],
        "next_legal_action": "STOP_FOR_PI_NO_AA1_AA2_AA3_AA4_AA5",
    }
    write_json(ROOT_OUT, root_seal)
    root_hash = sha256(ROOT / ROOT_OUT)
    (ROOT / SIDECAR_OUT).write_text(f"{root_hash}  {Path(ROOT_OUT).name}\n", encoding="utf-8")
    print(json.dumps({
        "status": root_seal["status"],
        "fresh_inventory": inventory["analysis_pool_after_aa1_reservation"]["count"],
        "aa1_canaries": inventory["aa1_engineering_canary_reservation"]["count"],
        "old_clean_contact_loss": audit["full_matrix_clean_class_counts"].get("GRIPPER_CONTACT_LOSS", 0),
        "z3_parents": len(load(Z3_TERMINAL)["physical_results"]["parent_rows"]),
        "root_sha256": root_hash,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
