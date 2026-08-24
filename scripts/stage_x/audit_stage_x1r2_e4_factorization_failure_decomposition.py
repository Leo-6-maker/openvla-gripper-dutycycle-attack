#!/usr/bin/env python3
"""Offline, sealed-artifact E4 decomposition for the E3 candidate audit."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any


EXPECTED_DECISION = "E3_STRICT_SELECTIVE_REALIZABILITY_PARTIAL_SUITE_DEPENDENT"
EXPECTED_SOURCES = ["delta0", *[f"pgd_iteration_{i}" for i in range(1, 6)]]
CLASS_NAMES = (
    "ARM_EXACT_AND_NATIVE_OPEN",
    "ARM_EXACT_BUT_NOT_NATIVE_OPEN",
    "NATIVE_OPEN_BUT_ARM_DRIFT",
    "NEITHER_OPEN_NOR_ARM_EXACT",
    "EVIDENCE_MISSING",
)
REQUIRED_FIELDS = (
    "candidate_index",
    "candidate_source",
    "arm_token_ids_equal",
    "arm_mismatch_dimensions",
    "direct_generated_token_ids",
    "direct_generated_arm_token_ids",
    "direct_generated_gripper_token_id",
    "direct_generated_gripper_is_native_open",
    "gripper_token_changed",
    "processor_input_sha256",
    "pixel_budget_adv_inputs_linf",
    "delta_sha256",
)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def git_value(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], check=True, capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "UNAVAILABLE"


def verify_e3_seal(decision_path: Path, e3_root: Path) -> dict[str, Any]:
    decision = load_json(decision_path)
    seal_path = e3_root / "E3_ROOT_SEAL_V1.json"
    sidecar_path = e3_root / "E3_ROOT_SEAL_V1.sha256"
    seal = load_json(seal_path)
    seal_hash = sha256_file(seal_path)
    sidecar = sidecar_path.read_text(encoding="utf-8").split()[0]
    if sidecar != seal_hash:
        raise ValueError("E3 root seal sidecar mismatch")
    if seal.get("decision_sha256") != sha256_file(decision_path):
        raise ValueError("E3 decision hash does not match E3 root seal")
    if decision.get("status") != EXPECTED_DECISION:
        raise ValueError(f"unexpected E3 decision: {decision.get('status')}")
    if decision.get("fixed_denominator") != 12 or len(decision.get("parent_rows", [])) != 12:
        raise ValueError("E3 denominator is not the frozen 12-parent denominator")
    if decision.get("incomplete_evidence_rows") or decision.get("runtime_hold_rows"):
        raise ValueError("E3 contains incomplete or held parent rows")
    missing = []
    mismatched = []
    for entry in seal.get("artifact_manifest", []):
        path = e3_root / str(entry["path"])
        if not path.is_file():
            missing.append(str(entry["path"]))
            continue
        if int(path.stat().st_size) != int(entry["bytes"]) or sha256_file(path) != entry["sha256"]:
            mismatched.append(str(entry["path"]))
    if missing or mismatched:
        raise ValueError(json.dumps({"missing_e3_artifacts": missing, "mismatched_e3_artifacts": mismatched}, sort_keys=True))
    return {
        "decision_path": str(decision_path),
        "decision_sha256": sha256_file(decision_path),
        "root_seal_path": str(seal_path),
        "root_seal_sha256": seal_hash,
        "root_seal_status": seal.get("status"),
        "artifact_count": seal.get("artifact_count"),
        "e3_status": decision["status"],
    }


def candidate_class(candidate: dict[str, Any]) -> tuple[str, list[str], list[str]]:
    missing = [field for field in REQUIRED_FIELDS if field not in candidate]
    errors: list[str] = []
    if candidate.get("candidate_source") not in EXPECTED_SOURCES:
        errors.append("unexpected_candidate_source")
    if candidate.get("candidate_index") not in range(6):
        errors.append("unexpected_candidate_index")
    if not isinstance(candidate.get("arm_token_ids_equal"), bool):
        errors.append("arm_token_ids_equal_not_bool")
    if not isinstance(candidate.get("direct_generated_gripper_is_native_open"), bool):
        errors.append("native_open_not_bool")
    if not isinstance(candidate.get("arm_mismatch_dimensions"), list):
        errors.append("arm_mismatch_dimensions_not_list")
    if not isinstance(candidate.get("direct_generated_token_ids"), list) or len(candidate.get("direct_generated_token_ids", [])) != 7:
        errors.append("direct_token_count_not_7")
    if not isinstance(candidate.get("direct_generated_arm_token_ids"), list) or len(candidate.get("direct_generated_arm_token_ids", [])) != 6:
        errors.append("direct_arm_token_count_not_6")
    if missing:
        return "EVIDENCE_MISSING", missing, errors
    if errors:
        return "EVIDENCE_MISSING", [], errors
    exact = bool(candidate["arm_token_ids_equal"])
    native_open = bool(candidate["direct_generated_gripper_is_native_open"])
    if exact and native_open:
        return "ARM_EXACT_AND_NATIVE_OPEN", [], []
    if exact:
        return "ARM_EXACT_BUT_NOT_NATIVE_OPEN", [], []
    if native_open:
        return "NATIVE_OPEN_BUT_ARM_DRIFT", [], []
    return "NEITHER_OPEN_NOR_ARM_EXACT", [], []


def parent_category(rows: list[dict[str, Any]]) -> str:
    classes = [row["classification"] for row in rows]
    if "EVIDENCE_MISSING" in classes:
        return "EVIDENCE_MISSING"
    if "ARM_EXACT_AND_NATIVE_OPEN" in classes:
        return "STRICT_REALIZABLE"
    any_native = any(row["native_open"] for row in rows)
    any_exact = any(row["arm_exact"] for row in rows)
    if not any_native:
        return "TARGETABILITY_LIMITED"
    if not any_exact:
        return "SELECTIVITY_LIMITED"
    return "JOINT_LIMITED"


def parent_row(parent: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    receipt_path = Path(str(parent["receipt_path"]))
    receipt = load_json(receipt_path)
    true = receipt.get("true_receipt") or {}
    candidates = list(true.get("candidate_audit") or [])
    if len(candidates) != 6:
        raise ValueError(f"{parent['fixture_id']}: expected six candidate rows, got {len(candidates)}")
    rows = []
    for candidate in sorted(candidates, key=lambda item: int(item.get("candidate_index", -1))):
        classification, missing, errors = candidate_class(candidate)
        rows.append({
            "candidate_index": candidate.get("candidate_index"),
            "candidate_source": candidate.get("candidate_source"),
            "classification": classification,
            "evidence_missing_fields": missing,
            "structural_errors": errors,
            "arm_exact": candidate.get("arm_token_ids_equal") if isinstance(candidate.get("arm_token_ids_equal"), bool) else None,
            "native_open": candidate.get("direct_generated_gripper_is_native_open") if isinstance(candidate.get("direct_generated_gripper_is_native_open"), bool) else None,
            "gripper_token_changed": candidate.get("gripper_token_changed"),
            "direct_generated_token_ids": candidate.get("direct_generated_token_ids"),
            "direct_generated_arm_token_ids": candidate.get("direct_generated_arm_token_ids"),
            "direct_generated_gripper_token_id": candidate.get("direct_generated_gripper_token_id"),
            "arm_mismatch_dimensions": candidate.get("arm_mismatch_dimensions"),
            "processor_input_sha256": candidate.get("processor_input_sha256"),
            "pixel_budget_adv_inputs_linf": candidate.get("pixel_budget_adv_inputs_linf"),
            "delta_sha256": candidate.get("delta_sha256"),
        })
    category = parent_category(rows)
    exact_indices = [row["candidate_index"] for row in rows if row["arm_exact"] is True]
    native_indices = [row["candidate_index"] for row in rows if row["native_open"] is True]
    strict_indices = [row["candidate_index"] for row in rows if row["classification"] == "ARM_EXACT_AND_NATIVE_OPEN"]
    mismatch_histogram = Counter(str(dim) for row in rows for dim in (row["arm_mismatch_dimensions"] or []))
    summary = {
        "suite": parent["suite"],
        "fixture_id": parent["fixture_id"],
        "canonical_parent_key": parent["canonical_parent_key"],
        "receipt_path": str(receipt_path),
        "receipt_status": receipt.get("status"),
        "selected_candidate_index": true.get("selected_candidate_index"),
        "candidate_count": len(rows),
        "candidate_class_counts": dict(Counter(row["classification"] for row in rows)),
        "any_candidate_native_open": bool(native_indices),
        "any_candidate_exact_arm": bool(exact_indices),
        "any_strict_valid_candidate": bool(strict_indices),
        "earliest_native_open_candidate_index": min(native_indices) if native_indices else None,
        "earliest_exact_arm_candidate_index": min(exact_indices) if exact_indices else None,
        "earliest_strict_valid_candidate_index": min(strict_indices) if strict_indices else None,
        "arm_mismatch_dimension_histogram": dict(sorted(mismatch_histogram.items())),
        "parent_failure_category": category,
        "candidate_evidence_complete": not any(row["classification"] == "EVIDENCE_MISSING" for row in rows),
    }
    return summary, rows


def suite_summary(parents: list[dict[str, Any]]) -> dict[str, Any]:
    suites = sorted({str(parent["suite"]) for parent in parents})
    result = {}
    for suite in suites:
        rows = [parent for parent in parents if parent["suite"] == suite]
        histogram = Counter()
        for row in rows:
            histogram.update(row["arm_mismatch_dimension_histogram"])
        result[suite] = {
            "parents": len(rows),
            "parent_failure_categories": dict(Counter(row["parent_failure_category"] for row in rows)),
            "any_native_open": sum(row["any_candidate_native_open"] for row in rows),
            "any_exact_arm": sum(row["any_candidate_exact_arm"] for row in rows),
            "any_strict_valid": sum(row["any_strict_valid_candidate"] for row in rows),
            "arm_mismatch_dimension_histogram": dict(sorted(histogram.items())),
        }
    return result


def synthesis_rows() -> list[dict[str, Any]]:
    # These are claim-safe handoff summaries; populations are deliberately not joined.
    return [
        {"stage": "X0", "status": "STAGE_X_PHYSICAL_DUTY_CYCLE_MECHANISM_SUPPORTED", "population_or_denominator": "1,126 complete three-dose probe rows; 40 Stage V + 16 Stage VI-B2 parents", "result_summary": "OPEN dose response T3/T5/T10 raw positive rates 0.39438/0.67758/0.87300; aperture excess, contact loss, and displacement increase", "claim": "Dose- and phase-dependent physical OPEN duty-cycle mechanism is supported descriptively and mechanistically; no formal mediation claim.", "source": "docs/handoffs/STAGE_X_X0_RESULT_20260817.md", "descriptive": True, "predictive": False, "mechanistic": True, "causal_counterfactual_bounded": True, "model_side_exploitability": False, "attack_efficacy": False, "engineering_diagnostic_only": False, "identity_join": "NONE"},
        {"stage": "VI-B2", "status": "STUDENT_HELDOUT_GENERALIZATION_NOT_ESTABLISHED", "population_or_denominator": "16 fresh parents; T5 333 consumable and 51 abstain/censored", "result_summary": "Held-out B2-C overall AUROC 0.624643; ECE-10 0.460636; suite failures material", "claim": "Clean/context timing criticality did not establish stable held-out causal localization or actionable generalization.", "source": "docs/handoffs/STAGE_VI_B2_FRESH_M4_AND_NEGATIVE_CAUSAL_HANDOFF_20260816.md", "descriptive": True, "predictive": True, "mechanistic": False, "causal_counterfactual_bounded": False, "model_side_exploitability": False, "attack_efficacy": False, "engineering_diagnostic_only": False, "identity_join": "NONE"},
        {"stage": "VII", "status": "STAGE_VII_DEVELOPMENT_NO_GENERALIZABLE_DETECTOR", "population_or_denominator": "Three frozen detector candidates with suite-level promotion gates", "result_summary": "S7-A/S7-B/S7-C all failed at least one cross-suite generalization/selectivity gate; none promoted", "claim": "No frozen candidate established a generalizable cross-suite timing detector under the predeclared gates.", "source": "docs/handoffs/STAGE_VII_DEVELOPMENT_NEGATIVE_HANDOFF_20260816.md", "descriptive": True, "predictive": True, "mechanistic": False, "causal_counterfactual_bounded": False, "model_side_exploitability": False, "attack_efficacy": False, "engineering_diagnostic_only": False, "identity_join": "NONE"},
        {"stage": "VIII", "status": "STAGE_VIII_R1_NO_GENERALIZABLE_RELATIVE_SELECTOR", "population_or_denominator": "Referenced sealed Stage VIII result; direct handoff file is not present in this checkout", "result_summary": "Stage IX sealed handoff preserves the Stage VIII negative status; no identity-level join is attempted", "claim": "Relative timing selector generalization was not established; this row is included only as a sealed-status reference.", "source": "docs/handoffs/STAGE_IX_F0_RESULT_20260817.md", "descriptive": True, "predictive": True, "mechanistic": False, "causal_counterfactual_bounded": False, "model_side_exploitability": False, "attack_efficacy": False, "engineering_diagnostic_only": False, "identity_join": "NONE"},
        {"stage": "IX", "status": "STAGE_IX_NO_MODEL_SIDE_TIMING_SIGNAL", "population_or_denominator": "1,344 sealed no-environment rows", "result_summary": "Model-side scores were high (DEVTEST AUROC 0.870743–0.900510) while factorized parent-macro AUC was 0.483698–0.523390 and LOSO remained weak", "claim": "Model-side targetability did not provide reliable physical timing utility; this is the factorization gap, not physical attack efficacy.", "source": "docs/handoffs/STAGE_IX_F0_RESULT_20260817.md", "descriptive": True, "predictive": True, "mechanistic": False, "causal_counterfactual_bounded": False, "model_side_exploitability": True, "attack_efficacy": False, "engineering_diagnostic_only": False, "identity_join": "NONE"},
        {"stage": "E2", "status": "HOLD_E2_FOUR_SUITE_BRANCH_QUALIFICATION_INCOMPLETE_NO_LEGAL_GOAL_EMIT", "population_or_denominator": "Three bounded Goal successor identities; no TRUE probe", "result_summary": "All three Goal clean references had no legal Student emit; this is a timing/scheduler feasibility hold", "claim": "E2 supplies bounded timing-selector non-emission evidence and is not a strict visual-method negative.", "source": "docs/handoffs/STAGE_X_X1R2_Q3R3_E2_SUCCESSOR_BRANCH_QUALIFICATION_HOLD_20260821.md", "descriptive": True, "predictive": False, "mechanistic": False, "causal_counterfactual_bounded": False, "model_side_exploitability": False, "attack_efficacy": False, "engineering_diagnostic_only": True, "identity_join": "NONE"},
        {"stage": "E3", "status": EXPECTED_DECISION, "population_or_denominator": "12 fresh engineering-only parents; 72 ordered candidate slots", "result_summary": "Strict-valid parents: libero_10 1/3, libero_spatial 1/3, libero_goal 0/3, libero_object 0/3", "claim": "Timing-decoupled strict selective visual realizability exists sparsely and suite/state-dependently under the frozen method; it is not physical efficacy.", "source": "reports/STAGE_X1R2_E3_FACTORIZED_SELECTIVE_REALIZABILITY_20260821/E3_DECISION_TABLE_V1.json", "descriptive": True, "predictive": False, "mechanistic": False, "causal_counterfactual_bounded": False, "model_side_exploitability": True, "attack_efficacy": False, "engineering_diagnostic_only": False, "identity_join": "NONE"},
    ]


def write_synthesis(path: Path) -> None:
    rows = synthesis_rows()
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def build_claim_ledger(decomposition: dict[str, Any], synthesis_path: str, input_binding: dict[str, Any]) -> dict[str, Any]:
    strict = decomposition["global_summary"]["parents_with_strict_valid_candidate"]
    return {
        "schema": "STAGE_X_X1R2_E4_FINAL_CLAIM_LEDGER_V1",
        "status": "STAGE_X_X1R2_E4_PAPER_LOCK_READY",
        "preferred_thesis": "Gripper-targeted manipulation failures exhibit a dose- and phase-dependent OPEN duty-cycle physical mechanism, while clean timing criticality, physical vulnerability, and visual exploitability only partially align. Clean/context timing selectors fail to generalize reliably across suites, and even timing-decoupled strict gripper-selective visual realizability is sparse and suite/state dependent under the frozen method.",
        "input_binding": input_binding,
        "e3_structural_summary": {
            "fixed_parent_denominator": decomposition["global_summary"]["parents"],
            "candidate_slots": decomposition["global_summary"]["candidate_slots"],
            "parents_with_strict_valid_candidate": strict,
            "suite_summary": decomposition["suite_summary"],
        },
        "promotable_claims": [
            {"id": "X0_MECHANISM", "claim": "X0 supports a dose- and phase-dependent physical OPEN duty-cycle mechanism through command delivery, aperture excess, contact loss, and displacement.", "role": "descriptive_mechanistic_bounded_counterfactual", "source": "docs/handoffs/STAGE_X_X0_RESULT_20260817.md", "boundary": "No formal mediation and no protected evaluation."},
            {"id": "TIMING_NEGATIVE_CASCADE", "claim": "VI-B2, VII, and VIII do not establish stable cross-suite timing-selector generalization under their frozen gates.", "role": "descriptive_predictive_negative", "source": synthesis_path, "boundary": "Populations and estimands remain separate; no identity-level join."},
            {"id": "IX_FACTORIZATION_GAP", "claim": "Stage IX separates model-side targetability from physical timing utility; high model-side scores did not establish factorized timing utility.", "role": "descriptive_predictive_model_side", "source": "docs/handoffs/STAGE_IX_F0_RESULT_20260817.md", "boundary": "Not physical attack efficacy."},
            {"id": "E2_TIMING_HOLD", "claim": "E2 is bounded Goal timing/scheduler non-emission evidence, not a strict visual-method negative.", "role": "engineering_diagnostic", "source": "docs/handoffs/STAGE_X_X1R2_Q3R3_E2_SUCCESSOR_BRANCH_QUALIFICATION_HOLD_20260821.md", "boundary": "No TRUE, physical, V_phys, Eval160, or protected read."},
            {"id": "E3_SPARSE_REALIZABILITY", "claim": f"E3 establishes bounded model-side strict selective realizability in {strict} of 12 engineering-only parents, with suite/state dependence.", "role": "model_side_exploitability_descriptive", "source": input_binding["decision_path"], "boundary": "No physical efficacy, prevalence, cross-suite general capability, or impossibility claim."},
        ],
        "not_promotable_claims": [
            "universal or generalizable detector",
            "cross-suite visual-PGD physical efficacy",
            "Goal/Object visual attack impossibility",
            "detector caused E3 failure",
            "formal mediation from X0",
            "attack efficacy inferred from E3-valid parents",
            "protected or Eval160 validation",
        ],
        "identity_join_policy": "No identity-level joins across X0, VI-B2, VII, VIII, IX, E2, and E3; synthesis is claim-safe and stage-level only.",
        "attack_efficacy": False,
        "protected_evaluation": "UNREAD",
        "mandatory_stop": "OWNER_PI_REVIEW_REQUIRED; no new attack objective, tuning, detector retraining, enlarged pool, RAND, SHUFFLED, R0/R1/R2, physical arm, Eval160, or protected read",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--e3-decision", type=Path, required=True)
    parser.add_argument("--e3-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise SystemExit(f"refusing non-empty output root: {args.output_root}")
    args.output_root.mkdir(parents=True, exist_ok=True)

    e3_binding = verify_e3_seal(args.e3_decision, args.e3_root)
    decision = load_json(args.e3_decision)
    parent_summaries = []
    candidate_rows = []
    for parent in decision["parent_rows"]:
        summary, rows = parent_row(parent)
        parent_summaries.append(summary)
        candidate_rows.append({"parent": summary, "candidates": rows})
    candidate_slots = sum(len(row["candidates"]) for row in candidate_rows)
    class_counts = Counter(row["classification"] for parent in candidate_rows for row in parent["candidates"])
    global_summary = {
        "parents": len(parent_summaries),
        "candidate_slots": candidate_slots,
        "candidate_class_counts": dict(class_counts),
        "parents_with_native_open": sum(row["any_candidate_native_open"] for row in parent_summaries),
        "parents_with_exact_arm": sum(row["any_candidate_exact_arm"] for row in parent_summaries),
        "parents_with_strict_valid_candidate": sum(row["any_strict_valid_candidate"] for row in parent_summaries),
        "parent_failure_categories": dict(Counter(row["parent_failure_category"] for row in parent_summaries)),
        "evidence_missing_parent_count": sum(not row["candidate_evidence_complete"] for row in parent_summaries),
    }
    if candidate_slots != 72 or global_summary["evidence_missing_parent_count"]:
        raise ValueError("E4 requires 12 parents x 6 complete candidate rows")
    synthesis_path = "STAGE_X_X1R2_E4_FACTORIZATION_SYNTHESIS_TABLE_V1.csv"
    input_binding = {
        **e3_binding,
        "source_commit": git_value("rev-parse", "HEAD"),
        "source_tree": git_value("rev-parse", "HEAD^{tree}"),
        "source_status": git_value("status", "--porcelain=v1"),
        "execution_mode": "offline_cpu_static_sealed_artifact_analysis",
        "forbidden_execution": {"gpu": 0, "openvla_inference": 0, "simulator": 0, "env_step": 0, "pgd": 0, "backward": 0, "physical_intervention": 0, "vphys": 0, "eval160": 0, "protected": 0},
        "candidate_slot_denominator": "12 parents x 6 ordered candidates = 72; parent-level aggregation is primary",
    }
    decomposition = {
        "schema": "STAGE_X_X1R2_E4_E3_CANDIDATE_FAILURE_DECOMPOSITION_V1",
        "status": "STAGE_X_X1R2_E4_DECOMPOSITION_PASS",
        "input_binding": input_binding,
        "classification_definitions": {
            "ARM_EXACT_AND_NATIVE_OPEN": "arm_token_ids_equal=true and direct_generated_gripper_is_native_open=true",
            "ARM_EXACT_BUT_NOT_NATIVE_OPEN": "arm_token_ids_equal=true and direct_generated_gripper_is_native_open=false",
            "NATIVE_OPEN_BUT_ARM_DRIFT": "arm_token_ids_equal=false and direct_generated_gripper_is_native_open=true",
            "NEITHER_OPEN_NOR_ARM_EXACT": "arm_token_ids_equal=false and direct_generated_gripper_is_native_open=false",
            "EVIDENCE_MISSING": "a required sealed candidate field is absent or structurally invalid; no field is inferred",
        },
        "parent_failure_category_definitions": {
            "STRICT_REALIZABLE": "at least one candidate is ARM_EXACT_AND_NATIVE_OPEN",
            "TARGETABILITY_LIMITED": "no candidate reaches native OPEN",
            "SELECTIVITY_LIMITED": "native OPEN occurs, but no candidate preserves exact arm tokens",
            "JOINT_LIMITED": "native OPEN and exact-arm candidates occur separately, but no candidate satisfies both",
            "EVIDENCE_MISSING": "one or more candidate rows cannot be classified from sealed fields",
        },
        "global_summary": global_summary,
        "suite_summary": suite_summary(parent_summaries),
        "parent_rows": [{"summary": item["parent"], "candidates": item["candidates"]} for item in candidate_rows],
    }
    decomposition_path = args.output_root / "STAGE_X1R2_E4_E3_CANDIDATE_FAILURE_DECOMPOSITION_V1.json"
    synthesis_file = args.output_root / "STAGE_X1R2_E4_FACTORIZATION_SYNTHESIS_TABLE_V1.csv"
    ledger_path = args.output_root / "STAGE_X1R2_E4_FINAL_CLAIM_LEDGER_V1.json"
    handoff_path = args.output_root / "STAGE_X1R2_E4_PAPER_LOCK_20260821.md"
    binding_path = args.output_root / "E4_INPUT_BINDING_V1.json"
    write_json(binding_path, input_binding)
    write_json(decomposition_path, decomposition)
    write_synthesis(synthesis_file)
    write_json(ledger_path, build_claim_ledger(decomposition, synthesis_path, input_binding))
    handoff_path.write_text(
        "# E4 factorization failure decomposition and paper lock\n\n"
        "Status: `STAGE_X_X1R2_E4_PAPER_LOCK_READY`.\n\n"
        "E4 is a sealed-artifact, offline CPU analysis of the E3 denominator: 12 parents × 6 ordered candidates = 72 candidate slots. It does not perform OpenVLA inference, simulator construction, environment stepping, PGD/backward, physical intervention, V_phys, Eval160, or protected reads.\n\n"
        "The candidate-level decomposition is primary descriptive evidence; parent-level categories are the scientific aggregation. No candidate slots are treated as iid observations. X0, VI-B2, VII, VIII, IX, E2, and E3 are synthesized at stage level without identity-level joins or formal mediation claims.\n\n"
        "The preferred paper thesis is mechanism-first: a dose- and phase-dependent physical OPEN duty-cycle mechanism coexists with repeated timing-selector negatives and sparse, suite/state-dependent timing-decoupled strict visual realizability. Model-side exploitability is not physical attack efficacy.\n\n"
        "Mandatory stop: Owner/PI review required. No new attack objective, tuning, detector retraining, enlarged pool, RAND, SHUFFLED, R0/R1/R2, physical arm, Eval160, or protected read is authorized from E4.\n",
        encoding="utf-8",
    )
    entries = []
    for path in sorted(item for item in args.output_root.rglob("*") if item.is_file()):
        if path.name in {"E4_ROOT_SEAL_V1.json", "E4_ROOT_SEAL_V1.sha256"}:
            continue
        entries.append({"path": path.relative_to(args.output_root).as_posix(), "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    seal = {
        "schema": "STAGE_X_X1R2_E4_FACTORIZATION_FAILURE_DECOMPOSITION_ROOT_SEAL_V1",
        "status": "SEALED_STAGE_X_X1R2_E4_FACTORIZATION_FAILURE_DECOMPOSITION_AND_PAPER_LOCK",
        "input_decision_sha256": e3_binding["decision_sha256"],
        "input_e3_root_seal_sha256": e3_binding["root_seal_sha256"],
        "artifact_count": len(entries),
        "artifact_manifest": entries,
        "protected_boundary": input_binding["forbidden_execution"],
        "mandatory_stop": "OWNER_PI_REVIEW_REQUIRED",
    }
    seal_path = args.output_root / "E4_ROOT_SEAL_V1.json"
    write_json(seal_path, seal)
    (args.output_root / "E4_ROOT_SEAL_V1.sha256").write_text(f"{sha256_file(seal_path)}  E4_ROOT_SEAL_V1.json\n", encoding="utf-8")
    print(json.dumps({"status": seal["status"], "root": str(args.output_root), "parents": len(parent_summaries), "candidate_slots": candidate_slots, "categories": global_summary["parent_failure_categories"], "artifact_count": len(entries)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
