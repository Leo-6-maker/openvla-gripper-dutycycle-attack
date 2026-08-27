#!/usr/bin/env python3
"""Audit AA2 eligibility evidence without rerunning any model or environment.

The AA2 receipts intentionally retain summaries rather than full clean rows.
This audit therefore proves what is and is not reconstructible from the sealed
bytes; it never invents a corrected denominator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


MODELS = ("M0_OPENVLA", "M1_OPENVLA_OFT", "M2_PI05_LIBERO")
SUITES = ("libero_10", "libero_object", "libero_spatial")
FORBIDDEN_COUNTERS = (
    "open_intervention_steps",
    "attacked_env_steps",
    "pgd_calls",
    "aa_v_phys_reads",
    "attack_outcome_reads",
    "task_success_reads",
    "eval160_reads",
    "protected_reads",
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value in (0, 1):
        return bool(value)
    raise ValueError(f"BOOLEAN_FIELD_INVALID:{value!r}")


def display_path(path: Path, root: Path) -> str:
    try:
        relative = path.relative_to(root)
    except ValueError:
        return str(path)
    return str(relative).replace("\\", "/")


def binding(path: Path, root: Path) -> dict[str, Any]:
    return {
        "path": display_path(path, root),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def lines_with(text: str, needles: tuple[str, ...]) -> list[dict[str, Any]]:
    result = []
    for number, line in enumerate(text.splitlines(), 1):
        if any(needle in line for needle in needles):
            result.append({"line": number, "text": line.strip()})
    return result


def evidence_rows(anchor: Any) -> list[dict[str, Any]]:
    if not isinstance(anchor, dict):
        return []
    rows = anchor.get("evidence_rows")
    return rows if isinstance(rows, list) else []


def distance_summary(receipts: list[dict[str, Any]]) -> dict[str, Any]:
    selected = []
    for item in receipts:
        anchor = item["receipt"].get("eligibility", {}).get("critical_anchor")
        rows = evidence_rows(anchor)
        if not rows:
            continue
        distances = [row.get("object_eef_distance_m") for row in rows if isinstance(row.get("object_eef_distance_m"), (int, float))]
        if distances:
            selected.append(
                {
                    "cell_id": item["cell_id"],
                    "model_family": item["model_family"],
                    "suite": item["suite"],
                    "anchor_step": anchor.get("step") if isinstance(anchor, dict) else None,
                    "evidence_rows": len(rows),
                    "distance_min_m": min(float(value) for value in distances),
                    "distance_max_m": max(float(value) for value in distances),
                }
            )
    return {
        "accepted_critical_anchor_count_with_evidence": len(selected),
        "accepted_critical_anchor_evidence": selected,
        "rejected_candidate_distance_rows_available": False,
        "distance_failure_count_computable": False,
        "reason": "Rejected candidates retain reason counts but not per-row distance values or failure-reason intersections.",
    }


def audit(args: argparse.Namespace) -> dict[str, Any]:
    root = args.root.resolve()
    index_path = args.receipt_index if args.receipt_index.is_absolute() else root / args.receipt_index
    terminal_path = args.terminal if args.terminal.is_absolute() else root / args.terminal
    root_seal_path = args.root_seal if args.root_seal.is_absolute() else root / args.root_seal
    manifest_path = args.manifest if args.manifest.is_absolute() else root / args.manifest
    aa0_path = args.aa0 if args.aa0.is_absolute() else root / args.aa0
    protocol_path = args.protocol if args.protocol.is_absolute() else root / args.protocol
    runner_path = args.runner if args.runner.is_absolute() else root / args.runner
    taxonomy_path = args.taxonomy if args.taxonomy.is_absolute() else root / args.taxonomy

    index = load_json(index_path)
    terminal = load_json(terminal_path)
    root_seal = load_json(root_seal_path)
    manifest = load_json(manifest_path)
    aa0 = load_json(aa0_path)
    protocol = load_json(protocol_path)
    runner_text = runner_path.read_text(encoding="utf-8")
    taxonomy_text = taxonomy_path.read_text(encoding="utf-8")

    indexed = index.get("receipts")
    if not isinstance(indexed, list) or len(indexed) != 324:
        raise RuntimeError("RECEIPT_INDEX_NOT_324")
    if manifest.get("cell_count") != 324 or len(manifest.get("cells", [])) != 324:
        raise RuntimeError("MANIFEST_NOT_324")

    manifest_by_id = {row["cell_id"]: row for row in manifest["cells"]}
    if len(manifest_by_id) != 324:
        raise RuntimeError("MANIFEST_CELL_IDS_NOT_UNIQUE")

    receipts: list[dict[str, Any]] = []
    verified_receipts = 0
    raw_rows = 0
    raw_actions = 0
    critical_evidence_groups = 0
    noncritical_evidence_groups = 0
    critical_evidence_rows = 0
    noncritical_evidence_rows = 0
    forbidden = Counter()
    by_model: dict[str, dict[str, Any]] = defaultdict(lambda: {"cells": 0, "complete": 0, "truncated": 0, "truncated_with_at_least_20_steps": 0, "critical_eligible": 0, "noncritical_eligible": 0, "censored_candidate_attempts": 0})
    by_suite: dict[str, dict[str, Any]] = defaultdict(lambda: {"cells": 0, "complete": 0, "truncated": 0, "critical_eligible": 0, "noncritical_eligible": 0, "censored_candidate_attempts": 0})
    reason_counts = defaultdict(Counter)
    seen_ids = set()

    for item in indexed:
        cell_id = item.get("cell_id")
        if cell_id in seen_ids:
            raise RuntimeError(f"DUPLICATE_CELL:{cell_id}")
        seen_ids.add(cell_id)
        if cell_id not in manifest_by_id:
            raise RuntimeError(f"CELL_NOT_IN_MANIFEST:{cell_id}")
        path_value = Path(str(item["path"]))
        receipt_path = path_value if path_value.is_absolute() else root / path_value
        if not receipt_path.is_file():
            raise RuntimeError(f"RECEIPT_MISSING:{cell_id}:{receipt_path}")
        if receipt_path.stat().st_size != int(item["bytes"]):
            raise RuntimeError(f"RECEIPT_BYTES_MISMATCH:{cell_id}")
        if sha256_file(receipt_path) != item["sha256"]:
            raise RuntimeError(f"RECEIPT_SHA_MISMATCH:{cell_id}")
        receipt = load_json(receipt_path)
        cell = manifest_by_id[cell_id]
        for field in ("cell_id", "model_family", "suite", "canonical_parent_key", "seed"):
            if receipt.get(field) != cell.get(field) or receipt.get(field) != item.get(field):
                raise RuntimeError(f"RECEIPT_BINDING_MISMATCH:{cell_id}:{field}")
        verified_receipts += 1
        clean = receipt.get("clean", {})
        eligibility = receipt.get("eligibility", {})
        complete = bool_value(clean.get("complete_trajectory"))
        steps = int(clean.get("steps_captured", -1))
        horizon = int(clean.get("horizon", -1))
        critical = bool_value(eligibility.get("critical"))
        noncritical = bool_value(eligibility.get("noncritical"))
        model = str(item["model_family"])
        suite = str(item["suite"])
        model_row = by_model[model]
        suite_row = by_suite[suite]
        model_row["cells"] += 1
        suite_row["cells"] += 1
        model_row["complete"] += int(complete)
        model_row["truncated"] += int(not complete)
        model_row["truncated_with_at_least_20_steps"] += int(not complete and steps >= 20)
        suite_row["complete"] += int(complete)
        suite_row["truncated"] += int(not complete)
        model_row["critical_eligible"] += int(critical)
        model_row["noncritical_eligible"] += int(noncritical)
        suite_row["critical_eligible"] += int(critical)
        suite_row["noncritical_eligible"] += int(noncritical)
        for key, value in (clean.get("critical_reason_counts") or {}).items():
            reason_counts[model][key] += int(value)
        censored = int((clean.get("critical_reason_counts") or {}).get("CLEAN_TRAJECTORY_INCOMPLETE_OR_HORIZON_CENSORED", 0))
        model_row["censored_candidate_attempts"] += censored
        suite_row["censored_candidate_attempts"] += censored
        for key in FORBIDDEN_COUNTERS:
            forbidden[key] += int(receipt.get("runtime_counters", {}).get(key, 0))

        clean_rows = clean.get("rows")
        clean_actions = clean.get("actions")
        raw_rows += int(isinstance(clean_rows, list) and bool(clean_rows))
        raw_actions += int(isinstance(clean_actions, list) and bool(clean_actions))
        crit_rows = evidence_rows(eligibility.get("critical_anchor"))
        noncrit_rows = evidence_rows(eligibility.get("noncritical_anchor"))
        critical_evidence_groups += int(bool(crit_rows))
        noncritical_evidence_groups += int(bool(noncrit_rows))
        critical_evidence_rows += len(crit_rows)
        noncritical_evidence_rows += len(noncrit_rows)
        receipts.append({
            "cell_id": cell_id,
            "model_family": model,
            "suite": suite,
            "receipt": receipt,
            "complete": complete,
            "steps_captured": steps,
            "horizon": horizon,
            "critical": critical,
            "noncritical": noncritical,
            "path": display_path(receipt_path, root),
        })

    if len(seen_ids) != 324 or verified_receipts != 324:
        raise RuntimeError("RECEIPT_VERIFICATION_NOT_324")

    protocol_clean = protocol.get("clean_eligibility", {})
    aa0_clean = aa0.get("aa2_clean_screen_and_anchor_freeze", {}).get("clean_trajectory_requirements", {})
    source_findings = {
        "runner": {
            "path": display_path(runner_path, root),
            "sha256": sha256_file(runner_path),
            "distance_gate_lines": lines_with(runner_text, ('float(row["object_eef_distance_m"]) <= 0.04',)),
            "full_horizon_definition_lines": lines_with(runner_text, ("complete = len(rows) == horizon",)),
            "continuation_censor_lines": lines_with(runner_text, ("if len(continuation) != 20 or not complete:",)),
            "candidate_loop_lines": lines_with(runner_text, ("for step in range(max(0, len(rows) - 19)):",)),
        },
        "telemetry": {
            "path": display_path(taxonomy_path, root),
            "sha256": sha256_file(taxonomy_path),
            "object_eef_distance_definition_lines": lines_with(taxonomy_text, ("math.sqrt(sum((position[index] - eef[index]) ** 2",)),
        },
        "frozen_contract_values": {
            "aa0_complete_trajectory": aa0_clean.get("complete_trajectory"),
            "aa0_continuation_steps": aa0_clean.get("clean_continuation_steps_after_anchor"),
            "aa0_carry_relative_distance_max_m": aa0_clean.get("carry_relative_distance_max_m"),
            "aa2_complete_trajectory": protocol_clean.get("complete_trajectory"),
            "aa2_continuation_steps": protocol_clean.get("clean_stable_continuation_steps"),
            "aa2_carry_relative_distance_max_m": protocol_clean.get("carry_relative_distance_max_m"),
        },
    }

    source_bindings = {
        "receipt_index": binding(index_path, root),
        "terminal": binding(terminal_path, root),
        "root_seal": binding(root_seal_path, root),
        "manifest": binding(manifest_path, root),
        "aa0_protocol": binding(aa0_path, root),
        "aa2_protocol": binding(protocol_path, root),
        "runner": binding(runner_path, root),
        "telemetry": binding(taxonomy_path, root),
    }

    report: dict[str, Any] = {
        "schema": "STAGE_AA_AA2_ELIGIBILITY_CONSTRUCT_VALIDITY_AUDIT_V1",
        "status": "STAGE_AA_AA2_ELIGIBILITY_CONSTRUCT_VALIDITY_ALARM_EVIDENCE_INSUFFICIENT",
        "read_only": True,
        "new_model_inference": 0,
        "new_env_steps": 0,
        "new_open_intervention_steps": 0,
        "new_protected_reads": 0,
        "new_aa_v_phys_reads": 0,
        "claim_boundary": "Retrospective evidence sufficiency and implementation-forensics only; no denominator amendment and no AA3 result.",
        "historical_result_preservation": {
            "aa2r2_v2_root_unchanged": True,
            "sealed_common_denominator_not_rewritten": True,
            "historical_n_common": terminal.get("common_denominator", {}).get("n_common"),
            "historical_terminal_status": terminal.get("status"),
            "root_seal_declared_status": root_seal.get("status"),
        },
        "authority": {
            "server_root": str(root),
            "receipt_index_count": len(indexed),
            "verified_receipt_count": verified_receipts,
            "source_bindings": source_bindings,
        },
        "receipt_persistence": {
            "raw_clean_rows_field_present_and_nonempty": raw_rows,
            "raw_clean_actions_field_present_and_nonempty": raw_actions,
            "critical_anchor_evidence_groups": critical_evidence_groups,
            "critical_anchor_evidence_rows": critical_evidence_rows,
            "noncritical_anchor_evidence_groups": noncritical_evidence_groups,
            "noncritical_anchor_evidence_rows": noncritical_evidence_rows,
            "rejected_candidate_per_step_rows_available": False,
            "rejected_candidate_distance_values_available": False,
            "rejected_candidate_reason_intersections_available": False,
        },
        "coverage": {
            "by_model": {key: dict(value) for key, value in sorted(by_model.items())},
            "by_suite": {key: dict(value) for key, value in sorted(by_suite.items())},
            "critical_eligible_total": sum(value["critical_eligible"] for value in by_model.values()),
            "noncritical_eligible_total": sum(value["noncritical_eligible"] for value in by_model.values()),
            "critical_eligible_cells": [
                {"cell_id": item["cell_id"], "model_family": item["model_family"], "suite": item["suite"], "path": item["path"]}
                for item in receipts
                if item["critical"]
            ],
        },
        "reason_counts": {key: dict(sorted(value.items())) for key, value in sorted(reason_counts.items())},
        "aggregate_historical_forbidden_counters": dict(sorted(forbidden.items())),
        "source_findings": source_findings,
        "distance_alarm": {
            "status": "UNRESOLVED_CONTRACT_AMBIGUITY",
            "current_implementation_uses_object_eef_distance_for_both_012m_and_004m": True,
            "separate_authoritative_relative_to_anchor_formula_found": False,
            "relative_to_anchor_reconstruction_possible_from_persisted_bytes": False,
            "diagnostic": distance_summary(receipts),
            "interpretation": "The code-level 0.04 gate is proven. Whether that is wrong cannot be established from the frozen contract or rejected-cell bytes: object_eef_distance_m is a persisted Euclidean object-to-EEF separation, while no separate anchor-relative field/formula is persisted.",
        },
        "truncation_alarm": {
            "status": "IMPLEMENTATION_BEHAVIOR_CONFIRMED_CONSTRUCT_VALIDITY_UNRESOLVED",
            "truncated_cells": sum(value["truncated"] for value in by_model.values()),
            "truncated_cells_with_at_least_20_captured_steps": sum(value["truncated_with_at_least_20_steps"] for value in by_model.values()),
            "censored_candidate_attempts": sum(value["censored_candidate_attempts"] for value in by_model.values()),
            "implementation_behavior": "complete_trajectory is len(rows)==horizon; every candidate in a truncated receipt receives CLEAN_TRAJECTORY_INCOMPLETE_OR_HORIZON_CENSORED.",
            "literal_protocol_constraint": "Both AA0 and AA2 freeze complete_trajectory=true, so dropping this gate would be a protocol amendment, not an unmarked bug fix.",
            "local_20_step_validity_reconstructible": False,
            "interpretation": "The scanner demonstrably rejects local 20-row windows when the episode is not full-horizon, but missing raw rows prevent testing whether any rejected window otherwise satisfied all frozen geometry/contact rules.",
        },
        "diagnostic_variants": {
            "original_frozen_implementation": {"computable": True, "parent_set_authority": "sealed AA2R2 V2 receipt eligibility flags"},
            "relative_distance_corrected": {"computable": False, "reason": "No authoritative alternative formula and no rejected per-step positions/anchor reference."},
            "truncation_continuation_corrected": {"computable": False, "reason": "No persisted rows/actions for rejected windows; complete flag alone cannot prove local stability."},
            "combined_corrected": {"computable": False, "reason": "Both required inputs are unavailable for rejected candidates."},
        },
        "disposition": "HOLD_AA2_RETROSPECTIVE_ELIGIBILITY_EVIDENCE_INSUFFICIENT",
        "next_legal_action": "STOP_FOR_PI_REVIEW_BEFORE_ANY_AA2_DENOMINATOR_AMENDMENT_OR_AA3",
    }
    report["canonical_payload_sha256"] = sha256_bytes(canonical(report))
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt-index", type=Path, default=Path("reports/STAGE_AA_AA2R2_PHASE_B_V2_RECEIPT_INDEX_V1.json"))
    parser.add_argument("--terminal", type=Path, default=Path("reports/STAGE_AA_AA2R2_PHASE_B_V2_CENSUS_TERMINAL_V1.json"))
    parser.add_argument("--root-seal", type=Path, default=Path("reports/STAGE_AA_AA2R2_PHASE_B_V2_ROOT_SEAL_V1.json"))
    parser.add_argument("--manifest", type=Path, default=Path("reports/STAGE_AA_AA2_CLEAN_SCREEN_LAUNCH_MANIFEST_V1.json"))
    parser.add_argument("--aa0", type=Path, default=Path("configs/STAGE_AA_AA0_PROSPECTIVE_PROTOCOL_V1.json"))
    parser.add_argument("--protocol", type=Path, default=Path("configs/STAGE_AA_AA2_CLEAN_SCREEN_PROTOCOL_V1.json"))
    parser.add_argument("--runner", type=Path, default=Path("scripts/stage_aa/run_stage_aa2_clean_screen.py"))
    parser.add_argument("--taxonomy", type=Path, default=Path("src/gripper_attack/stage_v_m3_5_physical_taxonomy.py"))
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise RuntimeError(f"APPEND_ONLY_OUTPUT_EXISTS:{output}")
    report = audit(args)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(canonical(report))
    sidecar = output.with_suffix(output.suffix + ".sha256")
    if sidecar.exists():
        raise RuntimeError(f"APPEND_ONLY_SIDECAR_EXISTS:{sidecar}")
    sidecar.write_text(f"{sha256_file(output)}  {output.name}\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "verified_receipts": report["authority"]["verified_receipt_count"], "critical_eligible": report["coverage"]["critical_eligible_total"], "output_sha256": sha256_file(output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
