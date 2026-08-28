#!/usr/bin/env python3
"""Freeze censoring-aware coverage from the sealed AC3 G2 branch index."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
MODELS = ("M0_OPENVLA", "M1_OPENVLA_OFT", "M2_PI05_LIBERO")
SUITES = ("libero_10", "libero_object", "libero_spatial")
DOSES = (3, 5, 10)
CONDITIONS = {"OPEN_T3": 3, "OPEN_T5": 5, "OPEN_T10": 10}
PASS = "PASS"
INVALID = "ENGINEERING_INVALID_OR_HORIZON_CENSORED"
TARGET = "AC3-65bcfd948a45dd0be9ac"
HORIZON = "TRUE_SIMULATOR_TERMINAL_HORIZON_CENSOR"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def canonical_hash(value: Any) -> str:
    return sha256(canonical(value))


def read_json(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    data = path.read_bytes()
    return json.loads(data.decode("utf-8")), {"path": str(path), "bytes": len(data), "sha256": sha256(data)}


def require(ok: bool, message: str) -> None:
    if not ok:
        raise RuntimeError(message)


def write_new(path: Path, value: dict[str, Any]) -> dict[str, Any]:
    require(not path.exists(), f"AC3_G2R1_C_OUTPUT_EXISTS:{path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    path.write_bytes(data)
    return {"path": str(path), "bytes": len(data), "sha256": sha256(data)}


def verify_record(path: Path, expected: dict[str, Any], label: str) -> dict[str, Any]:
    actual = read_json(path)[1]
    require(actual["bytes"] == int(expected["bytes"]), f"AC3_G2R1_C_{label}_BYTES")
    require(actual["sha256"] == str(expected["sha256"]), f"AC3_G2R1_C_{label}_SHA")
    return actual


def bounds(events: int, observed: int, horizon: int, action_unknown: int, denominator: int = 32) -> dict[str, Any]:
    require(0 <= events <= observed <= denominator, "AC3_G2R1_C_BOUND_COUNTS")
    return {
        "events": events,
        "observed_complete": observed,
        "horizon_censored": horizon,
        "action_semantics_unknown": action_unknown,
        "unknown_is_not_zero": True,
        "conditional_rate_observed": events / observed if observed else None,
        "p_lower_fixed_32": events / denominator,
        "p_upper_horizon_only_fixed_32": (events + horizon) / denominator,
        "p_upper_all_unknown_fixed_32": (events + horizon + action_unknown) / denominator,
        "denominator": denominator,
    }


def compact_row(row: dict[str, Any]) -> dict[str, Any]:
    validation = row.get("validation") or {}
    counters = validation.get("runtime_counters") or {}
    return {
        "branch_id": row["branch_id"],
        "model_family": row["model_family"],
        "suite": row["suite"],
        "canonical_parent_key": row["canonical_parent_key"],
        "condition": row["condition"],
        "dose": int(row["dose"]),
        "status": row["status"],
        "physical_class": validation.get("physical_class"),
        "v_phys_label": validation.get("v_phys_label"),
        "rows": validation.get("rows"),
        "physical_telemetry_reads": int(counters.get("physical_telemetry_reads", 0)),
        "physical_endpoint_reads": int(counters.get("physical_endpoint_reads", 0)),
    }


def build(args: argparse.Namespace) -> dict[str, Any]:
    index, index_record = read_json(args.g2_index)
    g2_root, g2_root_record = read_json(args.g2_root)
    adjudication, adjudication_record = read_json(args.adjudication)
    reseal, reseal_record = read_json(args.structural_reseal)
    reseal_root, reseal_root_record = read_json(args.structural_root)

    require(index["schema"] == "STAGE_AC_AC3_G2_BRANCH_RECEIPT_INDEX_V1", "AC3_G2R1_C_INDEX_SCHEMA")
    require(index["status"] == "HOLD_AC3_G2_ENGINEERING_OR_HORIZON", "AC3_G2R1_C_INDEX_STATUS")
    require(index["counts"]["manifest_branches"] == 384, "AC3_G2R1_C_MANIFEST_COUNT")
    require(index["counts"]["pass_branches"] == 372 and index["counts"]["invalid_or_horizon_censored_branches"] == 12, "AC3_G2R1_C_INDEX_COUNTS")
    require(canonical_hash(g2_root["root_payload"]) == g2_root["root_payload_sha256"], "AC3_G2R1_C_G2_ROOT_PAYLOAD")
    verify_record(args.g2_index, g2_root["root_payload"]["receipt_index"], "G2_INDEX")
    verify_record(args.g2_terminal, g2_root["root_payload"]["terminal"], "G2_TERMINAL")

    require(adjudication["schema"] == "STAGE_AC_AC3_G2R1_A_CENSOR_ADJUDICATION_V1", "AC3_G2R1_C_ADJUDICATION_SCHEMA")
    require(len(adjudication["branches"]) == 12, "AC3_G2R1_C_ADJUDICATION_COUNT")
    require(reseal["schema"] == "STAGE_AC_AC3_G2R1_B1R1_STRUCTURAL_RESEAL_V1", "AC3_G2R1_C_RESEAL_SCHEMA")
    require(reseal["status"] == "STAGE_AC_AC3_G2R1_B1R1_STRUCTURAL_RESEAL_PASS_CONTINUE_TO_G2R1_C", "AC3_G2R1_C_RESEAL_STATUS")
    require(canonical_hash(reseal_root["root_payload"]) == reseal_root["root_payload_sha256"], "AC3_G2R1_C_RESEAL_ROOT_PAYLOAD")
    require(reseal["counts"] == {"authoritative_branches": 384, "pass_branches": 372, "target_unknown_action_semantics": 1, "true_horizon_censors": 11}, "AC3_G2R1_C_RESEAL_COUNTS")

    rows = index["rows"]
    require(len(rows) == 384 and len({row["branch_id"] for row in rows}) == 384, "AC3_G2R1_C_ROW_SET")
    target_updates = reseal["target_update"]
    require(target_updates["branch_id"] == TARGET and target_updates["post_b1r1_classification"] == "UNKNOWN_ACTION_SEMANTICS_AFTER_B1R1", "AC3_G2R1_C_TARGET_UPDATE")
    horizon_rows = {row["branch_id"]: row for row in reseal["preserved_true_horizon_censors"]}
    require(len(horizon_rows) == 11 and TARGET not in horizon_rows, "AC3_G2R1_C_HORIZON_SET")
    adjudicated = {row["branch_id"]: row for row in adjudication["branches"]}
    require(set(horizon_rows) | {TARGET} == set(adjudicated), "AC3_G2R1_C_ADJUDICATION_SET")
    require(all(row["detail"]["physical_outcome_read"] is False for row in adjudication["branches"]), "AC3_G2R1_C_A_OUTCOME_FIREWALL")

    invalid = {row["branch_id"]: row for row in index["invalid_or_horizon_censored"]}
    require(set(invalid) == set(adjudicated), "AC3_G2R1_C_INDEX_INVALID_SET")
    unknown_type: dict[str, str] = {TARGET: "ACTION_SEMANTICS_UNKNOWN"}
    unknown_type.update({branch_id: HORIZON for branch_id in horizon_rows})
    require(set(unknown_type) == set(invalid), "AC3_G2R1_C_UNKNOWN_SET")

    parent_cells: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    branch_inventory: list[dict[str, Any]] = []
    valid_treatment_rows: list[dict[str, Any]] = []
    for row in rows:
        model = str(row["model_family"])
        suite = str(row["suite"])
        condition = str(row["condition"])
        require(model in MODELS and suite in SUITES and condition in {"CLEAN_REFERENCE", *CONDITIONS}, f"AC3_G2R1_C_BINDING:{row['branch_id']}")
        key = (model, str(row["canonical_parent_key"]))
        require(condition not in parent_cells[key], f"AC3_G2R1_C_DUPLICATE_CELL:{row['branch_id']}")
        parent_cells[key][condition] = row
        if row["status"] == PASS:
            validation = row.get("validation") or {}
            if condition != "CLEAN_REFERENCE":
                require(validation.get("v_phys_label") in {"V_PHYS", "NO_PHYSICAL_VULNERABILITY"}, f"AC3_G2R1_C_TREATMENT_LABEL:{row['branch_id']}")
                valid_treatment_rows.append(compact_row(row))
            branch_inventory.append(compact_row(row))
        else:
            require(row["branch_id"] in unknown_type, f"AC3_G2R1_C_UNEXPECTED_INVALID:{row['branch_id']}")
            branch_inventory.append({
                "branch_id": row["branch_id"],
                "model_family": model,
                "suite": suite,
                "canonical_parent_key": row["canonical_parent_key"],
                "condition": condition,
                "dose": int(row["dose"]),
                "status": "UNKNOWN",
                "unknown_type": unknown_type[row["branch_id"]],
                "physical_outcome_read": False,
            })

    require(len(parent_cells) == 96, "AC3_G2R1_C_PARENT_COUNT")
    require(all(set(cells) == {"CLEAN_REFERENCE", *CONDITIONS} for cells in parent_cells.values()), "AC3_G2R1_C_PARENT_CONDITIONS")
    require(all(sum(1 for key in parent_cells if key[0] == model) == 32 for model in MODELS), "AC3_G2R1_C_MODEL_PARENT_COUNT")

    model_summary: dict[str, Any] = {}
    complete_pair_sets: dict[str, set[str]] = {}
    complete_triplet_sets: dict[str, set[str]] = {}
    for model in MODELS:
        model_parents = {parent: cells for (family, parent), cells in parent_cells.items() if family == model}
        by_dose: dict[str, Any] = {}
        for condition, dose in CONDITIONS.items():
            complete = []
            unknown = []
            events = 0
            physical_classes = Counter()
            vphys_labels = Counter()
            telemetry = {"rows": 0, "physical_telemetry_reads": 0, "physical_endpoint_reads": 0}
            for parent, cells in sorted(model_parents.items()):
                row = cells[condition]
                if row["status"] == PASS:
                    complete.append(parent)
                    validation = row["validation"]
                    physical_classes[str(validation.get("physical_class"))] += 1
                    vphys_labels[str(validation.get("v_phys_label"))] += 1
                    if validation.get("v_phys_label") == "V_PHYS":
                        events += 1
                    counters = validation.get("runtime_counters") or {}
                    telemetry["rows"] += int(validation.get("rows", 0))
                    telemetry["physical_telemetry_reads"] += int(counters.get("physical_telemetry_reads", 0))
                    telemetry["physical_endpoint_reads"] += int(counters.get("physical_endpoint_reads", 0))
                else:
                    unknown.append({"parent": parent, "branch_id": row["branch_id"], "unknown_type": unknown_type[row["branch_id"]]})
            horizon_count = sum(item["unknown_type"] == HORIZON for item in unknown)
            action_count = sum(item["unknown_type"] == "ACTION_SEMANTICS_UNKNOWN" for item in unknown)
            by_dose[str(dose)] = {
                "condition": condition,
                "dose": dose,
                "total_fixed_parents": len(model_parents),
                "complete_parent_count": len(complete),
                "unknown_parent_count": len(unknown),
                "unknown_parent_rows": unknown,
                "bounds": bounds(events, len(complete), horizon_count, action_count),
                "auto_valid_telemetry": telemetry,
                "auto_valid_physical_class_counts": dict(sorted(physical_classes.items())),
                "auto_valid_v_phys_label_counts": dict(sorted(vphys_labels.items())),
                "unknown_breakdown": {"true_horizon_censor": horizon_count, "action_semantics_unknown": action_count},
            }
        pairs = {parent for parent, cells in model_parents.items() if cells["OPEN_T3"]["status"] == PASS and cells["OPEN_T10"]["status"] == PASS}
        triplets = {parent for parent, cells in model_parents.items() if all(cells[condition]["status"] == PASS for condition in CONDITIONS)}
        pair_patterns = Counter()
        triplet_patterns = Counter()
        for parent in pairs:
            cells = model_parents[parent]
            pair_patterns["".join("1" if cells[condition]["validation"].get("v_phys_label") == "V_PHYS" else "0" for condition in ("OPEN_T3", "OPEN_T10"))] += 1
        for parent in triplets:
            cells = model_parents[parent]
            triplet_patterns["".join("1" if cells[condition]["validation"].get("v_phys_label") == "V_PHYS" else "0" for condition in ("OPEN_T3", "OPEN_T5", "OPEN_T10"))] += 1
        monotone_patterns = {"000", "001", "011", "111"}
        complete_pair_sets[model] = pairs
        complete_triplet_sets[model] = triplets
        model_summary[model] = {
            "fixed_parent_count": len(model_parents),
            "complete_t3_t10_pair_count": len(pairs),
            "complete_t3_t5_t10_triplet_count": len(triplets),
            "pair_floor_24_pass": len(pairs) >= 24,
            "triplet_floor_24_pass": len(triplets) >= 24,
            "complete_t3_t10_pair_patterns": dict(sorted(pair_patterns.items())),
            "complete_t3_t5_t10_patterns": dict(sorted(triplet_patterns.items())),
            "complete_triplet_monotone_patterns": dict(sorted((key, value) for key, value in triplet_patterns.items() if key in monotone_patterns)),
            "complete_triplet_nonmonotone_patterns": dict(sorted((key, value) for key, value in triplet_patterns.items() if key not in monotone_patterns)),
            "by_dose": by_dose,
        }

    require({model: len(complete_pair_sets[model]) for model in MODELS} == {"M0_OPENVLA": 30, "M1_OPENVLA_OFT": 31, "M2_PI05_LIBERO": 28}, "AC3_G2R1_C_PAIR_COUNTS")
    require({model: len(complete_triplet_sets[model]) for model in MODELS} == {"M0_OPENVLA": 30, "M1_OPENVLA_OFT": 31, "M2_PI05_LIBERO": 28}, "AC3_G2R1_C_TRIPLET_COUNTS")
    require(sum(row["status"] == PASS for row in branch_inventory) == 372, "AC3_G2R1_C_PASS_INVENTORY")

    suite_summary: dict[str, Any] = {}
    for model in MODELS:
        suite_summary[model] = {}
        for suite in SUITES:
            rows_for_suite = [row for row in branch_inventory if row["model_family"] == model and row["suite"] == suite]
            suite_summary[model][suite] = {
                "pass_by_condition": dict(sorted(Counter(row["condition"] for row in rows_for_suite if row["status"] == PASS).items())),
                "unknown_by_condition": dict(sorted(Counter(row["condition"] for row in rows_for_suite if row["status"] == "UNKNOWN").items())),
            }

    unknown_inventory = [row for row in branch_inventory if row["status"] == "UNKNOWN"]
    require(len(unknown_inventory) == 12, "AC3_G2R1_C_UNKNOWN_INVENTORY")
    status = "STAGE_AC_AC3_G2R1_C_CENSORING_AWARE_ANALYSIS_FROZEN_CONTINUE_TO_G3R1"
    next_action = "G3R1_CENSORING_AWARE_STATISTICS"
    report = {
        "schema": "STAGE_AC_AC3_G2R1_C_CENSORING_AWARE_ANALYSIS_V1",
        "status": status,
        "gate": "STAGE_AC_AC3_G2R1_C_CENSORING_AWARE_ANALYSIS_V1",
        "claim_boundary": "Read-only censoring/coverage freeze from sealed G2 validation rows; no new execution and no statistical promotion.",
        "source_authority": {
            "g2_index": index_record,
            "g2_root": g2_root_record,
            "g2r1_a_adjudication": adjudication_record,
            "b1r1_structural_reseal": reseal_record,
            "b1r1_structural_root": reseal_root_record,
        },
        "counts": {
            "fixed_authoritative_branches": 384,
            "pass_branches": 372,
            "true_horizon_censored_branches": 11,
            "action_semantics_unknown_branches": 1,
            "physical_outcome_reclassified": 0,
            "new_execution": 0,
        },
        "unknown_branch_policy": {
            "horizon_censor_is_unknown": True,
            "action_semantics_unknown_is_unknown": True,
            "unknown_is_not_zero": True,
            "target_branch_not_recovered": True,
            "eleven_horizon_branches_not_recovered": True,
            "fixed_parent_denominator_per_model": 32,
        },
        "model_summary": model_summary,
        "suite_summary": suite_summary,
        "unknown_branches": sorted(unknown_inventory, key=lambda row: row["branch_id"]),
        "sealed_pass_branch_inventory": sorted(valid_treatment_rows, key=lambda row: row["branch_id"]),
        "analysis_notes": {
            "primary_partial_identification": "p_lower=observed V_phys events/32; p_upper_horizon_only=(observed events+true horizon censors)/32",
            "action_unknown_sensitivity": "reported separately in p_upper_all_unknown_fixed_32; never silently treated as horizon censor or physical failure",
            "telemetry_scope": "sealed G2 validation metadata only: row counts, telemetry-read counts, endpoint labels/classes; no raw remote receipt re-execution",
            "pair_and_triplet_scope": "availability/floor and descriptive complete-parent pattern freeze only; exact paired tests are deferred to G3R1",
        },
        "outcome_firewall": {
            "new_model_inference": 0,
            "new_env_steps": 0,
            "new_open_interventions": 0,
            "new_physical_outcome_reads": 0,
            "new_pgd": 0,
            "new_protected_reads": 0,
        },
        "next_legal_action": next_action,
    }
    report_artifact = write_new(args.output_dir / "STAGE_AC_AC3_G2R1_C_CENSORING_AWARE_ANALYSIS_V1.json", report)
    root_payload = {
        "gate": report["gate"],
        "status": status,
        "report": report_artifact,
        "source_authority": report["source_authority"],
        "counts": report["counts"],
        "model_summary": model_summary,
        "unknown_branch_ids": [row["branch_id"] for row in report["unknown_branches"]],
        "outcome_firewall": report["outcome_firewall"],
        "claim_boundary": report["claim_boundary"],
        "next_legal_action": next_action,
    }
    root = {
        "schema": "STAGE_AC_AC3_G2R1_C_ROOT_SEAL_V1",
        "status": status,
        "root_payload": root_payload,
        "root_payload_sha256": canonical_hash(root_payload),
        "artifacts": {"analysis": report_artifact},
        "claim_boundary": report["claim_boundary"],
        "next_legal_action": next_action,
    }
    root_artifact = write_new(args.output_dir / "STAGE_AC_AC3_G2R1_C_ROOT_SEAL_V1.json", root)
    return {"status": status, "counts": report["counts"], "model_summary": model_summary, "artifacts": {"analysis": report_artifact, "root": root_artifact}, "root_payload_sha256": root["root_payload_sha256"]}


def self_test() -> None:
    assert bounds(3, 10, 2, 1)["p_lower_fixed_32"] == 3 / 32
    assert CONDITIONS == {"OPEN_T3": 3, "OPEN_T5": 5, "OPEN_T10": 10}
    print(json.dumps({"status": "AC3_G2R1_C_STATIC_SELF_TEST_PASS", "unknown_is_not_zero": True}, sort_keys=True))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--g2-index", type=Path, default=ROOT / "reports/STAGE_AC_AC3_G2_BRANCH_RECEIPT_INDEX_V1.json")
    parser.add_argument("--g2-terminal", type=Path, default=ROOT / "reports/STAGE_AC_AC3_G2_TERMINAL_V1.json")
    parser.add_argument("--g2-root", type=Path, default=ROOT / "reports/STAGE_AC_AC3_G2_ROOT_SEAL_V1.json")
    parser.add_argument("--adjudication", type=Path, default=ROOT / "reports/STAGE_AC_AC3_G2R1_A_CENSOR_ADJUDICATION_V1.json")
    parser.add_argument("--structural-reseal", type=Path, default=ROOT / "reports/STAGE_AC_AC3_G2R1_B1R1_STRUCTURAL_RESEAL_V1/STAGE_AC_AC3_G2R1_B1R1_STRUCTURAL_RESEAL_V1.json")
    parser.add_argument("--structural-root", type=Path, default=ROOT / "reports/STAGE_AC_AC3_G2R1_B1R1_STRUCTURAL_RESEAL_V1/STAGE_AC_AC3_G2R1_B1R1_STRUCTURAL_RESEAL_ROOT_V1.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports/STAGE_AC_AC3_G2R1_C_CENSORING_AWARE_ANALYSIS_V1")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    print(json.dumps(build(args), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
