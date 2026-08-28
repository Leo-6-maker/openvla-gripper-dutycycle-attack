#!/usr/bin/env python3
"""Compute censoring-aware G3R1 statistics from the sealed G2R1-C report."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from math import comb
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
MODELS = ("M0_OPENVLA", "M1_OPENVLA_OFT", "M2_PI05_LIBERO")
DOSES = (3, 5, 10)
CONDITIONS = ("OPEN_T3", "OPEN_T5", "OPEN_T10")
HORIZON = "TRUE_SIMULATOR_TERMINAL_HORIZON_CENSOR"
TARGET_UNKNOWN = "ACTION_SEMANTICS_UNKNOWN"
MONOTONE = {"000", "001", "011", "111"}


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
    require(not path.exists(), f"AC3_G3R1_OUTPUT_EXISTS:{path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    path.write_bytes(data)
    return {"path": str(path), "bytes": len(data), "sha256": sha256(data)}


def exact_mcnemar_p(discordant_t3_yes_t10_no: int, discordant_t3_no_t10_yes: int) -> float:
    b = int(discordant_t3_yes_t10_no)
    c = int(discordant_t3_no_t10_yes)
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(comb(n, k) for k in range(min(b, c) + 1)) / (2 ** n)
    return min(1.0, 2.0 * tail)


def holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(p_values.items(), key=lambda item: (item[1], item[0]))
    adjusted: dict[str, float] = {}
    running = 0.0
    total = len(ordered)
    for index, (label, p_value) in enumerate(ordered):
        value = min(1.0, (total - index) * p_value)
        running = max(running, value)
        adjusted[label] = running
    return adjusted


def build(args: argparse.Namespace) -> dict[str, Any]:
    c_report, c_record = read_json(args.g2r1_c)
    c_root, c_root_record = read_json(args.g2r1_c_root)
    require(c_report["schema"] == "STAGE_AC_AC3_G2R1_C_CENSORING_AWARE_ANALYSIS_V1", "AC3_G3R1_C_REPORT_SCHEMA")
    require(c_report["status"] == "STAGE_AC_AC3_G2R1_C_CENSORING_AWARE_ANALYSIS_FROZEN_CONTINUE_TO_G3R1", "AC3_G3R1_C_REPORT_STATUS")
    require(c_root["schema"] == "STAGE_AC_AC3_G2R1_C_ROOT_SEAL_V1", "AC3_G3R1_C_ROOT_SCHEMA")
    require(canonical_hash(c_root["root_payload"]) == c_root["root_payload_sha256"], "AC3_G3R1_C_ROOT_PAYLOAD")
    require(c_root["root_payload"]["report"]["sha256"] == c_record["sha256"], "AC3_G3R1_C_REPORT_ROOT_SHA")
    require(c_report["counts"] == {"action_semantics_unknown_branches": 1, "fixed_authoritative_branches": 384, "new_execution": 0, "pass_branches": 372, "physical_outcome_reclassified": 0, "true_horizon_censored_branches": 11}, "AC3_G3R1_C_COUNTS")
    require(c_report["outcome_firewall"]["new_model_inference"] == 0 and c_report["outcome_firewall"]["new_env_steps"] == 0 and c_report["outcome_firewall"]["new_open_interventions"] == 0, "AC3_G3R1_C_FIREWALL")

    cells: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    valid_rows = c_report["sealed_pass_branch_inventory"]
    unknown_rows = c_report["unknown_branches"]
    require(len(valid_rows) == 276 and len(unknown_rows) == 12, "AC3_G3R1_ROW_COUNTS")
    for row in valid_rows:
        model = str(row["model_family"])
        condition = str(row["condition"])
        require(model in MODELS and condition in CONDITIONS, f"AC3_G3R1_VALID_BINDING:{row['branch_id']}")
        key = (model, str(row["canonical_parent_key"]))
        require(condition not in cells[key], f"AC3_G3R1_DUPLICATE_VALID:{row['branch_id']}")
        require(row["v_phys_label"] in {"V_PHYS", "NO_PHYSICAL_VULNERABILITY"}, f"AC3_G3R1_VALID_LABEL:{row['branch_id']}")
        cells[key][condition] = {"status": "PASS", "event": row["v_phys_label"] == "V_PHYS", "branch_id": row["branch_id"], "physical_class": row["physical_class"], "suite": row["suite"]}
    for row in unknown_rows:
        model = str(row["model_family"])
        condition = str(row["condition"])
        require(model in MODELS and condition in CONDITIONS, f"AC3_G3R1_UNKNOWN_BINDING:{row['branch_id']}")
        key = (model, str(row["canonical_parent_key"]))
        require(condition not in cells[key], f"AC3_G3R1_DUPLICATE_UNKNOWN:{row['branch_id']}")
        require(row["unknown_type"] in {HORIZON, TARGET_UNKNOWN}, f"AC3_G3R1_UNKNOWN_TYPE:{row['branch_id']}")
        cells[key][condition] = {"status": "UNKNOWN", "event": None, "unknown_type": row["unknown_type"], "branch_id": row["branch_id"], "suite": row["suite"]}

    require(len(cells) == 96, "AC3_G3R1_PARENT_COUNT")
    require(all(len(cells[key]) == 3 for key in cells), "AC3_G3R1_DOSE_COVERAGE")
    require(all(sum(key[0] == model for key in cells) == 32 for model in MODELS), "AC3_G3R1_MODEL_PARENT_COUNTS")

    model_stats: dict[str, Any] = {}
    p_values: dict[str, float] = {}
    for model in MODELS:
        parents = {parent: value for (family, parent), value in cells.items() if family == model}
        dose_stats: dict[str, Any] = {}
        for condition, dose in zip(CONDITIONS, DOSES):
            entries = [value[condition] for value in parents.values()]
            complete = [value for value in entries if value["status"] == "PASS"]
            unknown = [value for value in entries if value["status"] == "UNKNOWN"]
            events = sum(bool(value["event"]) for value in complete)
            horizon = sum(value.get("unknown_type") == HORIZON for value in unknown)
            action_unknown = sum(value.get("unknown_type") == TARGET_UNKNOWN for value in unknown)
            physical = Counter(value.get("physical_class") for value in complete)
            dose_stats[str(dose)] = {
                "condition": condition,
                "fixed_parent_count": 32,
                "complete_parent_count": len(complete),
                "unknown_parent_count": len(unknown),
                "events": events,
                "non_events": len(complete) - events,
                "horizon_censored": horizon,
                "action_semantics_unknown": action_unknown,
                "conditional_rate_observed": events / len(complete) if complete else None,
                "p_lower_fixed_32": events / 32,
                "p_upper_horizon_only_fixed_32": (events + horizon) / 32,
                "p_upper_all_unknown_fixed_32": (events + horizon + action_unknown) / 32,
                "auto_valid_physical_class_counts": dict(sorted((str(key), value) for key, value in physical.items())),
                "unknown_is_not_zero": True,
            }

        pairs = [parent for parent, value in parents.items() if value["OPEN_T3"]["status"] == "PASS" and value["OPEN_T10"]["status"] == "PASS"]
        pair_pattern = Counter("".join("1" if parents[parent][condition]["event"] else "0" for condition in ("OPEN_T3", "OPEN_T10")) for parent in pairs)
        b = pair_pattern["10"]
        c = pair_pattern["01"]
        p_value = exact_mcnemar_p(b, c)
        p_values[model] = p_value

        triplets = [parent for parent, value in parents.items() if all(value[condition]["status"] == "PASS" for condition in CONDITIONS)]
        triplet_pattern = Counter("".join("1" if parents[parent][condition]["event"] else "0" for condition in CONDITIONS) for parent in triplets)
        observed_rates = [dose_stats[str(dose)]["conditional_rate_observed"] for dose in DOSES]
        lower_rates = [dose_stats[str(dose)]["p_lower_fixed_32"] for dose in DOSES]
        horizon_upper_rates = [dose_stats[str(dose)]["p_upper_horizon_only_fixed_32"] for dose in DOSES]
        all_unknown_upper_rates = [dose_stats[str(dose)]["p_upper_all_unknown_fixed_32"] for dose in DOSES]
        horizon_lower_diff = lower_rates[2] - horizon_upper_rates[0]
        horizon_upper_diff = horizon_upper_rates[2] - lower_rates[0]
        all_unknown_lower_diff = lower_rates[2] - all_unknown_upper_rates[0]
        all_unknown_upper_diff = all_unknown_upper_rates[2] - lower_rates[0]
        model_stats[model] = {
            "fixed_parent_count": 32,
            "complete_t3_t10_pair_count": len(pairs),
            "complete_t3_t5_t10_triplet_count": len(triplets),
            "pair_floor_24_pass": len(pairs) >= 24,
            "triplet_floor_24_pass": len(triplets) >= 24,
            "by_dose": dose_stats,
            "complete_t3_t10_pair_patterns": dict(sorted(pair_pattern.items())),
            "complete_t3_t5_t10_patterns": dict(sorted(triplet_pattern.items())),
            "complete_triplet_monotone_patterns": dict(sorted((key, value) for key, value in triplet_pattern.items() if key in MONOTONE)),
            "complete_triplet_nonmonotone_patterns": dict(sorted((key, value) for key, value in triplet_pattern.items() if key not in MONOTONE)),
            "paired_exact_test": {
                "test": "two-sided exact McNemar conditional binomial",
                "t3_yes_t10_no": b,
                "t3_no_t10_yes": c,
                "discordant_total": b + c,
                "p_value": p_value,
            },
            "dose_response_diagnostic": {
                "observed_conditional_rates_t3_t5_t10": observed_rates,
                "observed_conditional_monotone": all(observed_rates[index] <= observed_rates[index + 1] for index in range(2)),
                "complete_triplet_monotone_count": sum(triplet_pattern[key] for key in MONOTONE),
                "complete_triplet_nonmonotone_count": sum(triplet_pattern[key] for key in triplet_pattern if key not in MONOTONE),
                "fixed_32_lower_rates": lower_rates,
                "fixed_32_horizon_only_upper_rates": horizon_upper_rates,
                "fixed_32_all_unknown_upper_rates": all_unknown_upper_rates,
                "t10_minus_t3_observed_conditional": observed_rates[2] - observed_rates[0],
                "t10_minus_t3_horizon_only_interval": [horizon_lower_diff, horizon_upper_diff],
                "t10_minus_t3_all_unknown_interval": [all_unknown_lower_diff, all_unknown_upper_diff],
                "threshold_020": {
                    "horizon_only_robustly_at_least_020": horizon_lower_diff >= 0.20,
                    "horizon_only_can_reach_020": horizon_upper_diff >= 0.20,
                    "all_unknown_robustly_at_least_020": all_unknown_lower_diff >= 0.20,
                    "all_unknown_can_reach_020": all_unknown_upper_diff >= 0.20,
                    "interpretation": "intervals preserve UNKNOWN; no promotion decision is made here",
                },
            },
        }

    adjusted = holm_adjust(p_values)
    for model in MODELS:
        model_stats[model]["paired_exact_test"]["holm_adjusted_p_value_across_models"] = adjusted[model]

    global_status = "STAGE_AC_AC3_G2R1_G3R1_CENSORING_AWARE_STATISTICS_COMPLETE_CONTINUE_TO_AC4"
    next_action = "AC4_FROZEN_BLINDED_AUDIT"
    report = {
        "schema": "STAGE_AC_AC3_G2R1_G3R1_CENSORING_AWARE_STATISTICS_V1",
        "status": global_status,
        "gate": "STAGE_AC_AC3_G2R1_G3R1_CENSORING_AWARE_STATISTICS_V1",
        "claim_boundary": "Static G3R1 censoring-aware statistics only; no new execution, endpoint relabeling, or scientific promotion.",
        "source_authority": {"g2r1_c_report": c_record, "g2r1_c_root": c_root_record},
        "counts": {"fixed_parents_per_model": 32, "models": 3, "authoritative_treatment_rows": 288, "complete_treatment_rows": 276, "unknown_treatment_rows": 12, "true_horizon_unknown": 11, "action_semantics_unknown": 1, "new_execution": 0},
        "model_summary": model_stats,
        "holm_correction": {"family": "three model paired exact tests", "raw_p_values": p_values, "adjusted_p_values": adjusted, "alpha_or_promotion_rule": "not applied in G3R1 closeout"},
        "censoring_policy": {"unknown_is_not_zero": True, "horizon_censor_in_primary_bounds": True, "action_unknown_separate_sensitivity": True, "fixed_denominator": 32, "pair_test_complete_pair_only": True, "triplet_pattern_complete_triplet_only": True},
        "global_diagnostic": {
            "models_with_pair_floor_24": sum(model_stats[model]["pair_floor_24_pass"] for model in MODELS),
            "models_with_triplet_floor_24": sum(model_stats[model]["triplet_floor_24_pass"] for model in MODELS),
            "all_models_observed_conditional_dose_monotone": all(model_stats[model]["dose_response_diagnostic"]["observed_conditional_monotone"] for model in MODELS),
            "all_models_complete_triplet_nonmonotone_count": sum(sum(model_stats[model]["complete_triplet_nonmonotone_patterns"].values()) for model in MODELS),
            "promotion_status": "NOT_APPLIED_AC4_REQUIRED_AND_CENSORING_SENSITIVITY_REMAINS_EXPLICIT",
        },
        "outcome_firewall": {"new_model_inference": 0, "new_env_steps": 0, "new_open_interventions": 0, "new_physical_outcome_reads": 0, "new_pgd": 0, "new_protected_reads": 0},
        "next_legal_action": next_action,
    }
    report_artifact = write_new(args.output_dir / "STAGE_AC_AC3_G2R1_G3R1_CENSORING_AWARE_STATISTICS_V1.json", report)
    root_payload = {"gate": report["gate"], "status": global_status, "report": report_artifact, "source_authority": report["source_authority"], "counts": report["counts"], "model_summary": model_stats, "holm_correction": report["holm_correction"], "global_diagnostic": report["global_diagnostic"], "outcome_firewall": report["outcome_firewall"], "next_legal_action": next_action}
    root = {"schema": "STAGE_AC_AC3_G2R1_G3R1_ROOT_SEAL_V1", "status": global_status, "root_payload": root_payload, "root_payload_sha256": canonical_hash(root_payload), "artifacts": {"statistics": report_artifact}, "claim_boundary": report["claim_boundary"], "next_legal_action": next_action}
    root_artifact = write_new(args.output_dir / "STAGE_AC_AC3_G2R1_G3R1_ROOT_SEAL_V1.json", root)
    return {"status": global_status, "counts": report["counts"], "model_summary": model_stats, "holm": report["holm_correction"], "artifacts": {"statistics": report_artifact, "root": root_artifact}, "root_payload_sha256": root["root_payload_sha256"]}


def self_test() -> None:
    assert math.isclose(exact_mcnemar_p(8, 0), 0.0078125)
    assert holm_adjust({"a": 0.001, "b": 0.01, "c": 0.1}) == {"a": 0.003, "b": 0.02, "c": 0.1}
    print(json.dumps({"status": "AC3_G3R1_STATIC_SELF_TEST_PASS", "unknown_is_not_zero": True}, sort_keys=True))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--g2r1-c", type=Path, default=ROOT / "reports/STAGE_AC_AC3_G2R1_C_CENSORING_AWARE_ANALYSIS_V1/STAGE_AC_AC3_G2R1_C_CENSORING_AWARE_ANALYSIS_V1.json")
    parser.add_argument("--g2r1-c-root", type=Path, default=ROOT / "reports/STAGE_AC_AC3_G2R1_C_CENSORING_AWARE_ANALYSIS_V1/STAGE_AC_AC3_G2R1_C_ROOT_SEAL_V1.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports/STAGE_AC_AC3_G2R1_G3R1_CENSORING_AWARE_STATISTICS_V1")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    print(json.dumps(build(args), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
