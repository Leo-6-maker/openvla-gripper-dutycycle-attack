#!/usr/bin/env python3
"""Analyze an audited C2g matched-load five-condition result matrix.

The script consumes only a PASS closed-world audit report. It reports condition
success rates, paired success-flip tables, exact two-sided McNemar/binomial tests,
timing and objective effects, and the 2x2 interaction on failure probability.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.gripper_attack.c2g_matched_load_manifest import CORE_CONDITIONS


def exact_two_sided_binomial(discordant_a: int, discordant_b: int) -> float:
    n = int(discordant_a + discordant_b)
    if n <= 0:
        return 1.0
    tail = min(discordant_a, discordant_b)
    probability = sum(math.comb(n, value) for value in range(tail + 1)) / (2.0 ** n)
    return min(1.0, 2.0 * probability)


def paired_comparison(
    parents: Sequence[Mapping[str, Any]],
    first: str,
    second: str,
) -> dict[str, Any]:
    pairs: list[tuple[bool, bool]] = []
    for parent in parents:
        success = parent["success"]
        left, right = success.get(first), success.get(second)
        if type(left) is bool and type(right) is bool:
            pairs.append((left, right))
    first_only = sum(left and not right for left, right in pairs)
    second_only = sum(right and not left for left, right in pairs)
    both_success = sum(left and right for left, right in pairs)
    both_failure = sum((not left) and (not right) for left, right in pairs)
    first_rate = sum(left for left, _ in pairs) / max(1, len(pairs))
    second_rate = sum(right for _, right in pairs) / max(1, len(pairs))
    return {
        "first": first,
        "second": second,
        "paired_n": len(pairs),
        "first_success_rate": first_rate,
        "second_success_rate": second_rate,
        "success_rate_difference_first_minus_second": first_rate - second_rate,
        "failure_rate_difference_first_minus_second": (1.0 - first_rate) - (1.0 - second_rate),
        "both_success": both_success,
        "first_success_second_failure": first_only,
        "first_failure_second_success": second_only,
        "both_failure": both_failure,
        "discordant_n": first_only + second_only,
        "exact_two_sided_mcnemar_p": exact_two_sided_binomial(first_only, second_only),
    }


def condition_rates(parents: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for condition in CORE_CONDITIONS:
        values = [parent["success"].get(condition) for parent in parents]
        valid = [value for value in values if type(value) is bool]
        output[condition] = {
            "n": len(valid),
            "success_count": sum(valid),
            "failure_count": len(valid) - sum(valid),
            "success_rate": sum(valid) / max(1, len(valid)),
        }
    return output


def factorial_effects(rates: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    def failure(condition: str) -> float | None:
        row = rates[condition]
        return None if int(row["n"]) == 0 else 1.0 - float(row["success_rate"])

    dg = failure("DET_GRIPPER_VIS_PGD")
    dr = failure("DET_RANDOM_VIS_ATTACK")
    rg = failure("RANDTIME_GRIPPER_VIS_PGD")
    rr = failure("RANDTIME_RANDOM_VIS_ATTACK")
    if any(value is None for value in (dg, dr, rg, rr)):
        return {"available": False}
    return {
        "available": True,
        "failure_rates": {
            "DET_GRIPPER": dg,
            "DET_RANDOM": dr,
            "RANDTIME_GRIPPER": rg,
            "RANDTIME_RANDOM": rr,
        },
        "timing_effect_under_gripper": dg - rg,
        "timing_effect_under_random_objective": dr - rr,
        "objective_effect_under_detector_timing": dg - dr,
        "objective_effect_under_random_timing": rg - rr,
        "difference_in_differences_interaction": (dg - rg) - (dr - rr),
    }


def analyze(report: Mapping[str, Any], denominator_report: Mapping[str, Any] | None = None) -> dict[str, Any]:
    if report.get("status") != "PASS_C2G_MATCHED_LOAD_RUN_AUDIT":
        raise ValueError("analysis requires a PASS closed-world runtime audit")
    parents = list(report.get("parents", []))
    if not parents:
        raise ValueError("audit report contains no parent summaries")
    rates = condition_rates(parents)
    comparisons = {
        "timing_value_gripper_objective": paired_comparison(
            parents, "DET_GRIPPER_VIS_PGD", "RANDTIME_GRIPPER_VIS_PGD"
        ),
        "objective_specificity_detector_timing": paired_comparison(
            parents, "DET_GRIPPER_VIS_PGD", "DET_RANDOM_VIS_ATTACK"
        ),
        "timing_value_random_objective": paired_comparison(
            parents, "DET_RANDOM_VIS_ATTACK", "RANDTIME_RANDOM_VIS_ATTACK"
        ),
        "objective_specificity_random_timing": paired_comparison(
            parents, "RANDTIME_GRIPPER_VIS_PGD", "RANDTIME_RANDOM_VIS_ATTACK"
        ),
        "clean_vs_detector_gripper": paired_comparison(
            parents, "CLEAN", "DET_GRIPPER_VIS_PGD"
        ),
    }
    by_suite: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    job_suite = {
        str(job["parent_key"]): str(job.get("suite") or str(job["parent_key"]).split("/", 1)[0])
        for job in report.get("jobs", [])
        if job.get("parent_key")
    }
    for parent in parents:
        suite = job_suite.get(str(parent["parent_key"]), "unknown")
        by_suite[suite].append(parent)
    denominator = dict(denominator_report or {})
    return {
        "gate": "C2G_MATCHED_LOAD_RESULT_ANALYSIS",
        "status": "PASS_C2G_MATCHED_LOAD_RESULT_ANALYSIS",
        "audited_parent_count": len(parents),
        "condition_rates": rates,
        "paired_comparisons": comparisons,
        "factorial_effects": factorial_effects(rates),
        "per_suite": {
            suite: {
                "parent_count": len(rows),
                "condition_rates": condition_rates(rows),
                "timing_value_gripper_objective": paired_comparison(
                    rows, "DET_GRIPPER_VIS_PGD", "RANDTIME_GRIPPER_VIS_PGD"
                ),
                "objective_specificity_detector_timing": paired_comparison(
                    rows, "DET_GRIPPER_VIS_PGD", "DET_RANDOM_VIS_ATTACK"
                ),
            }
            for suite, rows in sorted(by_suite.items())
        },
        "detector_coverage_denominator": {
            "input_parent_count": denominator.get("input_parent_count"),
            "included_parent_count": denominator.get("included_parent_count"),
            "excluded_parent_count": denominator.get("excluded_parent_count"),
            "detector_emit_burst_feasible_coverage": denominator.get(
                "detector_emit_burst_feasible_coverage"
            ),
            "excluded_reason_counts": denominator.get("excluded_reason_counts", {}),
        },
        "interpretation_contract": {
            "timing_primary": "DET_GRIPPER_VIS_PGD vs RANDTIME_GRIPPER_VIS_PGD",
            "objective_primary": "DET_GRIPPER_VIS_PGD vs DET_RANDOM_VIS_ATTACK",
            "selection_or_thresholds_changed_from_attacked_outcomes": False,
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-report", type=Path, required=True)
    parser.add_argument("--job-build-report", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    audit_report = json.loads(args.audit_report.read_text(encoding="utf-8"))
    denominator = (
        json.loads(args.job_build_report.read_text(encoding="utf-8"))
        if args.job_build_report else None
    )
    result = analyze(audit_report, denominator)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
