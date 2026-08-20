#!/usr/bin/env python3
"""Aggregate four sealed Q3R2 clean-prefix suite reports without new inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


SUITES = ("libero_10", "libero_goal", "libero_object", "libero_spatial")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reports = {}
    for suite in SUITES:
        path = args.root / suite / "SUITE_CLEAN_DETERMINISM_REPORT_V1.json"
        if not path.is_file():
            raise SystemExit(f"MISSING_SUITE_REPORT:{suite}")
        reports[suite] = json.loads(path.read_text(encoding="utf-8"))
    statuses = {suite: report.get("status") for suite, report in reports.items()}
    passed = all(status == "PASS_SUITE_CLEAN_PREFIX_DETERMINISM" for status in statuses.values())
    counters = {"model_inference_calls": 0, "env_step_calls": 0, "pgd_calls": 0, "physical_interventions": 0, "vphys_reads": 0, "attack_outcome_reads": 0, "eval160_reads": 0, "protected_reads": 0}
    result = {"schema": "STAGE_X_X1R2_Q3R2_CLEAN_PREFIX_DETERMINISM_AUDIT_V1", "status": "STAGE_X1R2_Q3R2_CLEAN_PREFIX_DETERMINISM_PASS" if passed else "OWNER_REVIEW_Q3R2_CLEAN_PREFIX_DETERMINISM_NOT_ESTABLISHED", "suite_status": statuses, "reports": reports, "protected_boundary": {"eval160": "UNREAD", "protected_evaluation": "UNREAD", "counters": counters}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "suite_status": statuses}, sort_keys=True))
    raise SystemExit(0 if passed else 2)


if __name__ == "__main__":
    main()
