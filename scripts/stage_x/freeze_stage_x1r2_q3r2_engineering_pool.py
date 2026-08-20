#!/usr/bin/env python3
"""Freeze an outcome-blind, permanently excluded Q3R2 fixture pool."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[2]
SALT = "STAGE_X_X1R2_Q3R2_ENGINEERING_FIXTURE_POOL_V1_20260820"
SUITES = ("libero_10", "libero_goal", "libero_object", "libero_spatial")
PER_SUITE = 12
G10_LEDGER = REPO / "reports/STAGE_X_X1R_T1D0R_G10_IDENTITY_EXCLUSION_LEDGER_V1.json"
SCIENCE_LEDGER = REPO / "reports/STAGE_X_X1R_T1D0R2_PARENT_SEED_INVARIANCE_V1.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    try:
        value = json.loads(text)
        if isinstance(value, dict) and isinstance(value.get("rows"), list):
            return value["rows"]
    except json.JSONDecodeError:
        pass
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def rank(key: str) -> str:
    return hashlib.sha256(f"{SALT}|{key}".encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=REPO / "reports/STAGE_X_X1R2_Q3R2_ENGINEERING_FIXTURE_POOL_V1.json")
    args = parser.parse_args()

    g10 = rows(G10_LEDGER)
    science = rows(SCIENCE_LEDGER)
    science_keys = {str(row["canonical_parent_key"]) for row in science}
    explicit_historical = {
        "libero_10/task_04/state_30", "libero_goal/task_02/state_34",
        "libero_object/task_04/state_35", "libero_spatial/task_01/state_38",
        "libero_10/task_09/state_43",
    }
    excluded = science_keys | explicit_historical
    fresh = [row for row in g10 if not bool(row.get("excluded_union"))]
    eligible = [row for row in fresh if str(row["canonical_parent_key"]) not in excluded]
    if len({str(row["canonical_parent_key"]) for row in eligible}) != len(eligible):
        raise SystemExit("DUPLICATE_ENGINEERING_IDENTITY")

    selected = []
    for suite in SUITES:
        ordered = sorted((row for row in eligible if row["suite"] == suite), key=lambda row: (rank(str(row["canonical_parent_key"])), str(row["canonical_parent_key"])))
        if len(ordered) < PER_SUITE:
            raise SystemExit(f"INSUFFICIENT_ENGINEERING_FIXTURES:{suite}:{len(ordered)}")
        for position, row in enumerate(ordered[:PER_SUITE], start=1):
            selected.append({
                "fixture_id": f"Q3R2-{suite.upper()}-{position:02d}",
                "suite": row["suite"], "task_idx": int(row["task_idx"]), "state_id": int(row["state_id"]),
                "canonical_parent_key": row["canonical_parent_key"], "rank_sha256": rank(str(row["canonical_parent_key"])),
                "source_row": {"excluded_union": row["excluded_union"], "fresh_after_exclusion": row["fresh_after_exclusion"], "prior_clean_attempt": row["prior_clean_attempt"], "prior_exposure": row["prior_exposure"]},
                "permanent_exclusion": True, "outcome_read": False, "scientific_use": False,
            })

    report = {
        "schema": "STAGE_X_X1R2_Q3R2_ENGINEERING_FIXTURE_POOL_V1",
        "status": "STAGE_X1R2_Q3R2_ENGINEERING_FIXTURE_POOL_FROZEN",
        "purpose": "Outcome-blind engineering qualification only; every selected identity is permanently excluded from all scientific populations.",
        "selection": {"salt": SALT, "candidate_order": "sha256(salt|canonical_parent_key), then canonical_parent_key", "per_suite_count": PER_SUITE, "selected_count": len(selected)},
        "candidate_universe": {"path": G10_LEDGER.relative_to(REPO).as_posix(), "sha256": sha256(G10_LEDGER), "rows": len(g10), "fresh_rows_before_stage_x_exclusion": len(fresh), "fresh_rows_after_stage_x_exclusion": len(eligible)},
        "exclusions": {"stage_x1r_scientific_parent_ledger": {"path": SCIENCE_LEDGER.relative_to(REPO).as_posix(), "sha256": sha256(SCIENCE_LEDGER), "rows": len(science)}, "explicit_historical_q3_keys": sorted(explicit_historical), "protected_boundary": "Eval160/protected identities are never eligible; no protected identity is admitted by this source-universe join; counters remain zero"},
        "selected": selected,
        "selection_inputs": {"student_scores": False, "clean_outcomes": False, "emit_outcomes": False, "attack_outcomes": False, "v_phys": False, "manual_outcomes": False, "protected_data": False},
        "runtime": {"authority": "configs/STAGE_X_X1R2_Q3R2_RUNTIME_AUTHORITY_V1.json", "model_inference_authorized": False, "clean_rollout_authorized": False, "pgd_authorized": False, "env_step_authorized": False},
        "protected_boundary": {"eval160": "UNREAD", "protected_evaluation": "UNREAD", "counters": {"model_inference_calls": 0, "env_step_calls": 0, "pgd_calls": 0, "physical_interventions": 0, "vphys_reads": 0, "protected_reads": 0}},
        "next_gate": "STAGE_X1R2_Q3R2_CLEAN_PREFIX_DETERMINISM",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "selected": len(selected), "by_suite": {suite: sum(row["suite"] == suite for row in selected) for suite in SUITES}}, sort_keys=True))


if __name__ == "__main__":
    main()
