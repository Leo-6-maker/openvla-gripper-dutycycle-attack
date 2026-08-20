#!/usr/bin/env python3
"""Create an append-only repair manifest for the known branch-receipt schema omission."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


SUITES = ("libero_10", "libero_goal", "libero_object", "libero_spatial")
FORBIDDEN = ("pgd_calls", "physical_interventions", "vphys_reads", "attack_outcome_reads", "attacked_env_steps", "protected_reads")


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execution-source-commit", required=True)
    parser.add_argument("--execution-source-tree", required=True)
    parser.add_argument("--repair-source-commit", required=True)
    parser.add_argument("--repair-source-tree", required=True)
    args = parser.parse_args()
    errors: list[str] = []
    repairs: list[dict[str, Any]] = []
    for suite in SUITES:
        report_path = args.root / suite / "SUITE_BRANCH_REPLAY_REPORT_V1.json"
        if not report_path.is_file():
            errors.append(f"MISSING_REPORT:{report_path}")
            continue
        report = load(report_path)
        if report.get("source", {}).get("commit") != args.execution_source_commit or report.get("source", {}).get("tree") != args.execution_source_tree:
            errors.append(f"SOURCE_MISMATCH:{suite}")
        fixture = report.get("selected_fixture")
        for branch in report.get("branch_receipts", []):
            boundary = dict(branch.get("protected_boundary", {}))
            for field in FORBIDDEN:
                if field != "protected_reads" and int(boundary.get(field, -1)) != 0:
                    errors.append(f"NONZERO:{suite}:repeat={branch.get('repeat')}:{field}")
            if boundary.get("protected_reads") is not None:
                if int(boundary["protected_reads"]) != 0:
                    errors.append(f"NONZERO:{suite}:repeat={branch.get('repeat')}:protected_reads")
                continue
            raw_path = args.root / suite / str(fixture) / f"branch_repeat_{int(branch['repeat'])}.json"
            if not raw_path.is_file():
                errors.append(f"MISSING_RAW_BRANCH_RECEIPT:{raw_path}")
                continue
            repairs.append({"suite": suite, "repeat": int(branch["repeat"]), "canonical_parent_key": branch.get("canonical_parent_key"), "fixture_id": fixture, "raw_receipt_path": raw_path.relative_to(args.root).as_posix(), "raw_receipt_sha256": sha256_file(raw_path), "field_added": "protected_reads", "derived_value": 0, "reason": "success receipt schema omission; no protected read path exists in the bound engineering runner; raw receipt is unchanged"})
    result = {"schema": "STAGE_X1R2_Q3R3_BRANCH_RECEIPT_SCHEMA_REPAIR_V1", "status": "PASS_APPEND_ONLY_RECEIPT_SCHEMA_REPAIR" if not errors else "HOLD_RECEIPT_SCHEMA_REPAIR", "execution_source": {"commit": args.execution_source_commit, "tree": args.execution_source_tree}, "repair_source": {"commit": args.repair_source_commit, "tree": args.repair_source_tree}, "raw_receipts_unchanged": True, "repairs": repairs, "errors": errors, "protected_boundary": {"pgd_calls": 0, "physical_interventions": 0, "vphys_reads": 0, "attack_outcome_reads": 0, "attacked_env_steps": 0, "protected_reads": 0, "eval160": "UNREAD", "protected_evaluation": "UNREAD"}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "repairs": len(repairs), "errors": errors}, ensure_ascii=False, sort_keys=True))
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
