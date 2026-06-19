#!/usr/bin/env python3
"""Build a Teacher/privileged-label coverage table from a CLEAN master ledger."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def truthy(v: Any) -> bool:
    return v is True or str(v).lower() == "true"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def build_coverage(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[(row.get("suite", ""), row.get("condition", ""))].append(row)
    out: list[dict[str, Any]] = []
    for (suite, condition), rs in sorted(groups.items()):
        valid = [r for r in rs if r.get("status") == "COMPLETE_VALID"]
        privileged = [r for r in valid if truthy(r.get("privileged_valid"))]
        abstain = [r for r in valid if truthy(r.get("teacher_abstain"))]
        out.append({
            "suite": suite,
            "condition": condition,
            "valid_episode_count": len(valid),
            "privileged_valid_count": len(privileged),
            "teacher_abstain_count": len(abstain),
            "teacher_label_coverage_rate": round(len(privileged) / len(valid), 6) if valid else "",
            "teacher_timing_label_count": len(privileged),
            "teacher_timing_eval_status": "REQUIRES_OFFLINE_RESOLVER_AND_PREREG_GATE",
            "object_target_binding_valid_count": "",
            "anchor_present_count": "",
            "event_type_valid_count": "",
            "anonymous_object_identity_count": "",
            "usable_for_clean_sr_only": len(valid) > 0 and len(privileged) == 0,
        })
    return out


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--master-ledger", required=True)
    ap.add_argument("--output-csv", required=True)
    ap.add_argument("--output-json", required=True)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_rows(Path(args.master_ledger))
    coverage = build_coverage(rows)
    write_csv(Path(args.output_csv), coverage)
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps({"coverage": coverage}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"result": "TEACHER_COVERAGE_DONE", "groups": len(coverage)}, sort_keys=True))


if __name__ == "__main__":
    main()
