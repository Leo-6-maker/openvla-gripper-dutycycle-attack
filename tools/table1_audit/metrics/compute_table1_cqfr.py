from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def compute(path: Path) -> dict:
    groups: dict[str, list[dict]] = defaultdict(list)
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            groups[row["condition_id"]].append({
                "emitted": str(row.get("detector_emitted", "")).lower() == "true",
                "failure": str(row.get("contact_quality_failure", "")).lower(),
            })
    return {"schema_version": "table1_cqfr_result.v1", "conditions": {k: summarize(v) for k, v in sorted(groups.items())}}


def summarize(rows: list[dict]) -> dict:
    emitted = [r for r in rows if r["emitted"]]
    failures = sum(1 for r in rows if r["failure"] == "true")
    emitted_failures = sum(1 for r in emitted if r["failure"] == "true")
    return {
        "itt_count": len(rows),
        "emitted_count": len(emitted),
        "no_emission_count": len(rows) - len(emitted),
        "unknown_count": sum(1 for r in rows if r["failure"] == "unknown"),
        "cqfr_itt": failures / len(rows) if rows else 0.0,
        "cqfr_conditional_emitted": emitted_failures / len(emitted) if emitted else 0.0,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels-csv", type=Path, required=True)
    ap.add_argument("--output-json", type=Path, required=True)
    args = ap.parse_args()
    args.output_json.write_text(json.dumps(compute(args.labels_csv), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
