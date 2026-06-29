from __future__ import annotations

import argparse
import json
from pathlib import Path


def aggregate(rnad_path: Path, cqfr_path: Path) -> dict:
    rnad = json.loads(rnad_path.read_text(encoding="utf-8"))["conditions"]
    cqfr = json.loads(cqfr_path.read_text(encoding="utf-8"))["conditions"]
    conditions = sorted(set(rnad) | set(cqfr))
    return {
        "schema_version": "table1_metric_aggregation.v1",
        "conditions": {
            c: {
                "condition_id": c,
                "row_count": max(rnad.get(c, {}).get("itt_count", 0), cqfr.get(c, {}).get("itt_count", 0)),
                "rnad": rnad.get(c),
                "cqfr": cqfr.get(c),
            }
            for c in conditions
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rnad-json", type=Path, required=True)
    ap.add_argument("--cqfr-json", type=Path, required=True)
    ap.add_argument("--output-json", type=Path, required=True)
    args = ap.parse_args()
    out = aggregate(args.rnad_json, args.cqfr_json)
    args.output_json.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
