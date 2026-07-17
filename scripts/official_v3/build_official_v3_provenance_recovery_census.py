#!/usr/bin/env python3
"""Build a discovery-only V3 provenance recovery census from normalized tables."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from gripper_attack.official_v3_contract import sha256_file
from gripper_attack.official_v3_recovery import build_recovery_rows, write_recovery_bundle


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--artifact-rows", type=Path, required=True)
    parser.add_argument("--worker-start-rows", type=Path, required=True)
    parser.add_argument("--lease-rows", type=Path, required=True)
    parser.add_argument("--completion-rows", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--snapshot-root-sha256", default="")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config.get("schema") != "OFFICIAL_V3_PROVENANCE_RECOVERY_CENSUS_V1":
        raise SystemExit("invalid recovery census config schema")
    if config.get("status") != "DISCOVERY_ONLY" or config.get("formal_decision_allowed") is not False:
        raise SystemExit("recovery census config is not discovery-only")
    raw_inputs = {
        "artifact_rows": args.artifact_rows,
        "worker_start_rows": args.worker_start_rows,
        "lease_rows": args.lease_rows,
        "completion_rows": args.completion_rows,
    }
    parsed = {name: read_csv(path) for name, path in raw_inputs.items()}
    record_rows = {
        name: (rows if name in {"artifact_rows", "worker_start_rows"}
               else [row for row in rows if row.get("canonical_parent_key", "").strip()])
        for name, rows in parsed.items()
    }
    rows, summary = build_recovery_rows(
        record_rows["artifact_rows"], record_rows["worker_start_rows"],
        record_rows["lease_rows"], record_rows["completion_rows"],
        snapshot_root_sha256=args.snapshot_root_sha256,
    )
    input_binding = {
        name: {
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
            "raw_row_count": len(parsed[name]),
            "record_row_count": len(record_rows[name]),
        }
        for name, path in raw_inputs.items()
    }
    input_binding["config"] = {
        "path": str(args.config.resolve()),
        "sha256": sha256_file(args.config),
        "schema": config["schema"],
    }
    write_recovery_bundle(rows, summary, args.output_root, input_binding=input_binding)
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
