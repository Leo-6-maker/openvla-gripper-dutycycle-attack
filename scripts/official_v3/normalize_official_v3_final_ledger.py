#!/usr/bin/env python3
"""Create discovery-only normalized completion rows from the frozen V3 ledger."""

from __future__ import annotations

import argparse
import json
import csv
from pathlib import Path

from gripper_attack.official_v3_contract import sha256_file
from gripper_attack.official_v3_provenance_sources import (
    normalize_final_ledger_rows,
    write_normalized_completion_bundle,
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--artifact-rows", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    ledger_sha = sha256_file(args.ledger)
    rows, summary = normalize_final_ledger_rows(
        read_csv(args.ledger), read_csv(args.artifact_rows),
        ledger_source_path=str(args.ledger.resolve()), ledger_source_sha256=ledger_sha,
    )
    summary = dict(summary)
    summary["ledger_source_sha256"] = ledger_sha
    summary["artifact_rows_source_sha256"] = sha256_file(args.artifact_rows)
    write_normalized_completion_bundle(rows, summary, args.output_root)
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
