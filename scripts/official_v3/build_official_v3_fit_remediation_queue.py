#!/usr/bin/env python3
"""Build an exact-identity FIT remediation queue from a bridge report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gripper_attack.official_v3_sprint0 import (
    Sprint0ContractViolation,
    build_fit_remediation_queue,
    read_csv_rows,
    read_json,
    write_sealed_csv,
    write_sealed_json,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--canonical-manifest", type=Path, required=True)
    parser.add_argument("--bridge-report", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--queue-epoch-id", required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-summary", type=Path, required=True)
    parser.add_argument("--formal-registry", type=Path)
    args = parser.parse_args()
    try:
        formal = read_csv_rows(args.formal_registry) if args.formal_registry else []
        rows, summary = build_fit_remediation_queue(
            read_csv_rows(args.canonical_manifest),
            read_json(args.bridge_report),
            read_csv_rows(args.ledger),
            queue_epoch_id=args.queue_epoch_id,
            formal_registry_rows=formal,
        )
        write_sealed_csv(args.output_csv, rows)
        summary["queue_csv_sha256"] = __import__("hashlib").sha256(args.output_csv.read_bytes()).hexdigest()
        write_sealed_json(args.output_summary, summary)
    except (OSError, json.JSONDecodeError, Sprint0ContractViolation) as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps({"identity_count": summary["identity_count"], "queue_epoch_id": summary["queue_epoch_id"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
