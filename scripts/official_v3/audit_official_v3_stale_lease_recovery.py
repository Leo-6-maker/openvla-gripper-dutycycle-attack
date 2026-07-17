#!/usr/bin/env python3
"""Read-only stale-lease fencing audit for Official V3."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gripper_attack.official_v3_sprint0 import (
    Sprint0ContractViolation,
    audit_stale_lease_recovery,
    read_csv_rows,
    read_json,
    write_sealed_json,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--process-snapshot", type=Path, required=True)
    parser.add_argument("--formal-results", type=Path, required=True)
    parser.add_argument("--recovery-records", type=Path, required=True)
    parser.add_argument("--now-epoch", type=float, required=True)
    parser.add_argument("--expected-stale-keys", type=Path)
    parser.add_argument("--stale-after-seconds", type=float, default=600.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        process = read_json(args.process_snapshot).get("processes", [])
        recovery = read_json(args.recovery_records).get("recoveries", [])
        expected = None
        if args.expected_stale_keys:
            expected = read_json(args.expected_stale_keys).get("canonical_parent_keys")
        report = audit_stale_lease_recovery(
            read_csv_rows(args.ledger),
            process,
            read_csv_rows(args.formal_results),
            recovery,
            now_epoch=args.now_epoch,
            stale_after_seconds=args.stale_after_seconds,
            expected_stale_keys=expected,
        )
        write_sealed_json(args.output, report)
    except (OSError, json.JSONDecodeError, Sprint0ContractViolation) as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps({"status": report["status"], "stale_keys": report["stale_keys"]}, sort_keys=True))
    return 0 if report["status"] != "HOLD" else 2


if __name__ == "__main__":
    raise SystemExit(main())
