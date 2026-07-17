#!/usr/bin/env python3
"""Read-only stale-lease fencing audit for Official V3."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gripper_attack.official_v3_sprint0 import (
    Sprint0ContractViolation,
    _input_binding,
    _runner_binding,
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
    parser.add_argument("--late-quarantine-records", type=Path, required=True)
    parser.add_argument("--now-epoch", type=float, required=True)
    parser.add_argument("--expected-stale-keys", type=Path, required=True)
    parser.add_argument("--stale-after-seconds", type=float, default=600.0)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--runner-repo", type=Path, required=True)
    parser.add_argument("--expected-runner-head", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        process_payload = read_json(args.process_snapshot)
        recovery_payload = read_json(args.recovery_records)
        expected_payload = read_json(args.expected_stale_keys)
        process = process_payload.get("processes", [])
        recovery = recovery_payload.get("recoveries", [])
        expected = expected_payload.get("canonical_parent_keys")
        if not isinstance(expected, list) or not all(isinstance(key, str) for key in expected):
            raise Sprint0ContractViolation("expected-stale-keys requires canonical_parent_keys list")
        ledger_rows = read_csv_rows(args.ledger)
        formal_rows = read_csv_rows(args.formal_results)
        quarantine_rows = read_csv_rows(args.late_quarantine_records)
        report = audit_stale_lease_recovery(
            ledger_rows,
            process,
            formal_rows,
            recovery,
            now_epoch=args.now_epoch,
            stale_after_seconds=args.stale_after_seconds,
            expected_stale_keys=expected,
            late_quarantine_rows=quarantine_rows,
            input_snapshots={
                "ledger": _input_binding(args.ledger, schema="OFFICIAL_V3_GLOBAL_CELL_LEDGER_V1", row_count=len(ledger_rows), identity_count=len(ledger_rows)),
                "process_snapshot": _input_binding(args.process_snapshot, schema=str(process_payload.get("schema", "JSON")), row_count=len(process), identity_count=len(process)),
                "formal_results": _input_binding(args.formal_results, schema="OFFICIAL_V3_FORMAL_RESULTS_V1", row_count=len(formal_rows), identity_count=len(formal_rows)),
                "recovery_records": _input_binding(args.recovery_records, schema=str(recovery_payload.get("schema", "JSON")), row_count=len(recovery), identity_count=len(recovery)),
                "late_quarantine_records": _input_binding(args.late_quarantine_records, schema="OFFICIAL_V3_LATE_RESULT_QUARANTINE_V1", row_count=len(quarantine_rows), identity_count=len(quarantine_rows)),
                "expected_stale_keys": _input_binding(args.expected_stale_keys, schema="OFFICIAL_V3_EXPECTED_STALE_KEYS_V1", identity_count=len(expected)),
                "config": _input_binding(args.config, schema="OFFICIAL_V3_SPRINT0_PROVENANCE_V1"),
            },
            runner_binding=_runner_binding(
                runner_repo=args.runner_repo,
                expected_runner_head=args.expected_runner_head,
                config_path=args.config,
                runner_script_path=Path(__file__),
            ),
        )
        write_sealed_json(args.output, report)
    except (OSError, json.JSONDecodeError, Sprint0ContractViolation) as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps({"status": report["status"], "stale_keys": report["stale_keys"]}, sort_keys=True))
    return 0 if report["status"] != "HOLD" else 2


if __name__ == "__main__":
    raise SystemExit(main())
