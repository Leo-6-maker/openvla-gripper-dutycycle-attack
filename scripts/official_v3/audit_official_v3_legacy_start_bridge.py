#!/usr/bin/env python3
"""Audit legacy-start provenance without reading Teacher or attack evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gripper_attack.official_v3_sprint0 import (
    Sprint0ContractViolation,
    audit_legacy_bridge,
    read_csv_rows,
    read_json,
    write_sealed_json,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--expected-keys", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    expected = None
    if args.expected_keys:
        expected_payload = read_json(args.expected_keys)
        expected = expected_payload.get("canonical_parent_keys")
        if not isinstance(expected, list) or not all(isinstance(key, str) for key in expected):
            raise SystemExit("expected-keys JSON requires canonical_parent_keys list")
    try:
        report = audit_legacy_bridge(read_csv_rows(args.inventory), read_json(args.baseline), expected_keys=expected)
        write_sealed_json(args.output, report)
    except (OSError, json.JSONDecodeError, Sprint0ContractViolation) as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps({key: report[key] for key in ("overall_status", "identity_count", "exact_remediation_required_count")}, sort_keys=True))
    return 0 if report["overall_status"] == "BRIDGE_PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
