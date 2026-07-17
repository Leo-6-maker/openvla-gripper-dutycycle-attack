#!/usr/bin/env python3
"""Audit legacy-start provenance without reading Teacher or attack evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gripper_attack.official_v3_sprint0 import (
    Sprint0ContractViolation,
    _attach_run_binding,
    _input_binding,
    _runner_binding,
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
    parser.add_argument("--exploratory-partial", action="store_true")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--runner-head", required=True)
    parser.add_argument("--runner-worktree-clean", choices=("true", "false"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.expected_keys and not args.exploratory_partial:
        raise SystemExit("--expected-keys is required for a formal bridge audit; use --exploratory-partial explicitly for a non-formal subset")
    inventory_rows = read_csv_rows(args.inventory)
    baseline = read_json(args.baseline)
    expected = None
    if args.expected_keys:
        expected_payload = read_json(args.expected_keys)
        expected = expected_payload.get("canonical_parent_keys")
        if not isinstance(expected, list) or not all(isinstance(key, str) for key in expected):
            raise SystemExit("expected-keys JSON requires canonical_parent_keys list")
    try:
        report = audit_legacy_bridge(inventory_rows, baseline, expected_keys=expected, allow_partial=args.exploratory_partial)
        inputs = {
            "inventory": _input_binding(args.inventory, schema="OFFICIAL_V3_LEGACY_PROVENANCE_INVENTORY_V1", row_count=len(inventory_rows), identity_count=len(inventory_rows)),
            "baseline": _input_binding(args.baseline, schema=str(baseline.get("schema", "JSON"))),
            "config": _input_binding(args.config, schema="OFFICIAL_V3_SPRINT0_PROVENANCE_V1"),
        }
        if args.expected_keys:
            inputs["expected_keys"] = _input_binding(args.expected_keys, schema="OFFICIAL_V3_CANONICAL_IDENTITY_SET_V1", identity_count=len(expected or []))
        report = _attach_run_binding(
            report,
            inputs=inputs,
            runner=_runner_binding(
                runner_head=args.runner_head,
                worktree_clean=args.runner_worktree_clean == "true",
                config_path=args.config,
            ),
        )
        write_sealed_json(args.output, report)
    except (OSError, json.JSONDecodeError, Sprint0ContractViolation) as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps({key: report[key] for key in ("overall_status", "identity_count", "exact_remediation_required_count")}, sort_keys=True))
    return 0 if report["official_v3_overall_status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
