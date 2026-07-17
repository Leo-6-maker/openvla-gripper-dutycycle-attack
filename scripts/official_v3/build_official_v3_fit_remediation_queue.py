#!/usr/bin/env python3
"""Build an exact-identity FIT remediation queue from a bridge report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gripper_attack.official_v3_sprint0 import (
    Sprint0ContractViolation,
    _input_binding,
    _runner_binding,
    build_fit_remediation_queue,
    read_csv_rows,
    read_json,
    write_sealed_remediation_bundle,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--canonical-manifest", type=Path, required=True)
    parser.add_argument("--bridge-report", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--queue-epoch-id", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--formal-registry", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--runner-repo", type=Path, required=True)
    parser.add_argument("--expected-runner-head", required=True)
    args = parser.parse_args()
    try:
        manifest_rows = read_csv_rows(args.canonical_manifest)
        bridge_report = read_json(args.bridge_report)
        ledger_rows = read_csv_rows(args.ledger)
        formal = read_csv_rows(args.formal_registry) if args.formal_registry else []
        rows, summary = build_fit_remediation_queue(
            manifest_rows,
            bridge_report,
            ledger_rows,
            queue_epoch_id=args.queue_epoch_id,
            formal_registry_rows=formal,
            input_snapshots={
                "canonical_manifest": _input_binding(args.canonical_manifest, schema="OFFICIAL_V3_CANONICAL_MANIFEST_V1", row_count=len(manifest_rows), identity_count=len(manifest_rows)),
                "bridge_report": _input_binding(args.bridge_report, schema=str(bridge_report.get("schema", "JSON")), row_count=len(bridge_report.get("records", [])), identity_count=len(bridge_report.get("records", []))),
                "ledger": _input_binding(args.ledger, schema="OFFICIAL_V3_GLOBAL_CELL_LEDGER_V1", row_count=len(ledger_rows), identity_count=len({_row.get('canonical_parent_key') or _row.get('cell_id', '') for _row in ledger_rows})),
                "config": _input_binding(args.config, schema="OFFICIAL_V3_SPRINT0_PROVENANCE_V1"),
            },
            runner_binding=_runner_binding(
                runner_repo=args.runner_repo,
                expected_runner_head=args.expected_runner_head,
                config_path=args.config,
                runner_script_path=Path(__file__),
            ),
        )
        if args.formal_registry:
            summary["input_snapshots"]["formal_registry"] = _input_binding(args.formal_registry, schema="OFFICIAL_V3_FORMAL_REGISTRY_V1", row_count=len(formal), identity_count=len(formal))
        write_sealed_remediation_bundle(args.output_root, rows, summary)
    except (OSError, json.JSONDecodeError, Sprint0ContractViolation) as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps({"identity_count": summary["identity_count"], "queue_epoch_id": summary["queue_epoch_id"], "output_root": str(args.output_root)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
