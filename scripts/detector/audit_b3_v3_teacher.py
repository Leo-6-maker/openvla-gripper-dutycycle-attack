#!/usr/bin/env python3
"""Independent structural audit of a V3 S1 Teacher materialization root."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gripper_attack.b3_official_v3_s1 import (
    audit_materialized_root,
    build_s1_runner_binding,
    load_formal_fit_registry,
    rebuild_retention_features,
    sha256_file,
    write_sealed_json,
)
from gripper_attack.official_v3_contract import load_contract


def audit_root(
    materialized_root: Path, registry_rows: list[dict[str, str]], feature_order_sha256: str,
    *, expected_runner_binding: dict[str, object], expected_input_sha256: dict[str, str],
) -> dict[str, object]:
    report = audit_materialized_root(
        materialized_root, registry_rows, require_runner_binding=True,
        feature_order_sha256=feature_order_sha256, expected_runner_binding=expected_runner_binding,
        expected_input_sha256=expected_input_sha256,
    )
    report["materialized_root"] = str(materialized_root.resolve())
    report["teacher_audit_script_sha256"] = sha256_file(Path(__file__).resolve())
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--materialized-root", type=Path, required=True)
    parser.add_argument("--registry-csv", type=Path, required=True)
    parser.add_argument("--registry-summary", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--runner-repo", type=Path, required=True)
    parser.add_argument("--expected-runner-head", required=True)
    parser.add_argument("--runner-config", type=Path, required=True)
    parser.add_argument("--runner-script", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    registry_rows = load_formal_fit_registry(args.registry_csv, args.registry_summary)
    contract = load_contract(args.contract)
    expected_runner_binding = build_s1_runner_binding(
        runner_repo=args.runner_repo, expected_runner_head=args.expected_runner_head,
        config_path=args.runner_config, runner_script_path=args.runner_script,
    )
    report = audit_root(
        args.materialized_root.resolve(), registry_rows, contract["feature_order_sha256"],
        expected_runner_binding=expected_runner_binding,
        expected_input_sha256={
            "registry_csv_sha256": sha256_file(args.registry_csv),
            "registry_summary_sha256": sha256_file(args.registry_summary),
            "source_contract_sha256": sha256_file(args.contract),
            "protocol_sha256": sha256_file(args.protocol),
            "feature_rebuilder_sha256": sha256_file(Path(rebuild_retention_features.__code__.co_filename).resolve()),
        },
    )
    write_sealed_json(args.output, report)
    print(json.dumps({"status": report["status"], "identity_count": report["identity_count"]}, sort_keys=True))
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
