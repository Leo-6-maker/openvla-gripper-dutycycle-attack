#!/usr/bin/env python3
"""Independent structural audit of a V3 S1 Teacher materialization root."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gripper_attack.b3_official_v3_s1 import (
    audit_materialized_root,
    load_formal_fit_registry,
    sha256_file,
    write_sealed_json,
)


def audit_root(materialized_root: Path, registry_rows: list[dict[str, str]]) -> dict[str, object]:
    report = audit_materialized_root(materialized_root, registry_rows, require_runner_binding=True)
    report["materialized_root"] = str(materialized_root.resolve())
    report["teacher_audit_script_sha256"] = sha256_file(Path(__file__).resolve())
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--materialized-root", type=Path, required=True)
    parser.add_argument("--registry-csv", type=Path, required=True)
    parser.add_argument("--registry-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    registry_rows = load_formal_fit_registry(args.registry_csv, args.registry_summary)
    report = audit_root(args.materialized_root.resolve(), registry_rows)
    write_sealed_json(args.output, report)
    print(json.dumps({"status": report["status"], "identity_count": report["identity_count"]}, sort_keys=True))
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
