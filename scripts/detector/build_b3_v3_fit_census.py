#!/usr/bin/env python3
"""Freeze the exact 800-row Official V3 FIT census from a formal registry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gripper_attack.b3_official_v3_s1 import build_fit_census, sha256_file, write_sealed_csv, write_sealed_json


FIELDS = [
    "canonical_parent_key", "suite", "task_idx", "state_id", "split",
    "selected_artifact_root", "selected_artifact_recursive_sha256",
    "artifact_audit_sha256", "provenance_class",
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry-csv", type=Path, required=True)
    parser.add_argument("--registry-summary", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-summary", type=Path, required=True)
    args = parser.parse_args()
    rows, summary = build_fit_census(args.registry_csv, args.registry_summary)
    write_sealed_csv(args.output_csv, rows, FIELDS)
    summary["census_csv_sha256"] = sha256_file(args.output_csv)
    write_sealed_json(args.output_summary, summary)
    print(json.dumps({"status": summary["status"], "identity_count": summary["identity_count"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
