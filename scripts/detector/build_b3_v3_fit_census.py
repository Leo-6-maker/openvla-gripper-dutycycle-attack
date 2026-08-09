#!/usr/bin/env python3
"""Freeze the exact 800-row Official V3 FIT census from a formal registry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gripper_attack.b3_official_v3_s1 import build_fit_census, write_census_bundle


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry-csv", type=Path, required=True)
    parser.add_argument("--registry-summary", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    rows, summary = build_fit_census(args.registry_csv, args.registry_summary)
    write_census_bundle(args.output_root, rows, summary)
    print(json.dumps({"status": summary["status"], "identity_count": summary["identity_count"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
