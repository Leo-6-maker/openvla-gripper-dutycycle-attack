#!/usr/bin/env python3
"""Build an attack manifest only; this command never launches a rollout."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from gripper_attack.b3_v3_attack_protocol import build_attack_manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parents-csv", type=Path, required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--check-status", required=True)
    parser.add_argument("--cs200-manifest-sha256", required=True)
    parser.add_argument("--cs200-manifest", type=Path, required=True)
    parser.add_argument("--check-report-sha256", required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--calibration-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite: {args.output}")
    with args.parents_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    manifest = build_attack_manifest(
        rows, protocol_sha256=args.protocol_sha256, check_status=args.check_status,
        cs200_manifest_sha256=args.cs200_manifest_sha256, check_report_sha256=args.check_report_sha256,
        checkpoint_sha256=args.checkpoint_sha256, calibration_sha256=args.calibration_sha256,
        cs200_manifest_path=args.cs200_manifest,
    )
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": manifest["status"], "cell_count": manifest["cell_count"], "attack_execution_authorized": False}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
