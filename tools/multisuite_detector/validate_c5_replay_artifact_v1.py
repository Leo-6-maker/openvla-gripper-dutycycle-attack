#!/usr/bin/env python3
"""Validate C5 replay artifact layout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REQUIRED_FILES = [
    "replay_manifest.json",
    "detector_freeze_identity.json",
    "dataset_identity.json",
    "threshold_identity.json",
    "metrics_overall.json",
    "metrics_by_suite.csv",
    "metrics_by_task.csv",
    "timing_error_report.json",
    "emission_rate_report.json",
    "safety_false_trigger_report.json",
    "SHA256SUMS",
    "SHA256SUMS.sha256",
]


def validate(root: str | Path) -> dict[str, object]:
    path = Path(root)
    missing = [name for name in REQUIRED_FILES if not (path / name).is_file()]
    if missing:
        raise ValueError("missing required files: " + ", ".join(missing))
    return {"status": "PASS", "schema_version": "c5_replay_artifact_validation_v1", "root": str(path)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--output-json")
    args = parser.parse_args()
    try:
        report = validate(args.root)
    except (OSError, ValueError) as exc:
        print(str(exc))
        return 1
    if args.output_json:
        Path(args.output_json).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
