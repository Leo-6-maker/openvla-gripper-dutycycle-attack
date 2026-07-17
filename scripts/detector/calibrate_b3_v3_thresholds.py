#!/usr/bin/env python3
"""Preparation-only CAL threshold contract; no CHECK or attack access."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def validate_cal_records(records: list[dict]) -> None:
    if not records or any(row.get("split") != "CAL" or not 24 <= int(row.get("state_id", -1)) <= 26 for row in records):
        raise ValueError("CAL input must contain only states 24-26")
    if any(row.get("attack_enabled") is True for row in records):
        raise ValueError("CAL cannot read attack-enabled records")


def build_calibration_plan() -> dict:
    return {
        "schema": "B3_OFFICIAL_V3_CALIBRATION_PLAN_V1",
        "persistence_steps": 3,
        "persistence_required": 2,
        "thresholds_selected": ["retention_active", "retention_continuation_t10", "release_imminent_veto"],
        "status": "PREPARATION_ONLY",
        "formal_training_authorized": False,
        "formal_attack_authorized": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite: {args.output}")
    args.output.write_text(json.dumps(build_calibration_plan(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
