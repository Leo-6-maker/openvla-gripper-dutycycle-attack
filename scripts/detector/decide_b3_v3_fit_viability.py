#!/usr/bin/env python3
"""Build the sealed PASS/HOLD decision consumed by full-FIT refit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gripper_attack.b3_v3_viability_decision import build_viability_decision


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aggregate-root", type=Path, required=True)
    parser.add_argument("--decision-config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    decision = build_viability_decision(args.aggregate_root, args.decision_config, args.output_root)
    print(json.dumps({"status": decision["status"], "selected_variants": decision["selected_variants"], "formal_training_authorized": False, "formal_attack_authorized": False}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
