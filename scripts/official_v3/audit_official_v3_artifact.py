#!/usr/bin/env python3
"""Audit one Official V3 CLEAN artifact without opening Teacher labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gripper_attack.official_v3_contract import audit_artifact, load_contract


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--equivalence-status", choices=("PASS", "HOLD"), default="HOLD")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite audit output: {args.output}")
    result = audit_artifact(args.artifact, load_contract(args.contract), equivalence_status=args.equivalence_status)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: result.get(key) for key in ("status", "canonical_parent_key", "error_code")}, sort_keys=True))
    return 0 if result["status"] in {"PASS_FORMAL_CANDIDATE", "PASS_DATA_CONTRACT_PROVENANCE_HOLD"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
