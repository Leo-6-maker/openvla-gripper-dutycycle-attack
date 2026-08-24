#!/usr/bin/env python3
"""One-time CHECK receipt builder; execution remains disabled in preparation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_check_receipt(*, checkpoint_status: str, check_access_key: str) -> dict:
    if checkpoint_status != "FIT_DEV_SELECTED":
        raise ValueError("CHECK requires a FIT_DEV_SELECTED checkpoint")
    return {
        "schema": "B3_OFFICIAL_V3_CHECK_ACCESS_RECEIPT_V1",
        "check_access_key": check_access_key,
        "status": "PREPARATION_ONLY",
        "check_executed": False,
        "formal_training_authorized": False,
        "formal_attack_authorized": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-status", required=True)
    parser.add_argument("--check-access-key", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite: {args.output}")
    args.output.write_text(json.dumps(build_check_receipt(checkpoint_status=args.checkpoint_status, check_access_key=args.check_access_key), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
