#!/usr/bin/env python3
"""Materialize one V3 FIT episode after the exact-800 registry gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gripper_attack.b3_official_v3_s1 import load_formal_fit_registry, materialize_episode


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry-csv", type=Path, required=True)
    parser.add_argument("--registry-summary", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--canonical-parent-key", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    rows = load_formal_fit_registry(args.registry_csv, args.registry_summary)
    row = next((item for item in rows if item["canonical_parent_key"] == args.canonical_parent_key), None)
    if row is None:
        raise SystemExit(f"identity is not in formal FIT registry: {args.canonical_parent_key}")
    manifest = materialize_episode(row, args.contract, args.protocol, args.output_root)
    print(json.dumps({"status": "PASS", "schema": manifest["schema"], "identity": args.canonical_parent_key}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
