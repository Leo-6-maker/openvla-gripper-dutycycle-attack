#!/usr/bin/env python3
"""Materialize one V3 FIT episode after the exact-800 registry gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gripper_attack.b3_official_v3_s1 import build_s1_runner_binding, load_formal_fit_registry, materialize_episode


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry-csv", type=Path, required=True)
    parser.add_argument("--registry-summary", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--canonical-parent-key", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--runner-repo", type=Path, required=True)
    parser.add_argument("--expected-runner-head", required=True)
    parser.add_argument("--runner-config", type=Path, required=True)
    parser.add_argument("--campaign-contract", type=Path)
    args = parser.parse_args()
    rows = load_formal_fit_registry(args.registry_csv, args.registry_summary)
    row = next((item for item in rows if item["canonical_parent_key"] == args.canonical_parent_key), None)
    if row is None:
        raise SystemExit(f"identity is not in formal FIT registry: {args.canonical_parent_key}")
    binding = build_s1_runner_binding(
        runner_repo=args.runner_repo,
        expected_runner_head=args.expected_runner_head,
        config_path=args.runner_config,
        runner_script_path=Path(__file__).resolve(),
    )
    manifest = materialize_episode(row, args.contract, args.protocol, args.output_root, binding, args.campaign_contract)
    print(json.dumps({"status": "PASS", "schema": manifest["schema"], "identity": args.canonical_parent_key}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
