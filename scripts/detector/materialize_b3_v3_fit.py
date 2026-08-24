#!/usr/bin/env python3
"""Transactional 800-episode Official V3 FIT S1 materialization."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gripper_attack.b3_official_v3_s1 import build_s1_runner_binding, materialize_fit


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry-csv", type=Path, required=True)
    parser.add_argument("--registry-summary", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--runner-repo", type=Path, required=True)
    parser.add_argument("--expected-runner-head", required=True)
    parser.add_argument("--runner-config", type=Path, required=True)
    parser.add_argument("--campaign-contract", type=Path)
    args = parser.parse_args()
    binding = build_s1_runner_binding(
        runner_repo=args.runner_repo,
        expected_runner_head=args.expected_runner_head,
        config_path=args.runner_config,
        runner_script_path=Path(__file__).resolve(),
    )
    manifest = materialize_fit(
        args.registry_csv, args.registry_summary, args.contract, args.protocol, args.output_root, binding,
        campaign_contract_path=args.campaign_contract,
    )
    print(json.dumps({"status": manifest["status"], "identity_count": manifest["identity_count"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
