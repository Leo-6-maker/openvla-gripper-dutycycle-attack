#!/usr/bin/env python3
"""Read-only audit for a machine-built training authorization bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gripper_attack.b3_training_protocol import _measure_generator_provenance, _snapshot_sha, load_training_authorization_bundle
from gripper_attack.b3_formal import AUTHORIZATION_INPUT_NAMES
from gripper_attack.b3_official_v3_s1 import build_s1_runner_binding


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--authorization-root", type=Path, required=True)
    parser.add_argument("--runner-repo", type=Path, required=True)
    parser.add_argument("--runner-config", type=Path, required=True)
    parser.add_argument("--runner-script", type=Path, required=True)
    parser.add_argument("--generator-script", type=Path, required=True)
    parser.add_argument("--input", action="append", nargs=2, metavar=("NAME", "PATH"), required=True)
    args = parser.parse_args()
    payload = load_training_authorization_bundle(args.authorization_root)
    measured = build_s1_runner_binding(
        runner_repo=args.runner_repo, expected_runner_head=payload["runner_head"],
        config_path=args.runner_config, runner_script_path=args.runner_script,
    )
    if measured != payload["runner_binding"]:
        raise SystemExit("authorization runner binding mismatch")
    generator = _measure_generator_provenance(
        runner_repo=args.runner_repo, generator_script=args.generator_script,
        expected_head=payload["authorization_generation"]["generator_head"],
    )
    if generator != payload["authorization_generation"]:
        raise SystemExit("authorization generator provenance mismatch")
    inputs = {name: Path(path) for name, path in args.input}
    if len(inputs) != len(args.input) or set(inputs) != set(AUTHORIZATION_INPUT_NAMES):
        raise SystemExit("authorization input inventory is not exactly the frozen 13-item set")
    for name, path in inputs.items():
        if _snapshot_sha(name, path) != payload["input_snapshots"][name]:
            raise SystemExit(f"authorization input SHA mismatch: {name}")
    print(json.dumps({"status": "PASS", "variant": payload["variant"], "fold_id": payload["fold_id"], "seed": payload["seed"], "formal_training_authorized": True, "formal_attack_authorized": False}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
