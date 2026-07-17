#!/usr/bin/env python3
"""Read-only audit for a machine-built training authorization bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gripper_attack.b3_training_protocol import load_training_authorization_bundle, sha256_file
from gripper_attack.b3_formal import json_sha


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--authorization-root", type=Path, required=True)
    parser.add_argument("--generator-script", type=Path)
    parser.add_argument("--runner-binding-json", type=Path)
    parser.add_argument("--input", action="append", nargs=2, metavar=("NAME", "PATH"))
    args = parser.parse_args()
    payload = load_training_authorization_bundle(args.authorization_root)
    if args.generator_script is not None and sha256_file(args.generator_script) != payload["authorization_generation"]["generator_script_sha256"]:
        raise SystemExit("authorization generator SHA mismatch")
    if args.runner_binding_json is not None:
        expected = json.loads(args.runner_binding_json.read_text(encoding="utf-8"))
        if json_sha(expected) != json_sha(payload["runner_binding"]):
            raise SystemExit("authorization runner binding mismatch")
    for name, path in args.input or []:
        if name not in payload["input_snapshots"] or sha256_file(Path(path)) != payload["input_snapshots"][name]:
            raise SystemExit(f"authorization input SHA mismatch: {name}")
    print(json.dumps({"status": "PASS", "variant": payload["variant"], "fold_id": payload["fold_id"], "seed": payload["seed"], "formal_training_authorized": True, "formal_attack_authorized": False}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
