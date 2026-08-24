#!/usr/bin/env python3
"""Machine-build one fold/seed authorization bundle.

This command is deliberately not a generic JSON writer: the output can only
be produced after the explicit preparation gate and every input is bound by
its measured SHA-256 digest.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gripper_attack.b3_formal import AUTHORIZATION_INPUT_NAMES
from gripper_attack.b3_training_protocol import build_training_authorization_from_paths, load_training_authorization_bundle


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--variant", choices=("B3_25D", "B3_25D9D"), required=True)
    parser.add_argument("--fold-id", required=True)
    parser.add_argument("--fit-scope", choices=("FIT_FOLD", "FULL_FIT"), default="FIT_FOLD")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--runner-repo", type=Path, required=True)
    parser.add_argument("--expected-runner-head", required=True)
    parser.add_argument("--runner-config", type=Path, required=True)
    parser.add_argument("--runner-script", type=Path, required=True)
    parser.add_argument("--policy-intent-root", type=Path)
    parser.add_argument("--input", action="append", nargs=2, metavar=("NAME", "PATH"), required=True)
    parser.add_argument("--execute-preparation", action="store_true")
    args = parser.parse_args()
    if not args.execute_preparation:
        raise SystemExit("FORMAL_TRAINING_HOLD: authorization generation is not authorized without --execute-preparation")
    if args.variant == "B3_25D" and args.policy_intent_root is not None:
        raise SystemExit("B3_25D must not receive --policy-intent-root")
    if args.variant == "B3_25D9D" and args.policy_intent_root is None:
        raise SystemExit("B3_25D9D requires --policy-intent-root")
    input_paths = {name: Path(path) for name, path in args.input}
    if len(input_paths) != len(args.input) or set(input_paths) != set(AUTHORIZATION_INPUT_NAMES):
        raise SystemExit(f"authorization input set mismatch: expected {list(AUTHORIZATION_INPUT_NAMES)} got {sorted(input_paths)}")
    fold_id = "FULL_FIT" if args.fit_scope == "FULL_FIT" else int(args.fold_id)
    payload = build_training_authorization_from_paths(
        args.output_root, variant=args.variant, fit_scope=args.fit_scope, fold_id=fold_id, seed=args.seed,
        input_paths=input_paths, runner_repo=args.runner_repo, expected_runner_head=args.expected_runner_head,
        runner_config=args.runner_config, runner_script=args.runner_script,
        generator_script=Path(__file__).resolve(), policy_intent_root=args.policy_intent_root,
    )
    load_training_authorization_bundle(args.output_root)
    print(json.dumps({"status": "PASS_PREPARATION_ONLY", "variant": args.variant, "fold_id": args.fold_id, "seed": args.seed, "authorization_payload_sha256": payload["authorization_payload_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
