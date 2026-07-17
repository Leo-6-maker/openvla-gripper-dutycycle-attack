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

from gripper_attack.b3_training_protocol import build_training_authorization, load_training_authorization_bundle, sha256_file


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--variant", choices=("B3_25D", "B3_25D9D"), required=True)
    parser.add_argument("--fold-id", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--runner-binding-json", type=Path, required=True)
    parser.add_argument("--generator-script", type=Path, required=True)
    parser.add_argument("--input", action="append", nargs=2, metavar=("NAME", "PATH"), required=True)
    parser.add_argument("--execute-preparation", action="store_true")
    args = parser.parse_args()
    if not args.execute_preparation:
        raise SystemExit("FORMAL_TRAINING_HOLD: authorization generation is not authorized without --execute-preparation")
    snapshots = {name: sha256_file(Path(path)) for name, path in args.input}
    required = {
        "formal_fit_registry_sha256", "formal_registry_summary_sha256", "formal_registry_root_sha256",
        "s1_corpus_sha256", "s1_root_audit_sha256", "teacher_aggregate_sha256",
        "training_protocol_sha256", "normalization_bundle_sha256", "normalization_sha256", "fold_manifest_sha256",
    }
    if set(snapshots) != required:
        raise SystemExit(f"authorization input set mismatch: expected {sorted(required)} got {sorted(snapshots)}")
    binding = json.loads(args.runner_binding_json.read_text(encoding="utf-8"))
    payload = build_training_authorization(
        args.output_root, variant=args.variant, fold_id=args.fold_id, seed=args.seed,
        input_snapshots=snapshots, runner_binding=binding, generator_script_sha256=sha256_file(args.generator_script),
    )
    load_training_authorization_bundle(args.output_root)
    print(json.dumps({"status": "PASS_PREPARATION_ONLY", "variant": args.variant, "fold_id": args.fold_id, "seed": args.seed, "authorization_payload_sha256": payload["authorization_payload_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
