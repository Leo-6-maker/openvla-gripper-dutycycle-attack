#!/usr/bin/env python3
"""Orchestrate the pre-registered 24-run viability matrix.

The default mode only seals a plan.  Actual execution requires the explicit
flag and one machine-built authorization/normalization bundle per run.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from gripper_attack.b3_training_protocol import VIABILITY_SEEDS, load_fit_fold_bundle, sha256_file


def build_matrix_plan(fold_root: Path) -> dict:
    fold = load_fit_fold_bundle(fold_root)
    runs = [
        {"run_id": f"fold{fold_id}_{variant}_seed{seed}", "fold_id": fold_id, "variant": variant, "seed": seed, "train_count": 600, "validation_count": 200}
        for fold_id in range(4) for variant in ("B3_25D", "B3_25D9D") for seed in VIABILITY_SEEDS
    ]
    return {"schema": "B3_OFFICIAL_V3_FIT_VIABILITY_MATRIX_V1", "fold_bundle_sha256": sha256_file(fold_root / "SHA256SUMS"), "fold_registry_sha256": fold["registry_sha256"], "run_count": 24, "runs": runs, "status": "PREPARATION_ONLY", "formal_training_authorized": False, "formal_attack_authorized": False}


def execute_matrix(*, plan: dict, command_template: list[str], output_root: Path) -> list[dict]:
    required_flags = {"--execute-formal", "--authorization", "--fold-id", "--seed", "--variant", "--output-checkpoint-bundle"}
    if not required_flags.issubset(set(command_template)):
        raise ValueError(f"matrix command is missing formal-run flags: {sorted(required_flags - set(command_template))}")
    if output_root.exists():
        raise FileExistsError(output_root)
    output_root.mkdir(parents=True)
    results = []
    for run in plan["runs"]:
        command = [item.format(**run) for item in command_template]
        completed = subprocess.run(command, check=False, text=True, capture_output=True)
        result = dict(run, returncode=completed.returncode, stdout=completed.stdout[-4000:], stderr=completed.stderr[-4000:])
        results.append(result)
        if completed.returncode != 0:
            raise RuntimeError(f"viability matrix run failed: {run['run_id']}")
    (output_root / "matrix_results.json").write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold-root", type=Path, required=True)
    parser.add_argument("--output-plan", type=Path, required=True)
    parser.add_argument("--execute-formal", action="store_true")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--command-template", nargs="+")
    args = parser.parse_args()
    plan = build_matrix_plan(args.fold_root)
    if args.output_plan.exists():
        raise SystemExit(f"refusing to overwrite: {args.output_plan}")
    args.output_plan.parent.mkdir(parents=True, exist_ok=True)
    args.output_plan.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not args.execute_formal:
        print(json.dumps({"status": "PASS_PREPARATION_ONLY", "run_count": 24, "formal_training_authorized": False}, sort_keys=True))
        return 0
    if args.output_root is None or not args.command_template:
        raise SystemExit("FORMAL_TRAINING_HOLD: --output-root and --command-template are required for execution")
    execute_matrix(plan=plan, command_template=args.command_template, output_root=args.output_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
