#!/usr/bin/env python3
"""Orchestrate the pre-registered 24-run viability matrix.

The default mode only seals a plan.  Actual execution requires the explicit
flag and one machine-built authorization/normalization bundle per run.
"""

from __future__ import annotations

import argparse
import json
import shutil
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


def execute_matrix(*, plan: dict, command_template: list[str], output_root: Path, authorization_template: str, normalization_template: str, registry_csv: Path, s1_root: Path, fold_root: Path, aggregate_output_root: Path, policy_intent_root: Path | None = None) -> list[dict]:
    required_flags = {"--execute-formal", "--authorization", "--fold-id", "--seed", "--variant", "--output-checkpoint-bundle"}
    if not required_flags.issubset(set(command_template)):
        raise ValueError(f"matrix command is missing formal-run flags: {sorted(required_flags - set(command_template))}")
    template_text = " ".join(command_template)
    required_placeholders = ("{run_id}", "{fold_id}", "{variant}", "{seed}", "{authorization}", "{normalization}", "{output_checkpoint_bundle}")
    if any(token not in template_text for token in required_placeholders):
        raise ValueError(f"matrix command must bind every run coordinate and sealed input: {required_placeholders}")
    if output_root.exists():
        raise FileExistsError(output_root)
    staging = output_root.with_name(f".{output_root.name}.staging")
    if staging.exists():
        raise FileExistsError(staging)
    staging.mkdir(parents=True)
    results = []
    try:
        for run in plan["runs"]:
            run_root = staging / run["run_id"]
            run = dict(run, run_root=str(run_root), authorization=authorization_template.format(**run), normalization=normalization_template.format(**run), output_checkpoint_bundle=str(run_root / "checkpoint"))
            run_root.mkdir(parents=True)
            command = [item.format(**run) for item in command_template]
            completed = subprocess.run(command, check=False, text=True, capture_output=True)
            result = dict(run, returncode=completed.returncode, stdout=completed.stdout[-4000:], stderr=completed.stderr[-4000:])
            results.append(result)
            if completed.returncode != 0:
                raise RuntimeError(f"viability matrix run failed: {run['run_id']}")
            validation_command = [
                sys.executable, str(Path(__file__).with_name("run_b3_v3_fold_validation.py")),
                "--checkpoint-root", str(run_root / "checkpoint"), "--registry-csv", str(registry_csv),
                "--s1-root", str(s1_root), "--fold-root", str(fold_root),
                "--output-root", str(run_root / "prediction"), "--fold-id", str(run["fold_id"]),
                "--seed", str(run["seed"]), "--variant", str(run["variant"]),
            ]
            if policy_intent_root is not None:
                validation_command += ["--policy-intent-root", str(policy_intent_root)]
            validated = subprocess.run(validation_command, check=False, text=True, capture_output=True)
            result["validation_returncode"] = validated.returncode
            result["validation_stdout"] = validated.stdout[-4000:]
            result["validation_stderr"] = validated.stderr[-4000:]
            if validated.returncode != 0:
                raise RuntimeError(f"held-out validation failed: {run['run_id']}")
        aggregate_command = [sys.executable, str(Path(__file__).with_name("aggregate_b3_v3_fit_viability.py")), "--output-root", str(staging / "viability_aggregate")]
        for run in plan["runs"]:
            aggregate_command += ["--run-root", str(staging / run["run_id"] / "prediction")]
        aggregated = subprocess.run(aggregate_command, check=False, text=True, capture_output=True)
        if aggregated.returncode != 0:
            raise RuntimeError(f"viability aggregate failed: {aggregated.stderr[-4000:]}")
        result_path = staging / "matrix_results.json"
        result_path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (result_path.with_name(result_path.name + ".sha256")).write_text(f"{sha256_file(result_path)}  {result_path.name}\n", encoding="utf-8")
        staging.rename(output_root)
        return results
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold-root", type=Path, required=True)
    parser.add_argument("--output-plan", type=Path, required=True)
    parser.add_argument("--execute-formal", action="store_true")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--command-template", nargs="+")
    parser.add_argument("--authorization-template")
    parser.add_argument("--normalization-template")
    parser.add_argument("--registry-csv", type=Path)
    parser.add_argument("--s1-root", type=Path)
    parser.add_argument("--policy-intent-root", type=Path)
    args = parser.parse_args()
    plan = build_matrix_plan(args.fold_root)
    if args.output_plan.exists():
        raise SystemExit(f"refusing to overwrite: {args.output_plan}")
    args.output_plan.parent.mkdir(parents=True, exist_ok=True)
    args.output_plan.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not args.execute_formal:
        print(json.dumps({"status": "PASS_PREPARATION_ONLY", "run_count": 24, "formal_training_authorized": False}, sort_keys=True))
        return 0
    if args.output_root is None or not args.command_template or not args.authorization_template or not args.normalization_template or args.registry_csv is None or args.s1_root is None:
        raise SystemExit("FORMAL_TRAINING_HOLD: output, command, auth/norm templates, registry, and S1 root are required")
    execute_matrix(plan=plan, command_template=args.command_template, output_root=args.output_root, authorization_template=args.authorization_template, normalization_template=args.normalization_template, registry_csv=args.registry_csv, s1_root=args.s1_root, fold_root=args.fold_root, aggregate_output_root=args.output_root / "viability_aggregate", policy_intent_root=args.policy_intent_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
