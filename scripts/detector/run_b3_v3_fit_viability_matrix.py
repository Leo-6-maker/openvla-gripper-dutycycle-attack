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


def execute_matrix(
    *, plan: dict, command_template: list[str], output_root: Path, authorization_template: str,
    normalization_template: str, registry_csv: Path, registry_summary: Path, s1_root: Path,
    s1_root_audit: Path, source_contract: Path, s1_protocol: Path, training_protocol: Path,
    feature_rebuilder: Path, runner_repo: Path, runner_config: Path, runner_script: Path,
    fold_root: Path, aggregate_output_root: Path, policy_intent_root: Path | None = None,
) -> list[dict]:
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
            run = dict(
                run,
                run_root=str(run_root),
                authorization=authorization_template.format(**run),
                normalization=normalization_template.format(**run),
                output_checkpoint_bundle=str(run_root / "checkpoint"),
                policy_intent_root=(str(policy_intent_root) if policy_intent_root is not None else ""),
            )
            run_root.mkdir(parents=True)
            command = [item.format(**run) for item in command_template]
            has_policy_flag = "--policy-intent-root" in command
            if run["variant"] == "B3_25D":
                if has_policy_flag:
                    raise ValueError("B3_25D matrix command must not receive a 9D policy-intent root")
            else:
                if policy_intent_root is None:
                    raise ValueError("B3_25D9D matrix execution requires a sealed 9D root")
                if not has_policy_flag:
                    raise ValueError("B3_25D9D matrix command must bind --policy-intent-root")
                run["policy_intent_root"] = str(policy_intent_root)
                command = [item.format(**run) for item in command_template]
            completed = subprocess.run(command, check=False, text=True, capture_output=True)
            result = dict(run, returncode=completed.returncode, stdout=completed.stdout[-4000:], stderr=completed.stderr[-4000:])
            results.append(result)
            if completed.returncode != 0:
                raise RuntimeError(f"viability matrix run failed: {run['run_id']}")
            validation_command = [
                sys.executable, str(Path(__file__).with_name("run_b3_v3_fold_validation.py")),
                "--checkpoint-root", str(run_root / "checkpoint"), "--authorization", str(run["authorization"]),
                "--registry-csv", str(registry_csv), "--registry-summary", str(registry_summary),
                "--s1-root", str(s1_root), "--s1-root-audit", str(s1_root_audit),
                "--source-contract", str(source_contract), "--s1-protocol", str(s1_protocol),
                "--training-protocol", str(training_protocol), "--feature-rebuilder", str(feature_rebuilder),
                "--normalization-root", str(run["normalization"]), "--fold-root", str(fold_root),
                "--output-root", str(run_root / "prediction"), "--fold-id", str(run["fold_id"]),
                "--seed", str(run["seed"]), "--variant", str(run["variant"]),
                "--runner-repo", str(runner_repo), "--runner-config", str(runner_config),
                "--runner-script", str(runner_script),
            ]
            if policy_intent_root is not None and run["variant"] == "B3_25D9D":
                validation_command += ["--policy-intent-root", str(policy_intent_root)]
            validated = subprocess.run(validation_command, check=False, text=True, capture_output=True)
            result["validation_returncode"] = validated.returncode
            result["validation_stdout"] = validated.stdout[-4000:]
            result["validation_stderr"] = validated.stderr[-4000:]
            if validated.returncode != 0:
                raise RuntimeError(f"held-out validation failed: {run['run_id']}")
        aggregate_command = [
            sys.executable, str(Path(__file__).with_name("aggregate_b3_v3_fit_viability.py")),
            "--output-root", str(staging / "viability_aggregate"), "--fold-root", str(fold_root),
        ]
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
    parser.add_argument("--registry-summary", type=Path)
    parser.add_argument("--s1-root", type=Path)
    parser.add_argument("--s1-root-audit", type=Path)
    parser.add_argument("--source-contract", type=Path)
    parser.add_argument("--s1-protocol", type=Path)
    parser.add_argument("--training-protocol", type=Path)
    parser.add_argument("--feature-rebuilder", type=Path)
    parser.add_argument("--runner-repo", type=Path)
    parser.add_argument("--runner-config", type=Path)
    parser.add_argument("--runner-script", type=Path)
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
    required = (args.output_root, args.command_template, args.authorization_template, args.normalization_template, args.registry_csv, args.registry_summary, args.s1_root, args.s1_root_audit, args.source_contract, args.s1_protocol, args.training_protocol, args.feature_rebuilder, args.runner_repo, args.runner_config, args.runner_script)
    if any(value is None for value in required):
        raise SystemExit("FORMAL_TRAINING_HOLD: output, command, auth/norm templates, registry, and S1 root are required")
    execute_matrix(
        plan=plan, command_template=args.command_template, output_root=args.output_root,
        authorization_template=args.authorization_template, normalization_template=args.normalization_template,
        registry_csv=args.registry_csv, registry_summary=args.registry_summary, s1_root=args.s1_root,
        s1_root_audit=args.s1_root_audit, source_contract=args.source_contract, s1_protocol=args.s1_protocol,
        training_protocol=args.training_protocol, feature_rebuilder=args.feature_rebuilder,
        runner_repo=args.runner_repo, runner_config=args.runner_config, runner_script=args.runner_script,
        fold_root=args.fold_root, aggregate_output_root=args.output_root / "viability_aggregate",
        policy_intent_root=args.policy_intent_root,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
