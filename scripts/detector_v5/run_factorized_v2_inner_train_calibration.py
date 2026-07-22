#!/usr/bin/env python3
"""Preparation and explicitly authorized inner-train Factorized V2 inference.

``--prepare-only`` writes only a sealed plan.  Execution requires both an
authorization bundle with ``execution_authorized=true`` and the explicit
environment value ``OFFLINE_FACTORIZED_INNER_TRAIN_INFERENCE=GO``.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from gripper_attack.b3_training_protocol import seal_directory  # noqa: E402
from gripper_attack.factorized_calibration import (  # noqa: E402
    CalibrationPlanError,
    validate_authorization_template,
    validate_inner_train_plan,
)


def prepare(args: argparse.Namespace) -> dict:
    if not args.prepare_only:
        raise CalibrationPlanError("OFFLINE_INFERENCE_AUTHORIZATION_REQUIRED")
    output = args.output_root.resolve()
    if output.exists():
        raise FileExistsError(f"OUTPUT_EXISTS:{output}")
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    authorization = json.loads(args.authorization.read_text(encoding="utf-8"))
    validate_authorization_template(authorization)
    checkpoints = plan.get("checkpoints")
    if not isinstance(checkpoints, list):
        raise CalibrationPlanError("PLAN_CHECKPOINTS_MISSING")
    summary = validate_inner_train_plan(
        checkpoints,
        forbidden_roots=plan.get("forbidden_roots", authorization.get("forbidden_roots", [])),
    )
    staging = output.with_name(f".{output.name}.{uuid.uuid4().hex}.staging")
    try:
        staging.mkdir(parents=True)
        (staging / "preparation_manifest.json").write_text(json.dumps({
            **summary,
            "status": "PREPARATION_ONLY",
            "authorization_schema": authorization["schema"],
            "production_inference_executed": False,
            "prediction_artifacts_written": False,
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (staging / "plan_snapshot.json").write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (staging / "authorization_snapshot.json").write_text(json.dumps(authorization, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        seal_directory(staging)
        os.replace(staging, output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return {"status": "PREPARATION_ONLY", "output_root": str(output), **summary}


def execute(args: argparse.Namespace) -> dict:
    if os.environ.get("OFFLINE_FACTORIZED_INNER_TRAIN_INFERENCE") != "GO":
        raise CalibrationPlanError("OFFLINE_INFERENCE_GO_REQUIRED")
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    authorization = json.loads(args.authorization.read_text(encoding="utf-8"))
    validate_authorization_template(authorization, allow_execution=True)
    if authorization.get("execution_authorized") is not True:
        raise CalibrationPlanError("EXECUTION_AUTHORIZATION_FALSE")
    checkpoints = plan.get("checkpoints")
    summary = validate_inner_train_plan(checkpoints, forbidden_roots=plan.get("forbidden_roots", authorization.get("forbidden_roots", [])))
    commands = plan.get("commands")
    if not isinstance(commands, list) or len(commands) != 12:
        raise CalibrationPlanError("EXECUTION_COMMAND_COUNT_MUST_BE_12")
    forbidden_tokens = ("attack", "cal", "check", "train", "full-fit", "rollout")
    for command in commands:
        if not isinstance(command, list) or not command or any(token in str(part).lower() for part in command for token in forbidden_tokens):
            raise CalibrationPlanError("EXECUTION_COMMAND_FORBIDDEN_OR_INVALID")
    output = args.output_root.resolve()
    if output.exists():
        raise FileExistsError(f"OUTPUT_EXISTS:{output}")
    staging = output.with_name(f".{output.name}.{uuid.uuid4().hex}.staging")
    try:
        staging.mkdir(parents=True)
        results = []
        for index, command in enumerate(commands):
            completed = subprocess.run(command, check=True, capture_output=True, text=True)
            results.append({"index": index, "command": command, "returncode": completed.returncode})
        (staging / "execution_manifest.json").write_text(json.dumps({
            **summary,
            "status": "EXECUTED_INNER_TRAIN_INFERENCE",
            "execution_authorized": True,
            "training": False,
            "full_fit": False,
            "attack": False,
            "results": results,
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        seal_directory(staging)
        os.replace(staging, output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return {"status": "EXECUTED_INNER_TRAIN_INFERENCE", "output_root": str(output), **summary}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--execute", action="store_true")
    try:
        args = parser.parse_args()
        if args.prepare_only == args.execute:
            raise CalibrationPlanError("SELECT_EXACTLY_ONE_PREPARE_OR_EXECUTE")
        print(json.dumps(prepare(args) if args.prepare_only else execute(args), sort_keys=True))
    except Exception as exc:
        print(f"HOLD:{type(exc).__name__}:{exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
