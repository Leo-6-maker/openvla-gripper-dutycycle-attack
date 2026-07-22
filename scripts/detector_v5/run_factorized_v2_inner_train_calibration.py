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
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from gripper_attack.b3_training_protocol import seal_directory, sha256_file, verify_sealed_directory  # noqa: E402
from gripper_attack.factorized_calibration import (  # noqa: E402
    CalibrationPlanError,
    AUTHORIZATION_SCHEMA_V2,
    EXECUTION_RECEIPT_SCHEMA,
    PLAN_SCHEMA_V2,
    validate_authorization_template,
    validate_execution_authorization_template_v2,
    validate_execution_authorization_v2,
    validate_inner_train_plan,
    validate_structured_inner_plan,
)


def prepare(args: argparse.Namespace) -> dict:
    if not args.prepare_only:
        raise CalibrationPlanError("OFFLINE_INFERENCE_AUTHORIZATION_REQUIRED")
    output = args.output_root.resolve()
    if output.exists():
        raise FileExistsError(f"OUTPUT_EXISTS:{output}")
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    authorization = json.loads(args.authorization.read_text(encoding="utf-8"))
    if plan.get("schema") == PLAN_SCHEMA_V2:
        validate_structured_inner_plan(plan)
        validate_execution_authorization_template_v2(authorization)
        summary = {"schema": PLAN_SCHEMA_V2, "split_count": len(plan["jobs"]), "split_names": [job["split"] for job in plan["jobs"]], "validation_or_cal_read": False, "formal_selection_eligible": False, "training_authorized": False, "attack_authorized": False}
    else:
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
    if plan.get("schema") != PLAN_SCHEMA_V2 or authorization.get("schema") != AUTHORIZATION_SCHEMA_V2:
        raise CalibrationPlanError("STRUCTURED_EXECUTION_REQUIRED")
    summary = validate_structured_inner_plan(plan, execute=True)
    output = args.output_root.resolve()
    if output.exists():
        raise FileExistsError(f"OUTPUT_EXISTS:{output}")
    validate_execution_authorization_v2(authorization, plan, output_root=output)
    staging = output.with_name(f".{output.name}.{uuid.uuid4().hex}.staging")
    try:
        staging.mkdir(parents=True)
        results: list[dict[str, object]] = []
        for job in sorted(plan["jobs"], key=lambda item: item["split"]):
            job_output = Path(job["output_root"]).resolve()
            if job_output.exists():
                raise FileExistsError(f"OUTPUT_EXISTS:{job_output}")
            command = [
                sys.executable, "-m", job["predictor_module"],
                "--checkpoint", job["checkpoint_path"],
                "--feature-root", job["feature_root"],
                "--identity-manifest", job["identity_manifest_path"],
                "--split", job["split"],
                "--output-root", str(job_output),
            ]
            completed = subprocess.run(command, check=True, capture_output=True, text=True)
            _audit_inference_output(job_output, job)
            results.append({"split": job["split"], "command": command, "returncode": completed.returncode, "stdout_sha256": _bytes_sha(completed.stdout.encode()), "stderr_sha256": _bytes_sha(completed.stderr.encode())})
        (staging / "execution_manifest.json").write_text(json.dumps({
            **summary,
            "status": "EXECUTED_INNER_TRAIN_INFERENCE",
            "execution_authorized": True,
            "training": False,
            "full_fit": False,
            "attack": False,
            "results": results,
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (staging / "FACTORIZED_V2_INFERENCE_EXECUTION_RECEIPT_V1.json").write_text(json.dumps({
            "schema": EXECUTION_RECEIPT_SCHEMA,
            "status": "PASS_12_OF_12",
            "split_names": [job["split"] for job in sorted(plan["jobs"], key=lambda item: item["split"])],
            "output_roots": [str(Path(job["output_root"]).resolve()) for job in sorted(plan["jobs"], key=lambda item: item["split"])],
            "execution_authorized": True,
            "formal_selection_eligible": False,
            "training": False,
            "full_fit": False,
            "cal_check": False,
            "attack": False,
            "artifact_audit": "PASS",
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        seal_directory(staging)
        os.replace(staging, output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return {"status": "EXECUTED_INNER_TRAIN_INFERENCE", "output_root": str(output), **summary}


def _bytes_sha(value: bytes) -> str:
    import hashlib
    return hashlib.sha256(value).hexdigest()


def _audit_inference_output(root: Path, job: dict) -> None:
    verify_sealed_directory(root)
    manifest_path = root / "manifest.json"
    stream_path = root / "prediction_records.jsonl"
    if not manifest_path.is_file() or not stream_path.is_file():
        raise CalibrationPlanError("INFERENCE_OUTPUT_SCHEMA_MISSING")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    required = {"schema", "split", "checkpoint_sha256", "record_count", "formal_selection_eligible", "training_authorized", "attack_enabled"}
    if not required <= set(manifest) or manifest["split"] != job["split"] or manifest["checkpoint_sha256"] != job["checkpoint_sha256"]:
        raise CalibrationPlanError("INFERENCE_OUTPUT_BINDING_MISMATCH")
    if any(manifest.get(flag) is not False for flag in ("formal_selection_eligible", "training_authorized", "attack_enabled")):
        raise CalibrationPlanError("INFERENCE_OUTPUT_AUTHORIZATION")
    forbidden = {"event_id", "teacher_phase", "known_mask", "strict_k10_feasible", "utility_probability", "regrasp_probability", "attack_outcome"}
    rows = [json.loads(line) for line in stream_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != manifest["record_count"] or not rows:
        raise CalibrationPlanError("INFERENCE_OUTPUT_RECORD_COUNT")
    for row in rows:
        if forbidden & set(row):
            raise CalibrationPlanError("INFERENCE_OUTPUT_TEACHER_FIELD")
        for head in ("grasp", "manipulation", "release"):
            if f"{head}_logit" not in row or f"{head}_probability" not in row:
                raise CalibrationPlanError("INFERENCE_OUTPUT_HEAD_MISSING")
            z = float(row[f"{head}_logit"])
            expected = 1.0 / (1.0 + math.exp(-z))
            if abs(expected - float(row[f"{head}_probability"])) > 1e-7:
                raise CalibrationPlanError("INFERENCE_OUTPUT_PROBABILITY_MISMATCH")


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
