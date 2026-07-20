#!/usr/bin/env python3
"""Build a sealed, preparation-only V2.1C four-fold OOF training plan.

This entry point has no training flag, does not import torch, and cannot write a
checkpoint.  A future geometry audit may be bound, but even a passing geometry
gate only changes the plan status to PASS_PREPARATION_ONLY.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import re
import shutil
import subprocess
import uuid
from pathlib import Path
from typing import Any

from gripper_attack.b3_training_protocol import seal_directory, sha256_file, verify_sealed_directory
from gripper_attack.v21c_training_prep import (
    PREP_PROTOCOL_SCHEMA,
    build_oof_preparation_plan,
    validate_geometry_gate,
    validate_v21c_teacher_root,
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _atomic_text(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _git_head(repo_root: Path) -> str:
    try:
        value = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(repo_root), text=True).strip()
    except Exception:
        raise RuntimeError(f"git rev-parse HEAD failed in {repo_root}") from None
    if re.fullmatch(r"[0-9a-f]{40}", value) is None:
        raise RuntimeError(f"invalid git HEAD: {value!r}")
    return value


def run(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output_root.resolve()
    if output.exists():
        raise FileExistsError(output)
    protocol_path = args.protocol.resolve()
    protocol = _read_json(protocol_path)
    if protocol.get("schema") != PREP_PROTOCOL_SCHEMA:
        raise ValueError("wrong V2.1C OOF preparation protocol schema")
    execution = protocol.get("execution")
    if not isinstance(execution, dict) or any(
        execution.get(name) is not False
        for name in (
            "training_executed",
            "checkpoint_write_authorized",
            "full_fit_authorized",
            "formal_training_authorized",
            "formal_attack_authorized",
            "fsm_change_authorized",
        )
    ):
        raise ValueError("preparation protocol must prohibit training/checkpoint/attack/FSM execution")

    teacher = validate_v21c_teacher_root(
        args.teacher_root.resolve(),
        expected_source_commit=args.expected_teacher_source_commit,
    )
    geometry = None
    if args.geometry_audit_root is not None:
        geometry = validate_geometry_gate(args.geometry_audit_root.resolve(), teacher=teacher)

    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    plan = build_oof_preparation_plan(
        fold_root=args.fold_root.resolve(),
        teacher=teacher,
        prep_protocol_sha256=sha256_file(protocol_path),
        seeds=seeds,
        geometry=geometry,
    )

    prep_module = importlib.import_module("gripper_attack.v21c_training_prep")
    module_path = Path(prep_module.__file__).resolve()
    script_path = Path(__file__).resolve()
    repo_root = module_path.parent.parent.parent
    source_commit = _git_head(repo_root)
    if args.expected_prep_source_commit is not None and source_commit != args.expected_prep_source_commit:
        raise RuntimeError(
            f"preparation source mismatch: expected {args.expected_prep_source_commit}, HEAD is {source_commit}"
        )

    source_binding = {
        "schema": "DETECTOR_V5_V21C_OOF_TRAINING_PREP_SOURCE_BINDING_V1",
        "source_git_commit": source_commit,
        "prep_module_path": str(module_path),
        "prep_module_sha256": sha256_file(module_path),
        "plan_builder_path": str(script_path),
        "plan_builder_sha256": sha256_file(script_path),
        "protocol_path": str(protocol_path),
        "protocol_sha256": sha256_file(protocol_path),
        "training_code_executed": False,
        "checkpoint_written": False,
    }

    staging = output.with_name(f".{output.name}.{uuid.uuid4().hex}.staging")
    try:
        staging.mkdir(parents=True)
        _atomic_text(staging / "training_plan.json", json.dumps(plan, indent=2, sort_keys=True) + "\n")
        _atomic_text(staging / "source_binding.json", json.dumps(source_binding, indent=2, sort_keys=True) + "\n")
        _atomic_text(staging / "protocol.json", json.dumps(protocol, indent=2, sort_keys=True) + "\n")
        summary = {
            "schema": "DETECTOR_V5_V21C_OOF_TRAINING_PREP_SUMMARY_V1",
            "status": plan["status"],
            "job_count": plan["job_count"],
            "fold_count": plan["fold_count"],
            "seed_count": plan["seed_count"],
            "teacher_root_sha256s_sha256": teacher.root_sha256s_sha256,
            "geometry_gate_bound": geometry is not None,
            "training_executed": False,
            "checkpoint_write_authorized": False,
            "formal_training_authorized": False,
            "formal_attack_authorized": False,
        }
        _atomic_text(staging / "summary.json", json.dumps(summary, indent=2, sort_keys=True) + "\n")
        seal_directory(staging)
        verify_sealed_directory(staging)
        os.replace(staging, output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return {
        "status": plan["status"],
        "output_root": str(output),
        "job_count": plan["job_count"],
        "training_executed": False,
        "checkpoint_written": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-root", type=Path, required=True)
    parser.add_argument("--fold-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seeds", default="20260720")
    parser.add_argument("--geometry-audit-root", type=Path)
    parser.add_argument("--expected-teacher-source-commit", required=True)
    parser.add_argument("--expected-prep-source-commit")
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
