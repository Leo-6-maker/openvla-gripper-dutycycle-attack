#!/usr/bin/env python3
"""Read-only production-root audit; never loads a model or runs inference."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import uuid
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from gripper_attack.b3_training_protocol import seal_directory, sha256_file, verify_sealed_directory  # noqa: E402

EXPECTED_SPLITS = tuple(f"o{outer}_i{inner}" for outer in range(4) for inner in range(3))


def _audit(plan: Path | None) -> dict[str, Any]:
    if plan is None or not plan.is_file():
        return {
            "schema": "FACTORIZED_V3_1_RUNTIME_PRODUCTION_AUDIT_RECEIPT_V1",
            "status": "BLOCKED_ROOTS_NOT_MOUNTED",
            "reason": "12 sealed prediction/student/runtime roots are not mounted in this worktree",
            "split_names": list(EXPECTED_SPLITS),
            "production_inference": False,
            "model_loaded": False,
            "protected_split_read": False,
            "cal_check_read": False,
            "attack": False,
        }
    try:
        value = json.loads(plan.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"schema": "FACTORIZED_V3_1_RUNTIME_PRODUCTION_AUDIT_RECEIPT_V1", "status": "BLOCKED_MANIFEST_INCOMPLETE", "reason": f"plan unreadable: {exc}", "production_inference": False, "model_loaded": False}
    jobs = value.get("jobs", value.get("splits")) if isinstance(value, dict) else None
    names = sorted(str(job.get("split")) for job in jobs) if isinstance(jobs, list) and all(isinstance(job, dict) for job in jobs) else []
    if names != list(EXPECTED_SPLITS):
        return {"schema": "FACTORIZED_V3_1_RUNTIME_PRODUCTION_AUDIT_RECEIPT_V1", "status": "BLOCKED_MANIFEST_INCOMPLETE", "reason": "exact 12 split closure is missing", "observed_splits": names, "production_inference": False, "model_loaded": False}
    missing = []
    for job in jobs:
        for field in ("prediction_root", "student_root", "runtime_root", "checkpoint"):
            if not Path(str(job.get(field, ""))).exists():
                missing.append(f"{job['split']}:{field}")
    status = "PASS_READ_ONLY_AUDIT" if not missing else "BLOCKED_ROOTS_NOT_MOUNTED"
    return {
        "schema": "FACTORIZED_V3_1_RUNTIME_PRODUCTION_AUDIT_RECEIPT_V1",
        "status": status,
        "reason": "all declared roots are present for further read-only checks" if not missing else "declared production roots are not fully mounted",
        "split_names": list(EXPECTED_SPLITS),
        "missing": missing,
        "production_inference": False,
        "model_loaded": False,
        "prediction_runtime_join_checked": False,
        "candidate_close_checked": False,
        "protected_split_read": False,
        "cal_check_read": False,
        "attack": False,
    }


def materialize(plan: Path | None, output_root: Path) -> dict[str, Any]:
    output_root = output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"OUTPUT_EXISTS:{output_root}")
    staging = output_root.with_name(f".{output_root.name}.{uuid.uuid4().hex}.staging")
    try:
        staging.mkdir(parents=True)
        summary = _audit(plan)
        (staging / "audit_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (staging / "source_plan.json").write_text(json.dumps({"plan": str(plan) if plan else None, "plan_present": bool(plan and plan.is_file())}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (staging / "MANIFEST.json").write_text(json.dumps({"schema": summary["schema"], "status": summary["status"], "model_loaded": False, "production_inference": False, "formal_selection_eligible": False, "training_authorized": False, "attack_authorized": False}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        seal_directory(staging)
        verify_sealed_directory(staging)
        os.replace(staging, output_root)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return {"status": summary["status"], "output_root": str(output_root), "sha256s_sha256": sha256_file(output_root / "SHA256SUMS")}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    result = materialize(args.plan, args.output_root)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] == "PASS_READ_ONLY_AUDIT" else 2


if __name__ == "__main__":
    raise SystemExit(main())
