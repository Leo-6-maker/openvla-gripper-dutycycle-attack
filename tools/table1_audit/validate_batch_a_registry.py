from __future__ import annotations

import argparse
import csv
from pathlib import Path

from tools.table1_audit.common import add_path_arg, canonical_json, is_valid_sha256, write_json


EXPECTED_CONDITIONS = {
    "CLEAN",
    "RAND_LINF",
    "SHUFFLED_GRADIENT",
    "UMA_UNTARGETED_CE_PGD",
    "ADAPTED_TMA_OPEN",
    "PREFIX_LOG_RATIO_OPEN_TRUE_T10",
}
REQUIRED_COLUMNS = [
    "condition_id",
    "authorized",
    "launch_gate",
    "script_sha256",
    "config_sha256",
    "epsilon",
    "epsilon_space",
    "step_size",
    "optimization_steps",
    "K",
    "timing_policy",
    "initialization",
    "preprocessing_backend",
    "termination_policy",
    "victim_checkpoint_sha256",
    "detector_global_freeze_sha256",
    "state_selection_sha256",
    "protocol_sha256",
    "runner_sha256",
    "worker_sha256",
    "bridge_sha256",
    "metric_schema_sha256",
    "retry_policy_sha256",
    "condition_spec_sha256",
    "manifest_sha256",
    "arm_lock_mode",
    "output_root",
]
UNRESOLVED = {"UNVERIFIED", "SERVER_SNAPSHOT_REQUIRED", "MISSING", ""}
ARM_LOCK = {"NOT_APPLICABLE", "NO_ARM_LOCK", "PRESERVE_ARM_QPOS"}
TIMING = {"NOT_APPLICABLE", "Student trigger"}


def validate(path: Path) -> dict:
    problems: list[dict] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    missing_cols = sorted(set(REQUIRED_COLUMNS) - set(rows[0].keys() if rows else []))
    if missing_cols:
        problems.append({"class": "missing_registry_columns", "columns": missing_cols})
    ids = [r.get("condition_id", "") for r in rows]
    if set(ids) != EXPECTED_CONDITIONS:
        problems.append({"class": "condition_id_set_mismatch", "expected": sorted(EXPECTED_CONDITIONS), "actual": sorted(set(ids))})
    if len(ids) != len(set(ids)):
        problems.append({"class": "duplicate_condition_id"})
    outputs = {}
    for row in rows:
        cid = row.get("condition_id", "")
        auth = str(row.get("authorized", "")).lower() == "true"
        launchable = auth and row.get("launch_gate") != "HOLD"
        if row.get("arm_lock_mode") not in ARM_LOCK:
            problems.append({"class": "invalid_arm_lock", "condition_id": cid, "value": row.get("arm_lock_mode")})
        if row.get("timing_policy") not in TIMING:
            problems.append({"class": "invalid_timing_policy", "condition_id": cid, "value": row.get("timing_policy")})
        if cid == "CLEAN":
            for field in ["epsilon", "epsilon_space", "step_size", "optimization_steps", "K", "initialization", "preprocessing_backend", "termination_policy"]:
                if row.get(field) not in {"NOT_APPLICABLE", ""}:
                    problems.append({"class": "clean_attack_field_not_applicable", "condition_id": cid, "field": field})
        if launchable:
            for field in REQUIRED_COLUMNS:
                value = row.get(field, "")
                if value in UNRESOLVED:
                    problems.append({"class": "authorized_unresolved_field", "condition_id": cid, "field": field})
            if "targeted CE or TMA-like" in row.get("objective_semantics", ""):
                problems.append({"class": "launchable_ambiguous_objective", "condition_id": cid})
            for field in [c for c in REQUIRED_COLUMNS if c.endswith("_sha256")]:
                if not is_valid_sha256(row.get(field, "")):
                    problems.append({"class": "authorized_invalid_sha256", "condition_id": cid, "field": field})
        out = row.get("output_root", "")
        if out and out != "SERVER_SNAPSHOT_REQUIRED":
            for seen, other in outputs.items():
                if out == seen or out.startswith(seen.rstrip("/") + "/") or seen.startswith(out.rstrip("/") + "/"):
                    problems.append({"class": "output_root_overlap", "condition_id": cid, "other": other, "output_root": out})
            outputs[out] = cid
        if not auth and row.get("launch_gate") != "HOLD":
            problems.append({"class": "unauthorized_row_not_hold", "condition_id": cid})
    return {"schema_version": "batch_a_registry_validation.v1", "registry": str(path), "row_count": len(rows), "validation_pass": not problems, "problems": problems}


def main() -> int:
    ap = argparse.ArgumentParser(description="Validate the minimal Batch-A condition registry.")
    add_path_arg(ap, "--registry", required=True)
    add_path_arg(ap, "--output-json")
    args = ap.parse_args()
    result = validate(args.registry)
    if args.output_json:
        write_json(args.output_json, result)
    print(canonical_json(result), end="")
    return 0 if result["validation_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
