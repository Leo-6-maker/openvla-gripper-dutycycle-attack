from __future__ import annotations

import argparse
import csv
from pathlib import Path

from tools.table1_audit.common import add_path_arg, canonical_json, is_valid_sha256, write_json


REQUIRED_COLUMNS = [
    "condition_id",
    "authorized",
    "launch_gate",
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


def validate(path: Path) -> dict:
    problems: list[dict] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    missing_cols = sorted(set(REQUIRED_COLUMNS) - set(rows[0].keys() if rows else []))
    if missing_cols:
        problems.append({"class": "missing_registry_columns", "columns": missing_cols})
    outputs = {}
    for row in rows:
        cid = row.get("condition_id", "")
        auth = str(row.get("authorized", "")).lower() == "true"
        launchable = auth and row.get("launch_gate") != "HOLD"
        if row.get("arm_lock_mode") == "optional":
            problems.append({"class": "optional_arm_lock", "condition_id": cid})
        if cid == "CLEAN":
            for field in ["epsilon", "step_size", "optimization_steps", "K"]:
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
            if out in outputs:
                problems.append({"class": "duplicate_output_root", "condition_id": cid, "other": outputs[out], "output_root": out})
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
