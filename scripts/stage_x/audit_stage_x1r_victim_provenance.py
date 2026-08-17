"""Independent offline audit for the provenance/parity-only boundary."""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON_OBJECT_REQUIRED:{path}")
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--workers", type=Path, nargs=4, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    contract = load(args.contract)
    errors: list[str] = []
    if contract.get("schema") != "STAGE_X_X1R_SUITE_MATCHED_VICTIM_CONTRACT_V1":
        errors.append("contract_schema")
    if contract.get("scientific_authority") != "X1R_NOT_AUTHORIZED":
        errors.append("x1r_must_remain_unauthorized")
    if contract.get("historical_boundary", {}).get("stage_v_launch_time_weight_identity") != "NOT_IDENTIFIABLE":
        errors.append("stage_v_historical_identity_not_downgraded")
    if contract.get("historical_boundary", {}).get("stage_vi_b2_launch_time_weight_identity") != "NOT_IDENTIFIABLE":
        errors.append("stage_vi_historical_identity_not_downgraded")
    workers = [load(path) for path in args.workers]
    expected_suites = sorted(contract.get("suites", {}))
    actual_suites = sorted(item.get("suite") for item in workers)
    if actual_suites != expected_suites:
        errors.append(f"suite_set:{actual_suites}")
    seen_gpus = set()
    for item in workers:
        if item.get("status") != "PASS_CLEAN_NOOP_PARITY":
            errors.append(f"worker_status:{item.get('suite')}:{item.get('status')}")
        gpu = item.get("gpu", {})
        if int(gpu.get("free_memory_mib", 0)) <= 20480:
            errors.append(f"gpu_gate:{item.get('suite')}")
        physical_gpu = gpu.get("physical_gpu")
        if physical_gpu in seen_gpus:
            errors.append(f"duplicate_gpu:{physical_gpu}")
        seen_gpus.add(physical_gpu)
        if item.get("counters") != {"pgd_calls": 0, "env_step_calls": 0, "physical_interventions": 0, "vphys_reads": 0, "protected_reads": 0, "eval160_reads": 0}:
            errors.append(f"protected_or_attack_counter:{item.get('suite')}")
        if item.get("eval160") != "UNREAD" or item.get("protected_evaluation") != "UNREAD":
            errors.append(f"protected_boundary:{item.get('suite')}")
        if len(item.get("rows", [])) != 2:
            errors.append(f"row_count:{item.get('suite')}")
        for row in item.get("rows", []):
            if not row.get("pass"):
                errors.append(f"parity_row:{item.get('suite')}:{row.get('snapshot_root')}")
            processor = row.get("processor", {})
            if not all(processor.get(key) for key in ("input_ids_exact", "attention_mask_exact", "pixel_values_exact_after_dtype_cast")):
                errors.append(f"processor_parity:{item.get('suite')}")
            if not row.get("clean_generation", {}).get("token_exact"):
                errors.append(f"token_parity:{item.get('suite')}")
            semantic = row.get("semantic_open_target", {})
            if not semantic.get("open_token_ids") or not semantic.get("close_token_ids"):
                errors.append(f"empty_semantic_token_set:{item.get('suite')}")
            if set(semantic.get("open_token_ids", [])) & set(semantic.get("close_token_ids", [])):
                errors.append(f"semantic_token_overlap:{item.get('suite')}")
    report = {
        "schema": "STAGE_X_X1R_VICTIM_PROVENANCE_INDEPENDENT_AUDIT_V1",
        "status": "PASS" if not errors else "HOLD_PROVENANCE_PARITY",
        "errors": errors,
        "contract": str(args.contract),
        "worker_reports": [str(path) for path in args.workers],
        "historical_stage_ix_f0": "IMMUTABLE_DIAGNOSTIC_NONPROMOTIONAL",
        "historical_stage_x1": "IMMUTABLE_DIAGNOSTIC_NONCONSUMABLE",
        "x1r_authorized": False,
        "eval160": "UNREAD",
        "protected_evaluation": "UNREAD",
        "protected_counters": {"eval160_reads": 0, "protected_reads": 0, "physical_interventions": 0, "attacked_env_steps": 0, "pgd_calls": 0, "vphys_reads": 0},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
