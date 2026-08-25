#!/usr/bin/env python3
"""Build the fixed Z3 jobs and outcome-blind manual-audit selection."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
ELIGIBILITY = ROOT / "reports/STAGE_Z_Z3_ELIGIBILITY_RECONCILIATION_V1.json"
PROTOCOL = ROOT / "configs/STAGE_Z_Z3_CROSS_MODEL_COMMAND_OPEN_PHYSICAL_MATRIX_PROTOCOL_V1.json"
OUT = ROOT / "reports/STAGE_Z_Z3_EXECUTION_MANIFEST_V1.json"
STORAGE_OUT = ROOT / "reports/STAGE_Z_Z3_STORAGE_PREFLIGHT_V1.json"
ARMS = (("CLEAN_BRANCH_CRITICAL", 0, "CRITICAL"), ("COMMAND_OPEN_T3_CRITICAL", 3, "CRITICAL"), ("COMMAND_OPEN_T5_CRITICAL", 5, "CRITICAL"), ("COMMAND_OPEN_T10_CRITICAL", 10, "CRITICAL"), ("COMMAND_OPEN_T5_NONCRITICAL_CONTROL", 5, "NONCRITICAL"))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=PROTOCOL)
    parser.add_argument("--eligibility", type=Path, default=ELIGIBILITY)
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--manual-salt", default="STAGE_Z_Z3_MANUAL_AUDIT_V1_20260823")
    parser.add_argument("--free-bytes", type=int)
    parser.add_argument("--host", default=None)
    parser.add_argument("--observed-at-utc", default=None)
    parser.add_argument("--filesystem", default="/mnt/sdc")
    parser.add_argument("--storage-output", type=Path, default=STORAGE_OUT)
    args = parser.parse_args()
    protocol = load(args.protocol)
    eligibility = load(args.eligibility)
    if protocol.get("status") != "STAGE_Z_Z3_SOURCE_AUTHORITY_FROZEN":
        raise RuntimeError("PROTOCOL_NOT_FROZEN")
    eligible = eligibility.get("primary_eligible_model_parent_pairs", [])
    if len(eligible) != 92:
        raise RuntimeError(f"ELIGIBLE_COUNT:{len(eligible)}")
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in eligible:
        grouped[(str(row["model_family"]), str(row["suite"]))].append(row)
    chosen: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for key, rows in grouped.items():
        chosen[key] = sorted(rows, key=lambda row: digest(f"{args.manual_salt}|{key[0]}|{key[1]}|{row['canonical_parent_key']}"))[:2]
    jobs: list[dict[str, Any]] = []
    for row in sorted(eligible, key=lambda item: (item["model_family"], item["suite"], item["canonical_parent_key"])):
        model = str(row["model_family"])
        parent = str(row["canonical_parent_key"])
        selected = any(item["model_family"] == model and item["suite"] == row["suite"] and item["canonical_parent_key"] == parent for item in chosen[(model, str(row["suite"]))])
        manual_id = f"MA-{digest(f'{args.manual_salt}|{model}|{parent}')[:16]}" if selected else None
        for arm, duration, anchor_class in ARMS:
            anchor = row["critical_anchor"] if anchor_class == "CRITICAL" else row["noncritical_anchor"]
            branch_seed = digest(f"STAGE_Z_Z3_BRANCH_V1|{model}|{parent}|{arm}")
            jobs.append({
                "branch_id": f"Z3-{branch_seed[:20]}",
                "model_family": model,
                "suite": str(row["suite"]),
                "canonical_parent_key": parent,
                "anchor_class": anchor_class,
                "anchor_step": anchor["step"],
                "anchor_state_sha256": anchor["state_sha256"],
                "anchor_rank_digest": anchor.get("rank_digest"),
                "receipt_path": row["receipt_path"],
                "receipt_sha256": row["receipt_sha256"],
                "arm": arm,
                "duration": duration,
                "manual_audit_id": manual_id,
                "blinded_video_id": f"V-{digest(f'{manual_id}|{arm}')[:16]}" if manual_id else None,
                "outcome_status": "NOT_EXECUTED",
            })
    if len(jobs) != 460 or len({job["branch_id"] for job in jobs}) != 460:
        raise RuntimeError("JOB_MATRIX_INVALID")
    report = {
        "schema": "STAGE_Z_Z3_EXECUTION_MANIFEST_V1",
        "status": "STAGE_Z_Z3_EXECUTION_MANIFEST_FROZEN_NOT_EXECUTED",
        "protocol_sha256": sha(args.protocol),
        "eligibility_sha256": sha(args.eligibility),
        "population": {"eligible_model_parent_pairs": 92, "fixed_branches": 460, "arms_per_pair": 5},
        "manual_audit": {"salt": args.manual_salt, "selection_rule": "two lowest deterministic hashes per model x suite", "selected_model_parent_pairs": sorted({job["manual_audit_id"] for job in jobs if job["manual_audit_id"]}), "max_model_parent_pairs": 24, "max_videos": 120, "outcome_blind": True},
        "jobs": jobs,
        "fixed_incomplete_model_parent_pairs": eligibility["fixed_incomplete_model_parent_pairs"],
        "forbidden_counters_at_manifest": {"open_interventions": 0, "attacked_env_steps": 0, "v_phys_reads": 0, "protected_reads": 0, "eval160_reads": 0, "pgd_calls": 0},
        "next_legal_action": "Z3_B_ENGINEERING_SENTINELS",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.free_bytes is not None:
        estimates = {"compact_receipts_telemetry_bytes": 1024**3, "manual_videos_bytes": 3 * 1024**3, "overhead_bytes": 512 * 1024**2}
        reserve = 5 * 1024**3
        required = sum(estimates.values()) + reserve
        storage = {"schema": "STAGE_Z_Z3_STORAGE_PREFLIGHT_V1", "status": "PASS_Z3_STORAGE_PREFLIGHT" if args.free_bytes >= required else "HOLD_Z3_STORAGE_INSUFFICIENT", "filesystem": args.filesystem, "host": args.host, "observed_at_utc": args.observed_at_utc, "free_bytes": args.free_bytes, "estimates": estimates, "reserved_free_margin_bytes": reserve, "required_free_bytes": required, "free_after_estimate_and_margin_bytes": args.free_bytes - required, "no_historical_or_unrelated_deletion": True, "next_legal_action": "Z3_B_ENGINEERING_SENTINELS" if args.free_bytes >= required else "STOP_FOR_PI"}
        args.storage_output.parent.mkdir(parents=True, exist_ok=True)
        args.storage_output.write_text(json.dumps(storage, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps({"status": report["status"], "jobs": len(jobs), "storage": storage["status"]}, sort_keys=True))
    else:
        print(json.dumps({"status": report["status"], "jobs": len(jobs)}, sort_keys=True))


if __name__ == "__main__":
    main()
