#!/usr/bin/env python3
"""Audit a completed C2g five-condition matched-load execution matrix.

The audit is closed-world against the frozen job manifest. It verifies exact
condition closure, parent/provenance identity, fixed burst delivery, paired
objective seeds, detector/random timing, route-reported compute counts,
processor-space budgets, and pre-trigger clean-trajectory parity.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.gripper_attack.c2g_matched_load_manifest import (
    CORE_CONDITIONS,
    DETECTOR_TIMING_CONDITIONS,
    RANDOM_TIMING_CONDITIONS,
    validate_core_2x2_manifest,
)

PROTOCOL_NAME = "C2G_CLEAN_WINDOW_VIS_PGD"
PROTOCOL_VERSION = "2026-07-10.v1"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_no} must contain an object")
            rows.append(value)
    if not rows:
        raise ValueError(f"{path} contains no rows")
    return rows


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_action_prefix(rows: Sequence[Mapping[str, Any]], stop_step: int) -> str:
    payload = []
    for row in sorted(rows, key=lambda value: int(value.get("step", -1))):
        step = int(row.get("step", -1))
        if step < 0 or step >= stop_step:
            continue
        payload.append(
            {
                "step": step,
                "clean_raw_action": row.get("clean_raw_action"),
                "clean_env_action": row.get("clean_env_action"),
                "gripper_qpos_sum": row.get("gripper_qpos_sum"),
                "gripper_opening_proxy": row.get("gripper_opening_proxy"),
            }
        )
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def metadata_path(output_root: Path, parent_key: str, condition: str) -> Path:
    return output_root / parent_key / condition / "episode_metadata.json"


def steps_path(output_root: Path, parent_key: str, condition: str) -> Path:
    return output_root / parent_key / condition / "step_records.jsonl"


def audit(args: argparse.Namespace) -> dict[str, Any]:
    jobs = read_jsonl(args.jobs.resolve())
    manifest_summary = validate_core_2x2_manifest(
        jobs,
        strict_objective_seed_pairing=True,
    )
    output_root = args.output_root.resolve()
    expected = {(str(row["parent_key"]), str(row["condition"])): row for row in jobs}
    if len(expected) != len(jobs):
        raise ValueError("duplicate parent/condition jobs")

    discovered: set[tuple[str, str]] = set()
    for path in output_root.rglob("episode_metadata.json"):
        try:
            metadata = json.loads(path.read_text(encoding="utf-8"))
            discovered.add((str(metadata.get("parent_key", "")), str(metadata.get("condition", ""))))
        except Exception:
            discovered.add((f"MALFORMED:{path}", ""))
    missing = sorted(set(expected) - discovered)
    unexpected = sorted(discovered - set(expected))
    violations: list[dict[str, Any]] = []
    job_summaries: list[dict[str, Any]] = []
    rows_by_job: dict[tuple[str, str], list[dict[str, Any]]] = {}
    metadata_by_job: dict[tuple[str, str], dict[str, Any]] = {}

    for key, job in sorted(expected.items()):
        parent, condition = key
        meta_path = metadata_path(output_root, parent, condition)
        row_path = steps_path(output_root, parent, condition)
        if not meta_path.is_file() or not row_path.is_file():
            continue
        try:
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
            rows = read_jsonl(row_path)
        except Exception as exc:
            violations.append({
                "parent_key": parent,
                "condition": condition,
                "reason": "READ_ERROR",
                "error": str(exc),
            })
            continue
        metadata_by_job[key] = metadata
        rows_by_job[key] = rows
        if metadata.get("protocol_name") != PROTOCOL_NAME or metadata.get("protocol_version") != PROTOCOL_VERSION:
            violations.append({"parent_key": parent, "condition": condition, "reason": "PROTOCOL_MISMATCH"})
        for field in ("parent_key", "condition", "suite", "task_index", "state_id"):
            expected_value = job[field]
            if metadata.get(field) != expected_value:
                violations.append({
                    "parent_key": parent,
                    "condition": condition,
                    "reason": "IDENTITY_MISMATCH",
                    "field": field,
                    "expected": expected_value,
                    "actual": metadata.get(field),
                })
        for field, expected_value in (
            ("objective_family", job["objective_family"]),
            ("objective_seed", int(job["objective_seed"])),
            ("detector_checkpoint_sha256", job["detector_checkpoint_sha256"]),
        ):
            actual = metadata.get(field)
            if actual != expected_value:
                violations.append({
                    "parent_key": parent,
                    "condition": condition,
                    "reason": "FROZEN_JOB_FIELD_MISMATCH",
                    "field": field,
                    "expected": expected_value,
                    "actual": actual,
                })
        if not bool(metadata.get("runtime_valid")):
            violations.append({"parent_key": parent, "condition": condition, "reason": "RUNTIME_INVALID"})

        delivered = [row for row in rows if bool(row.get("attack_delivered"))]
        expected_frames = int(job["expected_attacked_frames"])
        if len(delivered) != expected_frames or int(metadata.get("attack_delivery_count", -1)) != expected_frames:
            violations.append({
                "parent_key": parent,
                "condition": condition,
                "reason": "DELIVERY_COUNT_MISMATCH",
                "expected": expected_frames,
                "rows": len(delivered),
                "metadata": metadata.get("attack_delivery_count"),
            })
        if delivered:
            delivered_steps = [int(row["step"]) for row in delivered]
            if delivered_steps != list(range(delivered_steps[0], delivered_steps[0] + expected_frames)):
                violations.append({"parent_key": parent, "condition": condition, "reason": "NONCONTIGUOUS_BURST"})
            planned = job.get("planned_start_step")
            if delivered_steps[0] != int(planned):
                violations.append({
                    "parent_key": parent,
                    "condition": condition,
                    "reason": "START_STEP_MISMATCH",
                    "expected": planned,
                    "actual": delivered_steps[0],
                })
            if metadata.get("first_attack_step") != delivered_steps[0]:
                violations.append({
                    "parent_key": parent,
                    "condition": condition,
                    "reason": "METADATA_FIRST_ATTACK_MISMATCH",
                    "expected": delivered_steps[0],
                    "actual": metadata.get("first_attack_step"),
                })

        load = job["load_spec"]
        epsilon = float(load["epsilon"])
        expected_loss_forwards = int(load["num_loss_forwards_per_frame"])
        expected_backwards = int(load["num_backwards_per_frame"])
        expected_adv_decodes = int(load["num_adv_decodes_per_frame"])
        metadata_load = metadata.get("attack_load", {})
        for field, expected_value in (
            ("burst_length", int(load["burst_length"])),
            ("epsilon", float(load["epsilon"])),
            ("step_size", float(load["step_size"])),
            ("pgd_steps", int(load["pgd_steps"])),
            ("temporal_init", str(load["temporal_init_policy"])),
            ("resize_size", int(load["image_height"])),
        ):
            if metadata_load.get(field) != expected_value:
                violations.append({
                    "parent_key": parent,
                    "condition": condition,
                    "reason": "ATTACK_LOAD_METADATA_MISMATCH",
                    "field": field,
                    "expected": expected_value,
                    "actual": metadata_load.get(field),
                })
        for row in delivered:
            step = row.get("step")
            if int(row.get("num_loss_forwards", -1)) != expected_loss_forwards:
                violations.append({
                    "parent_key": parent, "condition": condition, "step": step,
                    "reason": "LOSS_FORWARD_COUNT_MISMATCH",
                    "expected": expected_loss_forwards,
                    "actual": row.get("num_loss_forwards"),
                })
            if int(row.get("num_backwards", -1)) != expected_backwards:
                violations.append({
                    "parent_key": parent, "condition": condition, "step": step,
                    "reason": "BACKWARD_COUNT_MISMATCH",
                    "expected": expected_backwards,
                    "actual": row.get("num_backwards"),
                })
            if int(row.get("num_adv_decodes", -1)) != expected_adv_decodes:
                violations.append({
                    "parent_key": parent, "condition": condition, "step": step,
                    "reason": "ADV_DECODE_COUNT_MISMATCH",
                    "expected": expected_adv_decodes,
                    "actual": row.get("num_adv_decodes"),
                })
            linf = float(row.get("observation_perturb_linf", float("inf")))
            if linf > epsilon + args.epsilon_tolerance:
                violations.append({
                    "parent_key": parent,
                    "condition": condition,
                    "step": step,
                    "reason": "LINF_BUDGET_VIOLATION",
                    "epsilon": epsilon,
                    "linf": linf,
                })
        if condition == "CLEAN" and delivered:
            violations.append({"parent_key": parent, "condition": condition, "reason": "CLEAN_ATTACK_DELIVERED"})
        job_summaries.append(
            {
                "parent_key": parent,
                "condition": condition,
                "runtime_valid": bool(metadata.get("runtime_valid")),
                "success": metadata.get("success"),
                "total_steps": len(rows),
                "attack_delivery_count": len(delivered),
                "first_attack_step": delivered[0]["step"] if delivered else None,
                "metadata_sha256": sha256_file(meta_path),
                "steps_sha256": sha256_file(row_path),
            }
        )

    parent_summaries: list[dict[str, Any]] = []
    for parent in sorted({key[0] for key in expected}):
        keys = [(parent, condition) for condition in CORE_CONDITIONS]
        if any(key not in rows_by_job for key in keys):
            continue
        detector_starts = {
            metadata_by_job[key].get("first_attack_step")
            for key in keys if key[1] in DETECTOR_TIMING_CONDITIONS
        }
        random_starts = {
            metadata_by_job[key].get("first_attack_step")
            for key in keys if key[1] in RANDOM_TIMING_CONDITIONS
        }
        if len(detector_starts) != 1 or None in detector_starts:
            violations.append({
                "parent_key": parent,
                "reason": "DETECTOR_PAIR_START_MISMATCH",
                "values": sorted(detector_starts, key=str),
            })
        if len(random_starts) != 1 or None in random_starts:
            violations.append({
                "parent_key": parent,
                "reason": "RANDOM_PAIR_START_MISMATCH",
                "values": sorted(random_starts, key=str),
            })
        starts = [value for value in detector_starts | random_starts if value is not None]
        parity_stop = min(starts) if starts else 0
        prefix_hashes = {
            condition: canonical_action_prefix(rows_by_job[(parent, condition)], parity_stop)
            for condition in CORE_CONDITIONS
        }
        if len(set(prefix_hashes.values())) != 1:
            violations.append({
                "parent_key": parent,
                "reason": "PRETRIGGER_PARITY_MISMATCH",
                "hashes": prefix_hashes,
            })
        success = {
            condition: metadata_by_job[(parent, condition)].get("success")
            for condition in CORE_CONDITIONS
        }
        parent_summaries.append(
            {
                "parent_key": parent,
                "detector_start": next(iter(detector_starts)) if len(detector_starts) == 1 else None,
                "random_start": next(iter(random_starts)) if len(random_starts) == 1 else None,
                "pretrigger_stop": parity_stop,
                "pretrigger_hash": next(iter(prefix_hashes.values())) if len(set(prefix_hashes.values())) == 1 else None,
                "success": success,
            }
        )

    status = (
        "PASS_C2G_MATCHED_LOAD_RUN_AUDIT"
        if not missing and not unexpected and not violations
        else "HOLD_C2G_MATCHED_LOAD_RUN_AUDIT"
    )
    return {
        "gate": "C2G_MATCHED_LOAD_RUN_AUDIT",
        "status": status,
        "jobs_path": str(args.jobs.resolve()),
        "jobs_sha256": sha256_file(args.jobs.resolve()),
        "output_root": str(output_root),
        "manifest_validation": manifest_summary,
        "expected_job_count": len(expected),
        "processed_job_count": len(job_summaries),
        "missing_jobs": missing,
        "unexpected_jobs": unexpected,
        "violation_count": len(violations),
        "violations": violations,
        "jobs": job_summaries,
        "parents": parent_summaries,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--epsilon-tolerance", type=float, default=1e-6)
    args = parser.parse_args(argv)
    report = audit(args)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return 0 if report["status"].startswith("PASS_") else 2


if __name__ == "__main__":
    raise SystemExit(main())
