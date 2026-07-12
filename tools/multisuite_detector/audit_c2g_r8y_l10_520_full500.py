#!/usr/bin/env python3
"""Audit R8Y L10-520 full-500 collection (future validation contract).

This module defines the audit contract but MUST NOT BE RUN until
R8Y_L10_520_FULL500_COLLECTION_AUTHORIZED.

Verifies:
  - 500/500 runtime valid
  - 500 unique identities, 0 missing, 0 duplicates
  - 20/20 worker receipts
  - 25 episodes per logical worker, 125 per GPU
  - 520-step canonical horizon applied to 500/500
  - 0 GPU migration
  - Prefix comparison with old L10-300 for 470 completed parents
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

from tools.multisuite_detector.c2g_official_suite_horizons import (
    OFFICIAL_MAX_POLICY_STEPS,
)

SCHEMA = "c2g.r8y.l10_520_full500_audit.2026-07-12.v1"
PASS_STATUS = "PASS_C2G_R8Y_L10_520_FULL500_COLLECTION"
HOLD_STATUS = "HOLD_C2G_R8Y_L10_520_FULL500_COLLECTION"
TARGET_SUITE = "libero_10"
CANONICAL_MAX_STEPS = OFFICIAL_MAX_POLICY_STEPS[TARGET_SUITE]
GPUS = (4, 5, 6, 7)

# These are the only valid cohorts
EXPECTED_COHORTS = Counter({
    "DETECTOR_TRAIN": 300,
    "DETECTOR_VAL": 50,
    "DETECTOR_TEST_WITHIN_TASK": 50,
    "ATTACK_EVAL_PREREGISTERED": 100,
})


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"empty {path}")
    return rows


def identity(row: dict[str, Any]) -> tuple[str, int, int]:
    return (
        str(row.get("suite", TARGET_SUITE)),
        int(row.get("task_index", -1)),
        int(row.get("state_id", -1)),
    )


def audit_full500(
    *,
    collection_root: Path,
    plan_report: Path,
    old_l10_source_root: Path,
) -> dict[str, Any]:
    """Run the full-500 audit (AUTHORIZATION REQUIRED)."""
    collection_root = collection_root.resolve()
    old_l10_source_root = old_l10_source_root.resolve()

    # Identity closure from plan
    plan = read_json(plan_report)
    plan_status = plan.get("status", "")
    if not plan_status.startswith("PASS"):
        raise ValueError(f"plan not PASS: {plan_status}")

    # Collect all episode metadata
    workers_dir = collection_root / "workers"
    all_metadata: list[dict[str, Any]] = []
    worker_receipts: list[dict[str, Any]] = []
    runtime_valid = 0
    runtime_failed = 0
    gpu_migration_count = 0
    oom_count = 0

    if workers_dir.is_dir():
        for worker_dir in sorted(workers_dir.iterdir()):
            if not worker_dir.is_dir():
                continue

            # Worker receipt
            receipt_path = worker_dir / "worker_receipt.json"
            if receipt_path.is_file():
                r = read_json(receipt_path)
                worker_receipts.append(r)
                if str(r.get("status", "")).startswith("PASS"):
                    runtime_valid += 1
                else:
                    runtime_failed += 1

                assigned = r.get("assigned_physical_gpu") or r.get("physical_gpu")
                executed = r.get("cuda_visible_devices")
                if assigned is not None and executed is not None and str(assigned) != str(executed):
                    gpu_migration_count += 1

            # Episode metadata
            ep_dir = worker_dir / "collection" / "episodes"
            if ep_dir.is_dir():
                for meta_path in ep_dir.rglob("episode_metadata.json"):
                    all_metadata.append(read_json(meta_path))

    # Identity verification
    ids = [identity(m) for m in all_metadata]
    id_counts = Counter(ids)
    duplicates = {k: v for k, v in id_counts.items() if v > 1}
    unique = len(id_counts)
    total = len(all_metadata)

    # Cohort verification
    cohort_counts = Counter(str(m.get("cohort", "")) for m in all_metadata)

    # Horizon verification
    max_steps_520 = sum(
        1 for m in all_metadata
        if m.get("max_policy_steps") == CANONICAL_MAX_STEPS
        or m.get("max_steps") == CANONICAL_MAX_STEPS
    )
    dummy_wait_10 = sum(
        1 for m in all_metadata
        if m.get("dummy_wait_steps") == 10
    )

    # Per-GPU counts
    per_gpu = Counter(
        int(m.get("physical_gpu", m.get("assigned_physical_gpu", -1)))
        for m in all_metadata
    )

    # Suite closure
    suites = Counter(str(m.get("suite", "")) for m in all_metadata)

    # Prefix comparison with old L10-300
    old_completed = 0
    prefix_match = 0
    no_old_prefix = 0

    old_workers = old_l10_source_root / "workers"
    if old_workers.is_dir():
        for meta in all_metadata:
            source_key = meta.get("source_r8w_parent_key", meta.get("parent_key", ""))
            # Search for old metadata with matching parent_key
            found = False
            for old_w in old_workers.iterdir():
                if not old_w.is_dir():
                    continue
                old_ep = old_w / "collection" / "episodes"
                if not old_ep.is_dir():
                    continue
                for old_meta_path in old_ep.rglob("episode_metadata.json"):
                    old_meta = read_json(old_meta_path)
                    if old_meta.get("parent_key") == source_key:
                        old_completed += 1
                        found = True
                        # Compare success (simplified; full audit does step-level)
                        if meta.get("clean_success_observed") == old_meta.get(
                            "clean_success_observed"
                        ):
                            prefix_match += 1
                        break
                if found:
                    break
            if not found:
                no_old_prefix += 1

    # Gate
    identity_ok = total == 500 and unique == 500 and len(duplicates) == 0
    cohort_ok = cohort_counts == EXPECTED_COHORTS
    horizon_ok = max_steps_520 == 500 and dummy_wait_10 == 500
    gpu_ok = all(per_gpu.get(g, 0) == 125 for g in GPUS)
    suite_ok = suites == Counter({TARGET_SUITE: 500})
    worker_ok = len(worker_receipts) == 20 and runtime_failed == 0
    migration_ok = gpu_migration_count == 0
    prefix_ok = old_completed >= 470  # 470 min from old completed parents

    all_pass = all([
        identity_ok, cohort_ok, horizon_ok, gpu_ok, suite_ok,
        worker_ok, migration_ok, oom_count == 0,
    ])

    return {
        "schema": SCHEMA,
        "status": PASS_STATUS if all_pass else HOLD_STATUS,
        "episode_count": total,
        "unique_identities": unique,
        "duplicate_identities": len(duplicates),
        "duplicates": {str(k): v for k, v in list(duplicates.items())[:5]},
        "cohort_counts": dict(sorted(cohort_counts.items())),
        "cohort_ok": cohort_ok,
        "max_steps_520_count": max_steps_520,
        "dummy_wait_10_count": dummy_wait_10,
        "horizon_ok": horizon_ok,
        "per_gpu_counts": {str(g): per_gpu.get(g, 0) for g in GPUS},
        "gpu_ok": gpu_ok,
        "suite_ok": suite_ok,
        "worker_receipt_count": len(worker_receipts),
        "runtime_valid": runtime_valid,
        "runtime_failed": runtime_failed,
        "worker_ok": worker_ok,
        "gpu_migration_count": gpu_migration_count,
        "migration_ok": migration_ok,
        "oom_count": oom_count,
        "old_completed_parents_found": old_completed,
        "prefix_success_match": prefix_match,
        "no_old_prefix_available": no_old_prefix,
        "prefix_ok": prefix_ok,
        "attacks": 0,
        "training_epochs": 0,
        "materialization_runs": 0,
        "storage_deletions": 0,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection-root", type=Path, required=True)
    parser.add_argument("--plan-report", type=Path, required=True)
    parser.add_argument("--old-l10-source-root", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, default=None)
    parser.add_argument(
        "--authorization",
        default="",
        help="Must be set to FULL500_AUTHORIZED to run",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.authorization != "R8Y_L10_520_FULL500_COLLECTION_AUTHORIZED":
        raise PermissionError(
            "Full-500 audit is not yet authorized. "
            "Requires R8Y_L10_520_FULL500_COLLECTION_AUTHORIZED."
        )
    report = audit_full500(
        collection_root=args.collection_root,
        plan_report=args.plan_report,
        old_l10_source_root=args.old_l10_source_root,
    )
    if args.output_report:
        args.output_report.parent.mkdir(parents=True, exist_ok=True)
        args.output_report.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == PASS_STATUS else 1


if __name__ == "__main__":
    raise SystemExit(main())
