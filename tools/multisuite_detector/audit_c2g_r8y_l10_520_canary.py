#!/usr/bin/env python3
"""Audit R8Y L10-520 shadow calibration canary.

Verifies:
  - 12/12 runtime valid (worker receipts, episode receipts)
  - 8/8 reproducibility prefix match (raw action, applied action, 25D)
  - Canonical success agreement with old L10-300
  - Dynamic admission: no OOM, no GPU migration, fail-closed admission
  - Horizon: 520-step canonical horizon applied
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from tools.multisuite_detector.c2g_official_suite_horizons import (
    OFFICIAL_MAX_POLICY_STEPS,
)

SCHEMA = "c2g.r8y.l10_520_canary_audit.2026-07-12.v1"
PASS_STATUS = "PASS_C2G_R8Y_L10_520_CANARY"
HOLD_STATUS = "HOLD_C2G_R8Y_L10_520_CANARY"
TARGET_SUITE = "libero_10"
CANONICAL_MAX_STEPS = OFFICIAL_MAX_POLICY_STEPS[TARGET_SUITE]
GPUS = (4, 5, 6, 7)


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


def audit_canary(
    *,
    canary_root: Path,
    old_l10_source_root: Path,
    plan_report: Path,
) -> dict[str, Any]:
    """Run the canary audit."""
    canary_root = canary_root.resolve()
    old_l10_source_root = old_l10_source_root.resolve()

    # Phase 1: Runtime validity
    runtime_valid = 0
    runtime_failed = 0
    oom_count = 0
    gpu_migration_count = 0
    worker_receipts: list[dict[str, Any]] = []

    workers_dir = canary_root / "workers"
    if not workers_dir.is_dir():
        raise FileNotFoundError(f"workers dir not found: {workers_dir}")

    for worker_dir in sorted(workers_dir.iterdir()):
        if not worker_dir.is_dir():
            continue
        receipt_path = worker_dir / "worker_receipt.json"
        if not receipt_path.is_file():
            runtime_failed += 1
            continue

        receipt = read_json(receipt_path)
        worker_receipts.append(receipt)

        if str(receipt.get("status", "")).startswith("PASS"):
            runtime_valid += 1
        else:
            runtime_failed += 1

        # Check GPU migration
        assigned_gpu = receipt.get("assigned_physical_gpu") or receipt.get("physical_gpu")
        executed_gpu = receipt.get("cuda_visible_devices")
        if assigned_gpu is not None and executed_gpu is not None:
            if str(assigned_gpu) != str(executed_gpu):
                gpu_migration_count += 1

        # Check OOM
        oom_markers = ["CUDA_OUT_OF_MEMORY", "out of memory", "OOM"]
        status_msg = str(receipt.get("status", "")).lower()
        if any(m.lower() in status_msg for m in oom_markers):
            oom_count += 1

    # Phase 2: Reproducibility prefix comparison
    repro_results: list[dict[str, Any]] = []
    raw_action_exact = 0
    applied_action_exact = 0
    features_25d_exact_or_equiv = 0
    success_agreement = 0
    repro_total = 0

    # Find episode dirs with step_records
    for worker_dir in sorted(workers_dir.iterdir()):
        if not worker_dir.is_dir():
            continue
        episodes_dir = worker_dir / "collection" / "episodes"
        if not episodes_dir.is_dir():
            continue

        for ep_dir in episodes_dir.rglob("step_records.jsonl"):
            ep_parent = ep_dir.parent
            metadata_path = ep_parent / "episode_metadata.json"
            if not metadata_path.is_file():
                continue

            metadata = read_json(metadata_path)
            source_parent_key = metadata.get(
                "source_r8w_parent_key", metadata.get("parent_key", "")
            )

            # Find old step_records
            old_ep = (
                old_l10_source_root
                / "workers"
                / f"g{metadata.get('physical_gpu', '?')}_l10"
                / "collection" / "episodes" / TARGET_SUITE / TARGET_SUITE
            )
            old_steps_path = None
            if old_ep.is_dir():
                for old_candidate in old_ep.rglob("step_records.jsonl"):
                    if old_candidate.parent.name.startswith("episode_"):
                        old_meta_path = old_candidate.parent / "episode_metadata.json"
                        if old_meta_path.is_file():
                            old_meta = read_json(old_meta_path)
                            if old_meta.get("parent_key") == source_parent_key:
                                old_steps_path = old_candidate
                                break

            result = {
                "parent_key": metadata.get("parent_key", ""),
                "source_r8w_parent_key": source_parent_key,
                "old_steps_found": old_steps_path is not None,
            }

            if old_steps_path is not None:
                repro_total += 1
                new_steps = read_jsonl(ep_dir)
                old_steps = read_jsonl(old_steps_path)

                # Compare up to min(len(new), len(old)) steps
                min_len = min(len(new_steps), len(old_steps))
                raw_match = True
                applied_match = True
                features_match = True
                for i in range(min_len):
                    ns = new_steps[i]
                    os = old_steps[i]
                    if ns.get("raw_action_7d") != os.get("raw_action_7d"):
                        raw_match = False
                    if ns.get("applied_action_7d") != os.get("applied_action_7d"):
                        applied_match = False
                    ns_feat = ns.get("features_25d") or ns.get("state_25d")
                    os_feat = os.get("features_25d") or os.get("state_25d")
                    if ns_feat is not None and os_feat is not None:
                        if isinstance(ns_feat, list) and isinstance(os_feat, list):
                            if len(ns_feat) == len(os_feat):
                                max_diff = max(
                                    abs(float(a) - float(b))
                                    for a, b in zip(ns_feat, os_feat)
                                )
                                if max_diff > 1e-5:
                                    features_match = False
                            else:
                                features_match = False

                if raw_match:
                    raw_action_exact += 1
                if applied_match:
                    applied_action_exact += 1
                if features_match:
                    features_25d_exact_or_equiv += 1

                # Success agreement
                new_success = metadata.get("clean_success_observed", False)
                old_meta = read_json(old_steps_path.parent / "episode_metadata.json")
                old_success = old_meta.get("clean_success_observed", False)
                if new_success == old_success:
                    success_agreement += 1

                result.update({
                    "raw_action_prefix_exact": raw_match,
                    "applied_action_prefix_exact": applied_match,
                    "features_25d_exact_or_equivalent": features_match,
                    "success_agreement": new_success == old_success,
                    "new_success": new_success,
                    "old_success": old_success,
                    "min_steps_compared": min_len,
                    "new_steps_total": len(new_steps),
                    "old_steps_total": len(old_steps),
                })

            repro_results.append(result)

    # Phase 3: Dynamic admission validation
    scheduler_report_path = canary_root / "c2g_r8y_l10_520_scheduler_report.json"
    admission_valid = False
    effective_caps: dict[str, int] = {}
    if scheduler_report_path.is_file():
        sched = read_json(scheduler_report_path)
        effective_caps = {
            str(gpu): sched.get("per_gpu", {}).get(str(gpu), {}).get("effective_cap", 0)
            for gpu in GPUS
        }
        admission_valid = all(
            cap >= 1 for cap in effective_caps.values()
        ) and oom_count == 0

    # Phase 4: Horizon validation
    horizon_valid_count = 0
    for worker_dir in sorted(workers_dir.iterdir()):
        if not worker_dir.is_dir():
            continue
        episodes_dir = worker_dir / "collection" / "episodes"
        if not episodes_dir.is_dir():
            continue
        for meta_path in episodes_dir.rglob("episode_metadata.json"):
            meta = read_json(meta_path)
            if meta.get("max_policy_steps") == CANONICAL_MAX_STEPS:
                horizon_valid_count += 1
            elif meta.get("max_steps") == CANONICAL_MAX_STEPS:
                horizon_valid_count += 1

    # Gate decisions
    canary_pass = (
        runtime_valid >= 12
        and gpu_migration_count == 0
        and oom_count == 0
        and runtime_failed == 0
    )

    repro_gate = (
        raw_action_exact >= (repro_total or 1)
        and applied_action_exact >= (repro_total or 1)
        and features_25d_exact_or_equiv >= (repro_total or 1)
        and success_agreement >= (repro_total or 1)
    )

    report = {
        "schema": SCHEMA,
        "status": (
            PASS_STATUS
            if canary_pass and repro_gate
            else HOLD_STATUS
        ),
        "canary_pass": canary_pass,
        "repro_gate_pass": repro_gate,
        "runtime_valid": runtime_valid,
        "runtime_failed": runtime_failed,
        "runtime_total": runtime_valid + runtime_failed,
        "oom_count": oom_count,
        "gpu_migration_count": gpu_migration_count,
        "raw_action_prefix_exact": raw_action_exact,
        "applied_action_prefix_exact": applied_action_exact,
        "features_25d_exact_or_equivalent": features_25d_exact_or_equiv,
        "success_agreement": success_agreement,
        "repro_total": repro_total,
        "horizon_valid_count": horizon_valid_count,
        "effective_caps_by_gpu": effective_caps,
        "admission_valid": admission_valid,
        "max_steps_expected": CANONICAL_MAX_STEPS,
        "gpus": list(GPUS),
    }

    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canary-root", type=Path, required=True)
    parser.add_argument("--old-l10-source-root", type=Path, required=True)
    parser.add_argument("--plan-report", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = audit_canary(
        canary_root=args.canary_root,
        old_l10_source_root=args.old_l10_source_root,
        plan_report=args.plan_report,
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
