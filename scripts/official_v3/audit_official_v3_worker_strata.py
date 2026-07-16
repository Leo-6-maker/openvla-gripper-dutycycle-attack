#!/usr/bin/env python3
"""Audit worker-slot provenance and lease uniqueness."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from gripper_attack.official_v3_contract import sha256_file


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def audit_worker_strata(
    registry_rows: list[dict[str, Any]],
    ledger_rows: list[dict[str, str]],
    worker_manifests: list[dict[str, Any]],
    expected_slots: list[dict[str, Any]],
) -> dict[str, Any]:
    expected = {str(row["slot_id"]): row for row in expected_slots}
    manifests = {str(row.get("slot_id")): row for row in worker_manifests if row.get("slot_id")}
    active = [row for row in ledger_rows if row.get("status") in {"LEASED", "RUNNING"}]
    active_keys = [row.get("canonical_parent_key") or row.get("cell_id", "") for row in active]
    duplicate_active_keys = sorted(key for key, count in Counter(active_keys).items() if key and count > 1)
    duplicate_active_workers = sorted(
        key for key, count in Counter((row.get("worker_id"), key) for row, key in zip(active, active_keys)).items() if count > 1
    )
    by_slot: dict[str, dict[str, Any]] = {}
    for slot_id, expected_row in expected.items():
        manifest = manifests.get(slot_id, {})
        slot_records = [row for row in registry_rows if row.get("worker_id") == slot_id]
        by_slot[slot_id] = {
            "gpu_id": expected_row.get("gpu_id"),
            "manifest_present": bool(manifest),
            "first_canary_pass": bool(manifest.get("first_canary_pass")) or any(
                row.get("formal_eligible") is True for row in slot_records
            ),
            "artifact_count": len(slot_records),
            "task_success_count": sum(str(row.get("task_success")).lower() == "true" for row in slot_records),
            "task_failure_count": sum(str(row.get("task_success")).lower() == "false" for row in slot_records),
            "collector_heads": sorted({str(row.get("collector_head")) for row in slot_records if row.get("collector_head")}),
            "worker_script_shas": sorted({str(row.get("worker_script_sha256")) for row in slot_records if row.get("worker_script_sha256")}),
            "adapter_shas": sorted({str(row.get("adapter_sha256")) for row in slot_records if row.get("adapter_sha256")}),
            "protocol_shas": sorted({str(row.get("protocol_sha256")) for row in slot_records if row.get("protocol_sha256")}),
            "model_shas": sorted({str(row.get("model_tree_sha256")) for row in slot_records if row.get("model_tree_sha256")}),
            "processor_shas": sorted({str(row.get("processor_tree_sha256")) for row in slot_records if row.get("processor_tree_sha256")}),
        }
    all_slots = all(row["manifest_present"] and row["first_canary_pass"] for row in by_slot.values())
    return {
        "schema": "OFFICIAL_V3_WORKER_STRATA_AUDIT_V1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if all_slots and not duplicate_active_keys and not duplicate_active_workers else "HOLD",
        "expected_slot_count": len(expected),
        "observed_slot_count": len(manifests),
        "all_slots_have_clean_start_manifest": all(row["manifest_present"] for row in by_slot.values()),
        "all_slots_have_first_canary": all(row["first_canary_pass"] for row in by_slot.values()),
        "duplicate_active_canonical_keys": duplicate_active_keys,
        "duplicate_active_worker_keys": [list(key) for key in duplicate_active_workers],
        "by_slot": by_slot,
        "formal_training_authorized": False,
        "formal_attack_authorized": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--slot-manifests", type=Path, required=True)
    parser.add_argument("--expected-slots", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite worker audit: {args.output}")
    registry = read_csv(args.registry)
    ledger = read_csv(args.ledger)
    expected = json.loads(args.expected_slots.read_text(encoding="utf-8"))
    manifests = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(args.slot_manifests.glob("*.json"))]
    report = audit_worker_strata(registry, ledger, manifests, expected)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.output.with_name(args.output.name + ".sha256").write_text(
        f"{sha256_file(args.output)}  {args.output.name}\n", encoding="utf-8"
    )
    print(json.dumps({"status": report["status"], "observed_slot_count": report["observed_slot_count"]}, sort_keys=True))
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
