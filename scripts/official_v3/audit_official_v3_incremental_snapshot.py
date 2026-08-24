#!/usr/bin/env python3
"""Create a non-overwriting stable snapshot of new Official V3 artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from gripper_attack.official_v3_contract import audit_artifact, load_contract, sha256_file


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _ledger_key(row: dict[str, str]) -> str:
    if row.get("canonical_parent_key"):
        return row["canonical_parent_key"]
    cell = row.get("cell_id", "")
    return cell[len("CLEAN|"):] if cell.startswith("CLEAN|") else cell


def audit_snapshot(
    manifest_rows: list[dict[str, str]],
    ledger_rows: list[dict[str, str]],
    source_root: Path,
    contract: dict[str, Any],
    *,
    previous: dict[str, Any] | None = None,
) -> dict[str, Any]:
    ledger = {_ledger_key(row): row for row in ledger_rows if _ledger_key(row)}
    records: list[dict[str, Any]] = []
    for row in manifest_rows:
        key = row.get("canonical_parent_key", "")
        raw_root = row.get("artifact_root") or key
        artifact = Path(raw_root)
        if not artifact.is_absolute():
            artifact = source_root / artifact
        artifact = artifact.resolve()
        ledger_status = ledger.get(key, {}).get("status", "")
        active = ledger_status in {"LEASED", "RUNNING"}
        sealed = (artifact / "artifact_sha256.json").is_file() and not active
        result: dict[str, Any] = {
            "canonical_parent_key": key,
            "artifact_root": str(artifact),
            "ledger_status": ledger_status,
            "stable": sealed,
            "status": "NOT_SEALED" if not sealed else "AUDIT_NOT_RUN",
            "artifact_recursive_sha256": "",
        }
        if sealed:
            result = {**result, **audit_artifact(artifact, contract)}
        records.append(result)

    previous_records = {row.get("canonical_parent_key"): row for row in (previous or {}).get("records", []) if row.get("canonical_parent_key")}
    current_records = {row.get("canonical_parent_key"): row for row in records if row.get("canonical_parent_key")}
    changed = sorted(
        key for key, row in current_records.items()
        if key not in previous_records
        or row.get("artifact_recursive_sha256") != previous_records[key].get("artifact_recursive_sha256")
        or row.get("status") != previous_records[key].get("status")
    )
    artifact_keys = set(current_records)
    ledger_keys = set(ledger)
    return {
        "schema": "OFFICIAL_V3_INCREMENTAL_SNAPSHOT_V1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "records": records,
        "stable_snapshot": all(row["stable"] for row in records),
        "raw_sealed_count": sum(row["stable"] for row in records),
        "formal_selected_count": sum(row.get("status") == "PASS_FORMAL_CANDIDATE" for row in records),
        "status_counts": dict(Counter(row.get("status", "") for row in records)),
        "changed_or_new_keys": changed,
        "artifact_minus_ledger": sorted(artifact_keys - ledger_keys),
        "ledger_minus_artifact": sorted(ledger_keys - artifact_keys),
        "running_or_leased_excluded": sorted(row["canonical_parent_key"] for row in records if row["ledger_status"] in {"LEASED", "RUNNING"}),
        "formal_training_authorized": False,
        "formal_attack_authorized": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--previous-snapshot", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite snapshot: {args.output}")
    previous = json.loads(args.previous_snapshot.read_text(encoding="utf-8")) if args.previous_snapshot else None
    report = audit_snapshot(
        read_csv(args.manifest), read_csv(args.ledger), args.source_root.resolve(), load_contract(args.contract), previous=previous
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.output.with_name(args.output.name + ".sha256").write_text(
        f"{sha256_file(args.output)}  {args.output.name}\n", encoding="utf-8"
    )
    print(json.dumps({key: report[key] for key in ("raw_sealed_count", "formal_selected_count", "stable_snapshot")}, sort_keys=True))
    return 0 if not report["artifact_minus_ledger"] and not report["ledger_minus_artifact"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
