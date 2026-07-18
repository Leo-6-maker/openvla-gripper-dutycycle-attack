#!/usr/bin/env python3
"""Read-only census of V4 event/window identity semantics.

The current V2.1.2 derivative uses ``window_id == event_id``.  This auditor
checks whether an event is ever split across multiple phase segments before a
new Teacher derivative is considered necessary.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any

from gripper_attack.v4_contract import sha256_file, verify_checksum_manifest


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _seal(root: Path) -> None:
    excluded = {"SHA256SUMS", "SHA256SUMS.sha256"}
    payloads = sorted(
        (p for p in root.rglob("*") if p.is_file() and p.name not in excluded),
        key=lambda p: str(p.relative_to(root)).replace(os.sep, "/"),
    )
    sums = root / "SHA256SUMS"
    sums.write_text(
        "".join(f"{sha256_file(path)}  {str(path.relative_to(root)).replace(os.sep, '/') }\n" for path in payloads),
        encoding="utf-8",
    )
    (root / "SHA256SUMS.sha256").write_text(f"{sha256_file(sums)}  SHA256SUMS\n", encoding="utf-8")


def _phase_records(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, dict):
        value = value.get("phases", [])
    return value if isinstance(value, list) else []


def _audit_identity(labels_path: Path) -> dict[str, Any]:
    identity = "/".join(labels_path.relative_to(labels_path.parents[3]).parts[:3])
    rows = _jsonl(labels_path)
    phases_path = labels_path.parent / "close_phases.json"
    phases = _phase_records(json.loads(phases_path.read_text(encoding="utf-8"))) if phases_path.is_file() else []
    phase_ids = [int(item.get("event_id", -1)) for item in phases if int(item.get("event_id", -1)) >= 0]
    phase_id_counts: dict[int, int] = defaultdict(int)
    for event_id in phase_ids:
        phase_id_counts[event_id] += 1

    names_by_event: dict[int, set[str]] = defaultdict(set)
    rows_by_event: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        event_id = int(row.get("event_id", -1))
        if event_id < 0:
            continue
        names_by_event[event_id].add(str(row.get("phase_name", row.get("phase", "UNKNOWN"))))
        rows_by_event[event_id].append(row)

    multi_phase_events = sorted(event_id for event_id, names in names_by_event.items() if len(names) > 1)
    duplicate_phase_events = sorted(event_id for event_id, count in phase_id_counts.items() if count > 1)
    quality_veto_events = sorted(
        event_id for event_id, event_rows in rows_by_event.items()
        if any(bool(row.get("quality_valid", False)) for row in event_rows)
        and any(bool(row.get("veto_invalid", False)) for row in event_rows)
    )
    bounds_mismatch = []
    for event_id, event_rows in rows_by_event.items():
        observed = (min(int(row.get("step", 0)) for row in event_rows), max(int(row.get("step", 0)) for row in event_rows))
        declared = {
            (int(row.get("window_start", -1)), int(row.get("window_end", -1)))
            for row in event_rows
            if int(row.get("window_start", -1)) >= 0 and int(row.get("window_end", -1)) >= 0
        }
        if declared and observed not in declared:
            bounds_mismatch.append({"event_id": event_id, "observed": observed, "declared": sorted(declared)})

    return {
        "canonical_parent_key": identity,
        "step_count": len(rows),
        "close_phase_count": len(phases),
        "event_count": len(names_by_event),
        "duplicate_event_ids_in_close_phases": duplicate_phase_events,
        "event_ids_with_multiple_phase_names": multi_phase_events,
        "event_ids_with_quality_and_veto": quality_veto_events,
        "window_bounds_mismatch": bounds_mismatch,
        "window_id_event_id_identity": not multi_phase_events,
    }


def audit_window_semantics(teacher_root: Path, output_root: Path) -> dict[str, Any]:
    if output_root.exists():
        raise FileExistsError(output_root)
    source_seal = verify_checksum_manifest(teacher_root)
    label_paths = sorted(teacher_root.rglob("teacher_v212_labels.jsonl"))
    rows = [_audit_identity(path) for path in label_paths]
    duplicate = [row for row in rows if row["duplicate_event_ids_in_close_phases"]]
    multi_phase = [row for row in rows if row["event_ids_with_multiple_phase_names"]]
    bounds = [row for row in rows if row["window_bounds_mismatch"]]
    summary = {
        "schema": "DETECTOR_V4_WINDOW_SEMANTICS_AUDIT_V1",
        "status": "PASS" if len(rows) == 800 and not duplicate and not multi_phase and not bounds else "HOLD",
        "teacher_root_sha256s_sha256": source_seal["sha256sums_sha256"],
        "identity_count": len(rows),
        "duplicate_event_id_identity_count": len(duplicate),
        "multi_phase_event_identity_count": len(multi_phase),
        "window_bounds_mismatch_identity_count": len(bounds),
        "window_id_event_id_unique": not duplicate and not multi_phase and not bounds,
        "new_teacher_derivative_required": bool(duplicate or multi_phase),
        "formal_training_authorized": False,
        "formal_attack_authorized": False,
    }
    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = output_root.parent / f".{output_root.name}.{uuid.uuid4().hex}.staging"
    staging.mkdir(parents=True, exist_ok=False)
    try:
        (staging / "window_semantics_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (staging / "window_semantics_rows.jsonl").write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
        with (staging / "window_semantics_rows.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["canonical_parent_key", "step_count", "close_phase_count", "event_count", "window_id_event_id_identity"])
            writer.writeheader()
            for row in rows:
                writer.writerow({key: row[key] for key in writer.fieldnames})
        (staging / "input_binding.json").write_text(json.dumps({"teacher_root_sha256s_sha256": source_seal["sha256sums_sha256"]}, indent=2) + "\n", encoding="utf-8")
        _seal(staging)
        os.replace(staging, output_root)
        return summary
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit_window_semantics(args.teacher_root, args.output_root), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
