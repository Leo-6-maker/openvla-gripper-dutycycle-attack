#!/usr/bin/env python3
"""Independent structural audit of a V3 S1 Teacher materialization root."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gripper_attack.b3_official_v3_s1 import (
    aggregate_teacher_audit,
    audit_teacher_episode,
    load_formal_fit_registry,
    sha256_file,
    write_sealed_json,
)


def audit_root(materialized_root: Path, registry_rows: list[dict[str, str]]) -> dict[str, object]:
    reports = []
    for manifest_path in sorted(materialized_root.rglob("materialization_manifest.json")):
        episode_root = manifest_path.parent
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        identity = manifest.get("source_identity", {})
        key = identity.get("canonical_parent_key", "")
        teacher_path = episode_root / "teacher_retention_records.jsonl"
        event_path = episode_root / "retention_events.json"
        if not teacher_path.is_file() or not event_path.is_file():
            reports.append({"canonical_parent_key": key, "status": "HOLD", "violations": ["TEACHER_OR_EVENT_FILE_MISSING"]})
            continue
        rows = [json.loads(line) for line in teacher_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        events = json.loads(event_path.read_text(encoding="utf-8"))
        reports.append(audit_teacher_episode(rows, events, key))
    aggregate = aggregate_teacher_audit(reports, registry_rows)
    aggregate["materialized_root"] = str(materialized_root.resolve())
    aggregate["teacher_audit_script_sha256"] = sha256_file(Path(__file__).resolve())
    return aggregate


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--materialized-root", type=Path, required=True)
    parser.add_argument("--registry-csv", type=Path, required=True)
    parser.add_argument("--registry-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    registry_rows = load_formal_fit_registry(args.registry_csv, args.registry_summary)
    report = audit_root(args.materialized_root.resolve(), registry_rows)
    write_sealed_json(args.output, report)
    print(json.dumps({"status": report["status"], "identity_count": report["actual_identity_count"]}, sort_keys=True))
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
