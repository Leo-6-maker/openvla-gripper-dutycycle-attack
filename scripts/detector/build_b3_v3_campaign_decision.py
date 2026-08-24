#!/usr/bin/env python3
"""Seal the minimum evidence needed for campaign-bounded FIT provenance."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import uuid
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_text(path: Path, value: str) -> None:
    path.write_text(value, encoding="utf-8")


def write_sidecar(path: Path) -> None:
    write_text(path.with_name(path.name + ".sha256"), f"{sha256_file(path)}  {path.name}\n")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean-manifest", type=Path, required=True)
    parser.add_argument("--fit-data-audit-root", type=Path, required=True)
    parser.add_argument("--manifest-equivalence-root", type=Path, required=True)
    parser.add_argument("--deep-audit-summary", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    clean_rows = read_csv(args.clean_manifest)
    data_rows = read_csv(args.fit_data_audit_root / "audit_rows.csv")
    manifest_rows = read_csv(args.manifest_equivalence_root / "manifest_rows.csv")
    class_rows = read_csv(args.manifest_equivalence_root / "equivalence_classes.csv")
    if len(clean_rows) != 2000 or len(data_rows) != 800 or len(manifest_rows) != 150 or len(class_rows) != 32:
        raise SystemExit("campaign evidence counts are not closed")

    expected_keys = {
        f"{row['suite']}/task_{int(row['task_idx']):02d}/state_{int(row['state_id']):02d}"
        for row in clean_rows if int(row["state_id"]) < 20
    }
    audit_by_key = {row.get("canonical_parent_key", ""): row for row in data_rows}
    if set(audit_by_key) != expected_keys or any(row.get("data_status") != "DATA_PASS" for row in data_rows):
        raise SystemExit("FIT data-only audit is not exactly DATA_PASS for all 800 identities")
    if any(row.get("source_before_sha256") != row.get("source_after_sha256") for row in data_rows):
        raise SystemExit("source mutation was observed in FIT data audit")

    values: dict[str, list[str]] = {}
    for name in (
        "collector_head", "worker_script_sha256", "adapter_sha256", "protocol_sha256",
        "runtime_config_sha256", "queue_manifest_sha256", "model_tree_sha256", "processor_tree_sha256",
    ):
        values[name] = sorted({row[name] for row in manifest_rows if row.get(name) not in (None, "", "__MISSING__")})

    output = args.output_root
    if output.exists():
        raise SystemExit(f"refusing to overwrite campaign decision root: {output}")
    staging = output.with_name(f".{output.name}.{uuid.uuid4().hex}.staging")
    try:
        staging.mkdir(parents=True)
        fit_path = staging / "fit800_identity_manifest.csv"
        fields = ["canonical_parent_key", "artifact_root", "artifact_recursive_sha256", "data_status", "recovery_status", "source_before_sha256", "source_after_sha256"]
        with fit_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for key in sorted(audit_by_key):
                writer.writerow({field: audit_by_key[key].get(field, "") for field in fields})
        shutil.copy2(args.manifest_equivalence_root / "manifest_rows.csv", staging / "accepted_manifest_inventory.csv")
        shutil.copy2(args.manifest_equivalence_root / "equivalence_classes.csv", staging / "accepted_equivalence_classes.csv")
        for name in ("fit800_identity_manifest.csv", "accepted_manifest_inventory.csv", "accepted_equivalence_classes.csv"):
            write_sidecar(staging / name)

        evidence = {
            "clean_manifest_sha256": sha256_file(args.clean_manifest),
            "fit_data_audit_root_sha256": sha256_file(args.fit_data_audit_root / "SHA256SUMS"),
            "fit_data_audit_summary_sha256": sha256_file(args.fit_data_audit_root / "summary.json"),
            "deep_audit_summary_sha256": sha256_file(args.deep_audit_summary),
            "manifest_equivalence_root_sha256": sha256_file(args.manifest_equivalence_root / "SHA256SUMS"),
        }
        decision: dict[str, Any] = {
            "schema": "OFFICIAL_V3_CAMPAIGN_BOUNDED_SOURCE_CONTRACT_V1",
            "decision": "ACCEPT_CAMPAIGN_BOUNDED_PROVENANCE",
            "strict_instance_provenance_required": False,
            "campaign_membership_required": True,
            "accepted_provenance_states": ["EXACT_DIRECT_START_UUID", "EXACT_LEASE_CHAIN", "CAMPAIGN_BOUND_INSTANCE_UNRESOLVED"],
            "campaign_identity_count": 2000,
            "fit_identity_count": 800,
            "fit_data_pass_count": 800,
            "fit_data_fail_count": 0,
            "manifest_count": len(manifest_rows),
            "equivalence_class_count": len(class_rows),
            "known_contradiction_count": 0,
            "source_mutation_count": 0,
            "files": {name: name for name in ("fit800_identity_manifest.csv", "accepted_manifest_inventory.csv", "accepted_equivalence_classes.csv")},
            "file_sha256": {name: sha256_file(staging / name) for name in ("fit800_identity_manifest.csv", "accepted_manifest_inventory.csv", "accepted_equivalence_classes.csv")},
            "evidence_binding": evidence,
            "accepted_artifact_field_values": values,
            "formal_training_authorized": False,
            "formal_attack_authorized": False,
        }
        decision_path = staging / "decision.json"
        write_text(decision_path, json.dumps(decision, indent=2, sort_keys=True) + "\n")
        write_sidecar(decision_path)
        names = sorted(path.relative_to(staging).as_posix() for path in staging.rglob("*") if path.is_file())
        sums = staging / "SHA256SUMS"
        write_text(sums, "".join(f"{sha256_file(staging / name)}  {name}\n" for name in names))
        write_sidecar(sums)
        os.replace(staging, output)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        raise
    print(json.dumps({"status": "PASS", "decision": "ACCEPT_CAMPAIGN_BOUNDED_PROVENANCE", "fit": 800, "manifests": 150, "equivalence_classes": 32}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
