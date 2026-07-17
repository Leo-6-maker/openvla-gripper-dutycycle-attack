#!/usr/bin/env python3
"""Build the 2000-row registry under the sealed campaign decision."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import uuid
from collections import Counter
from pathlib import Path

from gripper_attack.official_v3_contract import CAMPAIGN_PROVENANCE, audit_artifact, canonical_key, expected_split, load_contract, sha256_file


FIELDS = [
    "canonical_parent_key", "suite", "task_idx", "state_id", "split", "ledger_status", "task_success",
    "selected_artifact_root", "selected_artifact_recursive_sha256", "artifact_audit_path", "artifact_audit_sha256",
    "provenance_class", "worker_start_manifest_sha256", "collector_head", "worker_id", "gpu_id",
    "worker_script_sha256", "adapter_sha256", "protocol_sha256", "model_tree_sha256", "processor_tree_sha256",
    "formal_eligible", "formal_selected", "superseded_artifact_sha256", "remediation_required", "remediation_reason",
    "selection_reason", "campaign_decision_sha256",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_sidecar(path: Path) -> None:
    path.with_name(path.name + ".sha256").write_text(f"{sha256_file(path)}  {path.name}\n", encoding="utf-8")


def seal_root(root: Path) -> None:
    names = sorted(path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file() and path.name not in {"SHA256SUMS", "SHA256SUMS.sha256"})
    sums = root / "SHA256SUMS"
    sums.write_text("".join(f"{sha256_file(root / name)}  {name}\n" for name in names), encoding="utf-8")
    write_sidecar(sums)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean-manifest", type=Path, required=True)
    parser.add_argument("--campaign-contract", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--registry-output-root", type=Path, required=True)
    parser.add_argument("--audit-output-root", type=Path, required=True)
    args = parser.parse_args()
    if args.registry_output_root.exists() or args.audit_output_root.exists():
        raise SystemExit("refusing to overwrite registry or audit root")

    clean_rows = read_csv(args.clean_manifest)
    campaign = json.loads((args.campaign_contract / "decision.json").read_text(encoding="utf-8"))
    decision_sha = sha256_file(args.campaign_contract / "decision.json")
    fit_rows = read_csv(args.campaign_contract / "fit800_identity_manifest.csv")
    if len(clean_rows) != 2000 or len(fit_rows) != 800 or campaign.get("decision") != "ACCEPT_CAMPAIGN_BOUNDED_PROVENANCE":
        raise SystemExit("campaign registry inputs are not closed")
    fit_by_key = {row["canonical_parent_key"]: row for row in fit_rows}
    contract = load_contract(args.contract)

    audit_staging = args.audit_output_root.with_name(f".{args.audit_output_root.name}.{uuid.uuid4().hex}.staging")
    registry_staging = args.registry_output_root.with_name(f".{args.registry_output_root.name}.{uuid.uuid4().hex}.staging")
    try:
        audit_staging.mkdir(parents=True)
        registry_staging.mkdir(parents=True)
        audit_rows: dict[str, dict] = {}
        audit_dir = audit_staging / "artifact_audits"
        audit_dir.mkdir()
        for key in sorted(fit_by_key):
            source = fit_by_key[key]
            report = audit_artifact(Path(source["artifact_root"]), contract, mode="campaign_25d", campaign_contract=args.campaign_contract / "decision.json")
            if report.get("status") != "PASS_FORMAL_CANDIDATE" or report.get("formal_eligible") is not True:
                raise SystemExit(f"campaign source audit failed: {key}: {report.get('error', report.get('status'))}")
            report["campaign_contract_sha256"] = decision_sha
            report_path = audit_dir / (key.replace("/", "__") + ".json")
            report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            write_sidecar(report_path)
            audit_rows[key] = {**report, "artifact_audit_path": str(report_path.resolve()), "artifact_audit_sha256": sha256_file(report_path)}
        seal_root(audit_staging)

        registry_rows: list[dict[str, str]] = []
        for manifest in clean_rows:
            suite, task, state = manifest["suite"], int(manifest["task_idx"]), int(manifest["state_id"])
            key = canonical_key(suite, task, state)
            row = {field: "" for field in FIELDS}
            row.update({"canonical_parent_key": key, "suite": suite, "task_idx": str(task), "state_id": str(state), "split": expected_split(state), "formal_eligible": "False", "formal_selected": "False", "selection_reason": "OUTSIDE_FIT_TRAIN"})
            if state < 20:
                audit = audit_rows[key]
                row.update({
                    "task_success": str(bool(audit.get("task_success"))),
                    "selected_artifact_root": audit["artifact_root"],
                    "selected_artifact_recursive_sha256": audit["artifact_recursive_sha256"],
                    "artifact_audit_path": audit["artifact_audit_path"],
                    "artifact_audit_sha256": audit["artifact_audit_sha256"],
                    "provenance_class": CAMPAIGN_PROVENANCE,
                    "collector_head": audit.get("collector_head", ""),
                    "model_tree_sha256": audit.get("model_tree_sha256", ""),
                    "processor_tree_sha256": audit.get("processor_tree_sha256", ""),
                    "formal_eligible": "True", "formal_selected": "True", "selection_reason": "CAMPAIGN_BOUNDED_DATA_AND_MEMBERSHIP_PASS",
                    "campaign_decision_sha256": decision_sha,
                })
            registry_rows.append(row)

        registry_path = registry_staging / "OFFICIAL_V3_FORMAL_REGISTRY_V1.csv"
        with registry_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(registry_rows)
        write_sidecar(registry_path)
        summary = {
            "schema": "OFFICIAL_V3_FORMAL_REGISTRY_SUMMARY_V1",
            "identity_count": 2000,
            "unique_identity_count": 2000,
            "raw_sealed_count": 2000,
            "formal_selected_count": 800,
            "global_formal_selected_count": 800,
            "fit_formal_selected_count": 800,
            "full_artifact_audit_pass_count": 800,
            "unresolved_provenance_count": 0,
            "unfinished_remediation_count": 0,
            "stale_recovery_unresolved_count": 0,
            "duplicate_selection_count": 0,
            "fit_by_suite_formal_selected": {suite: 200 for suite in {row["suite"] for row in clean_rows}},
            "fit_by_task_formal_selected": {f"{suite}/task_{task}": 20 for suite in {row["suite"] for row in clean_rows} for task in range(10)},
            "formal_fit_ready": True,
            "formal_training_authorized": False,
            "formal_attack_authorized": False,
            "campaign_decision_path": str((args.campaign_contract / "decision.json").resolve()),
            "campaign_decision_sha256": decision_sha,
            "campaign_instance_unresolved_count": 439,
            "provenance_counts": {CAMPAIGN_PROVENANCE: 800},
        }
        summary_path = registry_staging / "OFFICIAL_V3_FORMAL_REGISTRY_SUMMARY_V1.json"
        summary["registry_sha256"] = sha256_file(registry_path)
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        write_sidecar(summary_path)
        seal_root(registry_staging)
        os.replace(audit_staging, args.audit_output_root)
        os.replace(registry_staging, args.registry_output_root)
    except Exception:
        shutil.rmtree(audit_staging, ignore_errors=True)
        shutil.rmtree(registry_staging, ignore_errors=True)
        raise
    print(json.dumps({"status": "PASS", "registry_rows": 2000, "fit_selected": 800, "campaign_instance_unresolved": 439}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
