#!/usr/bin/env python3
"""Audit an R9Q matched attack run without changing its cell artifacts."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


CONDITIONS = ("CLEAN", "R9Q_DETECTOR_T10", "RAND_T10", "COMMAND_OPEN_ORACLE")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--expected-git-commit", required=True)
    parser.add_argument("--mode", choices=("canary", "panel", "full"), default="canary")
    parser.add_argument("--expected-cells", type=int, required=True)
    parser.add_argument("--detector-bundle", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = Path(args.manifest).resolve()
    run_root = Path(args.run_root).resolve()
    output = Path(args.output_root).resolve()
    if output.exists():
        raise SystemExit(f"refusing to overwrite audit root: {output}")
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != args.expected_cells:
        raise SystemExit(f"manifest cell count {len(rows)} != expected {args.expected_cells}")

    failures: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    ledgers: list[dict[str, Any]] = []

    # P0-2: verify detector bundle SHA closure (must be after failures init)
    bundle_verification: dict[str, Any] = {}
    if args.detector_bundle:
        bundle = Path(args.detector_bundle)
        expected_checkpoint = rows[0].get("detector_checkpoint_sha256", "")
        expected_config = rows[0].get("detector_config_sha256", "")
        expected_bundle = rows[0].get("detector_bundle_sha256", "")
        actual_checkpoint = sha256_file(bundle / "checkpoint.pt")
        actual_config = sha256_file(bundle / "detector_config.json")
        actual_normalization = sha256_file(bundle / "normalization.json")
        actual_sums = sha256_file(bundle / "SHA256SUMS") if (bundle / "SHA256SUMS").is_file() else ""
        actual_sums_dot_sha256 = sha256_file(bundle / "SHA256SUMS.sha256") if (bundle / "SHA256SUMS.sha256").is_file() else ""
        actual_normalization_sha = sha256_file(bundle / "normalization.json") if (bundle / "normalization.json").is_file() else ""

        # Verify SHA256SUMS internal entries
        bundle_sums_entries_valid = True
        bundle_fileset_match = True
        if (bundle / "SHA256SUMS").is_file():
            try:
                expected_entries = {}
                for line in (bundle / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    sha, _, relpath = line.partition("  ")
                    expected_entries[relpath] = sha
                for relpath, expected_sha in expected_entries.items():
                    fpath = bundle / relpath
                    if not fpath.is_file():
                        bundle_fileset_match = False
                        break
                    if sha256_file(fpath) != expected_sha:
                        bundle_sums_entries_valid = False
                        break
                actual_files = {str(p.relative_to(bundle).as_posix()) for p in bundle.rglob("*") if p.is_file() and p.name not in ("SHA256SUMS", "SHA256SUMS.sha256")}
                if actual_files != set(expected_entries.keys()):
                    bundle_fileset_match = False
            except Exception:
                bundle_sums_entries_valid = False
                bundle_fileset_match = False

        bundle_verification = {
            "checkpoint_sha256_match": actual_checkpoint == expected_checkpoint,
            "config_sha256_match": actual_config == expected_config,
            "bundle_sums_sha256_match": actual_sums == expected_bundle,
            "actual_checkpoint": actual_checkpoint,
            "expected_checkpoint": expected_checkpoint,
            "actual_config": actual_config,
            "expected_config": expected_config,
            "actual_normalization": actual_normalization,
            "expected_normalization": actual_normalization_sha,
            "actual_bundle_sums": actual_sums,
            "expected_bundle_sums": expected_bundle,
            "actual_sums_dot_sha256": actual_sums_dot_sha256,
            "sums_internal_entries_valid": bundle_sums_entries_valid,
            "sums_fileset_match": bundle_fileset_match,
        }
        if not all([
            actual_checkpoint == expected_checkpoint,
            actual_config == expected_config,
            actual_sums == expected_bundle,
            bundle_sums_entries_valid,
            bundle_fileset_match,
        ]):
            failures.append({"code": "BUNDLE_SHA_MISMATCH", "verification": bundle_verification})

    seen: set[tuple[str, str]] = set()
    triggers: list[dict[str, Any]] = []
    per_suite_condition: defaultdict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (str(row["parent_key"]), str(row["condition"]))
        if key in seen:
            failures.append({"code": "DUPLICATE_MANIFEST_CELL", "parent_key": key[0], "condition": key[1]})
            continue
        seen.add(key)
        cell = run_root / "cells" / str(row["suite"]) / str(row["parent_key"]) / str(row["condition"])
        metadata_path = cell / "episode_metadata.json"
        steps_path = cell / "step_records.jsonl"
        metadata: dict[str, Any] = {}
        records: list[dict[str, Any]] = []
        if metadata_path.is_file():
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            except Exception as exc:
                failures.append({"code": "METADATA_PARSE_ERROR", "cell": str(cell), "error": str(exc)})
        else:
            failures.append({"code": "MISSING_METADATA", "cell": str(cell)})
        if steps_path.is_file():
            try:
                records = [json.loads(line) for line in steps_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            except Exception as exc:
                failures.append({"code": "STEP_RECORD_PARSE_ERROR", "cell": str(cell), "error": str(exc)})
        else:
            failures.append({"code": "MISSING_STEP_RECORDS", "cell": str(cell)})
        valid = bool(metadata.get("runtime_valid") is True and isinstance(metadata.get("success"), bool) and records)
        if not valid:
            failures.append({"code": "INVALID_CELL", "cell": str(cell), "runtime_valid": metadata.get("runtime_valid"), "success": metadata.get("success"), "step_count": len(records)})
        for field, expected in (("parent_key", row["parent_key"]), ("condition", row["condition"]), ("git_commit", args.expected_git_commit)):
            if metadata.get(field) != expected:
                failures.append({"code": "PROVENANCE_MISMATCH", "cell": str(cell), "field": field, "expected": expected, "actual": metadata.get(field)})
        attack_count = int(metadata.get("attack_delivery_count", 0) or 0)
        condition = str(row["condition"])
        if condition == "RAND_T10" and attack_count != 10:
            failures.append({"code": "RAND_BURST_NOT_EXACT_T10", "cell": str(cell), "attack_count": attack_count})
        if condition in {"R9Q_DETECTOR_T10", "COMMAND_OPEN_ORACLE"} and metadata.get("first_attack_step") is not None and attack_count not in (0, 10):
            failures.append({"code": "DETECTOR_BURST_NOT_EXACT_T10", "cell": str(cell), "attack_count": attack_count})

        # R9Q step-level telemetry checks
        if records and condition == "R9Q_DETECTOR_T10":
            sg_enabled_count = sum(1 for r in records if r.get("detector_susceptibility_gate_enabled") is True)
            trigger_started_count = sum(1 for r in records if r.get("detector_trigger_started") is True)
            effective_valid_count = sum(1 for r in records if r.get("detector_effective_valid") is True)
            attack_steps = [r for r in records if r.get("attack_delivered")]
            attack_indices = [r.get("attack_index") for r in attack_steps if r.get("attack_index") is not None]

            if sg_enabled_count > 0:
                failures.append({"code": "SUSCEPTIBILITY_GATE_ENABLED_TRUE", "cell": str(cell), "count": sg_enabled_count})
            if trigger_started_count > 1:
                failures.append({"code": "MULTI_TRIGGER", "cell": str(cell), "trigger_started_count": trigger_started_count})
            if attack_indices and attack_indices != list(range(len(attack_indices))):
                failures.append({"code": "ATTACK_INDEX_NOT_SEQUENTIAL", "cell": str(cell), "attack_indices": attack_indices})
            if attack_count > 0 and effective_valid_count == 0:
                failures.append({"code": "TRIGGER_WITHOUT_EFFECTIVE_VALID", "cell": str(cell)})
        trigger_step = metadata.get("first_attack_step")
        if condition == "R9Q_DETECTOR_T10":
            triggers.append({"parent_key": row["parent_key"], "suite": row["suite"], "triggered": trigger_step is not None, "trigger_step": trigger_step})
        outcome = {
            "parent_key": row["parent_key"], "suite": row["suite"], "condition": condition,
            "task_index": row.get("task_index"), "state_id": row.get("state_id"),
            "runtime_valid": metadata.get("runtime_valid"), "success": metadata.get("success"),
            "step_count": len(records), "attack_delivery_count": attack_count,
            "first_attack_step": trigger_step, "detector_trigger_step": metadata.get("detector_trigger_step"),
            "cell": str(cell),
        }
        ledgers.append(outcome)
        per_suite_condition[(str(row["suite"]), condition)].append(outcome)
        if metadata.get("runtime_valid") is not True:
            failures.append({"code": "RUNTIME_INVALID", "cell": str(cell)})

    parent_conditions: defaultdict[str, set[str]] = defaultdict(set)
    for row in rows:
        parent_conditions[str(row["parent_key"])].add(str(row["condition"]))
    for parent, conditions in parent_conditions.items():
        if conditions != set(CONDITIONS):
            failures.append({"code": "PAIRED_CONDITION_CLOSURE", "parent_key": parent, "conditions": sorted(conditions)})

    r9q_triggers = [row["triggered"] for row in triggers]
    if args.mode == "canary" and (not r9q_triggers or all(r9q_triggers) or not any(r9q_triggers)):
        failures.append({"code": "R9Q_ALWAYS_OR_NEVER_TRIGGER", "triggered": sum(bool(value) for value in r9q_triggers), "count": len(r9q_triggers)})

    summary_rows: list[dict[str, Any]] = []
    for (suite, condition), entries in sorted(per_suite_condition.items()):
        valid_entries = [entry for entry in entries if entry["runtime_valid"] is True]
        triggered = [entry for entry in valid_entries if entry["first_attack_step"] is not None]
        delays = [int(entry["first_attack_step"]) for entry in triggered]
        summary_rows.append({
            "suite": suite, "condition": condition, "cells": len(entries),
            "runtime_valid": len(valid_entries),
            "successes": sum(bool(entry["success"]) for entry in valid_entries),
            "success_rate": (sum(bool(entry["success"]) for entry in valid_entries) / len(valid_entries)) if valid_entries else None,
            "triggered": len(triggered), "trigger_rate": (len(triggered) / len(valid_entries)) if valid_entries else None,
            "no_trigger": sum(entry["first_attack_step"] is None for entry in valid_entries),
            "full_t10": sum(entry["attack_delivery_count"] == 10 for entry in valid_entries if condition != "CLEAN"),
            "median_trigger_step": statistics.median(delays) if delays else None,
        })

    output.mkdir(parents=True)
    report = {
        "status": "PASS_C2G_R9Q_ATTACK_RUN_AUDITED" if not failures else "HOLD_C2G_R9Q_ATTACK_RUN_AUDIT",
        "mode": args.mode,
        "manifest": str(manifest),
        "manifest_sha256": sha256_file(manifest),
        "run_root": str(run_root),
        "expected_git_commit": args.expected_git_commit,
        "expected_cells": args.expected_cells,
        "observed_cells": len(ledgers),
        "failure_count": len(failures),
        "failures_by_code": dict(Counter(str(row["code"]) for row in failures)),
        "bundle_verification": bundle_verification,
        "r9q_triggered": sum(bool(value) for value in r9q_triggers),
        "r9q_trigger_count": len(r9q_triggers),
        "r9q_trigger_rate": (sum(bool(value) for value in r9q_triggers) / len(r9q_triggers)) if r9q_triggers else None,
        "attack_outcomes_read": True,
        "attack_outcomes_used_for_selection": False,
        "summary_by_suite_condition": summary_rows,
        "runtime_boundaries": {
            "openvla_models_loaded": "reported_by_worker",
            "libero_rollouts": "reported_by_worker",
            "training_epochs": 0,
        },
    }
    write_json(output / "r9q_attack_audit_report.json", report)
    with (output / "paired_cell_ledger.jsonl").open("w", encoding="utf-8") as handle:
        for entry in ledgers:
            handle.write(json.dumps(entry, sort_keys=True) + "\n")
    with (output / "trigger_ledger.jsonl").open("w", encoding="utf-8") as handle:
        for entry in triggers:
            handle.write(json.dumps(entry, sort_keys=True) + "\n")
    with (output / "failure_ledger.jsonl").open("w", encoding="utf-8") as handle:
        for entry in failures:
            handle.write(json.dumps(entry, sort_keys=True) + "\n")
    with (output / "paired_parent_ledger.jsonl").open("w", encoding="utf-8") as handle:
        for parent, conditions in sorted(parent_conditions.items()):
            handle.write(json.dumps({"parent_key": parent, "conditions": sorted(conditions), "complete": set(conditions) == set(CONDITIONS)}, sort_keys=True) + "\n")
    with (output / "cell_audit_ledger.jsonl").open("w", encoding="utf-8") as handle:
        for entry in ledgers:
            handle.write(json.dumps(entry, sort_keys=True) + "\n")
    with (output / "table_r9q_attack_preview.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = list(summary_rows[0]) if summary_rows else ["suite", "condition"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)
    write_json(output / "table_r9q_attack_preview.json", {"title": "R9Q Correct Detector - Partial-L10 Attack Main-Table Preview", "rows": summary_rows, "status": report["status"]})
    (output / "table_r9q_attack_preview.md").write_text(
        "# R9Q Correct Detector - Partial-L10 Attack Main-Table Preview\n\n"
        "PREVIEW\nPartial-L10 training\nNo canonical TEST claim\nNot final paper table\n\n"
        + "\n".join("| {suite} | {condition} | {success_rate} | {trigger_rate} |".format(**row) for row in summary_rows),
        encoding="utf-8",
    )
    files = sorted(path for path in output.rglob("*") if path.is_file())
    sums = output / "SHA256SUMS"
    sums.write_text("".join(f"{sha256_file(path)}  {path.relative_to(output).as_posix()}\n" for path in files), encoding="utf-8")
    (output / "SHA256SUMS.sha256").write_text(f"{sha256_file(sums)}  SHA256SUMS\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
