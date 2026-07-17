"""Normalize frozen V3 ledger rows into recovery input records."""

from __future__ import annotations

import csv
import json
import os
import shutil
import uuid
from pathlib import Path
from typing import Any, Iterable, Mapping

from .official_v3_contract import SUITES, canonical_key, json_sha, sha256_file

NORMALIZED_COMPLETION_FIELDS = [
    "canonical_parent_key", "selected_result", "quarantined",
    "artifact_recursive_sha256", "completion_record_sha256",
    "completion_source_path", "completion_source_sha256", "completion_row_index",
    "start_uuid", "worker_start_manifest_sha256", "lease_uuid", "lease_epoch",
    "fencing_token", "result_status", "task_success",
]


class ProvenanceSourceViolation(ValueError):
    pass


def _text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _key(row: Mapping[str, Any]) -> str:
    value = _text(row.get("canonical_parent_key"))
    parts = value.split("/")
    if len(parts) != 3 or parts[0] not in SUITES:
        raise ProvenanceSourceViolation(f"invalid canonical identity: {value!r}")
    try:
        parsed = canonical_key(parts[0], int(parts[1].split("_")[1]), int(parts[2].split("_")[1]))
    except (IndexError, TypeError, ValueError) as exc:
        raise ProvenanceSourceViolation(f"invalid canonical identity: {value!r}") from exc
    if parsed != value:
        raise ProvenanceSourceViolation(f"canonical identity columns are inconsistent: {value!r}")
    return value


def normalize_final_ledger_rows(
    ledger_rows: Iterable[Mapping[str, Any]],
    artifact_rows: Iterable[Mapping[str, Any]],
    *,
    ledger_source_path: str = "",
    ledger_source_sha256: str = "",
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    """Join the immutable final ledger with the artifact inventory.

    The ledger's worker_start_uuid is an exact direct-start candidate only when
    the corresponding worker-start manifest is independently present and
    validated by the recovery census.  This function only normalizes records;
    it never marks formal eligibility.
    """
    artifacts: dict[str, Mapping[str, Any]] = {}
    for row in artifact_rows:
        key = _key(row)
        if key in artifacts:
            raise ProvenanceSourceViolation(f"duplicate artifact identity: {key}")
        artifacts[key] = row

    output: list[dict[str, str]] = []
    seen: set[str] = set()
    allowed_results = {"PASS", "TASK_FAILURE"}
    for index, raw in enumerate(ledger_rows):
        key = _key(raw)
        if key in seen:
            raise ProvenanceSourceViolation(f"duplicate ledger identity: {key}")
        seen.add(key)
        artifact = artifacts.get(key)
        if artifact is None:
            raise ProvenanceSourceViolation(f"ledger identity missing from artifact inventory: {key}")
        ledger_root = _text(raw.get("artifact_root"))
        artifact_root = _text(artifact.get("artifact_root"))
        if ledger_root and artifact_root and ledger_root != artifact_root:
            raise ProvenanceSourceViolation(f"artifact root mismatch: {key}")
        result_status = _text(raw.get("result_status") or raw.get("status"))
        if result_status not in allowed_results:
            raise ProvenanceSourceViolation(f"unsupported final result status for {key}: {result_status!r}")
        canonical_row = {str(name): _text(value) for name, value in raw.items()}
        output.append({
            "canonical_parent_key": key,
            "selected_result": "true",
            "quarantined": "false",
            "artifact_recursive_sha256": _text(artifact.get("artifact_recursive_sha256")),
            "completion_record_sha256": json_sha(canonical_row),
            "completion_source_path": ledger_source_path,
            "completion_source_sha256": ledger_source_sha256,
            "completion_row_index": str(index),
            "start_uuid": _text(raw.get("worker_start_uuid")),
            "worker_start_manifest_sha256": _text(raw.get("worker_start_manifest_sha256")),
            "lease_uuid": "",
            "lease_epoch": "",
            "fencing_token": "",
            "result_status": result_status,
            "task_success": "true" if result_status == "PASS" else "false",
        })
    missing = sorted(set(artifacts) - seen)
    if missing:
        raise ProvenanceSourceViolation(f"artifact identities missing from final ledger: {len(missing)}")
    summary = {
        "schema": "OFFICIAL_V3_PROVENANCE_NORMALIZED_SOURCES_V1",
        "status": "DISCOVERY_ONLY",
        "formal_decision_allowed": False,
        "ledger_row_count": len(output),
        "artifact_row_count": len(artifacts),
        "direct_start_uuid_row_count": sum(bool(row["start_uuid"]) for row in output),
        "task_success_count": sum(row["task_success"] == "true" for row in output),
        "task_failure_count": sum(row["task_success"] == "false" for row in output),
        "formal_training_authorized": False,
        "formal_attack_authorized": False,
    }
    return output, summary


def write_normalized_completion_bundle(
    rows: list[Mapping[str, Any]],
    summary: Mapping[str, Any],
    output_root: Path,
) -> None:
    if output_root.exists():
        raise ProvenanceSourceViolation(f"refusing to overwrite source bundle: {output_root}")
    staging = output_root.with_name(f".{output_root.name}.{uuid.uuid4().hex}.staging")
    staging.mkdir(parents=True)
    try:
        with (staging / "normalized_completion_rows.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=NORMALIZED_COMPLETION_FIELDS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        (staging / "summary.json").write_text(json.dumps(dict(summary), sort_keys=True, indent=2) + "\n", encoding="utf-8")
        files = sorted(path for path in staging.iterdir() if path.is_file())
        (staging / "SHA256SUMS").write_text("\n".join(f"{sha256_file(path)}  {path.name}" for path in files) + "\n", encoding="utf-8")
        sums = staging / "SHA256SUMS"
        (staging / "SHA256SUMS.sha256").write_text(f"{sha256_file(sums)}  SHA256SUMS\n", encoding="utf-8")
        os.replace(staging, output_root)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


__all__ = [
    "NORMALIZED_COMPLETION_FIELDS", "ProvenanceSourceViolation",
    "normalize_final_ledger_rows", "write_normalized_completion_bundle",
]
