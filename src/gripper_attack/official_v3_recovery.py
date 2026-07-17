"""Fail-closed recovery of artifact provenance from normalized history tables."""

from __future__ import annotations

import csv
import json
import os
import shutil
import uuid
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

from .official_v3_contract import SUITES, canonical_key, sha256_file


RECOVERY_SCHEMA = "OFFICIAL_V3_PROVENANCE_RECOVERY_CENSUS_V1"
EXACT_DIRECT = "EXACT_DIRECT_START_UUID"
EXACT_LEASE = "EXACT_LEASE_CHAIN"
WEAK = "UNIQUE_WEAK_DERIVATION"
AMBIGUOUS = "AMBIGUOUS"
MISSING = "MISSING"
CONTRADICTORY = "CONTRADICTORY"
FORMAL_METHODS = {EXACT_DIRECT, EXACT_LEASE}

RECOVERY_FIELDS = [
    "canonical_parent_key", "artifact_root", "artifact_recursive_sha256",
    "candidate_start_uuid_list", "candidate_manifest_sha_list", "candidate_count",
    "direct_binding_fields", "lease_chain_fields", "weak_matching_fields",
    "recovery_status", "recovery_method", "recovery_reason", "formal_eligible",
    "start_uuid", "worker_start_manifest_sha256", "worker_start_manifest_sidecar_sha256",
    "lease_uuid", "lease_epoch", "fencing_token", "worker_start_gate_record_sha256",
    "assignment_record_sha256", "completion_record_sha256", "completion_artifact_sha256",
    "snapshot_root_sha256", "recovery_census_sha256",
]


class RecoveryContractViolation(ValueError):
    pass


def _text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _bool(value: Any) -> bool:
    return value is True or _text(value).lower() in {"1", "true", "yes", "pass", "selected"}


def _key(row: Mapping[str, Any]) -> str:
    value = _text(row.get("canonical_parent_key"))
    parts = value.split("/")
    if len(parts) != 3 or parts[0] not in SUITES:
        raise RecoveryContractViolation(f"invalid canonical identity: {value!r}")
    try:
        parsed = canonical_key(parts[0], int(parts[1].split("_")[1]), int(parts[2].split("_")[1]))
    except (IndexError, TypeError, ValueError) as exc:
        raise RecoveryContractViolation(f"invalid canonical identity: {value!r}") from exc
    if parsed != value:
        raise RecoveryContractViolation(f"canonical identity columns are inconsistent: {value!r}")
    return value


def _unique(rows: Iterable[Mapping[str, Any]], field: str) -> list[str]:
    return sorted({_text(row.get(field)) for row in rows if _text(row.get(field))})


def _row_for_start(worker_rows: list[Mapping[str, Any]], start_uuid: str) -> Mapping[str, Any] | None:
    matches = [row for row in worker_rows if _text(row.get("start_uuid")) == start_uuid]
    if len(matches) > 1:
        raise RecoveryContractViolation(f"duplicate worker start UUID: {start_uuid}")
    return matches[0] if matches else None


def _artifact_rows_by_key(rows: Iterable[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        key = _key(row)
        if key in result:
            raise RecoveryContractViolation(f"duplicate artifact identity: {key}")
        result[key] = row
    return result


def _candidate_workers(artifact: Mapping[str, Any], workers: list[Mapping[str, Any]]) -> tuple[list[Mapping[str, Any]], list[str]]:
    fields = (("worker_id", "collector_worker_id"), ("pid", "collector_pid"), ("collector_head", "collector_git_head"))
    candidates: list[Mapping[str, Any]] = []
    used: list[str] = []
    for row in workers:
        row_matches = True
        row_used: list[str] = []
        for left, right in fields:
            artifact_value = _text(artifact.get(left) or artifact.get(right))
            worker_value = _text(row.get(left) or row.get(right))
            if not artifact_value or not worker_value:
                continue
            if artifact_value != worker_value:
                row_matches = False
                break
            row_used.append(f"{left}={artifact_value}")
        if row_matches and row_used:
            candidates.append(row)
            used.extend(row_used)
    return candidates, sorted(set(used))


def _base(key: str, artifact: Mapping[str, Any], snapshot_root_sha256: str) -> dict[str, Any]:
    return {
        "canonical_parent_key": key,
        "artifact_root": _text(artifact.get("artifact_root")),
        "artifact_recursive_sha256": _text(artifact.get("artifact_recursive_sha256")),
        "candidate_start_uuid_list": [],
        "candidate_manifest_sha_list": [],
        "candidate_count": 0,
        "direct_binding_fields": [],
        "lease_chain_fields": [],
        "weak_matching_fields": [],
        "recovery_status": MISSING,
        "recovery_method": "",
        "recovery_reason": "",
        "formal_eligible": False,
        "start_uuid": "",
        "worker_start_manifest_sha256": "",
        "worker_start_manifest_sidecar_sha256": "",
        "lease_uuid": "",
        "lease_epoch": "",
        "fencing_token": "",
        "worker_start_gate_record_sha256": "",
        "assignment_record_sha256": "",
        "completion_record_sha256": "",
        "completion_artifact_sha256": "",
        "snapshot_root_sha256": snapshot_root_sha256,
        "recovery_census_sha256": "",
    }


def build_recovery_rows(
    artifact_rows: Iterable[Mapping[str, Any]],
    worker_rows: Iterable[Mapping[str, Any]],
    lease_rows: Iterable[Mapping[str, Any]],
    completion_rows: Iterable[Mapping[str, Any]],
    *,
    snapshot_root_sha256: str = "",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    artifacts = _artifact_rows_by_key(artifact_rows)
    workers = [dict(row) for row in worker_rows]
    leases = [dict(row) for row in lease_rows]
    completions = [dict(row) for row in completion_rows]
    completion_by_key: dict[str, list[dict[str, Any]]] = defaultdict(list)
    lease_by_key: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in completions:
        completion_by_key[_key(row)].append(row)
    for row in leases:
        lease_by_key[_key(row)].append(row)

    output: list[dict[str, Any]] = []
    for key in sorted(artifacts):
        artifact = artifacts[key]
        result = _base(key, artifact, snapshot_root_sha256)
        records = completion_by_key.get(key, [])
        selected = [row for row in records if _bool(row.get("selected_result")) and not _bool(row.get("quarantined"))]
        if len(selected) > 1:
            result.update(recovery_status=AMBIGUOUS, recovery_reason="MULTIPLE_SELECTED_COMPLETIONS")
        elif records and not selected:
            result.update(recovery_status=CONTRADICTORY, recovery_reason="NO_SELECTED_COMPLETION")
        elif selected:
            completion = selected[0]
            result["completion_record_sha256"] = _text(completion.get("completion_record_sha256"))
            result["completion_artifact_sha256"] = _text(completion.get("artifact_recursive_sha256"))
            if result["completion_artifact_sha256"] != result["artifact_recursive_sha256"]:
                result.update(recovery_status=CONTRADICTORY, recovery_reason="COMPLETION_ARTIFACT_SHA_MISMATCH")
            else:
                direct_start = _text(completion.get("start_uuid") or artifact.get("start_uuid") or artifact.get("worker_start_uuid"))
                if direct_start:
                    worker = _row_for_start(workers, direct_start)
                    result["candidate_start_uuid_list"] = [direct_start]
                    if worker is None:
                        result.update(recovery_status=MISSING, recovery_reason="DIRECT_START_UUID_NOT_IN_MANIFEST_INVENTORY")
                    else:
                        result.update(
                            recovery_status=EXACT_DIRECT,
                            recovery_method=EXACT_DIRECT,
                            formal_eligible=True,
                            start_uuid=direct_start,
                            worker_start_manifest_sha256=_text(worker.get("manifest_sha256")),
                            worker_start_manifest_sidecar_sha256=_text(worker.get("manifest_sidecar_sha256")),
                            worker_start_gate_record_sha256=_text(worker.get("worker_start_gate_ack_sha256") or worker.get("worker_start_gate_ready_sha256")),
                            direct_binding_fields=["canonical_parent_key", "start_uuid", "artifact_recursive_sha256"],
                        )
                else:
                    lease_uuid = _text(completion.get("lease_uuid"))
                    lease_epoch = _text(completion.get("lease_epoch"))
                    fencing = _text(completion.get("fencing_token"))
                    chain = [row for row in lease_by_key.get(key, []) if _text(row.get("lease_uuid")) == lease_uuid and _text(row.get("lease_epoch")) == lease_epoch and _text(row.get("fencing_token")) == fencing]
                    if len(chain) > 1:
                        result.update(recovery_status=AMBIGUOUS, recovery_reason="MULTIPLE_EXACT_LEASE_ROWS")
                    elif len(chain) == 1:
                        lease = chain[0]
                        start = _text(lease.get("start_uuid"))
                        worker = _row_for_start(workers, start)
                        result.update(lease_uuid=lease_uuid, lease_epoch=lease_epoch, fencing_token=fencing, assignment_record_sha256=_text(lease.get("assignment_record_sha256")), lease_chain_fields=["artifact_recursive_sha256", "lease_uuid", "lease_epoch", "fencing_token", "start_uuid"])
                        if worker is None or not start:
                            result.update(recovery_status=MISSING, recovery_reason="LEASE_CHAIN_START_UUID_NOT_IN_MANIFEST_INVENTORY")
                        else:
                            result.update(recovery_status=EXACT_LEASE, recovery_method=EXACT_LEASE, formal_eligible=True, start_uuid=start, worker_start_manifest_sha256=_text(worker.get("manifest_sha256")), worker_start_manifest_sidecar_sha256=_text(worker.get("manifest_sidecar_sha256")), worker_start_gate_record_sha256=_text(worker.get("worker_start_gate_ack_sha256") or worker.get("worker_start_gate_ready_sha256")))
                    else:
                        result.update(recovery_status=MISSING, recovery_reason="LEASE_CHAIN_NOT_FOUND")
        if result["recovery_status"] == MISSING and not records:
            candidates, fields = _candidate_workers(artifact, workers)
            result["candidate_start_uuid_list"] = _unique(candidates, "start_uuid")
            result["candidate_manifest_sha_list"] = _unique(candidates, "manifest_sha256")
            result["candidate_count"] = len(candidates)
            result["weak_matching_fields"] = fields
            if len(candidates) == 1 and fields:
                result.update(recovery_status=WEAK, recovery_method=WEAK, recovery_reason="UNIQUE_PID_WORKER_OR_HEAD_MATCH_IS_NOT_FORMAL", formal_eligible=False)
            elif len(candidates) > 1:
                result.update(recovery_status=AMBIGUOUS, recovery_reason="MULTIPLE_WEAK_WORKER_CANDIDATES")
            else:
                result["recovery_reason"] = "NO_COMPLETION_OR_EXACT_LEASE_CHAIN"
        if not result["candidate_count"]:
            result["candidate_count"] = len(result["candidate_start_uuid_list"])
        output.append(result)

    counts = Counter(row["recovery_status"] for row in output)
    fit = [row for row in output if int(row["canonical_parent_key"].split("state_")[-1]) < 20]
    fit_counts = Counter(row["recovery_status"] for row in fit)
    summary = {
        "schema": RECOVERY_SCHEMA,
        "status": "DISCOVERY_ONLY",
        "formal_decision_allowed": False,
        "identity_count": len(output),
        "unique_identity_count": len(output),
        "global_recovery_counts": dict(sorted(counts.items())),
        "fit_recovery_counts": dict(sorted(fit_counts.items())),
        "formal_fit_exact_count": sum(row["formal_eligible"] for row in fit),
        "formal_training_authorized": False,
        "formal_attack_authorized": False,
    }
    return output, summary


def _write_csv(path: Path, rows: list[Mapping[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())


def write_recovery_bundle(rows: list[Mapping[str, Any]], summary: Mapping[str, Any], output_root: Path, *, input_binding: Mapping[str, Any] | None = None) -> None:
    if output_root.exists():
        raise RecoveryContractViolation(f"refusing to overwrite recovery root: {output_root}")
    staging = output_root.with_name(f".{output_root.name}.{uuid.uuid4().hex}.staging")
    staging.mkdir(parents=True)
    try:
        _write_csv(staging / "recovery_rows.csv", rows, RECOVERY_FIELDS)
        groups = {
            "exact_rows.csv": lambda row: row["recovery_status"] in FORMAL_METHODS,
            "weak_rows.csv": lambda row: row["recovery_status"] == WEAK,
            "ambiguous_rows.csv": lambda row: row["recovery_status"] == AMBIGUOUS,
            "missing_rows.csv": lambda row: row["recovery_status"] == MISSING,
            "contradictory_rows.csv": lambda row: row["recovery_status"] == CONTRADICTORY,
            "fit_unresolved_rows.csv": lambda row: int(row["canonical_parent_key"].split("state_")[-1]) < 20 and row["recovery_status"] not in FORMAL_METHODS,
        }
        for filename, predicate in groups.items():
            _write_csv(staging / filename, [row for row in rows if predicate(row)], RECOVERY_FIELDS)
        write_json = lambda path, value: (path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"))
        write_json(staging / "summary.json", dict(summary))
        write_json(staging / "input_binding.json", dict(input_binding or {}))
        files = sorted(path for path in staging.iterdir() if path.is_file())
        sums = "\n".join(f"{sha256_file(path)}  {path.name}" for path in files) + "\n"
        (staging / "SHA256SUMS").write_text(sums, encoding="utf-8")
        (staging / "SHA256SUMS.sha256").write_text(f"{sha256_file(staging / 'SHA256SUMS')}  SHA256SUMS\n", encoding="utf-8")
        os.replace(staging, output_root)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


__all__ = [
    "AMBIGUOUS", "CONTRADICTORY", "EXACT_DIRECT", "EXACT_LEASE", "MISSING", "RECOVERY_FIELDS",
    "RECOVERY_SCHEMA", "RecoveryContractViolation", "WEAK", "build_recovery_rows", "write_recovery_bundle",
]
