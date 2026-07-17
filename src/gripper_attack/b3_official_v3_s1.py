"""Preparation-only Official V3 FIT census and S1 materialization.

This module is deliberately independent from the legacy V2 materializer.  It
may read CLEAN source evidence only after an exact formal FIT registry passes;
it never authorizes training or attack.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import shutil
import uuid
from collections import Counter
from pathlib import Path
from typing import Any

from .b3_retention import RetentionConfig, rebuild_retention_features
from .official_v3_contract import SUITES, audit_artifact, canonical_key, load_contract


class V3S1ContractViolation(ValueError):
    """Raised when the V3 S1 input or output contract is not closed."""


HEADS = ("grasp_support", "retention_active", "retention_continuation_t10", "release_imminent")
MASKS = {
    "grasp_support": "grasp_support_mask",
    "retention_active": "retention_active_mask",
    "retention_continuation_t10": "retention_unknown_mask",
    "release_imminent": "release_imminent_mask",
}
REGISTRY_REQUIRED = {
    "canonical_parent_key", "suite", "task_idx", "state_id", "split",
    "selected_artifact_root", "selected_artifact_recursive_sha256",
    "artifact_audit_sha256", "formal_eligible", "formal_selected",
    "provenance_class",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise V3S1ContractViolation(f"invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise V3S1ContractViolation(f"JSON object required: {path}")
    return value


def _load_json_value(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise V3S1ContractViolation(f"invalid JSON: {path}") from exc


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise V3S1ContractViolation(f"cannot read JSONL: {path}") from exc
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except ValueError as exc:
            raise V3S1ContractViolation(f"invalid JSONL at {path}:{line_no}") from exc
        if not isinstance(value, dict):
            raise V3S1ContractViolation(f"JSON object required at {path}:{line_no}")
        rows.append(value)
    return rows


def _read_csv(path: Path) -> list[dict[str, str]]:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))
    except OSError as exc:
        raise V3S1ContractViolation(f"cannot read CSV: {path}") from exc


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return isinstance(value, str) and value.strip().lower() == "true"


def _int(value: Any, field: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise V3S1ContractViolation(f"invalid integer field {field}: {value!r}") from exc


def _finite_vector(value: Any, length: int) -> bool:
    return isinstance(value, list) and len(value) == length and all(
        isinstance(item, (int, float)) and not isinstance(item, bool) and math.isfinite(float(item))
        for item in value
    )


def _same(left: Any, right: Any, tolerance: float = 1e-6) -> bool:
    if isinstance(left, (int, float)) and not isinstance(left, bool) and isinstance(right, (int, float)) and not isinstance(right, bool):
        return math.isfinite(float(left)) and math.isfinite(float(right)) and abs(float(left) - float(right)) <= tolerance
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(_same(a, b, tolerance) for a, b in zip(left, right))
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(_same(left[key], right[key], tolerance) for key in left)
    return left == right


def load_s1_protocol(path: Path) -> tuple[dict[str, Any], RetentionConfig]:
    payload = _load_json(path)
    if payload.get("schema") != "B3_OFFICIAL_V3_S1_PROTOCOL_V1" or payload.get("status") != "PREPARATION_ONLY":
        raise V3S1ContractViolation("unexpected or executable V3 S1 protocol")
    if payload.get("formal_training_authorized") is not False or payload.get("formal_attack_authorized") is not False:
        raise V3S1ContractViolation("V3 S1 preparation cannot authorize training or attack")
    if payload.get("expected_fit_identities") != 800 or payload.get("expected_suite_identities") != 200 or payload.get("expected_task_identities") != 20:
        raise V3S1ContractViolation("V3 S1 exact FIT quotas are not frozen")
    if payload.get("fit_split") != "FIT_TRAIN" or payload.get("student_records", {}).get("primary") != "features_25d":
        raise V3S1ContractViolation("V3 S1 split/student boundary is not frozen")
    if tuple(payload.get("suites", ())) != SUITES:
        raise V3S1ContractViolation("V3 S1 suite order is not frozen")
    if tuple(payload.get("teacher_heads", ())) != HEADS or payload.get("teacher_masks") != MASKS:
        raise V3S1ContractViolation("V3 S1 Teacher head/mask schema is not frozen")
    forbidden = set(payload.get("student_records", {}).get("forbidden_fields", ()))
    required_forbidden = {
        "event_id", "event_ordinal", "retention_continuation_t10", "retention_unknown_mask",
        "release_imminent", "object_state", "mujoco_contact_pairs", "attack_outcome",
    }
    if not required_forbidden.issubset(forbidden):
        raise V3S1ContractViolation("V3 S1 Student/Teacher physical boundary is incomplete")
    invariants = payload.get("invariants", {})
    if invariants.get("unknown_is_negative") is not False or invariants.get("event_intervals_must_not_overlap") is not True:
        raise V3S1ContractViolation("V3 S1 Teacher invariant contract is unsafe")
    params = payload.get("retention_teacher_parameters")
    if not isinstance(params, dict):
        raise V3S1ContractViolation("retention teacher parameters are missing")
    try:
        config = RetentionConfig(**params)
    except (TypeError, ValueError) as exc:
        raise V3S1ContractViolation("invalid retention teacher parameters") from exc
    if config.t10 != 10:
        raise V3S1ContractViolation("V3 S1 requires exact T10 labels")
    return payload, config


def load_formal_fit_registry(registry_csv: Path, summary_json: Path) -> list[dict[str, str]]:
    rows = _read_csv(registry_csv)
    summary = _load_json(summary_json)
    required_summary = {
        "formal_fit_ready": True,
        "formal_selected_count": 800,
        "full_artifact_audit_pass_count": 800,
        "unresolved_provenance_count": 0,
        "unfinished_remediation_count": 0,
        "stale_recovery_unresolved_count": 0,
        "duplicate_selection_count": 0,
    }
    for name, expected in required_summary.items():
        if summary.get(name) != expected:
            raise V3S1ContractViolation(f"formal FIT registry gate failed: {name}={summary.get(name)!r}")
    for name in ("registry_sha256", "stale_recovery_summary_sha256"):
        value = summary.get(name)
        if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdefABCDEF" for char in value):
            raise V3S1ContractViolation(f"formal FIT registry input closure missing: {name}")
    if summary["registry_sha256"].lower() != sha256_file(registry_csv):
        raise V3S1ContractViolation("formal FIT registry CSV SHA does not match its summary")
    if summary.get("identity_count") != 2000 or summary.get("unique_identity_count") != 2000:
        raise V3S1ContractViolation("formal FIT registry is not bound to the complete 2000-identity universe")
    if summary.get("formal_training_authorized") is not False or summary.get("formal_attack_authorized") is not False:
        raise V3S1ContractViolation("formal registry contains an authorization flag")
    if len(rows) != 800:
        raise V3S1ContractViolation(f"formal FIT registry must contain 800 rows, got {len(rows)}")
    seen: set[str] = set()
    suite_counts: Counter[str] = Counter()
    task_counts: Counter[tuple[str, int]] = Counter()
    for row in rows:
        missing = sorted(REGISTRY_REQUIRED - set(row))
        if missing:
            raise V3S1ContractViolation(f"registry row missing fields: {missing}")
        suite = row["suite"]
        task = _int(row["task_idx"], "task_idx")
        state = _int(row["state_id"], "state_id")
        key = row["canonical_parent_key"]
        if key != canonical_key(suite, task, state) or suite not in SUITES or not 0 <= task < 10 or not 0 <= state < 20:
            raise V3S1ContractViolation(f"registry identity mismatch: {key}")
        if row["split"] != "FIT_TRAIN" or row["canonical_parent_key"] in seen:
            raise V3S1ContractViolation(f"registry split/duplicate violation: {key}")
        if not _bool(row["formal_eligible"]) or not _bool(row["formal_selected"]):
            raise V3S1ContractViolation(f"registry row is not formally selected: {key}")
        for name in ("selected_artifact_recursive_sha256", "artifact_audit_sha256"):
            if len(row[name]) != 64 or any(char not in "0123456789abcdefABCDEF" for char in row[name]):
                raise V3S1ContractViolation(f"registry SHA missing/invalid: {key}:{name}")
        if row["provenance_class"] not in {"A_CURRENT_HEAD_CLEAN_START_VERIFIED", "B_PREVIOUS_HEAD_EQUIVALENT"}:
            raise V3S1ContractViolation(f"registry provenance is not eligible: {key}")
        seen.add(key)
        suite_counts[suite] += 1
        task_counts[(suite, task)] += 1
    if set(suite_counts) != set(SUITES) or any(count != 200 for count in suite_counts.values()):
        raise V3S1ContractViolation(f"registry suite quota mismatch: {dict(suite_counts)}")
    if len(task_counts) != 40 or any(count != 20 for count in task_counts.values()):
        raise V3S1ContractViolation("registry task quota mismatch")
    return sorted(rows, key=lambda row: row["canonical_parent_key"])


def build_fit_census(registry_csv: Path, summary_json: Path) -> tuple[list[dict[str, str]], dict[str, Any]]:
    rows = load_formal_fit_registry(registry_csv, summary_json)
    census_rows = [
        {
            "canonical_parent_key": row["canonical_parent_key"],
            "suite": row["suite"],
            "task_idx": row["task_idx"],
            "state_id": row["state_id"],
            "split": row["split"],
            "selected_artifact_root": row["selected_artifact_root"],
            "selected_artifact_recursive_sha256": row["selected_artifact_recursive_sha256"],
            "artifact_audit_sha256": row["artifact_audit_sha256"],
            "provenance_class": row["provenance_class"],
        }
        for row in rows
    ]
    return census_rows, {
        "schema": "B3_OFFICIAL_V3_FIT_CENSUS_SUMMARY_V1",
        "status": "PASS",
        "registry_csv_sha256": sha256_file(registry_csv),
        "registry_summary_sha256": sha256_file(summary_json),
        "identity_count": len(census_rows),
        "formal_training_authorized": False,
        "formal_attack_authorized": False,
    }


def _strict_join(source_root: Path, meta: dict[str, Any]) -> list[dict[str, Any]]:
    streams = {name: _load_jsonl(source_root / filename) for name, filename in {
        "step": "step_records.jsonl",
        "policy": "policy_intent_records.jsonl",
        "sidecar": "privileged_teacher_sidecar.jsonl",
    }.items()}
    indexed: dict[str, dict[int, dict[str, Any]]] = {}
    lengths = {name: len(rows) for name, rows in streams.items()}
    if not lengths or len(set(lengths.values())) != 1 or not next(iter(lengths.values())):
        raise V3S1ContractViolation(f"source stream length mismatch: {lengths}")
    count = next(iter(lengths.values()))
    for name, rows in streams.items():
        steps = [_int(row.get("step", index), "step") for index, row in enumerate(rows)]
        if steps != list(range(count)):
            raise V3S1ContractViolation(f"source stream is not contiguous: {name}")
        indexed[name] = {step: row for step, row in zip(steps, rows)}
    merged: list[dict[str, Any]] = []
    identity = {name: meta[name] for name in ("suite", "task_idx", "state_id", "canonical_parent_key")}
    for step in range(count):
        row: dict[str, Any] = dict(identity)
        row["step"] = step
        for source in ("step", "policy", "sidecar"):
            for key, value in indexed[source][step].items():
                if key in row and not _same(row[key], value):
                    raise V3S1ContractViolation(f"source stream join conflict: step={step}, field={key}")
                row[key] = value
        if not _finite_vector(row.get("features_25d"), 25) or not _finite_vector(row.get("clean_policy_intent_9d"), 9):
            raise V3S1ContractViolation(f"student source vector invalid at step {step}")
        merged.append(row)
    return merged


def _unknown_reason(index: int, rows: list[dict[str, Any]]) -> str:
    if len(rows) - index < 10:
        return "INSUFFICIENT_FUTURE_HORIZON"
    future = rows[index:index + 10]
    if any(item.get("valid") is not True or item.get("event_evidence_valid") is not True for item in future):
        return "MISSING_OR_INVALID_EVIDENCE"
    return "UNKNOWN_UNEXPLAINED"


def _audit_source(row: dict[str, str], contract: dict[str, Any]) -> dict[str, Any]:
    root = Path(row["selected_artifact_root"]).resolve()
    if row.get("provenance_class") == "B_PREVIOUS_HEAD_EQUIVALENT":
        report = audit_artifact(root, contract, equivalence_status="PASS")
    else:
        report = audit_artifact(root, contract)
    if report.get("status") != "PASS_FORMAL_CANDIDATE" or report.get("formal_eligible") is not True:
        raise V3S1ContractViolation(f"source audit is not formal PASS: {row['canonical_parent_key']}")
    if report.get("canonical_parent_key") != row["canonical_parent_key"]:
        raise V3S1ContractViolation(f"source audit identity mismatch: {row['canonical_parent_key']}")
    if report.get("artifact_recursive_sha256") != row["selected_artifact_recursive_sha256"]:
        raise V3S1ContractViolation(f"source artifact SHA mismatch: {row['canonical_parent_key']}")
    return report


def dry_run_episode(row: dict[str, str], contract_path: Path, protocol_path: Path) -> dict[str, Any]:
    contract = load_contract(contract_path)
    protocol, retention_config = load_s1_protocol(protocol_path)
    before = _audit_source(row, contract)
    source_root = Path(row["selected_artifact_root"]).resolve()
    meta = _load_json(source_root / "episode_metadata.json")
    merged = _strict_join(source_root, meta)
    rebuilt = rebuild_retention_features(merged, retention_config)
    after = _audit_source(row, contract)
    if before["artifact_recursive_sha256"] != after["artifact_recursive_sha256"]:
        raise V3S1ContractViolation(f"source changed during dry-run: {row['canonical_parent_key']}")
    return {
        "registry_row": row,
        "protocol": protocol,
        "source_before_sha256": before["artifact_recursive_sha256"],
        "source_after_sha256": after["artifact_recursive_sha256"],
        "merged": merged,
        "rebuilt": rebuilt,
        "source_contract": contract,
        "source_contract_sha256": sha256_file(contract_path),
        "step_count": len(merged),
        "feature_rebuilder_sha256": sha256_file(Path(rebuild_retention_features.__code__.co_filename).resolve()),
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _teacher_rows(prepared: dict[str, Any]) -> list[dict[str, Any]]:
    row_keys = {
        "step", "valid", "event_evidence_valid", "event_id", "event_ordinal",
        "event_start_step", "event_end_step", "event_close_onset", "event_release_onset",
        "released_event_id", "event_support", "event_qpos_stable", "event_opening_stable",
        "grasp_support", "grasp_support_mask", "retention_active", "retention_active_mask",
        "retention_continuation_t10", "retention_unknown_mask", "release_imminent",
        "release_imminent_mask", "retention_unknown_reason",
    }
    identity = {name: prepared["registry_row"][name] for name in ("suite", "task_idx", "state_id", "canonical_parent_key")}
    rows = []
    rebuilt_rows = prepared["rebuilt"]["rows"]
    for index, row in enumerate(rebuilt_rows):
        output = {
            "schema": "B3_OFFICIAL_V3_TEACHER_RECORD_V1",
            **identity,
            **{key: row.get(key) for key in row_keys if key in row},
        }
        output["step"] = index
        if output.get("retention_unknown_mask") is True:
            output["retention_unknown_reason"] = _unknown_reason(index, rebuilt_rows)
        output["source_artifact_sha256"] = prepared["source_before_sha256"]
        output["feature_rebuilder_sha256"] = prepared["feature_rebuilder_sha256"]
        rows.append(output)
    return rows


def audit_teacher_episode(rows: list[dict[str, Any]], events: list[dict[str, Any]], identity: str) -> dict[str, Any]:
    violations: list[str] = []
    if not isinstance(events, list):
        violations.append("EVENT_FILE_NOT_LIST")
        events = []
    if not rows or [row.get("step") for row in rows] != list(range(len(rows))):
        violations.append("NON_CONTIGUOUS_OR_EMPTY_STUDENT_TEACHER_STEPS")
    event_ids = sorted({row.get("event_id") for row in rows if isinstance(row.get("event_id"), int) and row.get("event_id", -1) >= 0})
    if event_ids and event_ids != list(range(event_ids[-1] + 1)):
        violations.append("EVENT_ORDINAL_NOT_CONTIGUOUS")
    event_file_ids = [event.get("event_id") for event in events]
    if any(not isinstance(event_id, int) for event_id in event_file_ids) or sorted(event_file_ids) != event_ids:
        violations.append("EVENT_STREAM_FILE_ID_MISMATCH")
    previous_end = -1
    for expected_id, event in enumerate(events):
        if event.get("event_id") != expected_id:
            violations.append("EVENT_FILE_ORDINAL_NOT_CONTIGUOUS")
        start, end = event.get("start_step"), event.get("end_step")
        if not isinstance(start, int) or not isinstance(end, int) or end < start or start <= previous_end:
            violations.append(f"EVENT_{expected_id}_BOUNDS_OR_OVERLAP")
        previous_end = end if isinstance(end, int) else previous_end
    for index, row in enumerate(rows):
        if row.get("canonical_parent_key") != identity:
            violations.append(f"STEP_{index}_IDENTITY_MISMATCH")
        for head in HEADS:
            mask = MASKS[head]
            if not isinstance(row.get(mask), bool):
                violations.append(f"STEP_{index}_{mask}_NOT_BOOLEAN")
                continue
            known = not row[mask] if head == "retention_continuation_t10" else row[mask]
            value = row.get(head)
            if known and value not in (False, True):
                violations.append(f"STEP_{index}_{head}_NOT_BINARY")
            if not known and value is not None:
                violations.append(f"STEP_{index}_{head}_UNKNOWN_NOT_NULL")
        if not isinstance(row.get("valid"), bool) or not isinstance(row.get("event_evidence_valid"), bool):
            violations.append(f"STEP_{index}_VALIDITY_NOT_BOOLEAN")
        if row.get("retention_unknown_mask") is True and row.get("retention_unknown_reason") not in {
            "INSUFFICIENT_FUTURE_HORIZON", "MISSING_OR_INVALID_EVIDENCE", "UNKNOWN_UNEXPLAINED"
        }:
            violations.append(f"STEP_{index}_UNKNOWN_REASON_INVALID")
        if row.get("retention_continuation_t10") is True:
            future = rows[index:index + 10]
            event_id = row.get("event_id")
            if len(future) != 10 or not isinstance(event_id, int) or event_id < 0:
                violations.append(f"STEP_{index}_T10_POSITIVE_BOUNDARY")
            elif any(item.get("event_id") != event_id or item.get("valid") is not True or item.get("event_evidence_valid") is not True for item in future):
                violations.append(f"STEP_{index}_T10_POSITIVE_FUTURE_CONTRACT")
            if row.get("release_imminent") is True:
                violations.append(f"STEP_{index}_T10_RELEASE_CONFLICT")
    for index in range(max(0, len(rows) - 9), len(rows)):
        if rows[index].get("retention_unknown_mask") is not True:
            violations.append(f"STEP_{index}_TAIL_NOT_UNKNOWN")
    return {
        "schema": "B3_OFFICIAL_V3_TEACHER_EPISODE_AUDIT_V1",
        "canonical_parent_key": identity,
        "status": "PASS" if not violations else "HOLD",
        "step_count": len(rows),
        "event_count": len(event_ids),
        "event_ordinal_counts": dict(Counter(row.get("event_ordinal", row.get("event_id")) for row in events)),
        "t10_positive_count": sum(row.get("retention_continuation_t10") is True for row in rows),
        "violations": sorted(set(violations)),
    }


def aggregate_teacher_audit(reports: list[dict[str, Any]], registry_rows: list[dict[str, str]]) -> dict[str, Any]:
    expected = {row["canonical_parent_key"] for row in registry_rows}
    actual = [row.get("canonical_parent_key") for row in reports]
    duplicates = sorted(key for key, count in Counter(actual).items() if count != 1)
    suite_counts = Counter(row.get("canonical_parent_key", "").split("/", 1)[0] for row in reports)
    violations = sorted({item for row in reports for item in row.get("violations", [])})
    status = "PASS" if set(actual) == expected and len(actual) == 800 and not duplicates and not violations and all(suite_counts[suite] == 200 for suite in SUITES) and all(row.get("status") == "PASS" for row in reports) else "HOLD"
    return {
        "schema": "B3_OFFICIAL_V3_TEACHER_AGGREGATE_AUDIT_V1",
        "status": status,
        "expected_identity_count": len(expected),
        "actual_identity_count": len(actual),
        "duplicate_identity_keys": duplicates,
        "suite_episode_counts": dict(sorted(suite_counts.items())),
        "invariant_violation_count": len(violations),
        "violations": violations,
        "teacher_labels_read": True,
        "formal_training_ready": False,
        "formal_attack_ready": False,
    }


def _atomic_write_text(path: Path, content: str) -> None:
    if path.exists():
        raise V3S1ContractViolation(f"refusing to overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8", newline="") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except OSError:
        if temporary.exists():
            temporary.unlink(missing_ok=True)
        raise


def _write_sidecar(path: Path) -> None:
    _atomic_write_text(path.with_name(path.name + ".sha256"), f"{sha256_file(path)}  {path.name}\n")


def _write_episode(prepared: dict[str, Any], output_root: Path) -> dict[str, Any]:
    if output_root.exists():
        raise V3S1ContractViolation(f"refusing to overwrite episode output: {output_root}")
    output_root.mkdir(parents=True)
    row = prepared["registry_row"]
    identity = {name: row[name] for name in ("suite", "task_idx", "state_id", "canonical_parent_key")}
    rebuilt_rows = prepared["rebuilt"]["rows"]
    contract = prepared["source_contract"]
    student = [{
        "schema": "B3_OFFICIAL_V3_STUDENT_INPUT_V1",
        "source_schema": "OFFICIAL_25D_V1",
        "feature_order_sha256": contract["feature_order_sha256"],
        **identity,
        "step": index,
        "features_25d": item["features_25d"],
        "valid": bool(rebuilt_rows[index].get("valid", True)),
    } for index, item in enumerate(prepared["merged"])]
    policy = [{
        "schema": "B3_OFFICIAL_V3_POLICY_INTENT_9D_V1",
        "source_schema": "OFFICIAL_POLICY_INTENT_9D_V1",
        "policy_intent_order_sha256": contract["policy_intent_order_sha256"],
        **identity,
        "step": index,
        "clean_policy_intent_9d": item["clean_policy_intent_9d"],
    } for index, item in enumerate(prepared["merged"])]
    teacher = _teacher_rows(prepared)
    events = prepared["rebuilt"]["events"]
    _write_jsonl(output_root / "student_input_records.jsonl", student)
    _write_jsonl(output_root / "policy_intent_9d_records.jsonl", policy)
    _write_jsonl(output_root / "teacher_retention_records.jsonl", teacher)
    (output_root / "retention_events.json").write_text(json.dumps(events, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    files = [name for name in ("student_input_records.jsonl", "policy_intent_9d_records.jsonl", "teacher_retention_records.jsonl", "retention_events.json")]
    manifest = {
        "schema": "B3_OFFICIAL_V3_S1_EPISODE_V1",
        "mode": "FIT_TRAIN",
        "source_identity": identity,
        "source_artifact_sha256": prepared["source_before_sha256"],
        "source_before_sha256": prepared["source_before_sha256"],
        "source_after_sha256": prepared["source_after_sha256"],
        "source_contract_sha256": prepared["source_contract_sha256"],
        "source_unchanged": prepared["source_before_sha256"] == prepared["source_after_sha256"],
        "protocol_schema": prepared["protocol"]["schema"],
        "step_count": prepared["step_count"],
        "teacher_materialization": "COMPLETED",
        "student_teacher_physical_separation": True,
        "teacher_labels_are_attack_vulnerability": False,
        "files": [{"path": name, "sha256": sha256_file(output_root / name), "size": (output_root / name).stat().st_size} for name in files],
        "formal_training_ready": False,
        "formal_attack_ready": False,
    }
    manifest_path = output_root / "materialization_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_sidecar(manifest_path)
    sum_names = sorted(path.name for path in output_root.iterdir() if path.is_file())
    sums = output_root / "SHA256SUMS"
    sums.write_text("".join(f"{sha256_file(output_root / name)}  {name}\n" for name in sum_names), encoding="utf-8")
    _write_sidecar(sums)
    return manifest


def materialize_episode(row: dict[str, str], contract_path: Path, protocol_path: Path, output_root: Path) -> dict[str, Any]:
    prepared = dry_run_episode(row, contract_path, protocol_path)
    if output_root.exists():
        raise V3S1ContractViolation(f"refusing to overwrite episode output: {output_root}")
    staging = output_root.with_name(f".{output_root.name}.{uuid.uuid4().hex}.staging")
    if staging.exists():
        raise V3S1ContractViolation(f"staging root already exists: {staging}")
    try:
        manifest = _write_episode(prepared, staging)
        report = audit_teacher_episode(_load_jsonl(staging / "teacher_retention_records.jsonl"), _load_json_value(staging / "retention_events.json"), row["canonical_parent_key"])
        if report["status"] != "PASS":
            raise V3S1ContractViolation(f"episode Teacher audit failed: {report['violations']}")
        os.replace(staging, output_root)
        return manifest
    except (OSError, V3S1ContractViolation):
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        raise


def materialize_fit(registry_csv: Path, registry_summary: Path, contract_path: Path, protocol_path: Path, output_root: Path) -> dict[str, Any]:
    registry_rows = load_formal_fit_registry(registry_csv, registry_summary)
    if output_root.exists():
        raise V3S1ContractViolation(f"refusing to overwrite S1 output root: {output_root}")
    # Complete the source dry-run before creating a promotable staging tree.
    # The second pass intentionally re-reads each source so a source mutation
    # between validation and materialization is caught by the before/after SHA.
    for row in registry_rows:
        dry_run_episode(row, contract_path, protocol_path)
    staging = output_root.with_name(f".{output_root.name}.{uuid.uuid4().hex}.staging")
    if staging.exists():
        raise V3S1ContractViolation(f"staging root already exists: {staging}")
    try:
        staging.mkdir(parents=True)
        reports: list[dict[str, Any]] = []
        for row in registry_rows:
            item = dry_run_episode(row, contract_path, protocol_path)
            row = item["registry_row"]
            episode_root = staging / row["suite"] / f"task_{int(row['task_idx']):02d}" / f"state_{int(row['state_id']):02d}"
            _write_episode(item, episode_root)
            reports.append(audit_teacher_episode(_load_jsonl(episode_root / "teacher_retention_records.jsonl"), _load_json_value(episode_root / "retention_events.json"), row["canonical_parent_key"]))
        aggregate = aggregate_teacher_audit(reports, registry_rows)
        if aggregate["status"] != "PASS":
            raise V3S1ContractViolation(f"aggregate Teacher audit failed: {aggregate['violations']}")
        aggregate_path = staging / "B3_OFFICIAL_V3_TEACHER_AGGREGATE_AUDIT_V1.json"
        aggregate_path.write_text(json.dumps(aggregate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        _write_sidecar(aggregate_path)
        top = {
            "schema": "B3_OFFICIAL_V3_S1_MATERIALIZATION_V1",
            "status": "PASS_STRUCTURAL_TEACHER_AUDIT",
            "registry_csv_sha256": sha256_file(registry_csv),
            "registry_summary_sha256": sha256_file(registry_summary),
            "source_contract_sha256": sha256_file(contract_path),
            "protocol_sha256": sha256_file(protocol_path),
            "identity_count": len(registry_rows),
            "episode_audit_count": len(reports),
            "aggregate_audit": aggregate["status"],
            "teacher_labels_read": True,
            "formal_training_ready": False,
            "formal_attack_ready": False,
        }
        top_path = staging / "B3_OFFICIAL_V3_S1_MATERIALIZATION_MANIFEST_V1.json"
        top_path.write_text(json.dumps(top, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        _write_sidecar(top_path)
        names = sorted(path.relative_to(staging).as_posix() for path in staging.rglob("*") if path.is_file())
        sums = staging / "SHA256SUMS"
        sums.write_text("".join(f"{sha256_file(staging / name)}  {name}\n" for name in names), encoding="utf-8")
        _write_sidecar(sums)
        os.replace(staging, output_root)
        return top
    except (OSError, V3S1ContractViolation):
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        raise


def write_sealed_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    if path.exists() or path.with_name(path.name + ".sha256").exists():
        raise V3S1ContractViolation(f"refusing to overwrite: {path}")
    content = io.StringIO(newline="")
    writer = csv.DictWriter(content, fieldnames=fields)
    writer.writeheader()
    writer.writerows({field: row.get(field, "") for field in fields} for row in rows)
    _atomic_write_text(path, content.getvalue())
    _write_sidecar(path)


def write_sealed_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists() or path.with_name(path.name + ".sha256").exists():
        raise V3S1ContractViolation(f"refusing to overwrite: {path}")
    _atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    _write_sidecar(path)


__all__ = [
    "V3S1ContractViolation", "aggregate_teacher_audit", "audit_teacher_episode", "build_fit_census",
    "dry_run_episode", "load_formal_fit_registry", "load_s1_protocol", "materialize_episode",
    "materialize_fit", "sha256_file", "write_sealed_csv", "write_sealed_json",
]
