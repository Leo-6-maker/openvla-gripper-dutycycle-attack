"""Preparation-only Official V3 FIT census and S1 materialization.

This module is deliberately independent from the legacy V2 materializer.  It
may read CLEAN source evidence only after an exact formal FIT registry passes;
it never authorizes training or attack.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import io
import json
import math
import os
import shutil
import sys
import uuid
from collections import Counter
from pathlib import Path
from typing import Any

from .b3_retention import RetentionConfig, rebuild_retention_features
from .official_v3_contract import SUITES, audit_artifact, canonical_key, expected_split, load_contract
from .official_v3_sprint0 import _runner_binding


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
    "artifact_audit_path", "artifact_audit_sha256", "formal_eligible", "formal_selected",
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


def build_s1_runner_binding(
    *, runner_repo: Path, expected_runner_head: str, config_path: Path, runner_script_path: Path,
) -> dict[str, Any]:
    """Bind the real S1 runner to Git and the execution environment."""
    binding = _runner_binding(
        runner_repo=runner_repo,
        expected_runner_head=expected_runner_head,
        config_path=config_path,
        runner_script_path=runner_script_path,
    )
    def version(name: str) -> str:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            return "NOT_INSTALLED"
    return {
        "status": "PASS",
        **binding,
        "python_version": sys.version,
        "torch_version": version("torch"),
        "transformers_version": version("transformers"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    }


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
    all_rows = _read_csv(registry_csv)
    summary = _load_json(summary_json)
    required_summary = {
        "formal_fit_ready": True,
        "fit_formal_selected_count": 800,
        "full_artifact_audit_pass_count": 800,
        "unresolved_provenance_count": 0,
        "unfinished_remediation_count": 0,
        "stale_recovery_unresolved_count": 0,
        "duplicate_selection_count": 0,
    }
    for name, expected in required_summary.items():
        if summary.get(name) != expected:
            raise V3S1ContractViolation(f"formal FIT registry gate failed: {name}={summary.get(name)!r}")
    expected_fit_suites = {suite: 200 for suite in SUITES}
    if summary.get("fit_by_suite_formal_selected") != expected_fit_suites:
        raise V3S1ContractViolation("formal FIT registry suite counts are not closed")
    expected_fit_tasks = {f"{suite}/task_{task}": 20 for suite in SUITES for task in range(10)}
    if summary.get("fit_by_task_formal_selected") != expected_fit_tasks:
        raise V3S1ContractViolation("formal FIT registry task counts are not closed")
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
    if len(all_rows) != 2000:
        raise V3S1ContractViolation(f"formal V3 registry must contain 2000 rows, got {len(all_rows)}")
    all_keys = {row.get("canonical_parent_key", "") for row in all_rows}
    expected_all_keys = {canonical_key(suite, task, state) for suite in SUITES for task in range(10) for state in range(50)}
    if all_keys != expected_all_keys:
        raise V3S1ContractViolation("formal V3 registry is not the exact 2000-identity universe")
    for row in all_rows:
        try:
            if row.get("canonical_parent_key") != canonical_key(row["suite"], _int(row["task_idx"], "task_idx"), _int(row["state_id"], "state_id")):
                raise V3S1ContractViolation("formal V3 registry row identity columns do not match canonical key")
            if row.get("split") != expected_split(_int(row["state_id"], "state_id")):
                raise V3S1ContractViolation("formal V3 registry row split does not match state")
        except KeyError as exc:
            raise V3S1ContractViolation("formal V3 registry row is missing identity columns") from exc
    rows = [row for row in all_rows if row.get("split") == "FIT_TRAIN"]
    if len(rows) != 800:
        raise V3S1ContractViolation(f"formal FIT census input must contain exactly 800 FIT rows, got {len(rows)}")
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
        audit_path = Path(row["artifact_audit_path"])
        if not audit_path.is_absolute():
            audit_path = summary_json.parent / audit_path
        if not audit_path.is_file() or sha256_file(audit_path).lower() != row["artifact_audit_sha256"].lower():
            raise V3S1ContractViolation(f"registry artifact audit binding is not closed: {key}")
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
            "artifact_audit_path": row["artifact_audit_path"],
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
    """Join only the causal 25D source and privileged robot sidecar.

    The policy-intent stream is deliberately not opened here.  It is an
    optional, separate 9D ablation export and must never be a prerequisite for
    the primary 25D materialization.
    """
    streams = {name: _load_jsonl(source_root / filename) for name, filename in {
        "step": "step_records.jsonl",
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
        for source in ("step", "sidecar"):
            for key, value in indexed[source][step].items():
                if key in row and not _same(row[key], value):
                    raise V3S1ContractViolation(f"source stream join conflict: step={step}, field={key}")
                row[key] = value
        if not _finite_vector(row.get("features_25d"), 25):
            raise V3S1ContractViolation(f"student source vector invalid at step {step}")
        merged.append(row)
    return merged


def _load_policy_intent_stream(source_root: Path, meta: dict[str, Any]) -> list[dict[str, Any]]:
    """Read the optional 9D stream only for the separate ablation exporter."""
    rows = _load_jsonl(source_root / "policy_intent_records.jsonl")
    identity = {name: meta[name] for name in ("suite", "task_idx", "state_id", "canonical_parent_key")}
    if not rows or [_int(row.get("step", index), "step") for index, row in enumerate(rows)] != list(range(len(rows))):
        raise V3S1ContractViolation("policy-intent stream is empty or non-contiguous")
    for index, row in enumerate(rows):
        if any(row.get(name, identity[name]) != identity[name] for name in identity):
            raise V3S1ContractViolation(f"policy-intent identity mismatch at step {index}")
        if not _finite_vector(row.get("clean_policy_intent_9d"), 9):
            raise V3S1ContractViolation(f"policy-intent vector invalid at step {index}")
    return rows


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
        report = audit_artifact(root, contract, equivalence_status="PASS", mode="25d")
    else:
        report = audit_artifact(root, contract, mode="25d")
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


def export_policy_intent_9d(row: dict[str, str], contract_path: Path, output_root: Path) -> dict[str, Any]:
    """Export the optional 9D stream to an independent, non-primary root."""
    contract = load_contract(contract_path)
    source_root = Path(row["selected_artifact_root"]).resolve()
    meta = _load_json(source_root / "episode_metadata.json")
    # Full source audit is intentionally only used by this optional exporter.
    report = audit_artifact(source_root, contract, mode="full")
    if report.get("status") not in {"PASS_FORMAL_CANDIDATE", "PASS_DATA_CONTRACT_PROVENANCE_HOLD"}:
        raise V3S1ContractViolation(f"9D source audit failed: {row['canonical_parent_key']}")
    policy_rows = _load_policy_intent_stream(source_root, meta)
    if output_root.exists():
        raise V3S1ContractViolation(f"refusing to overwrite 9D output: {output_root}")
    staging = output_root.with_name(f".{output_root.name}.{uuid.uuid4().hex}.staging")
    try:
        staging.mkdir(parents=True)
        identity = {name: row[name] for name in ("suite", "task_idx", "state_id", "canonical_parent_key")}
        output = [{
            "schema": "B3_OFFICIAL_V3_POLICY_INTENT_9D_V1",
            "source_schema": "OFFICIAL_POLICY_INTENT_9D_V1",
            **identity,
            "step": index,
            "clean_policy_intent_9d": item["clean_policy_intent_9d"],
        } for index, item in enumerate(policy_rows)]
        _write_jsonl(staging / "policy_intent_9d_records.jsonl", output)
        manifest = {
            "schema": "B3_OFFICIAL_V3_POLICY_INTENT_9D_EXPORT_V1",
            "source_identity": identity,
            "source_artifact_sha256": row["selected_artifact_recursive_sha256"],
            "record_count": len(output),
            "formal_training_authorized": False,
            "formal_attack_authorized": False,
        }
        _atomic_write_text(staging / "policy_intent_9d_manifest.json", json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        for path in (staging / "policy_intent_9d_records.jsonl", staging / "policy_intent_9d_manifest.json"):
            _write_sidecar(path)
        os.replace(staging, output_root)
        return manifest
    except (OSError, V3S1ContractViolation):
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        raise


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
    if any(not isinstance(event_id, int) for event_id in event_file_ids) or len(set(event_file_ids)) != len(event_file_ids) or sorted(event_file_ids) != event_ids:
        violations.append("EVENT_STREAM_FILE_ID_MISMATCH")
    previous_end = -1
    for expected_id, event in enumerate(events):
        if event.get("event_id") != expected_id:
            violations.append("EVENT_FILE_ORDINAL_NOT_CONTIGUOUS")
        start, end = event.get("start_step"), event.get("end_step")
        if not isinstance(start, int) or not isinstance(end, int) or end < start or start < 0 or end >= len(rows) or start <= previous_end:
            violations.append(f"EVENT_{expected_id}_BOUNDS_OR_OVERLAP")
        event_rows = [row for row in rows if row.get("event_id") == expected_id]
        if isinstance(start, int) and isinstance(end, int) and [row.get("step") for row in event_rows] != list(range(start, end + 1)):
            violations.append(f"EVENT_{expected_id}_ROW_INTERVAL_MISMATCH")
        release_step = event.get("release_onset")
        if release_step is not None and not any(row.get("event_release_onset") == release_step for row in event_rows):
            violations.append(f"EVENT_{expected_id}_RELEASE_ONSET_MISMATCH")
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
    teacher_counts: dict[str, dict[str, int]] = {}
    for head in HEADS:
        known_count = positive_count = negative_count = unknown_count = 0
        positive_episodes = 0
        for row in rows:
            mask = MASKS[head]
            known = (not row[mask]) if head == "retention_continuation_t10" and isinstance(row.get(mask), bool) else row.get(mask) is True
            if not known:
                unknown_count += 1
            else:
                known_count += 1
                if row.get(head) is True:
                    positive_count += 1
                elif row.get(head) is False:
                    negative_count += 1
            if known and row.get(head) is True:
                positive_episodes = 1
        teacher_counts[head] = {
            "known": known_count,
            "unknown": unknown_count,
            "positive": positive_count,
            "negative": negative_count,
            "positive_episodes": positive_episodes,
            "all_unknown_episodes": int(known_count == 0),
            "all_negative_episodes": int(known_count > 0 and positive_count == 0),
        }
    ordinal_stats: dict[str, dict[str, int]] = {}
    for event in events:
        ordinal = event.get("event_ordinal", event.get("event_id"))
        if not isinstance(ordinal, int):
            continue
        bucket = ordinal_stats.setdefault(str(ordinal), {"events": 0, "known_t10_steps": 0, "positive_t10_steps": 0})
        bucket["events"] += 1
        for row in rows:
            if row.get("event_ordinal", row.get("event_id")) == ordinal and row.get("retention_unknown_mask") is not True:
                bucket["known_t10_steps"] += 1
                bucket["positive_t10_steps"] += int(row.get("retention_continuation_t10") is True)
    later_event_count = sum(int(isinstance(event.get("event_ordinal", event.get("event_id")), int) and event.get("event_ordinal", event.get("event_id")) >= 1) for event in events)
    later_known_t10 = sum(int(row.get("event_ordinal", row.get("event_id", -1)) >= 1 and row.get("retention_unknown_mask") is not True) for row in rows)
    later_positive_t10 = sum(int(row.get("event_ordinal", row.get("event_id", -1)) >= 1 and row.get("retention_continuation_t10") is True) for row in rows)
    return {
        "schema": "B3_OFFICIAL_V3_TEACHER_EPISODE_AUDIT_V1",
        "canonical_parent_key": identity,
        "status": "PASS" if not violations else "HOLD",
        "step_count": len(rows),
        "event_count": len(event_ids),
        "event_ordinal_counts": dict(Counter(event.get("event_ordinal", event.get("event_id")) for event in events)),
        "t10_positive_count": sum(row.get("retention_continuation_t10") is True for row in rows),
        "teacher_counts": teacher_counts,
        "event_ordinal_stats": ordinal_stats,
        "episodes_with_multiple_teacher_events": int(len(events) >= 2),
        "later_event_count": later_event_count,
        "later_event_known_t10_steps": later_known_t10,
        "later_event_positive_t10_steps": later_positive_t10,
        "violations": sorted(set(violations)),
    }


def aggregate_teacher_audit(reports: list[dict[str, Any]], registry_rows: list[dict[str, str]]) -> dict[str, Any]:
    expected = {row["canonical_parent_key"] for row in registry_rows}
    actual = [row.get("canonical_parent_key") for row in reports]
    duplicates = sorted(key for key, count in Counter(actual).items() if count != 1)
    suite_counts = Counter(row.get("canonical_parent_key", "").split("/", 1)[0] for row in reports)
    violations = sorted({item for row in reports for item in row.get("violations", [])})
    def empty_counts() -> dict[str, int]:
        return {"known": 0, "unknown": 0, "positive": 0, "negative": 0, "positive_episodes": 0, "all_unknown_episodes": 0, "all_negative_episodes": 0}

    def add_counts(target: dict[str, int], source: dict[str, Any]) -> None:
        for name in target:
            target[name] += int(source.get(name, 0))

    global_head_counts = {head: empty_counts() for head in HEADS}
    suite_head_counts = {suite: {head: empty_counts() for head in HEADS} for suite in SUITES}
    task_head_counts: dict[str, dict[str, dict[str, int]]] = {}
    ordinal_counts: dict[str, dict[str, int]] = {}
    l10 = {
        "episodes_with_multiple_teacher_events": 0,
        "teacher_event_count": 0,
        "later_event_known_t10_steps": 0,
        "later_event_positive_t10_steps": 0,
    }
    for report in reports:
        identity = str(report.get("canonical_parent_key", ""))
        suite, _, task = identity.partition("/")
        task_key = f"{suite}/{task}" if suite and task else identity
        task_head_counts.setdefault(task_key, {head: empty_counts() for head in HEADS})
        for head in HEADS:
            counts = report.get("teacher_counts", {}).get(head, {})
            add_counts(global_head_counts[head], counts)
            if suite in suite_head_counts:
                add_counts(suite_head_counts[suite][head], counts)
            add_counts(task_head_counts[task_key][head], counts)
        for ordinal, counts in report.get("event_ordinal_stats", {}).items():
            bucket = ordinal_counts.setdefault(str(ordinal), {"events": 0, "known_t10_steps": 0, "positive_t10_steps": 0})
            for name in bucket:
                bucket[name] += int(counts.get(name, 0))
        l10["episodes_with_multiple_teacher_events"] += int(suite == "libero_10" and report.get("episodes_with_multiple_teacher_events", 0))
        l10["teacher_event_count"] += int(suite == "libero_10" and report.get("event_count", 0))
        l10["later_event_known_t10_steps"] += int(suite == "libero_10" and report.get("later_event_known_t10_steps", 0))
        l10["later_event_positive_t10_steps"] += int(suite == "libero_10" and report.get("later_event_positive_t10_steps", 0))
    suite_known_t10_positive = {
        suite: suite_head_counts[suite]["retention_continuation_t10"]["positive"] for suite in SUITES
    }
    task_all_unknown = sorted(
        key for key, heads in task_head_counts.items()
        if heads["retention_continuation_t10"]["known"] == 0
    )
    structural_pass = (
        set(actual) == expected and len(actual) == 800 and not duplicates and not violations
        and all(suite_counts[suite] == 200 for suite in SUITES)
        and all(row.get("status") == "PASS" for row in reports)
    )
    nondegeneracy_gates = {
        "structural_invariants_zero": not violations,
        "every_suite_has_known_t10_positive": all(value > 0 for value in suite_known_t10_positive.values()),
        "every_task_has_known_t10": not task_all_unknown,
        "l10_has_later_teacher_event": l10["episodes_with_multiple_teacher_events"] > 0,
        "l10_later_event_has_known_t10": l10["later_event_known_t10_steps"] > 0,
    }
    status = "PASS" if structural_pass and all(nondegeneracy_gates.values()) else "HOLD"
    return {
        "schema": "B3_OFFICIAL_V3_TEACHER_AGGREGATE_AUDIT_V1",
        "status": status,
        "expected_identity_count": len(expected),
        "actual_identity_count": len(actual),
        "duplicate_identity_keys": duplicates,
        "suite_episode_counts": dict(sorted(suite_counts.items())),
        "invariant_violation_count": len(violations),
        "violations": violations,
        "global_teacher_counts": global_head_counts,
        "suite_teacher_counts": suite_head_counts,
        "task_teacher_counts": task_head_counts,
        "event_ordinal_counts": ordinal_counts,
        "suite_known_t10_positive": suite_known_t10_positive,
        "task_all_unknown": task_all_unknown,
        "l10_later_event": l10,
        "nondegeneracy_gates": nondegeneracy_gates,
        "teacher_labels_read": True,
        "formal_training_ready": False,
        "formal_attack_ready": False,
    }


def audit_materialized_episode(
    episode_root: Path, registry_row: dict[str, str], *, require_runner_binding: bool = False,
) -> dict[str, Any]:
    """Audit a promoted episode independently of the writer's in-memory state."""
    manifest_path = episode_root / "materialization_manifest.json"
    manifest = _load_json(manifest_path)
    identity = registry_row["canonical_parent_key"]
    if manifest.get("schema") != "B3_OFFICIAL_V3_S1_EPISODE_V1":
        raise V3S1ContractViolation(f"unexpected materialized episode schema: {identity}")
    source_identity = manifest.get("source_identity", {})
    if source_identity.get("canonical_parent_key") != identity:
        raise V3S1ContractViolation(f"materialized identity mismatch: {identity}")
    if manifest.get("source_artifact_sha256") != registry_row["selected_artifact_recursive_sha256"]:
        raise V3S1ContractViolation(f"materialized source binding mismatch: {identity}")
    if manifest.get("source_unchanged") is not True or manifest.get("student_teacher_physical_separation") is not True:
        raise V3S1ContractViolation(f"materialized source/separation gate failed: {identity}")
    runner_binding = manifest.get("runner_binding")
    if require_runner_binding and (not isinstance(runner_binding, dict) or runner_binding.get("status") != "PASS"):
        raise V3S1ContractViolation(f"materialized runner provenance is not formally bound: {identity}")
    if manifest.get("policy_intent_9d_exported") is not False or (episode_root / "policy_intent_9d_records.jsonl").exists():
        raise V3S1ContractViolation(f"9D policy stream leaked into primary 25D output: {identity}")
    student_path = episode_root / "student_input_records.jsonl"
    teacher_path = episode_root / "teacher_retention_records.jsonl"
    events_path = episode_root / "retention_events.json"
    if not student_path.is_file() or not teacher_path.is_file() or not events_path.is_file():
        raise V3S1ContractViolation(f"materialized episode files missing: {identity}")
    listed = {item.get("path"): item for item in manifest.get("files", []) if isinstance(item, dict)}
    expected_names = {student_path.name, teacher_path.name, events_path.name}
    if set(listed) != expected_names:
        raise V3S1ContractViolation(f"materialized manifest file set mismatch: {identity}")
    for name, item in listed.items():
        path = episode_root / name
        if not path.is_file() or item.get("sha256") != sha256_file(path) or int(item.get("size", -1)) != path.stat().st_size:
            raise V3S1ContractViolation(f"materialized file checksum mismatch: {identity}:{name}")
    manifest_sidecar = manifest_path.with_name(manifest_path.name + ".sha256")
    if not manifest_sidecar.is_file() or manifest_sidecar.read_text(encoding="utf-8").strip() != f"{sha256_file(manifest_path)}  {manifest_path.name}":
        raise V3S1ContractViolation(f"materialized manifest sidecar mismatch: {identity}")
    student_rows = _load_jsonl(student_path)
    teacher_rows = _load_jsonl(teacher_path)
    events = _load_json_value(events_path)
    if len(student_rows) != len(teacher_rows):
        raise V3S1ContractViolation(f"student/teacher length mismatch: {identity}")
    for index, row in enumerate(student_rows):
        if row.get("schema") != "B3_OFFICIAL_V3_STUDENT_INPUT_V1" or row.get("source_schema") != "OFFICIAL_25D_V1":
            raise V3S1ContractViolation(f"student schema mismatch: {identity}:{index}")
        if row.get("canonical_parent_key") != identity or row.get("step") != index or not _finite_vector(row.get("features_25d"), 25) or not isinstance(row.get("valid"), bool):
            raise V3S1ContractViolation(f"student row mismatch: {identity}:{index}")
    report = audit_teacher_episode(teacher_rows, events, identity)
    report["materialized_episode_root"] = str(episode_root.resolve())
    report["materialization_manifest_sha256"] = sha256_file(manifest_path)
    return report


def audit_materialized_root(
    root: Path, registry_rows: list[dict[str, str]], *, require_runner_binding: bool = False,
) -> dict[str, Any]:
    """Re-read a sealed corpus and require exact registry identity closure."""
    expected = {row["canonical_parent_key"]: row for row in registry_rows}
    top_path = root / "B3_OFFICIAL_V3_S1_MATERIALIZATION_MANIFEST_V1.json"
    top_sidecar = top_path.with_name(top_path.name + ".sha256")
    if not top_path.is_file() or not top_sidecar.is_file() or top_sidecar.read_text(encoding="utf-8").strip() != f"{sha256_file(top_path)}  {top_path.name}":
        raise V3S1ContractViolation("materialized root manifest or sidecar is missing/invalid")
    top = _load_json(top_path)
    if top.get("schema") != "B3_OFFICIAL_V3_S1_MATERIALIZATION_V1" or top.get("identity_count") != len(expected):
        raise V3S1ContractViolation("materialized root manifest schema/count mismatch")
    if require_runner_binding and (not isinstance(top.get("runner_binding"), dict) or top["runner_binding"].get("status") != "PASS"):
        raise V3S1ContractViolation("materialized root runner provenance is not formally bound")
    manifests = sorted(root.rglob("materialization_manifest.json")) if root.exists() else []
    reports: list[dict[str, Any]] = []
    seen: set[str] = set()
    for manifest_path in manifests:
        manifest = _load_json(manifest_path)
        identity = manifest.get("source_identity", {}).get("canonical_parent_key")
        if identity not in expected or identity in seen:
            raise V3S1ContractViolation(f"materialized root contains extra/duplicate identity: {identity}")
        seen.add(identity)
        reports.append(audit_materialized_episode(manifest_path.parent, expected[identity], require_runner_binding=require_runner_binding))
    if seen != set(expected) or len(reports) != len(expected):
        raise V3S1ContractViolation(f"materialized root identity closure failed: expected={len(expected)} actual={len(seen)}")
    aggregate = aggregate_teacher_audit(reports, registry_rows)
    aggregate_path = root / "B3_OFFICIAL_V3_TEACHER_AGGREGATE_AUDIT_V1.json"
    if not aggregate_path.is_file():
        raise V3S1ContractViolation("aggregate Teacher audit is missing")
    stored = _load_json(aggregate_path)
    aggregate_sidecar = aggregate_path.with_name(aggregate_path.name + ".sha256")
    if not aggregate_sidecar.is_file() or aggregate_sidecar.read_text(encoding="utf-8").strip() != f"{sha256_file(aggregate_path)}  {aggregate_path.name}":
        raise V3S1ContractViolation("aggregate Teacher audit sidecar is missing/invalid")
    sums = root / "SHA256SUMS"
    sums_sidecar = sums.with_name(sums.name + ".sha256")
    if not sums.is_file() or not sums_sidecar.is_file() or sums_sidecar.read_text(encoding="utf-8").strip() != f"{sha256_file(sums)}  {sums.name}":
        raise V3S1ContractViolation("materialized root checksum closure is missing/invalid")
    if stored.get("status") != aggregate.get("status") or stored.get("nondegeneracy_gates") != aggregate.get("nondegeneracy_gates"):
        raise V3S1ContractViolation("stored aggregate Teacher audit does not match independent recomputation")
    return {
        "schema": "B3_OFFICIAL_V3_S1_ROOT_AUDIT_V1",
        "status": "PASS" if aggregate["status"] == "PASS" else "HOLD",
        "identity_count": len(reports),
        "aggregate": aggregate,
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
    teacher = _teacher_rows(prepared)
    events = prepared["rebuilt"]["events"]
    _write_jsonl(output_root / "student_input_records.jsonl", student)
    _write_jsonl(output_root / "teacher_retention_records.jsonl", teacher)
    (output_root / "retention_events.json").write_text(json.dumps(events, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    files = [name for name in ("student_input_records.jsonl", "teacher_retention_records.jsonl", "retention_events.json")]
    manifest = {
        "schema": "B3_OFFICIAL_V3_S1_EPISODE_V1",
        "mode": "FIT_TRAIN",
        "source_identity": identity,
        "source_artifact_sha256": prepared["source_before_sha256"],
        "source_before_sha256": prepared["source_before_sha256"],
        "source_after_sha256": prepared["source_after_sha256"],
        "source_contract_sha256": prepared["source_contract_sha256"],
        "source_unchanged": prepared["source_before_sha256"] == prepared["source_after_sha256"],
        "runner_binding": prepared.get("runner_binding", {"status": "SYNTHETIC_TEST_ONLY"}),
        "protocol_schema": prepared["protocol"]["schema"],
        "step_count": prepared["step_count"],
        "teacher_materialization": "COMPLETED",
        "policy_intent_9d_exported": False,
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


def materialize_episode(
    row: dict[str, str], contract_path: Path, protocol_path: Path, output_root: Path,
    runner_binding: dict[str, Any] | None = None,
) -> dict[str, Any]:
    prepared = dry_run_episode(row, contract_path, protocol_path)
    prepared["runner_binding"] = runner_binding or {"status": "SYNTHETIC_TEST_ONLY"}
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


def materialize_fit(
    registry_csv: Path, registry_summary: Path, contract_path: Path, protocol_path: Path, output_root: Path,
    runner_binding: dict[str, Any] | None = None,
) -> dict[str, Any]:
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
            item["runner_binding"] = runner_binding or {"status": "SYNTHETIC_TEST_ONLY"}
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
            "feature_rebuilder_sha256": sha256_file(Path(rebuild_retention_features.__code__.co_filename).resolve()),
            "runner_binding": runner_binding or {"status": "SYNTHETIC_TEST_ONLY"},
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


def write_census_bundle(output_root: Path, rows: list[dict[str, str]], summary: dict[str, Any]) -> None:
    """Atomically seal the FIT census CSV, summary, and checksum closure."""
    if output_root.exists():
        raise V3S1ContractViolation(f"refusing to overwrite census root: {output_root}")
    staging = output_root.with_name(f".{output_root.name}.{uuid.uuid4().hex}.staging")
    if staging.exists():
        raise V3S1ContractViolation(f"census staging root already exists: {staging}")
    fields = [
        "canonical_parent_key", "suite", "task_idx", "state_id", "split",
        "selected_artifact_root", "selected_artifact_recursive_sha256",
        "artifact_audit_path", "artifact_audit_sha256", "provenance_class",
    ]
    try:
        staging.mkdir(parents=True)
        csv_path = staging / "OFFICIAL_V3_FIT_CENSUS_V1.csv"
        summary_path = staging / "OFFICIAL_V3_FIT_CENSUS_SUMMARY_V1.json"
        write_sealed_csv(csv_path, rows, fields)
        sealed_summary = dict(summary)
        sealed_summary["census_csv_sha256"] = sha256_file(csv_path)
        sealed_summary["census_schema"] = "B3_OFFICIAL_V3_FIT_CENSUS_V1"
        write_sealed_json(summary_path, sealed_summary)
        names = sorted(path.name for path in staging.iterdir() if path.is_file())
        _atomic_write_text(staging / "SHA256SUMS", "".join(f"{sha256_file(staging / name)}  {name}\n" for name in names))
        _write_sidecar(staging / "SHA256SUMS")
        os.replace(staging, output_root)
    except (OSError, V3S1ContractViolation):
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        raise


__all__ = [
    "V3S1ContractViolation", "aggregate_teacher_audit", "audit_teacher_episode", "audit_materialized_episode",
    "audit_materialized_root", "build_fit_census", "export_policy_intent_9d", "write_census_bundle",
    "build_s1_runner_binding",
    "dry_run_episode", "load_formal_fit_registry", "load_s1_protocol", "materialize_episode",
    "materialize_fit", "sha256_file", "write_sealed_csv", "write_sealed_json",
]
