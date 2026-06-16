#!/usr/bin/env python3
"""Independent auditor for M3 arm-v5 clean-capture artifacts.

This auditor intentionally does not import the producer capture runner or the
shared event-selection helper. It recomputes the frozen state pool, capture
attempt policy, earliest clean-CLOSE event, final eight-frame selection, exact
artifact bindings, and model bundle manifest independently from disk artifacts.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]

V5_HASH_SALT = "M3_ARM_V5_CLOSE_PANEL"
V5_TASKS = (
    "alphabet_soup",
    "bbq_sauce",
    "butter",
    "chocolate_pudding",
    "cream_cheese",
    "ketchup",
    "milk",
    "orange_juice",
    "salad_dressing",
    "tomato_sauce",
)
V5_STATE_IDS = tuple(range(50))
V5_PANEL_SIZE = 8
V5_EVENT_GRIPPER_TOKEN = 31872
V5_MIN_STEP = 0
V5_MAX_STEP = 279
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

EXACT_INPUT_REQUIRED_FIELDS = (
    "raw_image_path",
    "raw_image_sha256",
    "processed_tensor_path",
    "processed_tensor_sha256",
    "prompt_token_ids",
    "prompt_token_ids_sha256",
    "previous_raw_image_path",
    "previous_raw_image_sha256",
    "previous_processed_tensor_path",
    "previous_processed_tensor_sha256",
    "previous_prompt_token_ids",
    "previous_prompt_token_ids_sha256",
    "model_fingerprint",
    "model_checkpoint_sha256",
    "processor_config_sha256",
    "preprocess_config_sha256",
    "task_state_init_sha256",
    "clean_record_source_path",
    "clean_record_source_sha256",
    "runner_sha256",
    "config_sha256",
    "commit",
    "gpu_query",
    "worktree_status",
    "official_score_argmax_token_id",
    "previous_official_score_argmax_token_id",
)


@dataclass(frozen=True)
class Candidate:
    task: str
    state_id: int
    task_rank: int
    state_hash: str


@dataclass(frozen=True)
class Event:
    task: str
    state_id: int
    step: int
    state_hash: str
    record: Mapping[str, Any]
    previous: Mapping[str, Any]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical_json_sha256(obj: Any) -> str:
    return sha256_text(json.dumps(obj, sort_keys=True, separators=(",", ":")))


def is_sha256(value: Any) -> bool:
    return bool(SHA256_RE.match(str(value)))


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(str(tmp), str(path))
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


def write_json(path: Path, obj: Any) -> None:
    atomic_write_text(path, json.dumps(obj, indent=2, sort_keys=True))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fieldnames: list[str]) -> None:
    import io

    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        writer.writerow({key: row.get(key, "") for key in fieldnames})
    atomic_write_text(path, buf.getvalue())


def load_config(path: Path) -> dict[str, Any]:
    obj = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"config must be a mapping: {path}")
    return obj


def v5_state_hash(task: str, state_id: int) -> str:
    return sha256_text(f"{V5_HASH_SALT}|{task}|{int(state_id)}")


def parse_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def ledger_exclusions(path: Path) -> dict[str, set[int]]:
    rows = read_csv(path)
    exclusions: dict[str, set[int]] = {}
    for row in rows:
        if parse_bool(row.get("used_for_development", "")):
            exclusions.setdefault(str(row["task"]), set()).add(int(row["state_id"]))
    return exclusions


def derive_state_pool_from_ledger(path: Path) -> list[Candidate]:
    excluded = ledger_exclusions(path)
    pool: list[Candidate] = []
    for task in V5_TASKS:
        candidates = [
            (v5_state_hash(task, state_id), state_id)
            for state_id in V5_STATE_IDS
            if state_id not in excluded.get(task, set())
        ]
        candidates.sort()
        for rank, (state_hash, state_id) in enumerate(candidates[:2], start=1):
            pool.append(Candidate(task=task, state_id=state_id, task_rank=rank, state_hash=state_hash))
    return pool


def config_pool(config: Mapping[str, Any]) -> list[Candidate]:
    rows = config.get("task_state_pool", [])
    if not isinstance(rows, list):
        raise ValueError("task_state_pool must be a list")
    pool: list[Candidate] = []
    seen: set[tuple[str, int]] = set()
    for row in rows:
        task = str(row["task"])
        state_id = int(row["state_id"])
        if (task, state_id) in seen:
            raise ValueError(f"duplicate config pool state: {task}_s{state_id}")
        seen.add((task, state_id))
        pool.append(
            Candidate(
                task=task,
                state_id=state_id,
                task_rank=int(row["task_rank"]),
                state_hash=str(row["state_hash"]),
            )
        )
    return pool


def csv_pool(path: Path) -> list[Candidate]:
    return [
        Candidate(
            task=str(row["task"]),
            state_id=int(row["state_id"]),
            task_rank=int(row["task_rank"]),
            state_hash=str(row["state_hash"]),
        )
        for row in read_csv(path)
    ]


def candidate_key(pool: Iterable[Candidate]) -> list[tuple[str, int, int, str]]:
    return [(row.task, row.state_id, row.task_rank, row.state_hash) for row in pool]


def validate_state_pool(config: Mapping[str, Any], config_path: Path) -> list[Candidate]:
    ledger_path = Path(str(config["selection"]["prior_layer3_state_ledger"]))
    if not ledger_path.is_absolute():
        ledger_path = REPO_ROOT / ledger_path
    expected = derive_state_pool_from_ledger(ledger_path)
    configured = config_pool(config)
    csv_rows = csv_pool(REPO_ROOT / "tables" / "m3_arm_v5_preregistered_state_pool.csv")
    if candidate_key(configured) != candidate_key(expected):
        raise ValueError("config state pool does not match independently derived ledger pool")
    if candidate_key(csv_rows) != candidate_key(expected):
        raise ValueError("CSV state pool does not match independently derived ledger pool")
    for row in expected:
        if row.state_hash != v5_state_hash(row.task, row.state_id):
            raise ValueError(f"state hash mismatch: {row.task}_s{row.state_id}")
    if len(expected) != 20:
        raise ValueError(f"state pool size mismatch: {len(expected)}")
    return expected


def safe_rel(value: str) -> Path:
    path = Path(value)
    if not value or path.is_absolute() or any(part in ("", ".", "..") for part in path.parts):
        raise ValueError(f"unsafe relative path: {value!r}")
    return path


def classify_markers(attempt_dir: Path) -> tuple[str, str]:
    markers = {path.stem for path in attempt_dir.glob("*.marker")}
    if "CAPTURE_COMPLETED" in markers:
        return "CAPTURED", "true"
    if "FIRST_ACTION_TAKEN" in markers:
        return "CAPTURE_FAILED_POST_ACTION", "true"
    if "FIRST_ACTION_GENERATED" in markers:
        return "CAPTURE_FAILED_POST_GENERATION", "false"
    return "FIRST_ACTION_BEFORE_INFRA_FAILURE", "false"


def synthesize_attempt_rows(capture_root: Path, pool: Iterable[Candidate]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for candidate in pool:
        state_name = f"{candidate.task}_s{candidate.state_id}"
        dirs = sorted((capture_root / "attempts" / state_name).glob("attempt_*"))
        if not dirs:
            rows.append(
                {
                    "task": candidate.task,
                    "state_id": str(candidate.state_id),
                    "attempt_index": "",
                    "attempt_status": "NOT_STARTED",
                    "first_action_taken": "false",
                    "attempt_dir": "",
                    "clean_records_path": "",
                    "clean_records_sha256": "",
                    "failure_reason": "no attempt directory",
                }
            )
            continue
        for attempt_dir in dirs:
            status, first_action = classify_markers(attempt_dir)
            rel_records = Path("states") / state_name / attempt_dir.name / f"{state_name}_clean_records.json"
            record_path = capture_root / rel_records
            rows.append(
                {
                    "task": candidate.task,
                    "state_id": str(candidate.state_id),
                    "attempt_index": attempt_dir.name.removeprefix("attempt_"),
                    "attempt_status": status,
                    "first_action_taken": first_action,
                    "attempt_dir": str(attempt_dir.relative_to(capture_root)).replace("\\", "/"),
                    "clean_records_path": str(rel_records).replace("\\", "/") if record_path.exists() else "",
                    "clean_records_sha256": sha256_file(record_path) if record_path.exists() else "",
                    "failure_reason": "synthesized_from_phase_markers",
                }
            )
    return rows


def load_attempt_rows(capture_root: Path, pool: Iterable[Candidate]) -> tuple[list[dict[str, str]], bool]:
    ledger = capture_root / "m3_arm_v5_capture_attempt_ledger.csv"
    if ledger.exists():
        return read_csv(ledger), True
    return synthesize_attempt_rows(capture_root, pool), False


def validate_phase_markers(capture_root: Path, row: Mapping[str, Any]) -> None:
    if str(row.get("attempt_status", "")) == "NOT_STARTED":
        return
    rel = str(row.get("attempt_dir", ""))
    if not rel:
        raise ValueError("attempt_dir missing")
    attempt_dir = capture_root / safe_rel(rel)
    if not attempt_dir.is_dir():
        raise ValueError(f"attempt_dir missing: {rel}")
    markers = {path.stem for path in attempt_dir.glob("*.marker")}
    if "ATTEMPT_STARTED" not in markers:
        raise ValueError(f"missing ATTEMPT_STARTED marker: {rel}")
    status = str(row.get("attempt_status", ""))
    if status == "CAPTURED":
        required = {"MODEL_READY", "ENV_READY", "FIRST_ACTION_GENERATED", "FIRST_ACTION_TAKEN", "CAPTURE_COMPLETED"}
        missing = sorted(required - markers)
        if missing:
            raise ValueError(f"CAPTURED attempt missing markers {missing}: {rel}")
    if status == "FIRST_ACTION_BEFORE_INFRA_FAILURE" and ("FIRST_ACTION_GENERATED" in markers or "FIRST_ACTION_TAKEN" in markers):
        raise ValueError(f"pre-generation retry status conflicts with markers: {rel}")


def validate_attempt_policy(capture_root: Path, rows: Iterable[Mapping[str, Any]], pool: Iterable[Candidate]) -> None:
    rows = list(rows)
    expected_pairs = {(row.task, row.state_id) for row in pool}
    grouped: dict[tuple[str, int], list[Mapping[str, Any]]] = {}
    for row in rows:
        key = (str(row["task"]), int(row["state_id"]))
        if key not in expected_pairs:
            raise ValueError(f"attempt row outside frozen pool: {key[0]}_s{key[1]}")
        grouped.setdefault(key, []).append(row)
        validate_phase_markers(capture_root, row)
    if set(grouped) != expected_pairs:
        raise ValueError("attempt ledger coverage mismatch")
    for key, attempts in grouped.items():
        attempts = sorted(attempts, key=lambda row: int(row.get("attempt_index") or 0))
        if len(attempts) > 2:
            raise ValueError(f"too many attempts for {key}")
        captured = [row for row in attempts if str(row.get("attempt_status")) == "CAPTURED"]
        if len(captured) > 1:
            raise ValueError(f"multiple captured attempts for {key}")
        if len(attempts) == 2:
            first = attempts[0]
            if str(first.get("attempt_status")) != "FIRST_ACTION_BEFORE_INFRA_FAILURE" or parse_bool(first.get("first_action_taken")):
                raise ValueError(f"illegal retry for {key}")
        for row in captured:
            rel = str(row.get("clean_records_path", ""))
            expected_sha = str(row.get("clean_records_sha256", ""))
            if not rel or not expected_sha:
                raise ValueError(f"CAPTURED attempt missing clean records binding: {key}")
            path = capture_root / safe_rel(rel)
            if not path.is_file():
                raise ValueError(f"clean records missing: {rel}")
            if sha256_file(path) != expected_sha:
                raise ValueError(f"clean records sha mismatch: {rel}")


def enumerate_model_bundle(model_root: Path) -> list[dict[str, Any]]:
    include_exact = {
        "config.json",
        "generation_config.json",
        "model.safetensors.index.json",
        "tokenizer.json",
        "tokenizer.model",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "preprocessor_config.json",
    }
    suffixes = {".safetensors", ".bin", ".py"}
    rows: list[dict[str, Any]] = []
    for path in sorted(model_root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(model_root).as_posix()
        if rel in include_exact or path.suffix in suffixes:
            rows.append({"relative_path": rel, "size_bytes": path.stat().st_size, "sha256": sha256_file(path)})
    if not rows:
        raise ValueError(f"model bundle enumeration is empty: {model_root}")
    return rows


def verify_model_bundle_exact_set(manifest_path: Path, model_root: Path) -> str:
    manifest_rows = read_csv(manifest_path)
    actual_rows = enumerate_model_bundle(model_root)
    normalized_manifest = sorted(
        [
            {
                "relative_path": safe_rel(str(row["relative_path"])).as_posix(),
                "size_bytes": int(row["size_bytes"]),
                "sha256": str(row["sha256"]),
            }
            for row in manifest_rows
        ],
        key=lambda row: row["relative_path"],
    )
    normalized_actual = sorted(actual_rows, key=lambda row: row["relative_path"])
    if normalized_manifest != normalized_actual:
        raise ValueError("model bundle manifest exact-set mismatch")
    return canonical_json_sha256(normalized_actual)


def tokens_from_record(record: Mapping[str, Any]) -> list[int]:
    if "tokens" in record:
        return [int(x) for x in record["tokens"]]
    value = record.get("clean_exact_7_tokens", "")
    if value == "":
        return []
    if isinstance(value, str):
        return [int(x) for x in json.loads(value)]
    return [int(x) for x in value]


def gripper_token(record: Mapping[str, Any]) -> int | None:
    if record.get("gripper_token", "") != "":
        return int(record["gripper_token"])
    if record.get("clean_gripper_token", "") != "":
        return int(record["clean_gripper_token"])
    tokens = tokens_from_record(record)
    return int(tokens[-1]) if len(tokens) == 7 else None


def score_invariant_pass(record: Mapping[str, Any]) -> bool:
    if "score_invariant_status" in record:
        return str(record["score_invariant_status"]).upper() == "PASS"
    invariant = record.get("score_invariant", {})
    if isinstance(invariant, Mapping):
        return bool(invariant.get("tie_aware_pass", invariant.get("pass", False)))
    if "score_tie_aware_pass" in record:
        return parse_bool(record["score_tie_aware_pass"])
    return False


def record_status(record: Mapping[str, Any], *, task: str, state_id: int) -> tuple[str, int | None, list[int]]:
    if str(record.get("task", "")) != task:
        return "task_mismatch", None, []
    if int(record.get("state_id", -1)) != int(state_id):
        return "state_id_mismatch", None, []
    tokens = tokens_from_record(record)
    token = gripper_token(record)
    if len(tokens) != 7:
        return "invalid_exact_7_tokens", token, tokens
    if token is None or token != tokens[-1]:
        return "gripper_token_mismatch", token, tokens
    if not score_invariant_pass(record):
        return "score_invariant_not_pass", token, tokens
    argmax = record.get("official_score_argmax_token_id", "")
    if argmax == "":
        return "missing_official_argmax_evidence", token, tokens
    if int(argmax) != int(token):
        return "official_argmax_emitted_mismatch", token, tokens
    return "pass", int(token), tokens


def load_clean_records(path: Path) -> list[dict[str, Any]]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    records = obj.get("records", obj if isinstance(obj, list) else [])
    if not isinstance(records, list):
        raise ValueError(f"clean records must be a list or records mapping: {path}")
    return [dict(row) for row in records]


def first_clean_close_event(records: Iterable[Mapping[str, Any]], candidate: Candidate) -> tuple[str, Event | None, str]:
    previous: Mapping[str, Any] | None = None
    previous_step: int | None = None
    previous_token: int | None = None
    seen: set[int] = set()
    saw_any = False
    for record in records:
        saw_any = True
        if "step" not in record:
            return "V5_CLEAN_EVENT_INFRA_INVALID", None, "missing_step"
        step = int(record["step"])
        if step in seen:
            return "V5_CLEAN_EVENT_INFRA_INVALID", None, "duplicate_step"
        seen.add(step)
        if previous_step is not None and step != previous_step + 1:
            return "V5_CLEAN_EVENT_INFRA_INVALID", None, "step_gap"
        status, token, _tokens = record_status(record, task=candidate.task, state_id=candidate.state_id)
        if status != "pass":
            return "V5_CLEAN_EVENT_INFRA_INVALID", None, status
        if V5_MIN_STEP <= step <= V5_MAX_STEP and token == V5_EVENT_GRIPPER_TOKEN:
            if previous is None or previous_step != step - 1:
                return "V5_CLEAN_EVENT_INFRA_INVALID", None, "missing_adjacent_previous_step"
            if previous_token != V5_EVENT_GRIPPER_TOKEN:
                return "V5_CLEAN_EVENT_FOUND", Event(candidate.task, candidate.state_id, step, candidate.state_hash, record, previous), ""
        previous = record
        previous_step = step
        previous_token = token
    return "V5_CLEAN_EVENT_NOT_FOUND", None, "empty_records" if not saw_any else "no_clean_close_onset"


def captured_record_paths(attempt_rows: Iterable[Mapping[str, Any]]) -> dict[tuple[str, int], str]:
    out: dict[tuple[str, int], str] = {}
    for row in attempt_rows:
        if str(row.get("attempt_status")) == "CAPTURED":
            out[(str(row["task"]), int(row["state_id"]))] = str(row["clean_records_path"])
    return out


def artifact_map(event: Event) -> dict[str, Any]:
    current = dict(event.record)
    previous = dict(event.previous)
    out = {key: current.get(key, "") for key in EXACT_INPUT_REQUIRED_FIELDS if not key.startswith("previous_")}
    out["previous_official_score_argmax_token_id"] = previous.get("official_score_argmax_token_id", "")
    for key in ("raw_image_path", "raw_image_sha256", "processed_tensor_path", "processed_tensor_sha256", "prompt_token_ids", "prompt_token_ids_sha256"):
        out[f"previous_{key}"] = previous.get(key, "")
    return out


def event_row(event: Event) -> dict[str, Any]:
    row = {
        "task": event.task,
        "state_id": event.state_id,
        "state_hash": event.state_hash,
        "selected_step": event.step,
        "clean_gripper_token": gripper_token(event.record),
        "previous_gripper_token": gripper_token(event.previous),
    }
    row.update(artifact_map(event))
    return row


def select_events(capture_root: Path, pool: Iterable[Candidate], attempt_rows: Iterable[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], list[Event], str]:
    record_paths = captured_record_paths(attempt_rows)
    all_rows: list[dict[str, Any]] = []
    events: list[Event] = []
    for candidate in pool:
        key = (candidate.task, candidate.state_id)
        if key not in record_paths:
            all_rows.append({"task": candidate.task, "state_id": candidate.state_id, "state_hash": candidate.state_hash, "status": "V5_CLEAN_EVENT_NOT_CAPTURED", "reason": "no_captured_attempt"})
            continue
        path = capture_root / safe_rel(record_paths[key])
        status, event, reason = first_clean_close_event(load_clean_records(path), candidate)
        row = {"task": candidate.task, "state_id": candidate.state_id, "state_hash": candidate.state_hash, "status": status, "reason": reason, "selected_step": "" if event is None else event.step}
        all_rows.append(row)
        if event is not None:
            events.append(event)
    events.sort(key=lambda event: event.state_hash)
    if len(events) < V5_PANEL_SIZE:
        return all_rows, events, "V5_CAPTURE_POOL_INSUFFICIENT"
    return all_rows, events[:V5_PANEL_SIZE], "V5_EVENT_PANEL_INPUTS_FROZEN"


def capture_manifest(capture_root: Path) -> dict[str, str]:
    path = capture_root / "m3_arm_v5_clean_capture_manifest.csv"
    if not path.exists():
        return {}
    rows = read_csv(path)
    return rows[0] if rows else {}


def verify_exact_binding(row: Mapping[str, Any], *, capture_root: Path, expected_commit: str, expected_model_bundle_sha: str) -> None:
    for field in EXACT_INPUT_REQUIRED_FIELDS:
        if row.get(field, "") in ("", None):
            raise ValueError(f"missing exact input field: {field}")
    if str(row["commit"]) != expected_commit:
        raise ValueError("capture commit mismatch")
    if str(row["worktree_status"]) != "CLEAN":
        raise ValueError("capture worktree was not clean")
    if str(row["model_checkpoint_sha256"]) != expected_model_bundle_sha:
        raise ValueError("model bundle sha mismatch in selected row")
    if "GPU-" not in str(row["gpu_query"]):
        raise ValueError("invalid GPU query snapshot in selected row")
    for rel_field, sha_field in (
        ("raw_image_path", "raw_image_sha256"),
        ("processed_tensor_path", "processed_tensor_sha256"),
        ("previous_raw_image_path", "previous_raw_image_sha256"),
        ("previous_processed_tensor_path", "previous_processed_tensor_sha256"),
        ("clean_record_source_path", "clean_record_source_sha256"),
    ):
        path = capture_root / safe_rel(str(row[rel_field]))
        if not path.is_file():
            raise ValueError(f"bound artifact missing: {rel_field}")
        if sha256_file(path) != str(row[sha_field]):
            raise ValueError(f"bound artifact sha mismatch: {rel_field}")
    for field, sha_field in (
        ("prompt_token_ids", "prompt_token_ids_sha256"),
        ("previous_prompt_token_ids", "previous_prompt_token_ids_sha256"),
    ):
        if sha256_text(str(row[field])) != str(row[sha_field]):
            raise ValueError(f"prompt token sha mismatch: {field}")
    if int(row["official_score_argmax_token_id"]) != int(row["clean_gripper_token"]):
        raise ValueError("official argmax/current emitted mismatch")
    if int(row["previous_official_score_argmax_token_id"]) != int(row["previous_gripper_token"]):
        raise ValueError("official argmax/previous emitted mismatch")


def verify_selected_bindings(rows: Iterable[Mapping[str, Any]], *, capture_root: Path, expected_commit: str, expected_model_bundle_sha: str) -> None:
    seen_raw: set[str] = set()
    seen_tensor: set[str] = set()
    for row in rows:
        verify_exact_binding(row, capture_root=capture_root, expected_commit=expected_commit, expected_model_bundle_sha=expected_model_bundle_sha)
        raw = str(row["raw_image_path"])
        tensor = str(row["processed_tensor_path"])
        if raw in seen_raw or tensor in seen_tensor:
            raise ValueError("duplicate selected raw/tensor artifact")
        seen_raw.add(raw)
        seen_tensor.add(tensor)


def audit_capture_root(*, capture_root: Path, config_path: Path, expected_commit: str = "") -> dict[str, object]:
    try:
        cfg = load_config(config_path)
        pool = validate_state_pool(cfg, config_path)
        manifest = capture_manifest(capture_root)
        capture_commit = str(manifest.get("commit", ""))
        if not capture_commit:
            raise ValueError("capture commit missing from provenance manifest")
        if expected_commit and capture_commit != expected_commit:
            raise ValueError("expected commit does not match capture provenance commit")
        model_bundle_sha = verify_model_bundle_exact_set(capture_root / "m3_arm_v5_model_bundle_manifest.csv", Path(str(cfg["model"]["path"])))
        attempt_rows, ledger_present = load_attempt_rows(capture_root, pool)
        validate_attempt_policy(capture_root, attempt_rows, pool)
        all_rows, selected, selection_status = select_events(capture_root, pool, attempt_rows)
        selected_rows = [event_row(event) for event in selected]
        if selection_status == "V5_EVENT_PANEL_INPUTS_FROZEN":
            if len(selected_rows) != V5_PANEL_SIZE:
                raise ValueError("selected row count mismatch")
            verify_selected_bindings(selected_rows, capture_root=capture_root, expected_commit=capture_commit, expected_model_bundle_sha=model_bundle_sha)
        audit_status = "PASS" if selection_status == "V5_EVENT_PANEL_INPUTS_FROZEN" else "FAIL"
        failure_reason = "" if audit_status == "PASS" else selection_status
        return {
            "audit_status": audit_status,
            "failure_reason": failure_reason,
            "capture_root": str(capture_root),
            "config_path": str(config_path),
            "attempt_rows": len(attempt_rows),
            "ledger_present": ledger_present,
            "pool_size": len(pool),
            "selection_status": selection_status,
            "selected_count": len(selected_rows),
            "captured_count": sum(1 for row in attempt_rows if str(row.get("attempt_status")) == "CAPTURED"),
            "post_action_interrupted_count": sum(1 for row in attempt_rows if str(row.get("attempt_status")) == "CAPTURE_FAILED_POST_ACTION"),
            "not_started_count": sum(1 for row in attempt_rows if str(row.get("attempt_status")) == "NOT_STARTED"),
            "model_bundle_sha256": model_bundle_sha,
            "capture_commit": capture_commit,
            "expected_commit": expected_commit,
            "all_state_rows": all_rows,
            "selected_rows": selected_rows,
        }
    except Exception as exc:
        return {
            "audit_status": "FAIL",
            "failure_reason": repr(exc),
            "capture_root": str(capture_root),
            "config_path": str(config_path),
            "selected_count": 0,
        }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture_root", required=True)
    ap.add_argument("--config", default=str(REPO_ROOT / "configs" / "m3_arm_v5_clean_close_event_panel.yaml"))
    ap.add_argument("--expected_commit", default="")
    ap.add_argument("--audit_output", default="")
    args = ap.parse_args()
    out = audit_capture_root(capture_root=Path(args.capture_root), config_path=Path(args.config), expected_commit=str(args.expected_commit or ""))
    output_path = Path(args.audit_output) if args.audit_output else Path(args.capture_root) / "m3_arm_v5_clean_capture_external_audit.json"
    write_json(output_path, out)
    print(json.dumps(out, indent=2, sort_keys=True))
    if out["audit_status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
