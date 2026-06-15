"""Deterministic preregistration helpers for the M3 arm-v5 event panel."""

from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping


V5_HASH_SALT = "M3_ARM_V5_CLOSE_PANEL"
V5_STATE_ID_CANDIDATES = tuple(range(50))
V5_MIN_STEP = 0
V5_MAX_STEP = 279
V5_EVENT_GRIPPER_TOKEN = 31872
V5_TARGET_TOKEN = 31744
V5_PANEL_SIZE = 8

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

V5_EXCLUDED_DEVELOPMENT_STATES: dict[str, tuple[int, ...]] = {
    "butter": (2,),
    "cream_cheese": (2,),
    "tomato_sauce": (0,),
}

V5_ATTACK_SEED_HASH = hashlib.sha256(f"{V5_HASH_SALT}|attack_seed|v5.2|seed1".encode("utf-8")).hexdigest()
V5_FROZEN_ATTACK_SEED = int(V5_ATTACK_SEED_HASH[:8], 16) % 1_000_000


@dataclass(frozen=True)
class V5StateCandidate:
    task: str
    state_id: int
    state_hash: str
    task_rank: int


@dataclass(frozen=True)
class V5CleanCloseEvent:
    task: str
    state_id: int
    step: int
    gripper_token: int
    previous_gripper_token: int
    exact_7_tokens: tuple[int, ...]
    previous_exact_7_tokens: tuple[int, ...] = ()


@dataclass(frozen=True)
class V5EventSelectionResult:
    status: str
    event: V5CleanCloseEvent | None = None
    reason: str = ""
    invalid_step: int | None = None


@dataclass(frozen=True)
class V5PriorStateLedgerRow:
    task: str
    state_id: int
    prior_stage: str
    commit: str
    artifact_path: str
    used_for_development: bool
    exclusion_reason: str


def v5_state_hash(task: str, state_id: int) -> str:
    return hashlib.sha256(f"{V5_HASH_SALT}|{task}|{int(state_id)}".encode("utf-8")).hexdigest()


def select_two_states_per_task(
    *,
    tasks: Iterable[str] = V5_TASKS,
    state_ids: Iterable[int] = V5_STATE_ID_CANDIDATES,
    excluded: Mapping[str, Iterable[int]] = V5_EXCLUDED_DEVELOPMENT_STATES,
) -> list[V5StateCandidate]:
    selected: list[V5StateCandidate] = []
    for task in tasks:
        excluded_states = {int(x) for x in excluded.get(task, ())}
        candidates = [
            (v5_state_hash(task, state_id), int(state_id))
            for state_id in state_ids
            if int(state_id) not in excluded_states
        ]
        candidates.sort()
        if len(candidates) < 2:
            raise ValueError(f"not enough state candidates for task {task}")
        for rank, (state_hash, state_id) in enumerate(candidates[:2], start=1):
            selected.append(V5StateCandidate(task=task, state_id=state_id, state_hash=state_hash, task_rank=rank))
    return selected


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def load_prior_layer3_state_ledger(path: str | Path) -> list[V5PriorStateLedgerRow]:
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    ledger: list[V5PriorStateLedgerRow] = []
    for row in rows:
        ledger.append(
            V5PriorStateLedgerRow(
                task=str(row["task"]),
                state_id=int(row["state_id"]),
                prior_stage=str(row.get("prior_stage", "")),
                commit=str(row.get("commit", "")),
                artifact_path=str(row.get("artifact_path", "")),
                used_for_development=parse_bool(row.get("used_for_development", "")),
                exclusion_reason=str(row.get("exclusion_reason", "")),
            )
        )
    return ledger


def excluded_states_from_prior_ledger(
    rows: Iterable[V5PriorStateLedgerRow],
) -> dict[str, tuple[int, ...]]:
    excluded: dict[str, set[int]] = {}
    for row in rows:
        if row.used_for_development:
            excluded.setdefault(row.task, set()).add(row.state_id)
    return {task: tuple(sorted(states)) for task, states in sorted(excluded.items())}


def validate_state_pool_against_ledger(
    pool: Iterable[V5StateCandidate | Mapping[str, Any]],
    ledger_rows: Iterable[V5PriorStateLedgerRow],
) -> None:
    excluded = {
        (row.task, row.state_id)
        for row in ledger_rows
        if row.used_for_development
    }
    pool_pairs: set[tuple[str, int]] = set()
    for row in pool:
        if isinstance(row, V5StateCandidate):
            pair = (row.task, row.state_id)
        else:
            pair = (str(row["task"]), int(row["state_id"]))
        if pair in excluded:
            raise ValueError(f"state pool includes prior Layer3 development state: {pair[0]}_s{pair[1]}")
        if pair in pool_pairs:
            raise ValueError(f"state pool contains duplicate state: {pair[0]}_s{pair[1]}")
        pool_pairs.add(pair)


def _tokens_from_record(record: Mapping[str, Any]) -> list[int]:
    if "tokens" in record:
        return [int(x) for x in record["tokens"]]
    if "clean_exact_7_tokens" in record:
        value = record["clean_exact_7_tokens"]
        if isinstance(value, str):
            import json

            return [int(x) for x in json.loads(value)]
        return [int(x) for x in value]
    return []


def _score_invariant_pass(record: Mapping[str, Any]) -> bool:
    if "score_invariant_status" in record:
        return str(record["score_invariant_status"]).upper() == "PASS"
    invariant = record.get("score_invariant", {})
    if isinstance(invariant, Mapping):
        return bool(invariant.get("tie_aware_pass", invariant.get("pass", False)))
    if "score_tie_aware_pass" in record:
        return str(record["score_tie_aware_pass"]).lower() in {"true", "1", "yes"}
    return False


def _official_argmax_matches_emitted(record: Mapping[str, Any], token: int | None) -> bool:
    if token is None:
        return False
    for key in (
        "score_argmax_token_id",
        "official_score_argmax_token_id",
        "processed_score_argmax_token_id",
        "generation_score_argmax",
    ):
        if key in record and record[key] not in ("", None):
            return int(record[key]) == int(token)
    audit = record.get("generation_score_audit", record.get("official_generation_score_audit", {}))
    if isinstance(audit, Mapping):
        for key in ("argmax_token_id", "top1_token_id", "top_token_id"):
            if key in audit and audit[key] not in ("", None):
                return int(audit[key]) == int(token)
    return True


def clean_gripper_token(record: Mapping[str, Any]) -> int | None:
    if "gripper_token" in record:
        return int(record["gripper_token"])
    if "clean_gripper_token" in record:
        return int(record["clean_gripper_token"])
    tokens = _tokens_from_record(record)
    if len(tokens) == 7:
        return int(tokens[-1])
    return None


def _record_validation_status(record: Mapping[str, Any]) -> tuple[str, int | None, list[int]]:
    tokens = _tokens_from_record(record)
    token = clean_gripper_token(record)
    if len(tokens) != 7:
        return "invalid_exact_7_tokens", token, tokens
    if token is None:
        return "missing_gripper_token", token, tokens
    if int(token) != int(tokens[-1]):
        return "gripper_token_mismatch", token, tokens
    if not _score_invariant_pass(record):
        return "score_invariant_not_pass", token, tokens
    if not _official_argmax_matches_emitted(record, token):
        return "official_argmax_emitted_mismatch", token, tokens
    return "pass", int(token), tokens


def find_first_clean_close_onset_with_status(
    records: Iterable[Mapping[str, Any]],
    *,
    task: str,
    state_id: int,
    min_step: int = V5_MIN_STEP,
    max_step: int = V5_MAX_STEP,
) -> V5EventSelectionResult:
    previous_step: int | None = None
    previous_token: int | None = None
    previous_tokens: list[int] | None = None
    seen_steps: set[int] = set()
    saw_any = False
    for raw_record in records:
        saw_any = True
        record = dict(raw_record)
        if "step" not in record:
            return V5EventSelectionResult("V5_CLEAN_EVENT_INFRA_INVALID", reason="missing_step")
        step = int(record["step"])
        if step in seen_steps:
            return V5EventSelectionResult(
                "V5_CLEAN_EVENT_INFRA_INVALID",
                reason="duplicate_step",
                invalid_step=step,
            )
        seen_steps.add(step)
        if previous_step is not None:
            if step <= previous_step:
                return V5EventSelectionResult(
                    "V5_CLEAN_EVENT_INFRA_INVALID",
                    reason="non_increasing_step",
                    invalid_step=step,
                )
            if step != previous_step + 1:
                return V5EventSelectionResult(
                    "V5_CLEAN_EVENT_INFRA_INVALID",
                    reason="step_gap",
                    invalid_step=step,
                )
        status, token, tokens = _record_validation_status(record)
        if status != "pass":
            return V5EventSelectionResult(
                "V5_CLEAN_EVENT_INFRA_INVALID",
                reason=status,
                invalid_step=step,
            )
        in_window = min_step <= step <= max_step
        if in_window and token == V5_EVENT_GRIPPER_TOKEN:
            if previous_step is None or previous_step != step - 1:
                return V5EventSelectionResult(
                    "V5_CLEAN_EVENT_INFRA_INVALID",
                    reason="missing_adjacent_previous_step",
                    invalid_step=step,
                )
            if previous_token is None:
                return V5EventSelectionResult(
                    "V5_CLEAN_EVENT_INFRA_INVALID",
                    reason="invalid_previous_token",
                    invalid_step=step,
                )
            if previous_token != V5_EVENT_GRIPPER_TOKEN:
                return V5EventSelectionResult(
                    "V5_CLEAN_EVENT_FOUND",
                    event=V5CleanCloseEvent(
                        task=task,
                        state_id=int(state_id),
                        step=step,
                        gripper_token=int(token),
                        previous_gripper_token=int(previous_token),
                        exact_7_tokens=tuple(tokens),
                        previous_exact_7_tokens=tuple(previous_tokens or ()),
                    ),
                )
        previous_step = step
        previous_token = token
        previous_tokens = tokens
    if not saw_any:
        return V5EventSelectionResult("V5_CLEAN_EVENT_NOT_FOUND", reason="empty_records")
    return V5EventSelectionResult("V5_CLEAN_EVENT_NOT_FOUND", reason="no_clean_close_onset")


def find_first_clean_close_onset(
    records: Iterable[Mapping[str, Any]],
    *,
    task: str,
    state_id: int,
    min_step: int = V5_MIN_STEP,
    max_step: int = V5_MAX_STEP,
) -> V5CleanCloseEvent | None:
    result = find_first_clean_close_onset_with_status(
        records,
        task=task,
        state_id=state_id,
        min_step=min_step,
        max_step=max_step,
    )
    return result.event


def select_first_eligible_events_by_hash(
    events_by_task_state: Mapping[tuple[str, int], V5CleanCloseEvent | None],
    candidates: Iterable[V5StateCandidate],
    *,
    panel_size: int = V5_PANEL_SIZE,
) -> tuple[list[V5CleanCloseEvent], str]:
    eligible: list[tuple[str, V5CleanCloseEvent]] = []
    for candidate in candidates:
        event = events_by_task_state.get((candidate.task, candidate.state_id))
        if event is not None:
            eligible.append((candidate.state_hash, event))
    eligible.sort(key=lambda item: item[0])
    if len(eligible) < panel_size:
        return [event for _hash, event in eligible], "V5_CAPTURE_POOL_INSUFFICIENT"
    return [event for _hash, event in eligible[:panel_size]], "V5_EVENT_PANEL_INPUTS_FROZEN"
