"""Deterministic preregistration helpers for the M3 arm-v5 event panel."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
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


def clean_gripper_token(record: Mapping[str, Any]) -> int | None:
    if "gripper_token" in record:
        return int(record["gripper_token"])
    if "clean_gripper_token" in record:
        return int(record["clean_gripper_token"])
    tokens = _tokens_from_record(record)
    if len(tokens) == 7:
        return int(tokens[-1])
    return None


def find_first_clean_close_onset(
    records: Iterable[Mapping[str, Any]],
    *,
    task: str,
    state_id: int,
    min_step: int = V5_MIN_STEP,
    max_step: int = V5_MAX_STEP,
) -> V5CleanCloseEvent | None:
    ordered = sorted((dict(r) for r in records), key=lambda r: int(r["step"]))
    previous_token: int | None = None
    previous_step: int | None = None
    for record in ordered:
        step = int(record["step"])
        token = clean_gripper_token(record)
        tokens = _tokens_from_record(record)
        exact7 = len(tokens) == 7
        invariant_pass = _score_invariant_pass(record)
        if (
            min_step <= step <= max_step
            and exact7
            and invariant_pass
            and token == V5_EVENT_GRIPPER_TOKEN
            and previous_step is not None
            and previous_token != V5_EVENT_GRIPPER_TOKEN
        ):
            return V5CleanCloseEvent(
                task=task,
                state_id=int(state_id),
                step=step,
                gripper_token=int(token),
                previous_gripper_token=int(previous_token),
                exact_7_tokens=tuple(tokens),
            )
        if token is not None:
            previous_token = int(token)
            previous_step = step
    return None


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
