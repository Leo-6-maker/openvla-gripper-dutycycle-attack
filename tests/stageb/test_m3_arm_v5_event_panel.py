from __future__ import annotations

from pathlib import Path

import yaml

from gripper_attack.m3_event_panel import (
    V5_EVENT_GRIPPER_TOKEN,
    V5_EXCLUDED_DEVELOPMENT_STATES,
    V5_HASH_SALT,
    V5_MAX_STEP,
    V5_MIN_STEP,
    V5_TASKS,
    find_first_clean_close_onset,
    select_first_eligible_events_by_hash,
    select_two_states_per_task,
    v5_state_hash,
)


CONFIG = Path("configs/m3_arm_v5_clean_close_event_panel.yaml")


def _record(step, token, invariant=True, tokens=None):
    if tokens is None:
        tokens = [1, 2, 3, 4, 5, 6, token]
    return {
        "step": step,
        "tokens": tokens,
        "gripper_token": token,
        "score_invariant": {"tie_aware_pass": invariant},
    }


def test_v5_hash_salt_and_state_pool_are_frozen():
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert cfg["selection"]["hash_salt"] == V5_HASH_SALT
    assert cfg["selection"]["state_id_candidates"] == "0..49"
    assert cfg["selection"]["min_step"] == V5_MIN_STEP
    assert cfg["selection"]["max_step"] == V5_MAX_STEP
    assert cfg["selection"]["event_gripper_token"] == V5_EVENT_GRIPPER_TOKEN
    assert len(cfg["task_state_pool"]) == 20


def test_v5_deterministic_state_selection_matches_config():
    selected = select_two_states_per_task()
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    config_rows = [
        (row["task"], int(row["state_id"]), int(row["task_rank"]), row["state_hash"])
        for row in cfg["task_state_pool"]
    ]
    helper_rows = [(row.task, row.state_id, row.task_rank, row.state_hash) for row in selected]
    assert helper_rows == config_rows


def test_v5_excludes_known_development_states():
    selected = select_two_states_per_task()
    selected_by_task = {(row.task, row.state_id) for row in selected}
    for task, states in V5_EXCLUDED_DEVELOPMENT_STATES.items():
        for state in states:
            assert (task, state) not in selected_by_task


def test_v5_state_selection_uses_only_task_state_hash():
    task = "ketchup"
    state = 41
    expected = "0014853d62436e8368afb9230dfd883930c5406d4e02a882a4470e003573c22c"
    assert v5_state_hash(task, state) == expected


def test_v5_selects_two_states_per_each_task():
    selected = select_two_states_per_task()
    counts = {task: 0 for task in V5_TASKS}
    for row in selected:
        counts[row.task] += 1
    assert set(counts.values()) == {2}


def test_clean_close_event_requires_previous_non_close():
    records = [
        _record(10, 31744),
        _record(11, 31872),
    ]
    event = find_first_clean_close_onset(records, task="alphabet_soup", state_id=9)
    assert event is not None
    assert event.step == 11
    assert event.previous_gripper_token == 31744

    no_onset = [
        _record(10, 31872),
        _record(11, 31872),
    ]
    assert find_first_clean_close_onset(no_onset, task="alphabet_soup", state_id=9) is None


def test_clean_close_event_requires_exact7_and_score_invariant():
    bad_tokens = [_record(10, 31744), _record(11, 31872, tokens=[1, 2, 3])]
    assert find_first_clean_close_onset(bad_tokens, task="milk", state_id=35) is None

    bad_invariant = [_record(10, 31744), _record(11, 31872, invariant=False)]
    assert find_first_clean_close_onset(bad_invariant, task="milk", state_id=35) is None


def test_clean_close_event_uses_earliest_qualifying_event_per_state():
    records = [
        _record(10, 31744),
        _record(11, 31872),
        _record(12, 31744),
        _record(13, 31872),
    ]
    event = find_first_clean_close_onset(records, task="orange_juice", state_id=11)
    assert event is not None
    assert event.step == 11


def test_clean_close_event_respects_min_max_step():
    records = [
        _record(4, 31744),
        _record(5, 31872),
        _record(9, 31744),
        _record(10, 31872),
    ]
    event = find_first_clean_close_onset(records, task="salad_dressing", state_id=11, min_step=10, max_step=20)
    assert event is not None
    assert event.step == 10
    assert find_first_clean_close_onset(records, task="salad_dressing", state_id=11, min_step=12, max_step=20) is None


def test_first_eight_eligible_states_selected_by_hash_order():
    candidates = select_two_states_per_task()
    events = {}
    for candidate in candidates:
        events[(candidate.task, candidate.state_id)] = find_first_clean_close_onset(
            [_record(0, 31744), _record(1, 31872)],
            task=candidate.task,
            state_id=candidate.state_id,
        )
    selected, status = select_first_eligible_events_by_hash(events, candidates)
    expected = [candidate for candidate in sorted(candidates, key=lambda c: c.state_hash)[:8]]
    assert status == "V5_EVENT_PANEL_INPUTS_FROZEN"
    assert [(e.task, e.state_id) for e in selected] == [(c.task, c.state_id) for c in expected]


def test_insufficient_pool_stop_rule():
    candidates = select_two_states_per_task()
    events = {}
    for candidate in candidates[:7]:
        events[(candidate.task, candidate.state_id)] = find_first_clean_close_onset(
            [_record(0, 31744), _record(1, 31872)],
            task=candidate.task,
            state_id=candidate.state_id,
        )
    selected, status = select_first_eligible_events_by_hash(events, candidates)
    assert len(selected) == 7
    assert status == "V5_CAPTURE_POOL_INSUFFICIENT"
