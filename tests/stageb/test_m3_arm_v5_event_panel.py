from __future__ import annotations

from pathlib import Path

import yaml

from gripper_attack.m3_event_panel import (
    V5_ATTACK_SEED_HASH,
    V5_EVENT_GRIPPER_TOKEN,
    V5_EXCLUDED_DEVELOPMENT_STATES,
    V5_FROZEN_ATTACK_SEED,
    V5_HASH_SALT,
    V5_MAX_STEP,
    V5_MIN_STEP,
    V5_TASKS,
    excluded_states_from_prior_ledger,
    find_first_clean_close_onset,
    find_first_clean_close_onset_with_status,
    load_prior_layer3_state_ledger,
    select_first_eligible_events_by_hash,
    select_two_states_per_task,
    validate_state_pool_against_ledger,
    v5_state_hash,
)


CONFIG = Path("configs/m3_arm_v5_clean_close_event_panel.yaml")


def _record(step, token, invariant=True, tokens=None, *, task="alphabet_soup", state_id=9, argmax=None):
    if tokens is None:
        tokens = [1, 2, 3, 4, 5, 6, token]
    return {
        "task": task,
        "state_id": state_id,
        "step": step,
        "tokens": tokens,
        "gripper_token": token,
        "official_score_argmax_token_id": token if argmax is None else argmax,
        "score_invariant": {"tie_aware_pass": invariant},
    }


def test_v5_hash_salt_and_state_pool_are_frozen():
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert cfg["selection"]["hash_salt"] == V5_HASH_SALT
    assert cfg["selection"]["state_id_candidates"] == "0..49"
    assert cfg["selection"]["min_step"] == V5_MIN_STEP
    assert cfg["selection"]["max_step"] == V5_MAX_STEP
    assert cfg["selection"]["event_gripper_token"] == V5_EVENT_GRIPPER_TOKEN
    assert cfg["selection"]["prior_layer3_state_ledger"] == "tables/m3_arm_v5_prior_layer3_state_ledger.csv"
    assert cfg["selection"]["first_attack_seed"]["seed"] == V5_FROZEN_ATTACK_SEED
    assert cfg["selection"]["first_attack_seed"]["sha256"] == V5_ATTACK_SEED_HASH
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


def test_v5_exclusions_are_loaded_from_prior_layer3_ledger():
    ledger = load_prior_layer3_state_ledger("tables/m3_arm_v5_prior_layer3_state_ledger.csv")
    excluded = excluded_states_from_prior_ledger(ledger)
    assert excluded == V5_EXCLUDED_DEVELOPMENT_STATES
    validate_state_pool_against_ledger(select_two_states_per_task(), ledger)

    bad_pool = list(select_two_states_per_task())
    bad_pool[0] = type(bad_pool[0])(
        task="tomato_sauce",
        state_id=0,
        task_rank=1,
        state_hash=v5_state_hash("tomato_sauce", 0),
    )
    import pytest

    with pytest.raises(ValueError, match="prior Layer3 development state"):
        validate_state_pool_against_ledger(bad_pool, ledger)

    arbitrary = list(select_two_states_per_task())
    arbitrary[0] = type(arbitrary[0])(
        task="ketchup",
        state_id=0,
        task_rank=1,
        state_hash=v5_state_hash("ketchup", 0),
    )
    with pytest.raises(ValueError, match="ledger-derived hash selection"):
        validate_state_pool_against_ledger(arbitrary, ledger)


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


def test_clean_close_event_requires_strict_adjacent_unique_steps():
    duplicate = [_record(10, 31744), _record(10, 31872)]
    result = find_first_clean_close_onset_with_status(duplicate, task="alphabet_soup", state_id=9)
    assert result.status == "V5_CLEAN_EVENT_INFRA_INVALID"
    assert result.reason == "duplicate_step"

    gap = [_record(10, 31744), _record(12, 31872)]
    result = find_first_clean_close_onset_with_status(gap, task="alphabet_soup", state_id=9)
    assert result.status == "V5_CLEAN_EVENT_INFRA_INVALID"
    assert result.reason == "step_gap"

    non_increasing = [_record(10, 31744), _record(9, 31872)]
    result = find_first_clean_close_onset_with_status(non_increasing, task="alphabet_soup", state_id=9)
    assert result.status == "V5_CLEAN_EVENT_INFRA_INVALID"
    assert result.reason == "non_increasing_step"


def test_clean_close_event_rejects_token_field_mismatch_and_argmax_mismatch():
    mismatch = [
        _record(10, 31744, task="milk", state_id=35),
        _record(11, 12345, tokens=[1, 2, 3, 4, 5, 6, 31872], task="milk", state_id=35),
    ]
    result = find_first_clean_close_onset_with_status(mismatch, task="milk", state_id=35)
    assert result.status == "V5_CLEAN_EVENT_INFRA_INVALID"
    assert result.reason == "gripper_token_mismatch"

    argmax = [_record(10, 31744, task="milk", state_id=35), _record(11, 31872, task="milk", state_id=35, argmax=31744)]
    result = find_first_clean_close_onset_with_status(argmax, task="milk", state_id=35)
    assert result.status == "V5_CLEAN_EVENT_INFRA_INVALID"
    assert result.reason == "official_argmax_emitted_mismatch"


def test_clean_close_event_rejects_missing_argmax_evidence():
    records = [_record(10, 31744), _record(11, 31872)]
    records[1].pop("official_score_argmax_token_id")
    result = find_first_clean_close_onset_with_status(records, task="alphabet_soup", state_id=9)
    assert result.status == "V5_CLEAN_EVENT_INFRA_INVALID"
    assert result.reason == "missing_official_argmax_evidence"

    records = [_record(10, 31744), _record(11, 31872)]
    records[0].pop("official_score_argmax_token_id")
    result = find_first_clean_close_onset_with_status(records, task="alphabet_soup", state_id=9)
    assert result.status == "V5_CLEAN_EVENT_INFRA_INVALID"
    assert result.reason == "missing_official_argmax_evidence"


def test_clean_close_event_rejects_wrong_task_or_state():
    result = find_first_clean_close_onset_with_status(
        [_record(10, 31744, task="milk", state_id=35), _record(11, 31872, task="milk", state_id=35)],
        task="ketchup",
        state_id=35,
    )
    assert result.status == "V5_CLEAN_EVENT_INFRA_INVALID"
    assert result.reason == "task_mismatch"

    result = find_first_clean_close_onset_with_status(
        [_record(10, 31744, task="milk", state_id=35), _record(11, 31872, task="milk", state_id=35)],
        task="milk",
        state_id=0,
    )
    assert result.status == "V5_CLEAN_EVENT_INFRA_INVALID"
    assert result.reason == "state_id_mismatch"


def test_clean_close_event_requires_exact7_and_score_invariant():
    bad_tokens = [_record(10, 31744, task="milk", state_id=35), _record(11, 31872, tokens=[1, 2, 3], task="milk", state_id=35)]
    assert find_first_clean_close_onset(bad_tokens, task="milk", state_id=35) is None

    bad_invariant = [_record(10, 31744, task="milk", state_id=35), _record(11, 31872, invariant=False, task="milk", state_id=35)]
    assert find_first_clean_close_onset(bad_invariant, task="milk", state_id=35) is None
    result = find_first_clean_close_onset_with_status(bad_invariant, task="milk", state_id=35)
    assert result.status == "V5_CLEAN_EVENT_INFRA_INVALID"
    assert result.reason == "score_invariant_not_pass"


def test_clean_close_event_uses_earliest_qualifying_event_per_state():
    records = [
        _record(10, 31744, task="orange_juice", state_id=11),
        _record(11, 31872, task="orange_juice", state_id=11),
        _record(12, 31744, task="orange_juice", state_id=11),
        _record(13, 31872, task="orange_juice", state_id=11),
    ]
    event = find_first_clean_close_onset(records, task="orange_juice", state_id=11)
    assert event is not None
    assert event.step == 11


def test_clean_close_event_validates_corruption_after_apparent_event():
    records = [
        _record(10, 31744),
        _record(11, 31872),
        _record(12, 31744),
    ]
    records[2].pop("official_score_argmax_token_id")
    result = find_first_clean_close_onset_with_status(records, task="alphabet_soup", state_id=9)
    assert result.status == "V5_CLEAN_EVENT_INFRA_INVALID"
    assert result.reason == "missing_official_argmax_evidence"


def test_clean_close_event_respects_min_max_step():
    records = [
        _record(8, 31744, task="salad_dressing", state_id=11),
        _record(9, 31744, task="salad_dressing", state_id=11),
        _record(10, 31872, task="salad_dressing", state_id=11),
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
            [
                _record(0, 31744, task=candidate.task, state_id=candidate.state_id),
                _record(1, 31872, task=candidate.task, state_id=candidate.state_id),
            ],
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
            [
                _record(0, 31744, task=candidate.task, state_id=candidate.state_id),
                _record(1, 31872, task=candidate.task, state_id=candidate.state_id),
            ],
            task=candidate.task,
            state_id=candidate.state_id,
        )
    selected, status = select_first_eligible_events_by_hash(events, candidates)
    assert len(selected) == 7
    assert status == "V5_CAPTURE_POOL_INSUFFICIENT"
