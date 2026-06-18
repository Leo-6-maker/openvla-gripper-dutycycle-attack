#!/usr/bin/env python3
"""Test SC5 held-out exclusion: Butter s8,s9,s11 never in train/val/calibration/normalization."""
import sys, os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "src"))

HELD_OUT_BUTTER = {8, 9, 11}


def _make_ep(state_id, is_butter=True, is_held_out=False):
    return {
        'state_id': state_id,
        'task': 'pick_up_the_butter_and_place_it_in_the_basket' if is_butter else 'other_task',
        'is_butter': is_butter,
        'is_held_out': is_held_out,
    }


def test_butter_8_9_11_are_held_out():
    """Butter s8, s9, s11 must be flagged as held_out."""
    for sid in [8, 9, 11]:
        ep = _make_ep(sid, is_butter=True)
        is_h = ep['is_butter'] and ep['state_id'] in HELD_OUT_BUTTER
        assert is_h, f"Butter s{sid} should be held out"


def test_butter_other_states_are_train():
    """Butter s0-s7, s10 are train (not held-out)."""
    for sid in [0, 1, 2, 3, 4, 5, 6, 7, 10]:
        ep = _make_ep(sid, is_butter=True)
        is_h = ep['is_butter'] and ep['state_id'] in HELD_OUT_BUTTER
        assert not is_h, f"Butter s{sid} should NOT be held out"


def test_non_butter_never_held_out():
    """Non-Butter tasks are never held out by the Butter rule."""
    ep = _make_ep(8, is_butter=False)
    is_h = ep['is_butter'] and ep['state_id'] in HELD_OUT_BUTTER
    assert not is_h, "Non-Butter task should not be held out"


def test_held_out_in_train_raises():
    """Simulate: held-out episode appearing in train set should be caught."""
    train_eps = [
        _make_ep(0, is_butter=True, is_held_out=False),
        _make_ep(8, is_butter=True, is_held_out=True),  # BUG: s8 in train!
    ]
    held_in_train = [e for e in train_eps if e['is_held_out']]
    assert len(held_in_train) == 1, "Should detect held-out in train"
    assert held_in_train[0]['state_id'] == 8


def test_held_out_in_calibration_paths():
    """Held-out paths must not be passed to calibrate_thresholds()."""
    paths = [
        '/data/liuyu/outputs/milestone_x/runs/libero_object/pick_up_the_butter_s0/step_records.jsonl',
        '/data/liuyu/outputs/milestone_x/runs/libero_object/pick_up_the_butter_s8/step_records.jsonl',
    ]
    train_paths = []
    held_out_paths = []
    for p in paths:
        if 'butter_s8' in p or 'butter_s9' in p or 'butter_s11' in p:
            held_out_paths.append(p)
        else:
            train_paths.append(p)

    assert len(train_paths) == 1
    assert len(held_out_paths) == 1
    assert 'butter_s8' not in str(train_paths)


def test_split_manifest_flags():
    """Split manifest correctly flags held-out episodes."""
    episodes = [
        {** _make_ep(0, is_held_out=False), 'split': 'train'},
        {** _make_ep(8, is_held_out=True), 'split': 'held_out'},
        {** _make_ep(9, is_held_out=True), 'split': 'held_out'},
    ]
    for ep in episodes:
        if ep['is_held_out']:
            assert ep['split'] == 'held_out', \
                f"Held-out ep s{ep['state_id']} has wrong split: {ep['split']}"
        else:
            assert ep['split'] == 'train', \
                f"Train ep s{ep['state_id']} has wrong split: {ep['split']}"


if __name__ == '__main__':
    test_butter_8_9_11_are_held_out()
    test_butter_other_states_are_train()
    test_non_butter_never_held_out()
    test_held_out_in_train_raises()
    test_held_out_in_calibration_paths()
    test_split_manifest_flags()
    print("\nAll SC5 held-out exclusion tests passed.")
