#!/usr/bin/env python3
"""Test SC5 group-level split constraints: init-state, duplicate, parent-event groups don't cross splits."""
import sys, os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "src"))

from gripper_attack.sc5_dedup import validate_split_isolation


def test_init_state_group_not_cross_split():
    """Same init-state episodes must all go to same split."""
    episodes = [
        {'episode_id': 'a', 'is_held_out': False, 'initial_state_sha256': 'init_x'},
        {'episode_id': 'b', 'is_held_out': False, 'initial_state_sha256': 'init_x'},
        {'episode_id': 'c', 'is_held_out': True, 'initial_state_sha256': 'init_y'},
    ]
    result = validate_split_isolation(episodes, 'initial_state_sha256')
    assert result['valid'], f"Should be valid, got violations: {result['violations']}"


def test_init_state_group_cross_split_violation():
    """Same init-state in both train and held-out = violation."""
    episodes = [
        {'episode_id': 'a', 'is_held_out': False, 'initial_state_sha256': 'init_x'},
        {'episode_id': 'b', 'is_held_out': True, 'initial_state_sha256': 'init_x'},
    ]
    result = validate_split_isolation(episodes, 'initial_state_sha256')
    assert not result['valid']
    assert result['n_violations'] == 1


def test_empty_episodes_valid():
    """Empty episode list is trivially valid."""
    result = validate_split_isolation([], 'initial_state_sha256')
    assert result['valid']
    assert result['n_groups'] == 0


def test_no_hash_episodes_skipped():
    """Episodes without the group key are ignored."""
    episodes = [
        {'episode_id': 'a', 'is_held_out': False, 'initial_state_sha256': ''},
        {'episode_id': 'b', 'is_held_out': True, 'initial_state_sha256': ''},
    ]
    result = validate_split_isolation(episodes, 'initial_state_sha256')
    assert result['valid']  # empty hash = no group


def test_large_group_all_same_split():
    """Many episodes in same init-state group, all in train = valid."""
    episodes = [
        {'episode_id': f'e{i}', 'is_held_out': False, 'initial_state_sha256': 'init_z'}
        for i in range(20)
    ]
    result = validate_split_isolation(episodes, 'initial_state_sha256')
    assert result['valid']


if __name__ == '__main__':
    test_init_state_group_not_cross_split()
    test_init_state_group_cross_split_violation()
    test_empty_episodes_valid()
    test_no_hash_episodes_skipped()
    test_large_group_all_same_split()
    print("\nAll SC5 group split tests passed.")
