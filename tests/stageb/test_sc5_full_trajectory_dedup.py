#!/usr/bin/env python3
"""Test SC5 trajectory dedup: full-sequence hashing, duplicate groups, split isolation."""
import json, sys, os, tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "src"))

from gripper_attack.sc5_dedup import (
    sha256_hex, trajectory_content_hash, proprio_sequence_hash,
    privileged_sequence_hash, initial_state_hash,
    build_duplicate_groups, dedup_episodes, validate_split_isolation,
    write_duplicate_groups_csv, write_split_manifest_csv,
)


def _make_record(step_idx, eef_x=0.0, eef_y=0.0, eef_z=0.25,
                 gripper=0.3, has_priv=True):
    """Helper: create a minimal step record."""
    r = {
        'step_idx': step_idx,
        'eef_x': str(eef_x), 'eef_y': str(eef_y), 'eef_z': str(eef_z),
        'gripper_command': str(gripper),
        'gripper_qpos': '0.05', 'gripper_width': '0.02',
        'eef_vx': '0', 'eef_vy': '0', 'eef_vz': '0',
        'action_dx': '0', 'action_dy': '0', 'action_dz': '0',
        'action_gripper': '0',
        'teacher_privileged_state_available': has_priv,
    }
    if has_priv:
        r['object_pose_json'] = json.dumps([0.1, 0.2, 0.3])
        r['target_pose_json'] = json.dumps([0.5, 0.0, 0.3])
        r['object_to_target_distance'] = '0.4'
        r['object_eef_distance'] = '0.1'
    return r


def test_identical_trajectories_same_hash():
    """Two identical trajectories produce the same content hash."""
    r1 = [_make_record(i) for i in range(10)]
    r2 = [_make_record(i) for i in range(10)]
    h1 = trajectory_content_hash(r1)
    h2 = trajectory_content_hash(r2)
    assert h1 == h2
    assert len(h1) == 64  # SHA256 hex
    print("PASS: test_identical_trajectories_same_hash")


def test_different_trajectories_different_hash():
    """Different EEF trajectories produce different hashes."""
    r1 = [_make_record(i, eef_x=0.1) for i in range(10)]
    r2 = [_make_record(i, eef_x=0.2) for i in range(10)]
    h1 = trajectory_content_hash(r1)
    h2 = trajectory_content_hash(r2)
    assert h1 != h2
    print("PASS: test_different_trajectories_different_hash")


def test_proprio_hash_excludes_privileged():
    """proprio_sequence_hash should not include object/target pose."""
    r1 = [_make_record(i) for i in range(5)]
    r2 = [_make_record(i) for i in range(5)]
    # Modify privileged data only
    r2[2]['object_pose_json'] = json.dumps([9.9, 9.9, 9.9])
    r2[2]['object_to_target_distance'] = '9.9'
    assert proprio_sequence_hash(r1) == proprio_sequence_hash(r2), \
        "proprio hash should NOT depend on privileged fields"
    assert privileged_sequence_hash(r1) != privileged_sequence_hash(r2), \
        "privileged hash SHOULD differ"
    print("PASS: test_proprio_hash_excludes_privileged")


def test_initial_state_hash_same_init():
    """Same initial state produces same hash even with different later steps."""
    init_records = [_make_record(i) for i in range(5)]
    r1 = init_records + [_make_record(i, eef_x=1.0) for i in range(5, 10)]
    r2 = init_records + [_make_record(i, eef_x=9.0) for i in range(5, 10)]
    assert initial_state_hash(r1) == initial_state_hash(r2)
    assert trajectory_content_hash(r1) != trajectory_content_hash(r2)
    print("PASS: test_initial_state_hash_same_init")


def test_duplicate_groups():
    """Duplicate detection groups identical episodes correctly."""
    episodes = [
        {'episode_id': 'a', 'trajectory_content_sha256': 'h1', 'clean_status': 'CLEAN'},
        {'episode_id': 'b', 'trajectory_content_sha256': 'h1', 'clean_status': 'CLEAN'},
        {'episode_id': 'c', 'trajectory_content_sha256': 'h2', 'clean_status': 'CLEAN'},
    ]
    groups = build_duplicate_groups(episodes, 'trajectory_content_sha256')
    assert len(groups) == 1  # only h1 has duplicates
    assert groups[0]['n_members'] == 2
    assert 'a' in groups[0]['episode_ids']
    assert 'b' in groups[0]['episode_ids']
    print("PASS: test_duplicate_groups")


def test_dedup_keeps_highest_priority():
    """Dedup keeps the episode with best provenance."""
    episodes = [
        {'episode_id': 'low', 'trajectory_content_sha256': 'h1',
         'clean_status': 'UNCLEAN', 'schema_status': 'FAIL'},
        {'episode_id': 'high', 'trajectory_content_sha256': 'h1',
         'clean_status': 'CLEAN', 'schema_status': 'PASS'},
    ]
    unique, groups = dedup_episodes(episodes, 'trajectory_content_sha256')
    assert len(unique) == 1
    assert unique[0]['episode_id'] == 'high'
    print("PASS: test_dedup_keeps_highest_priority")


def test_no_duplicates():
    """All unique trajectories produce no duplicate groups."""
    episodes = [
        {'episode_id': 'a', 'trajectory_content_sha256': 'h1'},
        {'episode_id': 'b', 'trajectory_content_sha256': 'h2'},
        {'episode_id': 'c', 'trajectory_content_sha256': 'h3'},
    ]
    groups = build_duplicate_groups(episodes, 'trajectory_content_sha256')
    assert len(groups) == 0
    unique, _ = dedup_episodes(episodes, 'trajectory_content_sha256')
    assert len(unique) == 3
    print("PASS: test_no_duplicates")


def test_split_isolation_valid():
    """No cross-split violations when groups are isolated."""
    episodes = [
        {'is_held_out': False, 'initial_state_sha256': 'init_a'},
        {'is_held_out': False, 'initial_state_sha256': 'init_a'},  # same init, both train
        {'is_held_out': True, 'initial_state_sha256': 'init_b'},
    ]
    result = validate_split_isolation(episodes, 'initial_state_sha256')
    assert result['valid']
    assert result['n_violations'] == 0
    print("PASS: test_split_isolation_valid")


def test_split_isolation_violation():
    """Cross-split init-state group detected."""
    episodes = [
        {'is_held_out': False, 'initial_state_sha256': 'init_a'},
        {'is_held_out': True, 'initial_state_sha256': 'init_a'},  # same init in held-out!
    ]
    result = validate_split_isolation(episodes, 'initial_state_sha256')
    assert not result['valid']
    assert result['n_violations'] == 1
    print("PASS: test_split_isolation_violation")


def test_csv_output():
    """Duplicate groups and split manifest CSVs are written correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Duplicate groups CSV
        groups = [{
            'group_id': 'test_hash_12345',
            'hash': 'test_hash_1234567890abcdef1234567890abcdef',
            'n_members': 2,
            'episode_ids': ['ep_a', 'ep_b'],
            'keep_idx': 0,
            'drop_indices': [1],
        }]
        dup_path = os.path.join(tmpdir, 'dup_groups.csv')
        write_duplicate_groups_csv(groups, dup_path)
        assert os.path.isfile(dup_path)

        # Split manifest CSV
        episodes = [{
            'episode_id': 'ep_a', 'task': 'test_task', 'state_id': 0,
            'suite': 'libero_object', 'mechanism_tier': 'PRIMARY',
            'split': 'train', 'is_held_out': False,
            'trajectory_content_sha256': 'h1', 'proprio_sequence_sha256': 'h2',
            'initial_state_sha256': 'h3', 'source_file_sha256': 'h4',
            'init_state_group_id': 'g1', 'duplicate_group_id': '',
            'n_steps': 100, 'sc5_valid': True, 'sc5_anchor': 50,
        }]
        manifest_path = os.path.join(tmpdir, 'split_manifest.csv')
        write_split_manifest_csv(episodes, manifest_path)
        assert os.path.isfile(manifest_path)

        # Verify CSV content
        import csv
        with open(manifest_path) as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 1
        assert rows[0]['episode_id'] == 'ep_a'
        assert rows[0]['split'] == 'train'
    print("PASS: test_csv_output")


if __name__ == '__main__':
    test_identical_trajectories_same_hash()
    test_different_trajectories_different_hash()
    test_proprio_hash_excludes_privileged()
    test_initial_state_hash_same_init()
    test_duplicate_groups()
    test_dedup_keeps_highest_priority()
    test_no_duplicates()
    test_split_isolation_valid()
    test_split_isolation_violation()
    test_csv_output()
    print("\nAll SC5 trajectory dedup tests passed.")
