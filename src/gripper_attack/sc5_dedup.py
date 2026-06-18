#!/usr/bin/env python3
"""SC5 trajectory dedup — full-sequence hashing, group management, priority selection.

Reuses: build_sc5_student_dataset_v2.py content-based dedup pattern (first-5 EEF hash).
Extends: full-sequence SHA256, per-field hashing, duplicate group tracking.

All groups (duplicate, init-state, parent event) must NOT cross splits.
"""
from __future__ import annotations

import hashlib, json, csv, os
from collections import defaultdict
from typing import Optional, List, Dict, Tuple


def sha256_hex(data: str) -> str:
    """SHA256 hash of a string, returning hex digest."""
    return hashlib.sha256(data.encode('utf-8')).hexdigest()


MISSING_SENTINEL = -999999.0  # explicit marker: missing ≠ zero


def _safe_float(v, default=MISSING_SENTINEL):
    """Convert to float, handling empty strings, None, nan.

    Uses explicit sentinel (MISSING_SENTINEL) for missing values,
    NOT zero — so legitimate zeros and missing values are distinguishable.
    """
    if v is None:
        return default
    if isinstance(v, bool):
        return default
    if isinstance(v, str) and v.strip() in ('', 'nan', 'NaN', 'NAN', 'inf', '-inf', 'Infinity'):
        return default
    try:
        return float(v)
    except (ValueError, TypeError):
        return default


def trajectory_content_hash(records: List[dict]) -> str:
    """Full-sequence content hash: suite, task, state_id, EEF, gripper, action.

    Includes task/state/action identity so different tasks with similar motion
    are not incorrectly dedup'd. Missing values use MISSING_SENTINEL (-999999.0)
    to distinguish from legitimate zeros.
    """
    # Extract identity from first policy step
    suite = ''; task = ''; state_id = -1
    for r in records:
        if r.get('teacher_privileged_state_available'):
            suite = str(r.get('suite', ''))
            task = str(r.get('task_name', r.get('task_instruction', '')))
            state_id = int(r.get('state_id', -1))
            break

    identity = {'suite': suite, 'task': task, 'state_id': state_id,
                'n_steps': len(records)}

    motion = []
    for r in records:
        if not r.get('teacher_privileged_state_available'):
            continue
        motion.append({
            'step': int(r.get('step_idx', r.get('policy_step_idx', 0))),
            'eef_x': round(_safe_float(r.get('eef_x')), 4),
            'eef_y': round(_safe_float(r.get('eef_y')), 4),
            'eef_z': round(_safe_float(r.get('eef_z')), 4),
            'gripper': round(_safe_float(r.get('gripper_command')), 4),
            'action_dx': round(_safe_float(r.get('action_dx')), 6),
            'action_dy': round(_safe_float(r.get('action_dy')), 6),
            'action_dz': round(_safe_float(r.get('action_dz')), 6),
            'action_gripper': round(_safe_float(r.get('action_gripper')), 6),
        })

    return sha256_hex(json.dumps({'identity': identity, 'motion': motion},
                                  sort_keys=True))


def proprio_sequence_hash(records: List[dict]) -> str:
    """Hash of proprioceptive sequence only (EEF + gripper + action). No privileged fields."""
    seq = []
    for r in records:
        if not r.get('teacher_privileged_state_available'):
            continue
        row = {
            'gripper_command': round(_safe_float(r.get('gripper_command')), 4),
            'gripper_qpos': round(_safe_float(r.get('gripper_qpos')), 6),
            'gripper_width': round(_safe_float(r.get('gripper_width') or r.get('gripper_opening_proxy')), 6),
            'eef_x': round(_safe_float(r.get('eef_x')), 4),
            'eef_y': round(_safe_float(r.get('eef_y')), 4),
            'eef_z': round(_safe_float(r.get('eef_z')), 4),
            'eef_vx': round(_safe_float(r.get('eef_vx')), 6),
            'eef_vy': round(_safe_float(r.get('eef_vy')), 6),
            'eef_vz': round(_safe_float(r.get('eef_vz')), 6),
            'action_dx': round(_safe_float(r.get('action_dx')), 6),
            'action_dy': round(_safe_float(r.get('action_dy')), 6),
            'action_dz': round(_safe_float(r.get('action_dz')), 6),
            'action_gripper': round(_safe_float(r.get('action_gripper')), 6),
        }
        seq.append(row)
    if not seq:
        return ""
    return sha256_hex(json.dumps(seq, sort_keys=True))


def privileged_sequence_hash(records: List[dict]) -> str:
    """Hash of privileged fields only (object/target trajectory). NOT student input."""
    seq = []
    for r in records:
        if not r.get('teacher_privileged_state_available'):
            continue
        obj_str = r.get('object_pose_json', '')
        tgt_str = r.get('target_pose_json', '')
        row = {
            'object_pose': obj_str[:80] if obj_str else '',
            'target_pose': tgt_str[:80] if tgt_str else '',
            'obj_target_dist': round(_safe_float(r.get('object_to_target_distance')), 6),
            'obj_eef_dist': round(_safe_float(r.get('object_eef_distance')), 6),
        }
        seq.append(row)
    if not seq:
        return ""
    return sha256_hex(json.dumps(seq, sort_keys=True))


def initial_state_hash(records: List[dict]) -> str:
    """Hash of initial state (first 3 policy steps) for init-state grouping."""
    policy_steps = [r for r in records[:20] if r.get('teacher_privileged_state_available')]
    if not policy_steps:
        return ""
    first_3 = policy_steps[:3]
    state_vec = []
    for r in first_3:
        state_vec.append({
            'eef_x': round(_safe_float(r.get('eef_x')), 4),
            'eef_y': round(_safe_float(r.get('eef_y')), 4),
            'eef_z': round(_safe_float(r.get('eef_z')), 4),
            'gripper_qpos': round(_safe_float(r.get('gripper_qpos')), 6),
            'obj_z': round(json.loads(r.get('object_pose_json', '[0,0,0]'))[2], 4)
            if r.get('object_pose_json') else 0.0,
        })
    return sha256_hex(json.dumps(state_vec, sort_keys=True))


def source_file_sha256(filepath: str) -> str:
    """SHA256 of the raw step_records.jsonl file."""
    try:
        with open(filepath, 'rb') as f:
            return hashlib.sha256(f.read()).hexdigest()
    except (IOError, OSError):
        return ""


def compute_all_hashes(records: List[dict], filepath: str = "") -> dict:
    """Compute all 5 hashes for a trajectory."""
    return {
        'source_file_sha256': source_file_sha256(filepath) if filepath else "",
        'trajectory_content_sha256': trajectory_content_hash(records),
        'proprio_sequence_sha256': proprio_sequence_hash(records),
        'privileged_sequence_sha256': privileged_sequence_hash(records),
        'initial_state_sha256': initial_state_hash(records),
    }


def build_duplicate_groups(episodes: List[dict],
                           hash_key: str = 'trajectory_content_sha256') -> List[dict]:
    """Group episodes by content hash, identifying duplicate sets.

    Returns list of {'group_id': str, 'hash': str, 'n_members': int,
                     'episode_ids': list, 'keep_idx': int, 'drop_indices': list}
    """
    groups = defaultdict(list)
    for i, ep in enumerate(episodes):
        h = ep.get(hash_key, '')
        if h:
            groups[h].append((i, ep))

    result = []
    for h, members in groups.items():
        if len(members) < 2:
            continue
        # Priority: most complete provenance first (uses builder top-level fields)
        def priority_key(item):
            ep = item[1]
            score = 0
            if ep.get('candidate_tier') == 'LIBERO_OBJECT_SINGLE_OBJECT_CANDIDATE':
                score += 10
            elif ep.get('candidate_tier') == 'REQUIRES_OBJECT_TARGET_VALIDATION':
                score += 5
            if ep.get('privileged_sequence_sha256'):
                score += 3
            if ep.get('proprio_sequence_sha256'):
                score += 2
            return (-score, str(ep.get('episode_id', '')))

        sorted_members = sorted(members, key=priority_key)
        keep_idx = sorted_members[0][0]
        drop_indices = [m[0] for m in sorted_members[1:]]

        result.append({
            'group_id': h[:16],
            'hash': h,
            'n_members': len(members),
            'episode_ids': [ep.get('episode_id', '') for _, ep in sorted_members],
            'keep_idx': keep_idx,
            'drop_indices': drop_indices,
        })

    return result


def dedup_episodes(episodes: List[dict],
                   hash_key: str = 'trajectory_content_sha256') -> Tuple[List[dict], List[dict]]:
    """Deduplicate episodes by content hash, keeping highest-priority copy.

    Returns (unique_episodes, duplicate_groups).
    """
    groups = build_duplicate_groups(episodes, hash_key)
    drop_set = set()
    for g in groups:
        for idx in g['drop_indices']:
            drop_set.add(idx)

    unique = [ep for i, ep in enumerate(episodes) if i not in drop_set]
    return unique, groups


def validate_split_isolation(episodes: List[dict], group_key: str = 'initial_state_sha256') -> dict:
    """Validate that no group (init-state, duplicate) crosses train/held-out split.

    Returns {'valid': bool, 'violations': list of group_ids that cross splits}.
    """
    groups = defaultdict(lambda: {'train': 0, 'held_out': 0})
    for ep in episodes:
        gk = ep.get(group_key, '')
        if not gk:
            continue
        if ep.get('is_held_out') in (True, 'True', 'true', 1, '1'):
            groups[gk]['held_out'] += 1
        else:
            groups[gk]['train'] += 1

    violations = []
    for gk, counts in groups.items():
        if counts['train'] > 0 and counts['held_out'] > 0:
            violations.append(gk)

    return {
        'valid': len(violations) == 0,
        'violations': violations,
        'n_violations': len(violations),
        'n_groups': len(groups),
    }


def validate_multi_split_isolation(episodes: List[dict],
                                    group_key: str = 'initial_state_sha256',
                                    split_key: str = 'split') -> dict:
    """Validate group isolation across ALL split pairs (not just held_out vs rest).

    Checks train↔val, train↔held_out, val↔held_out independently.
    Returns detailed per-pair audit.
    """
    split_values = sorted(set(ep.get(split_key, 'train') for ep in episodes))
    groups = defaultdict(lambda: defaultdict(int))

    for ep in episodes:
        gk = ep.get(group_key, '')
        sp = ep.get(split_key, 'train')
        if gk:
            groups[gk][sp] += 1

    violations_by_pair = {}
    for i, s1 in enumerate(split_values):
        for s2 in split_values[i+1:]:
            pair_key = f"{s1}↔{s2}"
            violating = []
            for gk, split_counts in groups.items():
                if split_counts.get(s1, 0) > 0 and split_counts.get(s2, 0) > 0:
                    violating.append(gk)
            violations_by_pair[pair_key] = violating

    total_violations = sum(len(v) for v in violations_by_pair.values())
    return {
        'valid': total_violations == 0,
        'split_values': split_values,
        'violations_by_pair': {k: len(v) for k, v in violations_by_pair.items()},
        'violation_details': violations_by_pair,
        'n_violations': total_violations,
        'n_groups': len(groups),
    }


def write_duplicate_groups_csv(groups: List[dict], output_path: str):
    """Write duplicate groups to CSV for audit."""
    if not groups:
        return
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['group_id', 'hash', 'n_members',
                                           'episode_ids', 'keep_idx', 'drop_indices'])
        w.writeheader()
        for g in groups:
            row = {k: v for k, v in g.items()}
            row['episode_ids'] = '|'.join(row['episode_ids'])
            row['drop_indices'] = '|'.join(str(x) for x in row['drop_indices'])
            w.writerow(row)


def write_split_manifest_csv(episodes: List[dict], output_path: str):
    """Write split manifest with full hash and group info."""
    if not episodes:
        return
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    fieldnames = ['episode_id', 'task', 'state_id', 'suite', 'mechanism_tier',
                  'split', 'is_held_out',
                  'trajectory_content_sha256', 'proprio_sequence_sha256',
                  'initial_state_sha256', 'source_file_sha256',
                  'init_state_group_id', 'duplicate_group_id',
                  'n_steps', 'sc5_valid', 'sc5_anchor']

    with open(output_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        w.writeheader()
        for ep in episodes:
            row = {k: ep.get(k, '') for k in fieldnames}
            w.writerow(row)
