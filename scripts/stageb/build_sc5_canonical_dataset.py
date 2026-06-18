#!/usr/bin/env python3
"""SC5 Canonical Dataset Builder: scan all privileged artifact sources, dedup, label, output unified CSV.

Reuses mature code from build_sc5_student_dataset_v2.py:
  - Same Teacher interface (calibrate on train, label all)
  - Same streaming feature adapter
  - Same content-based dedup
  - Same HELD_OUT_BUTTER = {8,9,11}

Extensions:
  - Multi-source scan with schema validation
  - Cross-source dedup (not just single-directory)
  - Mechanism classification (pick_and_place, drawer, stove, other)
  - Detailed source provenance tracking
  - Split isolation enforcement
"""
import csv, hashlib, json, os, sys
from collections import defaultdict
from pathlib import Path
import numpy as np

# Find repo root: try common locations
_script_dir = Path(__file__).resolve().parent
REPO = None
for candidate in [
    _script_dir.parents[2],                                          # scripts/stageb/ -> repo
    Path('/data/liuyu/repos/sc5_census_freeze_7ab15f1_20260618'),   # frozen on server
    Path('/data/liuyu/repos/sc5_census_freeze_cc356f3_20260618'),
    Path('/data/liuyu/repos/sc5_census_freeze_e69d9a1_20260618'),
]:
    if (candidate / 'src' / 'gripper_attack' / 'v2_privileged_teacher.py').exists():
        REPO = candidate
        break
if REPO is None:
    # Fallback: assume modules are in same directory
    REPO = _script_dir.parents[2]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "src"))

from gripper_attack.v2_privileged_teacher import (
    V2PrivilegedTeacher, TeacherConfig, calibrate_thresholds,
    find_sc5_anchor_v2, compute_sc5_valid_start_corridor)
from gripper_attack.sc5_streaming_features_v2 import (
    SC5StreamingFeatureAdapterV2, FEATURE_NAMES)
from gripper_attack.sc5_schema_adapter_v2 import SC5SchemaAdapterV2
from gripper_attack.sc5_event_segmenter_v2 import SC5EventSegmenterV2
from gripper_attack.sc5_dedup import (
    compute_all_hashes, build_duplicate_groups, dedup_episodes,
    validate_split_isolation,
)

K, GUARD = 10, 5
HELD_OUT_BUTTER = {8, 9, 11}

# Required proprio fields for feature adapter
REQUIRED_13 = [
    "gripper_command", "gripper_qpos", "gripper_width",
    "eef_x", "eef_y", "eef_z", "eef_vx", "eef_vy", "eef_vz",
    "action_dx", "action_dy", "action_dz", "action_gripper",
]

# Required privileged fields for Teacher labeling (in policy steps)
REQUIRED_PRIV = [
    "object_pose_json", "object_to_target_distance", "object_eef_distance",
]

# Mechanism classifiers (simplified for tier classification)
PICK_PLACE_KW = ['pick_up', 'place_in', 'place_on', 'put_the', 'push_the']
DRAWER_KW = ['open_the', 'drawer']
STOVE_KW = ['turn_on_the_stove', 'put_the_moka', 'moka_pot']


def classify_mechanism(task_name):
    task_lower = task_name.lower()
    is_pp = any(kw in task_lower for kw in PICK_PLACE_KW)
    is_drawer = any(kw in task_lower for kw in DRAWER_KW)
    is_stove = any(kw in task_lower for kw in STOVE_KW)
    if is_drawer:
        return 'drawer'
    elif is_stove:
        return 'stove'
    elif is_pp:
        return 'pick_and_place'
    else:
        return 'other'


def scan_sources(source_roots):
    """Scan all source roots for usable step_records.jsonl files.
    Returns list of {'root': str, 'run_dir': str, 'jsonl_path': str,
                     'task_name': str, 'state_id': int, 'success': bool,
                     'is_butter': bool, 'source_milestone': str}
    """
    found = []
    stats = defaultdict(int)

    for source_root in source_roots:
        if not os.path.isdir(source_root):
            print(f"  SKIP (not a dir): {source_root}")
            continue

        milestone = os.path.basename(source_root)
        runs_dir = os.path.join(source_root, 'runs')
        search_root = runs_dir if os.path.isdir(runs_dir) else source_root

        for dirpath, dirnames, filenames in os.walk(search_root):
            if 'step_records.jsonl' not in filenames:
                continue
            stats['jsonl_found'] += 1

            jsonl_path = os.path.join(dirpath, 'step_records.jsonl')
            manifest_path = os.path.join(dirpath, 'run_manifest.json')

            # Validate schema: check a policy step (skip wait steps)
            try:
                with open(jsonl_path) as f:
                    all_records = [json.loads(line) for line in f]
            except (json.JSONDecodeError, IOError):
                stats['jsonl_read_error'] += 1
                continue

            # Find first policy step to check schema
            policy_step = None
            for r in all_records:
                if r.get('teacher_privileged_state_available'):
                    policy_step = r
                    break

            if policy_step is None:
                stats['no_privileged_step'] += 1
                continue

            keys = list(policy_step.keys())
            has_proprio = all(k in keys for k in REQUIRED_13)
            has_priv = all(k in keys for k in REQUIRED_PRIV)

            if not (has_proprio and has_priv):
                stats['schema_incomplete'] += 1
                continue

            # Read manifest
            success = False
            task_name = 'unknown'
            state_id = -1
            if os.path.isfile(manifest_path):
                with open(manifest_path) as f:
                    m = json.load(f)
                success = m.get('success', False)
                task_name = m.get('task_name', 'unknown')
                state_id = m.get('state_id', -1)

            if not success:
                stats['clean_fail'] += 1
                continue
            stats['clean_success'] += 1

            is_butter = 'butter' in task_name.lower()

            found.append({
                'root': source_root,
                'run_dir': os.path.relpath(dirpath, search_root),
                'jsonl_path': jsonl_path,
                'manifest_path': manifest_path,
                'task_name': task_name,
                'state_id': state_id,
                'success': success,
                'is_butter': is_butter,
                'source_milestone': milestone,
                'mechanism': classify_mechanism(task_name),
                'n_steps': len(all_records),
            })

    return found, dict(stats)


def dedup_entries(entries):
    """Full-sequence content-based dedup with priority selection.

    Reuses: sc5_dedup.dedup_episodes for 5-hash dedup and priority logic.
    """
    # Load records and compute all 5 hashes
    for e in entries:
        try:
            with open(e['jsonl_path']) as f:
                records = [json.loads(line) for line in f]
        except (json.JSONDecodeError, IOError):
            e['trajectory_content_sha256'] = ''
            continue
        hashes = compute_all_hashes(records, e['jsonl_path'])
        for k, v in hashes.items():
            e[k] = v

    # Use dedup module
    unique, groups = dedup_episodes(entries, 'trajectory_content_sha256')

    # Add duplicate group info to kept entries
    kept_hashes = {e['trajectory_content_sha256'] for e in unique}
    for g in groups:
        for i, ep_id in enumerate(g['episode_ids']):
            for e in entries:
                if e.get('episode_id') == ep_id or e.get('run_dir') == ep_id:
                    e['duplicate_group_id'] = g['group_id']
                    e['duplicate_rank'] = i

    return unique, groups


def build_dataset(entries, teacher, output_dir, artifacts_dir):
    """Build canonical dataset CSV from dedup'd entries."""
    rows = []
    episodes = []
    stats = defaultdict(int)

    for e in entries:
        stats['total'] += 1

        try:
            with open(e['jsonl_path']) as f:
                records = [json.loads(line) for line in f]
        except (json.JSONDecodeError, IOError):
            stats['read_error'] += 1
            continue

        is_held_out = e['is_butter'] and e['state_id'] in HELD_OUT_BUTTER

        # Teacher labels
        labels = teacher.label_trajectory(records)
        sc5 = find_sc5_anchor_v2(labels, K=K, guard=GUARD)
        stats['labeled'] += 1

        if sc5['valid']:
            stats['sc5_valid'] += 1
        else:
            stats['sc5_invalid'] += 1

        # Real corridor
        corridor = None
        if sc5['valid']:
            corridor = compute_sc5_valid_start_corridor(labels, sc5['anchor'], K=K)

        # Feature rows
        adapter = SC5StreamingFeatureAdapterV2()
        first_step = None
        feat_rows = []

        for r in records:
            if not r.get('teacher_privileged_state_available'):
                continue
            step_raw = int(r.get('step_idx', r.get('policy_step_idx', 0)))
            if first_step is None:
                first_step = step_raw
            step = step_raw - first_step

            # Fail-closed: check required fields
            missing = [fld for fld in REQUIRED_13
                       if r.get(fld) in (None, '', 'nan')]
            if missing:
                continue

            raw_grip = float(r['gripper_command'])
            env_grip = -1.0 if raw_grip > 0.5 else 1.0

            try:
                result = adapter.update(
                    step_id=step, raw_gripper=raw_grip, env_gripper=env_grip,
                    gripper_qpos=float(r['gripper_qpos']),
                    gripper_opening_proxy=float(r.get('gripper_width', r.get('gripper_opening_proxy', 0))),
                    eef_x=float(r['eef_x']), eef_y=float(r['eef_y']), eef_z=float(r['eef_z']),
                    eef_vx=float(r.get('eef_vx', 0)), eef_vy=float(r.get('eef_vy', 0)),
                    eef_vz=float(r.get('eef_vz', 0)),
                    action_dx=float(r.get('action_dx', 0)),
                    action_dy=float(r.get('action_dy', 0)),
                    action_dz=float(r.get('action_dz', 0)),
                    action_gripper=float(r.get('action_gripper', raw_grip)))
            except ValueError:
                continue
            if not result['valid']:
                continue

            tl = labels[step_raw] if step_raw < len(labels) else None
            has_sc5 = sc5['valid']
            in_attack_window = has_sc5 and sc5['window'][0] <= step_raw <= sc5['window'][1]
            in_corridor = corridor is not None and step_raw in corridor['corridor_active_at_t']
            k10_valid = (corridor is not None and step_raw < len(corridor['full_k10_valid_at_t'])
                         and corridor['full_k10_valid_at_t'][step_raw])

            row = dict(result['features'])
            row['step_idx'] = step_raw
            row['state_id'] = e['state_id']
            row['task_name'] = e['task_name']
            row['is_butter'] = e['is_butter']
            row['is_held_out'] = is_held_out
            row['run_id'] = e['run_dir']
            row['source_milestone'] = e['source_milestone']
            row['mechanism'] = e['mechanism']
            row['teacher_phase'] = tl['phase'] if tl else 'abstain'
            row['teacher_sc5_anchor'] = sc5['anchor'] if has_sc5 else -1
            row['teacher_sc5_attack_window_active'] = int(in_attack_window)
            row['teacher_sc5_ready'] = int(has_sc5 and step_raw == sc5['anchor'])
            row['teacher_sc5_corridor_active'] = int(in_corridor)
            row['teacher_full_k10_valid_at_t'] = int(k10_valid)
            row['teacher_stable_carry_start'] = sc5['stable_carry_start'] if has_sc5 else -1
            row['teacher_confidence'] = tl['confidence'] if tl else 0.0
            feat_rows.append(row)

        if feat_rows:
            episodes.append({
                'run_id': e['run_dir'], 'task': e['task_name'],
                'state_id': e['state_id'], 'is_butter': e['is_butter'],
                'is_held_out': is_held_out,
                'source_milestone': e['source_milestone'],
                'mechanism': e['mechanism'],
                'n_steps': len(feat_rows), 'sc5_valid': sc5['valid'],
                'sc5_anchor': sc5['anchor'],
                'corridor_start': corridor['corridor_start'] if corridor else -1,
                'corridor_end': corridor['corridor_end'] if corridor else -1,
            })
            rows.extend(feat_rows)

    # Write outputs
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(artifacts_dir, exist_ok=True)

    csv_path = os.path.join(output_dir, 'v2_sc5_canonical_dataset.csv')
    ep_csv_path = os.path.join(output_dir, 'v2_sc5_canonical_episodes.csv')
    manifest_path = os.path.join(artifacts_dir, 'v2_sc5_canonical_manifest.json')

    if rows:
        with open(csv_path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)

    if episodes:
        with open(ep_csv_path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(episodes[0].keys()))
            w.writeheader(); w.writerows(episodes)

    manifest = {
        'K': K, 'guard': GUARD,
        'held_out_butter': list(HELD_OUT_BUTTER),
        'n_rows': len(rows), 'n_episodes': len(episodes),
        'n_train_episodes': len([e for e in episodes if not e['is_held_out']]),
        'n_held_out_episodes': len([e for e in episodes if e['is_held_out']]),
        'stats': dict(stats),
    }
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2, default=str)

    return rows, episodes, manifest


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--output_dir', default='tables')
    ap.add_argument('--artifacts_dir', default='artifacts')
    ap.add_argument('--source_roots', nargs='*',
                    default=[
                        '/data/liuyu/outputs/milestone_2e2_object100_privileged_artifact_rich_20260527',
                        '/data/liuyu/outputs/milestone_2e4_cross_suite300_privileged_artifact_rich_20260527',
                        '/data/liuyu/outputs/milestone_2e5_goal100_parser_v2_privileged_rerun_20260527',
                        '/data/liuyu/outputs/milestone_2e5_l10100_parser_v2_privileged_rerun_20260527',
                        '/data/liuyu/outputs/milestone_2d_phase_c2_privileged_artifact_rich_object_smoke_20260527',
                        '/data/liuyu/outputs/milestone_3a_crosssuite_proprio_shadow_20260531',
                    ])
    args = ap.parse_args()

    print("=== Phase 1: Scan sources ===")
    entries, scan_stats = scan_sources(args.source_roots)
    print(f"Scan stats: {json.dumps(scan_stats, indent=2)}")
    print(f"Clean-success entries (before dedup): {len(entries)}")

    # Breakdown
    mech_counts = defaultdict(int)
    milestone_counts = defaultdict(int)
    for e in entries:
        mech_counts[e['mechanism']] += 1
        milestone_counts[e['source_milestone']] += 1
    print(f"By mechanism: {dict(mech_counts)}")
    print(f"By milestone: {dict(milestone_counts)}")

    print("\n=== Phase 2: Dedup ===")
    unique, dup_groups = dedup_entries(entries)
    print(f"After dedup: {len(unique)} entries (removed {len(entries) - len(unique)} duplicates, {len(dup_groups)} groups)")

    # Add episode_id if missing
    for i, e in enumerate(unique):
        if 'episode_id' not in e:
            e['episode_id'] = e.get('run_dir', f'ep_{i}')

    # Split train/held-out
    train_entries = [e for e in unique if not (e['is_butter'] and e['state_id'] in HELD_OUT_BUTTER)]
    held_entries = [e for e in unique if e['is_butter'] and e['state_id'] in HELD_OUT_BUTTER]
    print(f"Train: {len(train_entries)}, Held-out: {len(held_entries)}")

    # Split isolation validation
    iso_result = validate_split_isolation(unique, 'initial_state_sha256')
    if not iso_result['valid']:
        print(f"WARNING: Split isolation violated! {iso_result['n_violations']} groups cross splits")
    else:
        print(f"Split isolation: PASS ({iso_result['n_groups']} groups, 0 violations)")

    # Train-only mechanism breakdown
    train_mech = defaultdict(int)
    for e in train_entries:
        train_mech[e['mechanism']] += 1
    print(f"Train mechanisms: {dict(train_mech)}")

    held_mech = defaultdict(int)
    for e in held_entries:
        held_mech[e['mechanism']] += 1
    print(f"Held-out mechanisms: {dict(held_mech)}")

    print("\n=== Phase 3: Teacher calibration (train-only) ===")
    train_jsonl_paths = [e['jsonl_path'] for e in train_entries]

    # Pre-filter: ensure all train JSONLs have valid object_to_target_distance values
    valid_train_paths = []
    for jp in train_jsonl_paths:
        try:
            with open(jp) as f:
                records = [json.loads(line) for line in f]
            ok = True
            for r in records:
                if not r.get('teacher_privileged_state_available'):
                    continue
                for field in ['object_to_target_distance', 'object_eef_distance',
                              'gripper_command', 'gripper_qpos', 'gripper_width',
                              'eef_x', 'eef_y', 'eef_z']:
                    v = r.get(field, None)
                    if v is None or v == '' or v == 'nan':
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                valid_train_paths.append(jp)
        except (json.JSONDecodeError, IOError):
            pass
    print(f"Valid train paths: {len(valid_train_paths)}/{len(train_jsonl_paths)} "
          f"(removed {len(train_jsonl_paths) - len(valid_train_paths)} with bad fields)")
    train_jsonl_paths = valid_train_paths

    teacher_config = calibrate_thresholds(train_jsonl_paths)
    teacher = V2PrivilegedTeacher(teacher_config)
    print(f"Calibrated on {len(train_jsonl_paths)} train trajectories")
    print(f"Config: grasp_close_sustain={teacher_config.grasp_close_sustain}, "
          f"lift_z={teacher_config.lift_z_threshold:.4f}, "
          f"eef_obj_dist_max={teacher_config.eef_obj_dist_max:.4f}")

    # Save Teacher config and SHA
    os.makedirs(args.artifacts_dir, exist_ok=True)
    teacher_config_json = {
        'grasp_close_sustain': teacher_config.grasp_close_sustain,
        'grasp_open_proxy_max': teacher_config.grasp_open_proxy_max,
        'eef_obj_dist_max': teacher_config.eef_obj_dist_max,
        'eef_obj_dist_stable_var': teacher_config.eef_obj_dist_stable_var,
        'lift_z_threshold': teacher_config.lift_z_threshold,
        'lift_sustain_steps': teacher_config.lift_sustain_steps,
        'carry_window': teacher_config.carry_window,
        'release_target_dist_max': teacher_config.release_target_dist_max,
        'preplace_target_dist_min': teacher_config.preplace_target_dist_min,
        'preplace_target_dist_max': teacher_config.preplace_target_dist_max,
        'stability_window': teacher_config.stability_window,
        'carry_obj_z_var_max': teacher_config.carry_obj_z_var_max,
        'calibrated_from': teacher_config.calibrated_from,
        'version': teacher_config.version,
    }
    config_path = os.path.join(args.artifacts_dir, 'v2_sc5_teacher_config.json')
    with open(config_path, 'w') as f:
        json.dump(teacher_config_json, f, indent=2)
    config_hash = hashlib.sha256(json.dumps(teacher_config_json, sort_keys=True).encode()).hexdigest()
    with open(os.path.join(args.artifacts_dir, 'v2_sc5_teacher_config.sha256'), 'w') as f:
        f.write(f"{config_hash}  v2_sc5_teacher_config.json\n")
    print(f"Teacher config SHA: {config_hash[:16]}")

    # Save duplicate groups
    from gripper_attack.sc5_dedup import write_duplicate_groups_csv
    write_duplicate_groups_csv(dup_groups,
                               os.path.join(args.output_dir, 'v2_sc5_duplicate_groups.csv'))

    print("\n=== Phase 4: Build dataset ===")
    rows, episodes, manifest = build_dataset(unique, teacher, args.output_dir, args.artifacts_dir)

    print(f"\n=== Final ===")
    print(f"Rows: {len(rows)}")
    print(f"Episodes: {len(episodes)}")
    print(f"Train episodes: {len([e for e in episodes if not e['is_held_out']])}")
    print(f"Held-out episodes: {len([e for e in episodes if e['is_held_out']])}")

    # Butter state breakdown
    butter_eps = [e for e in episodes if e['is_butter']]
    for e in sorted(butter_eps, key=lambda x: x['state_id']):
        print(f"  butter_s{e['state_id']}: sc5={e['sc5_valid']} anchor={e['sc5_anchor']} "
              f"cor=[{e['corridor_start']},{e['corridor_end']}] held={e['is_held_out']} "
              f"source={e['source_milestone'][:40]}")

    sc5_eps = [e for e in episodes if e['sc5_valid']]
    no_corridor = [e for e in sc5_eps if e['corridor_start'] < 0]
    sc5_pp = [e for e in sc5_eps if e.get('mechanism') == 'pick_and_place']
    print(f"SC5-valid: {len(sc5_eps)} ({len(sc5_pp)} pick-and-place, {len(no_corridor)} no-corridor)")

    # Per-mechanism SC5 validity
    for mech in ['pick_and_place', 'drawer', 'stove', 'other']:
        mech_eps = [e for e in episodes if e.get('mechanism') == mech]
        mech_sc5 = [e for e in mech_eps if e['sc5_valid']]
        if mech_eps:
            print(f"  {mech}: {len(mech_sc5)}/{len(mech_eps)} SC5-valid")

    return rows, episodes, manifest


if __name__ == '__main__':
    main()
