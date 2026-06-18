#!/usr/bin/env python3
"""SC5 Canonical Corpus Builder v2 — inventory-driven ingestion.

Reuses mature code from:
  - build_sc5_student_dataset_v2.py (Teacher/adapter/held-out pattern)
  - v2_privileged_teacher.py (calibrate on train, label all)
  - sc5_streaming_features_v2.py (25D continuous features)

Key changes vs v1:
  - PRIMARY input = frozen census inventory CSV (not hardcoded dirs)
  - Every census row gets a final disposition
  - Schema adapter and provenance validator are actually called
  - Real env_gripper from env_action, not synthesized from raw
  - Split policy enforced BEFORE calibration
"""
import csv, hashlib, json, os, sys
from collections import defaultdict, Counter
from pathlib import Path
import numpy as np

# ── Repo path resolution ──
_script_dir = Path(__file__).resolve().parent
REPO = None
for candidate in [
    _script_dir.parents[2],
    Path('/data/liuyu/repos/sc5_census_freeze_7ab15f1_20260618'),
    Path('/data/liuyu/repos/sc5_census_freeze_cc356f3_20260618'),
    Path('/data/liuyu/repos/sc5_census_freeze_e69d9a1_20260618'),
]:
    if (candidate / 'src' / 'gripper_attack' / 'v2_privileged_teacher.py').exists():
        REPO = candidate; break
if REPO is None:
    REPO = _script_dir.parents[2]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "src"))

from gripper_attack.v2_privileged_teacher import (
    V2PrivilegedTeacher, TeacherConfig, calibrate_thresholds,
    find_sc5_anchor_v2, compute_sc5_valid_start_corridor)
from gripper_attack.sc5_streaming_features_v2 import (
    SC5StreamingFeatureAdapterV2, FEATURE_NAMES)
from gripper_attack.sc5_schema_adapter_v2 import SC5SchemaAdapterV2
from gripper_attack.sc5_dedup import (
    compute_all_hashes, dedup_episodes, validate_split_isolation,
    _safe_float,
)

K, GUARD = 10, 5
HELD_OUT_BUTTER = {8, 9, 11}

# Eligible tiers from census (others are excluded)
ELIGIBLE_TIERS = {
    'LIBERO_OBJECT_SINGLE_OBJECT_CANDIDATE',         # PRIMARY (449)
    'REQUIRES_OBJECT_TARGET_VALIDATION',              # CONDITIONAL_PLACE (129)
    'REQUIRES_EVENT_SEGMENTATION',                    # CONDITIONAL_MULTI_STAGE (110)
}

REQUIRED_13 = [
    "gripper_command", "gripper_qpos", "gripper_width",
    "eef_x", "eef_y", "eef_z", "eef_vx", "eef_vy", "eef_vz",
    "action_dx", "action_dy", "action_dz", "action_gripper",
]

# ── Mechanism classification ──
PICK_PLACE_KW = ['pick_up', 'place_in', 'place_on', 'put_the', 'push_the']
DRAWER_KW = ['open_the', 'drawer']  # "cabinet" removed per audit
STOVE_KW = ['turn_on_the_stove', 'put_the_moka', 'moka_pot']


def classify_mechanism(task_name):
    t = task_name.lower()
    if any(kw in t for kw in DRAWER_KW):
        return 'drawer'
    if any(kw in t for kw in STOVE_KW):
        return 'stove'
    if any(kw in t for kw in PICK_PLACE_KW):
        return 'pick_and_place'
    return 'other'


def load_inventory(inventory_path):
    """Load frozen census inventory. Returns list of rows with parsed fields."""
    rows = []
    with open(inventory_path) as f:
        for r in csv.DictReader(f):
            rows.append(r)
    print(f"Loaded {len(rows)} inventory rows from {inventory_path}")
    return rows


def filter_eligible(inventory_rows):
    """Filter to eligible tiers. Track exclusions."""
    eligible = []
    excluded = defaultdict(list)
    for r in inventory_rows:
        tier = r.get('exclusion_reason', r.get('tier', ''))
        if tier in ELIGIBLE_TIERS:
            eligible.append(r)
        else:
            excluded[tier].append(r['episode_id'])
    print(f"Eligible (in ELIGIBLE_TIERS): {len(eligible)}")
    for tier, eps in sorted(excluded.items()):
        print(f"  Excluded {tier}: {len(eps)}")
    return eligible, excluded


def load_episode_records(inv_row):
    """Load step_records.jsonl and run_manifest.json for one inventory row."""
    jsonl_path = inv_row.get('step_records_path', '')
    manifest_path = inv_row.get('manifest_path', '')
    try:
        with open(jsonl_path) as f:
            records = [json.loads(line) for line in f]
    except (json.JSONDecodeError, IOError, FileNotFoundError):
        return None, None, "jsonl_read_error"

    manifest = None
    if manifest_path and os.path.isfile(manifest_path):
        try:
            with open(manifest_path) as f:
                manifest = json.load(f)
        except (json.JSONDecodeError, IOError):
            pass

    # Validate clean provenance via schema adapter
    adapter = SC5SchemaAdapterV2()
    has_attack = False
    for rec in records:
        if rec.get('teacher_privileged_state_available'):
            prov = adapter.validate_clean_provenance(rec, manifest)
            if not prov['clean_provenance']:
                has_attack = True
                break

    if has_attack:
        return records, manifest, "attack_contamination"

    if not manifest or not manifest.get('success', False):
        return records, manifest, "not_clean_success"

    return records, manifest, "ok"


def build_dataset(episodes, teacher, output_dir, artifacts_dir):
    """Build canonical dataset CSV from validated episodes.

    Each episode already has: records, manifest, hashes, mechanism, split, is_held_out.
    """
    rows = []; ep_manifest = []; stats = defaultdict(int)

    for ep in episodes:
        records = ep['records']
        is_held_out = ep['is_held_out']

        # Teacher labels (use label_by_step for correct alignment)
        labels = teacher.label_trajectory(records)
        label_by_step = {l['step_idx']: l for l in labels}

        sc5 = find_sc5_anchor_v2(labels, K=K, guard=GUARD)
        stats['labeled'] += 1
        if sc5['valid']: stats['sc5_valid'] += 1
        else: stats['sc5_invalid'] += 1

        corridor = None
        if sc5['valid']:
            corridor = compute_sc5_valid_start_corridor(labels, sc5['anchor'], K=K)

        # Feature extraction with local contiguous index.
        # Critical: streaming adapter requires consecutive step IDs.
        # Skipping invalid steps MUST use a local counter, not raw step_idx gaps.
        adapter = SC5StreamingFeatureAdapterV2()
        schema_adapter = SC5SchemaAdapterV2()
        local_step = 0
        feat_rows = []
        field_source_audit = Counter()

        for r in records:
            if not r.get('teacher_privileged_state_available'):
                continue
            step_raw = int(r.get('step_idx', r.get('policy_step_idx', 0)))

            # Schema validation (actually called, results tracked)
            provenances = schema_adapter.validate_record(r)
            for name, p in provenances.items():
                field_source_audit[f"{name}:{p.source_type}"] += 1
            if not schema_adapter.all_valid(provenances):
                continue  # skip invalid step, do NOT increment local_step

            # Real env_gripper from env_action, not synthesized
            env_action = r.get('env_action', None)
            if not (isinstance(env_action, (list, tuple)) and len(env_action) >= 7):
                continue
            env_grip = float(env_action[6])

            raw_grip = float(r['gripper_command'])

            # Gripper semantics: independent validation (raw_close vs env_close)
            raw_close = raw_grip <= 0.5
            env_close = env_grip > 0
            if raw_close != env_close:
                continue  # semantic conflict, fail-closed

            try:
                result = adapter.update(
                    step_id=local_step, raw_gripper=raw_grip, env_gripper=env_grip,
                    gripper_qpos=_safe_float(r.get('gripper_qpos')),
                    gripper_opening_proxy=_safe_float(r.get('gripper_width', r.get('gripper_opening_proxy'))),
                    eef_x=_safe_float(r.get('eef_x')), eef_y=_safe_float(r.get('eef_y')),
                    eef_z=_safe_float(r.get('eef_z')),
                    eef_vx=_safe_float(r.get('eef_vx')), eef_vy=_safe_float(r.get('eef_vy')),
                    eef_vz=_safe_float(r.get('eef_vz')),
                    action_dx=_safe_float(r.get('action_dx')),
                    action_dy=_safe_float(r.get('action_dy')),
                    action_dz=_safe_float(r.get('action_dz')),
                    action_gripper=_safe_float(r.get('action_gripper', r.get('gripper_command'))))
            except ValueError:
                continue
            if not result['valid']:
                continue

            local_step += 1  # only increment for successfully processed steps

            # Label lookup by step_idx (not array index)
            tl = label_by_step.get(step_raw)

            has_sc5 = sc5['valid']
            in_attack_window = has_sc5 and sc5['window'][0] <= step_raw <= sc5['window'][1]
            in_corridor = corridor is not None and step_raw in corridor['corridor_active_at_t']
            k10_valid = (corridor is not None and step_raw < len(corridor['full_k10_valid_at_t'])
                         and corridor['full_k10_valid_at_t'][step_raw])

            row = dict(result['features'])
            row['step_idx'] = step_raw
            row['state_id'] = ep.get('state_id', -1)
            row['task_name'] = ep.get('task_name', '')
            row['is_butter'] = ep.get('is_butter', False)
            row['is_held_out'] = is_held_out
            row['run_id'] = ep.get('run_id', '')
            row['source_milestone'] = ep.get('source_milestone', '')
            row['mechanism'] = ep.get('mechanism', '')
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
            ep_manifest.append({
                'run_id': ep.get('run_id', ''), 'task': ep.get('task_name', ''),
                'state_id': ep.get('state_id', -1), 'is_butter': ep.get('is_butter', False),
                'is_held_out': is_held_out, 'split': ep.get('split', 'train'),
                'source_milestone': ep.get('source_milestone', ''),
                'mechanism': ep.get('mechanism', ''),
                'n_steps': len(feat_rows), 'sc5_valid': sc5['valid'],
                'sc5_anchor': sc5['anchor'],
                'corridor_start': corridor['corridor_start'] if corridor else -1,
                'corridor_end': corridor['corridor_end'] if corridor else -1,
            })
            rows.extend(feat_rows)

    # Write outputs
    os.makedirs(output_dir, exist_ok=True); os.makedirs(artifacts_dir, exist_ok=True)
    if rows:
        with open(os.path.join(output_dir, 'v2_sc5_canonical_dataset.csv'), 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    if ep_manifest:
        with open(os.path.join(output_dir, 'v2_sc5_canonical_episodes.csv'), 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(ep_manifest[0].keys())); w.writeheader(); w.writerows(ep_manifest)
    if field_source_audit:
        with open(os.path.join(output_dir, 'v2_sc5_field_source_audit.csv'), 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=['field_source', 'count'])
            w.writeheader()
            for (fs, cnt) in field_source_audit.most_common():
                w.writerow({'field_source': fs, 'count': cnt})

    manifest = {'K': K, 'guard': GUARD, 'n_rows': len(rows), 'n_episodes': len(ep_manifest),
                'held_out_butter': list(HELD_OUT_BUTTER), 'stats': dict(stats)}
    with open(os.path.join(artifacts_dir, 'v2_sc5_canonical_manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=2, default=str)

    return rows, ep_manifest, manifest


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--inventory', default='/tmp/sc5_source_census_cc356f3_r1/tables/v2_sc5_episode_inventory.csv')
    ap.add_argument('--output_dir', default='tables')
    ap.add_argument('--artifacts_dir', default='artifacts')
    ap.add_argument('--exclusion_output', default='tables/v2_sc5_exclusion_reasons.csv')
    args = ap.parse_args()

    # ── Phase 1: Load frozen inventory ──
    print("=== Phase 1: Load frozen census inventory ===")
    inv_rows = load_inventory(args.inventory)
    eligible, excluded = filter_eligible(inv_rows)

    # Write exclusion manifest
    os.makedirs(args.output_dir, exist_ok=True)
    with open(args.exclusion_output, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['exclusion_reason', 'count', 'episode_ids'])
        w.writeheader()
        for tier, eps in sorted(excluded.items()):
            w.writerow({'exclusion_reason': tier, 'count': len(eps),
                        'episode_ids': '|'.join(eps[:50])})

    # ── Phase 2: Load and validate each eligible episode ──
    print(f"\n=== Phase 2: Load and validate {len(eligible)} eligible episodes ===")
    validated = []
    disposition = defaultdict(list)
    for i, inv_row in enumerate(eligible):
        records, manifest, status = load_episode_records(inv_row)
        disposition[status].append(inv_row['episode_id'])

        if status != 'ok':
            continue

        task = inv_row.get('task', manifest.get('task_name', '') if manifest else '')
        state_id = int(inv_row.get('state_id', manifest.get('state_id', -1) if manifest else -1))
        is_butter = 'butter' in task.lower()
        mechanism = classify_mechanism(task)

        # Compute hashes
        hashes = compute_all_hashes(records, inv_row.get('step_records_path', ''))

        # Special cases per split policy
        if is_butter and state_id == 5:
            disposition['butter_s5_audit_only'].append(inv_row['episode_id'])
            continue
        if is_butter and state_id == 3:
            disposition['butter_s3_supplementary_abstain'].append(inv_row['episode_id'])
            continue

        validated.append({
            'episode_id': inv_row['episode_id'],
            'run_id': inv_row.get('episode_dir', inv_row.get('run_id', '')),
            'records': records,
            'manifest': manifest,
            'task_name': task,
            'state_id': state_id,
            'is_butter': is_butter,
            'mechanism': mechanism,
            'source_milestone': os.path.basename(
                os.path.dirname(os.path.dirname(inv_row.get('step_records_path', '')))),
            'jsonl_path': inv_row.get('step_records_path', ''),
            **hashes,
        })

    print(f"Validated clean episodes: {len(validated)}")
    for status, eps in sorted(disposition.items()):
        print(f"  {status}: {len(eps)}")

    if not validated:
        print("FATAL: No validated episodes. Exiting.")
        return

    # ── Phase 3: Dedup ──
    print(f"\n=== Phase 3: Dedup ===")
    unique, dup_groups = dedup_episodes(validated, 'trajectory_content_sha256')
    print(f"After dedup: {len(unique)} (removed {len(validated) - len(unique)}, {len(dup_groups)} groups)")

    # ── Phase 4: Split assignment (BEFORE calibration) ──
    print(f"\n=== Phase 4: Split assignment ===")
    for ep in unique:
        is_h = ep['is_butter'] and ep['state_id'] in HELD_OUT_BUTTER
        ep['is_held_out'] = is_h
        ep['split'] = 'held_out' if is_h else 'train'

    train_eps = [e for e in unique if not e['is_held_out']]
    held_eps = [e for e in unique if e['is_held_out']]
    print(f"Train: {len(train_eps)}, Held-out: {len(held_eps)}")

    # Split isolation validation (AFTER assignment)
    iso = validate_split_isolation(unique, 'initial_state_sha256')
    if not iso['valid']:
        print(f"FATAL: Split isolation violation! {iso['n_violations']} groups cross splits")
        for v in iso['violations'][:5]:
            print(f"  {v}")
        return
    print(f"Split isolation: PASS ({iso['n_groups']} groups, 0 violations)")

    # ── Phase 5: Teacher calibration (train-only) ──
    print(f"\n=== Phase 5: Teacher calibration (train-only) ===")
    train_paths = [e['jsonl_path'] for e in train_eps]

    # Pre-filter: valid privileged fields
    valid_paths = []
    for jp in train_paths:
        try:
            with open(jp) as f:
                recs = [json.loads(line) for line in f]
        except Exception:
            continue
        ok = True
        for r in recs:
            if not r.get('teacher_privileged_state_available'):
                continue
            for fld in ['object_to_target_distance', 'object_eef_distance',
                        'gripper_command', 'eef_x', 'eef_y', 'eef_z']:
                v = r.get(fld)
                if v is None or v == '' or v == 'nan':
                    ok = False; break
            if not ok:
                break
        if ok:
            valid_paths.append(jp)

    teacher_config = calibrate_thresholds(valid_paths)
    teacher = V2PrivilegedTeacher(teacher_config)
    config_json = {
        'grasp_close_sustain': teacher_config.grasp_close_sustain,
        'lift_z_threshold': teacher_config.lift_z_threshold,
        'eef_obj_dist_max': teacher_config.eef_obj_dist_max,
        'release_target_dist_max': teacher_config.release_target_dist_max,
    }
    os.makedirs(args.artifacts_dir, exist_ok=True)
    config_path = os.path.join(args.artifacts_dir, 'v2_sc5_teacher_config.json')
    with open(config_path, 'w') as f:
        json.dump(config_json, f, indent=2)
    config_sha = hashlib.sha256(json.dumps(config_json, sort_keys=True).encode()).hexdigest()
    with open(os.path.join(args.artifacts_dir, 'v2_sc5_teacher_config.sha256'), 'w') as f:
        f.write(f"{config_sha}  v2_sc5_teacher_config.json\n")
    print(f"Calibrated on {len(valid_paths)}/{len(train_paths)} train paths")
    print(f"Teacher config SHA: {config_sha[:16]}")

    # ── Phase 6: Build dataset ──
    print(f"\n=== Phase 6: Build dataset ===")
    rows, ep_manifest, manifest = build_dataset(unique, teacher, args.output_dir, args.artifacts_dir)

    # Note: field_source_audit.csv is saved inside build_dataset()

    print(f"\n=== Final ===")
    print(f"Rows: {len(rows)}, Episodes: {len(ep_manifest)}")
    print(f"Train: {len([e for e in ep_manifest if e['split'] == 'train'])}")
    print(f"Held-out: {len([e for e in ep_manifest if e['split'] == 'held_out'])}")
    sc5_eps = [e for e in ep_manifest if e['sc5_valid']]
    print(f"SC5-valid: {len(sc5_eps)}")

    for e in sorted(ep_manifest, key=lambda x: (x['is_held_out'], x.get('state_id', 0))):
        if e['is_butter']:
            print(f"  butter_s{e['state_id']}: sc5={e['sc5_valid']} split={e['split']}")

    # Final disposition reconciliation
    print(f"\n=== Disposition Reconciliation ===")
    # Eligible: all 688 from ELIGIBLE_TIERS
    eligible_total = len(eligible)
    # Excluded at load time: 90+1244+876+50+65 = 2325
    excluded_total = sum(len(v) for v in excluded.values())
    # Dispositions from validation phase
    accounted = sum(len(v) for v in disposition.values())
    total = eligible_total + excluded_total
    print(f"Census total: {len(inv_rows)}")
    print(f"  Eligible (entered validation): {eligible_total}")
    print(f"  Excluded by tier: {excluded_total}")
    for tier, eps in sorted(excluded.items()):
        print(f"    {tier}: {len(eps)}")
    print(f"Validation dispositions:")
    for status, eps in sorted(disposition.items()):
        print(f"  {status}: {len(eps)}")
    print(f"  Sum of dispositions: {accounted}")
    print(f"  Entered corpus (ok): {len(validated)}")
    print(f"Total accounted: {excluded_total} + {accounted} = {excluded_total + accounted}")
    assert excluded_total + accounted == len(inv_rows), \
        f"Reconciliation FAILED: {excluded_total + accounted} != {len(inv_rows)}"

    return rows, ep_manifest, manifest


if __name__ == '__main__':
    main()
