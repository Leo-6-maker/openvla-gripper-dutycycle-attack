#!/usr/bin/env python3
"""SC5 Canonical Corpus Builder v2 — thin orchestrator over mature Layer 1/2.

Architecture (data glue only, no new Layer 1/2 logic):
  Frozen census inventory → SchemaAdapter (normalize) → Dedup/Split →
  mature V2PrivilegedTeacher (label) → mature SC5StreamingFeatureAdapterV2 (25D) →
  CSV compatible with train_sc5_student_v2.py

Reuses (mature, debugged, never re-implemented):
  - v2_privileged_teacher.py: V2PrivilegedTeacher, calibrate_thresholds,
    find_sc5_anchor_v2, compute_sc5_valid_start_corridor
  - sc5_streaming_features_v2.py: SC5StreamingFeatureAdapterV2 (25D features)
  - sc5_schema_adapter_v2.py: SC5SchemaAdapterV2 (field normalization + provenance)
  - sc5_event_segmenter_v2.py: SC5EventSegmenterV2 (multi-stage only)
  - sc5_dedup.py: compute_all_hashes, dedup_episodes, validate_split_isolation
  - build_sc5_student_dataset_v2.py: output CSV schema pattern
"""
import csv, hashlib, json, os, sys
from collections import defaultdict, Counter
from pathlib import Path

# ── Repo path ──
_script_dir = Path(__file__).resolve().parent
REPO = None
for c in [_script_dir.parents[2],
          Path('/data/liuyu/repos/sc5_census_freeze_cc356f3_20260618'),
          Path('/data/liuyu/repos/sc5_census_freeze_7ab15f1_20260618')]:
    if (c / 'src' / 'gripper_attack' / 'v2_privileged_teacher.py').exists():
        REPO = c; break
if REPO is None: REPO = _script_dir.parents[2]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "src"))

# ── Mature modules (thin glue — no re-implementation) ──
from gripper_attack.v2_privileged_teacher import (
    V2PrivilegedTeacher, calibrate_thresholds,
    find_sc5_anchor_v2, compute_sc5_valid_start_corridor)
from gripper_attack.sc5_streaming_features_v2 import SC5StreamingFeatureAdapterV2
from gripper_attack.sc5_schema_adapter_v2 import SC5SchemaAdapterV2
from gripper_attack.sc5_dedup import (
    compute_all_hashes, dedup_episodes, validate_split_isolation, _safe_float)

K, GUARD = 10, 5
HELD_OUT_BUTTER = {8, 9, 11}
ELIGIBLE_TIERS = {'LIBERO_OBJECT_SINGLE_OBJECT_CANDIDATE',
                  'REQUIRES_OBJECT_TARGET_VALIDATION',
                  'REQUIRES_EVENT_SEGMENTATION'}


def load_and_filter_inventory(path):
    """Load frozen census, filter to eligible tiers, return (eligible, excluded, all_rows)."""
    rows = []; eligible = []; excluded = defaultdict(list)
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append(r)
            tier = r.get('exclusion_reason', r.get('tier', ''))
            if tier in ELIGIBLE_TIERS:
                eligible.append(r)
            else:
                excluded[tier].append(r['episode_id'])
    return eligible, excluded, rows


def validate_episode(inv_row):
    """Load + provenance-validate one episode. Returns (records, manifest, status)."""
    jp = inv_row.get('step_records_path', '')
    mp = inv_row.get('manifest_path', '')
    try:
        with open(jp) as f: records = [json.loads(line) for line in f]
    except Exception: return None, None, "jsonl_read_error"

    manifest = None
    if mp and os.path.isfile(mp):
        try:
            with open(mp) as f: manifest = json.load(f)
        except Exception: pass

    # Provenance: check every policy step for attack markers
    adapter = SC5SchemaAdapterV2()
    for rec in records:
        if rec.get('teacher_privileged_state_available'):
            if not adapter.validate_clean_provenance(rec, manifest)['clean_provenance']:
                return records, manifest, "attack_contamination"

    if not manifest or not manifest.get('success', False):
        return records, manifest, "not_clean_success"
    return records, manifest, "ok"


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--inventory', default='/tmp/sc5_source_census_cc356f3_r1/tables/v2_sc5_episode_inventory.csv')
    ap.add_argument('--output_dir', default='tables')
    ap.add_argument('--artifacts_dir', default='artifacts')
    args = ap.parse_args()

    # ── Step 1: Load frozen inventory ──
    print("1. Loading frozen inventory...")
    eligible, excluded, all_rows = load_and_filter_inventory(args.inventory)
    print(f"   {len(all_rows)} total, {len(eligible)} eligible, {sum(len(v) for v in excluded.values())} excluded")

    # ── Step 2: Validate each eligible episode ──
    print("2. Validating provenance + schema...")
    episodes = []; disposition = defaultdict(list)
    for inv_row in eligible:
        records, manifest, status = validate_episode(inv_row)
        disposition[status].append(inv_row['episode_id'])
        if status != 'ok': continue

        task = inv_row.get('task', manifest.get('task_name', '') if manifest else '')
        sid = int(inv_row.get('state_id', manifest.get('state_id', -1) if manifest else -1))
        is_butter = 'butter' in task.lower()

        # Split policy: s5 = AUDIT_ONLY, s3 = SUPPLEMENTARY_ABSTAIN
        if is_butter and sid in (3, 5):
            disposition[f'butter_s{sid}_policy_exclude'].append(inv_row['episode_id'])
            continue

        hashes = compute_all_hashes(records, inv_row.get('step_records_path', ''))
        episodes.append({
            'episode_id': inv_row['episode_id'],
            'run_id': inv_row.get('episode_dir', inv_row.get('run_id', '')),
            'records': records, 'manifest': manifest,
            'task_name': task, 'state_id': sid, 'is_butter': is_butter,
            'jsonl_path': inv_row.get('step_records_path', ''),
            **hashes,
        })

    print(f"   {len(episodes)} clean episodes ({len(disposition['attack_contamination'])} attack, "
          f"{len(disposition.get('butter_s3_policy_exclude',[]))} s3, "
          f"{len(disposition.get('butter_s5_policy_exclude',[]))} s5)")

    if not episodes: print("FATAL: no episodes"); return

    # ── Step 3: Dedup + split ──
    print("3. Dedup + split assignment...")
    unique, dup_groups = dedup_episodes(episodes, 'trajectory_content_sha256')
    for ep in unique:
        ep['is_held_out'] = ep['is_butter'] and ep['state_id'] in HELD_OUT_BUTTER
        ep['split'] = 'held_out' if ep['is_held_out'] else 'train'
    train_eps = [e for e in unique if not e['is_held_out']]
    held_eps = [e for e in unique if e['is_held_out']]
    iso = validate_split_isolation(unique, 'initial_state_sha256')
    print(f"   {len(unique)} unique ({len(dup_groups)} dup groups), "
          f"{len(train_eps)} train, {len(held_eps)} held-out, "
          f"split isolation: {'PASS' if iso['valid'] else 'FAIL'}")

    # ── Step 4: Mature Teacher calibration (train-only) ──
    print("4. Calibrating V2PrivilegedTeacher on train-only...")
    valid_paths = []
    for ep in train_eps:
        try:
            with open(ep['jsonl_path']) as f: recs = [json.loads(line) for line in f]
        except Exception: continue
        ok = True
        for r in recs:
            if not r.get('teacher_privileged_state_available'): continue
            for fld in ['object_to_target_distance','object_eef_distance','gripper_command','eef_x','eef_y','eef_z']:
                if r.get(fld) in (None, '', 'nan'): ok = False; break
            if not ok: break
        if ok: valid_paths.append(ep['jsonl_path'])
    teacher = V2PrivilegedTeacher(calibrate_thresholds(valid_paths))
    print(f"   {len(valid_paths)}/{len(train_eps)} valid calibration paths")

    # ── Step 5: Build dataset using MATURE Teacher + MATURE streaming adapter ──
    print("5. Building dataset with mature Layer 1/2...")
    rows = []; ep_rows = []; field_audit = Counter()

    for ep in unique:
        records = ep['records']
        labels = teacher.label_trajectory(records)
        label_by_step = {l['step_idx']: l for l in labels}
        sc5 = find_sc5_anchor_v2(labels, K=K, guard=GUARD)
        corridor = compute_sc5_valid_start_corridor(labels, sc5['anchor'], K=K) if sc5['valid'] else None

        # ── MATURE: SC5StreamingFeatureAdapterV2 (25D causal features) ──
        adapter = SC5StreamingFeatureAdapterV2()
        schema_adapter = SC5SchemaAdapterV2()
        local_step = 0

        for r in records:
            if not r.get('teacher_privileged_state_available'): continue
            step_raw = int(r.get('step_idx', r.get('policy_step_idx', 0)))

            # Schema normalization (data glue only)
            provenances = schema_adapter.validate_record(r)
            for name, p in provenances.items():
                field_audit[f"{name}:{p.source_type}"] += 1
            if not schema_adapter.all_valid(provenances): continue

            # Real env_gripper validation
            env_action = r.get('env_action', None)
            if not (isinstance(env_action, (list, tuple)) and len(env_action) >= 7): continue
            env_grip = float(env_action[6]); raw_grip = float(r['gripper_command'])
            if (raw_grip <= 0.5) != (env_grip > 0): continue  # semantic conflict

            # MATURE: call streaming adapter with normalized fields
            try:
                result = adapter.update(
                    step_id=local_step, raw_gripper=raw_grip, env_gripper=env_grip,
                    gripper_qpos=_safe_float(r.get('gripper_qpos')),
                    gripper_opening_proxy=_safe_float(r.get('gripper_width', r.get('gripper_opening_proxy'))),
                    eef_x=_safe_float(r.get('eef_x')), eef_y=_safe_float(r.get('eef_y')),
                    eef_z=_safe_float(r.get('eef_z')),
                    eef_vx=_safe_float(r.get('eef_vx')), eef_vy=_safe_float(r.get('eef_vy')),
                    eef_vz=_safe_float(r.get('eef_vz')),
                    action_dx=_safe_float(r.get('action_dx')), action_dy=_safe_float(r.get('action_dy')),
                    action_dz=_safe_float(r.get('action_dz')),
                    action_gripper=_safe_float(r.get('action_gripper', r.get('gripper_command'))))
            except ValueError: continue
            if not result['valid']: continue
            local_step += 1

            # ── MATURE: Teacher label lookup by step_idx ──
            tl = label_by_step.get(step_raw)
            has_sc5 = sc5['valid']
            in_window = has_sc5 and sc5['window'][0] <= step_raw <= sc5['window'][1]
            in_corridor = corridor is not None and step_raw in corridor['corridor_active_at_t']
            k10_valid = (corridor is not None and step_raw < len(corridor['full_k10_valid_at_t'])
                         and corridor['full_k10_valid_at_t'][step_raw])

            row = dict(result['features'])
            row.update({
                'step_idx': step_raw, 'state_id': ep['state_id'],
                'task_name': ep['task_name'], 'is_butter': ep['is_butter'],
                'is_held_out': ep['is_held_out'], 'run_id': ep['run_id'],
                'teacher_phase': tl['phase'] if tl else 'abstain',
                'teacher_sc5_anchor': sc5['anchor'] if has_sc5 else -1,
                'teacher_sc5_attack_window_active': int(in_window),
                'teacher_sc5_ready': int(has_sc5 and step_raw == sc5['anchor']),
                'teacher_sc5_corridor_active': int(in_corridor),
                'teacher_full_k10_valid_at_t': int(k10_valid),
                'teacher_stable_carry_start': sc5['stable_carry_start'] if has_sc5 else -1,
                'teacher_confidence': tl['confidence'] if tl else 0.0,
            })
            rows.append(row)

        if local_step > 0:
            ep_rows.append({
                'run_id': ep['run_id'], 'task': ep['task_name'],
                'state_id': ep['state_id'], 'is_butter': ep['is_butter'],
                'is_held_out': ep['is_held_out'], 'split': ep['split'],
                'n_steps': local_step, 'sc5_valid': sc5['valid'],
                'sc5_anchor': sc5['anchor'],
                'corridor_start': corridor['corridor_start'] if corridor else -1,
                'corridor_end': corridor['corridor_end'] if corridor else -1,
            })

    # ── Step 6: Write outputs (compatible with train_sc5_student_v2.py) ──
    print("6. Writing outputs...")
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.artifacts_dir, exist_ok=True)
    if rows:
        with open(os.path.join(args.output_dir, 'v2_sc5_canonical_dataset.csv'), 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    if ep_rows:
        with open(os.path.join(args.output_dir, 'v2_sc5_canonical_episodes.csv'), 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(ep_rows[0].keys())); w.writeheader(); w.writerows(ep_rows)
    if field_audit:
        with open(os.path.join(args.output_dir, 'v2_sc5_field_source_audit.csv'), 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=['field_source','count']); w.writeheader()
            for fs, cnt in field_audit.most_common(): w.writerow({'field_source': fs, 'count': cnt})

    manifest = {'K': K, 'guard': GUARD, 'n_rows': len(rows), 'n_episodes': len(ep_rows)}
    with open(os.path.join(args.artifacts_dir, 'v2_sc5_canonical_manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=2)

    # Summary
    sc5_eps = [e for e in ep_rows if e['sc5_valid']]
    print(f"\nCorpus: {len(rows)} rows, {len(ep_rows)} episodes, {len(sc5_eps)} SC5-valid")
    print(f"Train: {len([e for e in ep_rows if e['split']=='train'])}")
    print(f"Held-out: {len([e for e in ep_rows if e['split']=='held_out'])}")
    for e in sorted(ep_rows, key=lambda x: x.get('state_id',0)):
        if e['is_butter']:
            print(f"  butter_s{e['state_id']}: sc5={e['sc5_valid']} split={e['split']}")


if __name__ == '__main__':
    main()
