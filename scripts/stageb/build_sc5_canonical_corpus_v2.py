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
import csv, hashlib, json, math, os, sys
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
from gripper_attack.sc5_event_segmenter_v2 import (
    SC5EventSegmenterV2, segment_events_from_labels, compute_event_sc5,
    _count_consecutive_stable_carry)
from gripper_attack.sc5_dedup import (
    compute_all_hashes, dedup_episodes, validate_split_isolation,
    validate_multi_split_isolation, _safe_float, MISSING_SENTINEL)

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

        candidate_tier = inv_row.get('exclusion_reason', inv_row.get('tier', ''))
        hashes = compute_all_hashes(records, inv_row.get('step_records_path', ''))
        episodes.append({
            'episode_id': inv_row['episode_id'],
            'run_id': inv_row.get('episode_dir', inv_row.get('run_id', '')),
            'records': records, 'manifest': manifest,
            'task_name': task, 'state_id': sid, 'is_butter': is_butter,
            'candidate_tier': candidate_tier,
            'jsonl_path': inv_row.get('step_records_path', ''),
            **hashes,
        })

    print(f"   {len(episodes)} clean episodes ({len(disposition['attack_contamination'])} attack, "
          f"{len(disposition.get('butter_s3_policy_exclude',[]))} s3, "
          f"{len(disposition.get('butter_s5_policy_exclude',[]))} s5)")

    if not episodes: print("FATAL: no episodes"); return

    # ── Step 3: Dedup + grouped split (held-out → train/val 75/25 by init-state group) ──
    print("3. Dedup + grouped split assignment...")
    unique, dup_groups = dedup_episodes(episodes, 'trajectory_content_sha256')

    # Assign held-out first
    for ep in unique:
        ep['is_held_out'] = ep['is_butter'] and ep['state_id'] in HELD_OUT_BUTTER
        ep['split'] = 'held_out' if ep['is_held_out'] else None  # None = to be assigned

    held_eps = [e for e in unique if e['is_held_out']]
    assignable = [e for e in unique if not e['is_held_out']]

    # Group by initial_state_sha256 for split assignment
    import random
    random.seed(42)
    init_groups = defaultdict(list)
    for ep in assignable:
        gk = ep.get('initial_state_sha256', '')
        if gk: init_groups[gk].append(ep)
        else: ep['split'] = 'train'  # no group key → train

    group_ids = sorted(init_groups.keys())
    random.shuffle(group_ids)
    n_train_grp = int(len(group_ids) * 0.75)
    train_groups = set(group_ids[:n_train_grp])
    val_groups = set(group_ids[n_train_grp:])

    for gk, members in init_groups.items():
        split = 'train' if gk in train_groups else 'val'
        for ep in members:
            ep['split'] = split

    train_eps = [e for e in assignable if e['split'] == 'train']
    val_eps = [e for e in assignable if e['split'] == 'val']
    iso = validate_split_isolation(unique, 'initial_state_sha256')
    print(f"   {len(unique)} unique ({len(dup_groups)} dup groups), "
          f"{len(train_eps)} train, {len(val_eps)} val, {len(held_eps)} held-out, "
          f"{len(train_groups)} train groups, {len(val_groups)} val groups, "
          f"split isolation: {'PASS' if iso['valid'] else 'FAIL'}")

    # ── Step 4: Tier validation + Teacher calibration ──
    TIER_A = 'LIBERO_OBJECT_SINGLE_OBJECT_CANDIDATE'
    TIER_B = 'REQUIRES_OBJECT_TARGET_VALIDATION'
    TIER_C = 'REQUIRES_EVENT_SEGMENTATION'
    print("4. Tier validation + Teacher calibration...")

    # Validate Tier B for ALL unique parents (not just train)
    tier_b_status = {}  # episode_id → status dict
    for ep in unique:
        tier = ep.get('candidate_tier', '')
        if tier == TIER_B:
            try:
                with open(ep['jsonl_path']) as f: recs = [json.loads(line) for line in f]
            except Exception:
                tier_b_status[ep['episode_id']] = {'status': 'TIER_B_JSONL_ERROR',
                                                     'reason': 'jsonl_read_error'}
                continue
            has_obj = any(r.get('object_pose_json', '') for r in recs
                         if r.get('teacher_privileged_state_available'))
            has_tgt = any(r.get('target_pose_json', '') for r in recs
                         if r.get('teacher_privileged_state_available'))
            if has_obj and has_tgt:
                tier_b_status[ep['episode_id']] = {'status': 'TIER_B_VALIDATED',
                                                     'reason': 'object_and_target_fields_present'}
            else:
                status = 'TIER_B_OBJECT_AMBIGUOUS' if not has_obj else 'TIER_B_TARGET_AMBIGUOUS'
                tier_b_status[ep['episode_id']] = {'status': status,
                                                     'reason': 'missing_required_privileged_field'}
        elif tier == TIER_A:
            tier_b_status[ep['episode_id']] = {'status': 'TIER_A', 'reason': ''}
        elif tier == TIER_C:
            tier_b_status[ep['episode_id']] = {'status': 'TIER_C', 'reason': ''}
        else:
            raise ValueError(f"Unknown candidate_tier '{tier}' for episode {ep['episode_id']}")

    # Teacher calibration: train Tier A + validated train Tier B only
    calib_eps = [e for e in train_eps
                 if e.get('candidate_tier') == TIER_A
                 or (e.get('candidate_tier') == TIER_B
                     and tier_b_status.get(e['episode_id'], {}).get('status') == 'TIER_B_VALIDATED')]
    valid_paths = []
    for ep in calib_eps:
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

    # Tier B statistics + CSV
    tb_counts = Counter(
        tier_b_status.get(e['episode_id'], {}).get('status', 'UNKNOWN')
        for e in unique if e.get('candidate_tier') == TIER_B)
    print(f"   Calibration: {len(valid_paths)}/{len(calib_eps)} paths")
    for status, cnt in sorted(tb_counts.items()):
        print(f"   {status}: {cnt}")

    # Write Tier B validation CSV (all Tier B episodes, all splits)
    tier_b_rows = []
    for ep in unique:
        if ep.get('candidate_tier') != TIER_B: continue
        st = tier_b_status.get(ep['episode_id'], {})
        tier_b_rows.append({
            'episode_id': ep['episode_id'],
            'task': ep['task_name'], 'state_id': ep['state_id'],
            'split': ep['split'],
            'candidate_tier': TIER_B,
            'validation_status': st.get('status', 'UNKNOWN'),
            'validation_reason': st.get('reason', ''),
            'note': 'FIELD_AVAILABILITY_ONLY_semantic_binding_not_proven',
        })
    if tier_b_rows:
        tb_path = os.path.join(args.output_dir, 'v2_sc5_tier_b_validation.csv')
        with open(tb_path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(tier_b_rows[0].keys()))
            w.writeheader(); w.writerows(tier_b_rows)

    # ── Step 5: Build dataset with row buffering + two-pass velocity recovery ──
    print("5. Building dataset with mature Layer 1/2 + event segmenter for Tier C...")
    rows = []; ep_rows = []; evt_rows = []; field_audit = Counter()
    post_dedup_disposition = defaultdict(list)
    tier_c_stats = Counter()
    segmenter = SC5EventSegmenterV2(teacher)

    # ── Build record_by_step index (avoids list-index assumption) ──
    def build_record_index(records):
        return {int(r.get('step_idx', r.get('policy_step_idx', -1))): r
                for r in records if r.get('teacher_privileged_state_available')}

    # ── Helper: build rows for one event span ──
    def build_event_rows(ep, labels, label_by_step, records, sc5, corridor,
                         output_start, output_end, continuity_span,
                         event_id, parent_ep_id, corpus_class='',
                         adapter=None, schema_adapter=None):
        """Extract 25D features. output_span controls what rows are emitted;
        continuity_span controls where gaps trigger fail-closed.
        adapter/schema_adapter carry history across events within same parent."""
        if adapter is None: adapter = SC5StreamingFeatureAdapterV2()
        if schema_adapter is None: schema_adapter = SC5SchemaAdapterV2()
        local_rows = []; local_step = 0; gap = False
        cspan = set(range(continuity_span[0], continuity_span[1] + 1))
        processed_step_ids = set()  # track which step IDs were actually valid

        for r in records:
            if not r.get('teacher_privileged_state_available'): continue
            step_raw = int(r.get('step_idx', r.get('policy_step_idx', 0)))
            if step_raw < output_start or step_raw > output_end: continue

            provenances = schema_adapter.validate_record_causal(r)
            for name, p in provenances.items():
                field_audit[f"{name}:{p.source_type}"] += 1
            if not schema_adapter.all_valid(provenances):
                if step_raw in cspan: gap = True; continue
            values = schema_adapter.extract_values(provenances)
            if any((isinstance(v, float) and (math.isnan(v) or v == MISSING_SENTINEL))
                   for v in values.values()):
                if step_raw in cspan: gap = True; continue
            env_action = r.get('env_action', None)
            if not (isinstance(env_action, (list, tuple)) and len(env_action) >= 7):
                if step_raw in cspan: gap = True; continue
            env_grip = float(env_action[6])
            raw_grip = float(values['gripper_command'])
            if (raw_grip <= 0.5) != (env_grip > 0):
                if step_raw in cspan: gap = True; continue
            try:
                result = adapter.update(
                    step_id=local_step, raw_gripper=raw_grip, env_gripper=env_grip,
                    gripper_qpos=values['gripper_qpos'],
                    gripper_opening_proxy=values['gripper_opening_proxy'],
                    eef_x=values['eef_x'], eef_y=values['eef_y'], eef_z=values['eef_z'],
                    eef_vx=values['eef_vx'], eef_vy=values['eef_vy'], eef_vz=values['eef_vz'],
                    action_dx=values['action_dx'], action_dy=values['action_dy'],
                    action_dz=values['action_dz'], action_gripper=values['action_gripper'])
            except ValueError:
                if step_raw in cspan: gap = True; continue
            if not result['valid']:
                if step_raw in cspan: gap = True; continue
            local_step += 1
            processed_step_ids.add(step_raw)

            tl = label_by_step.get(step_raw)
            has_sc5 = sc5['valid'] and not gap
            in_window = has_sc5 and sc5.get('window') and sc5['window'][0] <= step_raw <= sc5['window'][1]
            in_corridor = corridor is not None and step_raw in corridor.get('corridor_active_at_t', set())
            k10_valid = (corridor is not None and step_raw < len(corridor.get('full_k10_valid_at_t', []))
                         and corridor['full_k10_valid_at_t'][step_raw])

            row = dict(result['features'])
            row.update({
                'step_idx': step_raw, 'state_id': ep['state_id'],
                'task_name': ep['task_name'], 'is_butter': ep['is_butter'],
                'is_held_out': ep['is_held_out'], 'run_id': ep['run_id'],
                'episode_id': parent_ep_id or ep.get('episode_id', ''),
                'event_id': event_id, 'split': ep.get('split', ''),
                'candidate_tier': ep.get('candidate_tier', ''),
                'corpus_class': corpus_class,
                'initial_state_sha256': ep.get('initial_state_sha256', ''),
                'trajectory_content_sha256': ep.get('trajectory_content_sha256', ''),
                'teacher_phase': tl['phase'] if tl else 'abstain',
                'teacher_sc5_anchor': sc5['anchor'] if has_sc5 else -1,
                'teacher_sc5_attack_window_active': int(in_window),
                'teacher_sc5_ready': int(has_sc5 and step_raw == sc5['anchor']),
                'teacher_sc5_corridor_active': int(in_corridor),
                'teacher_full_k10_valid_at_t': int(k10_valid),
                'teacher_stable_carry_start': sc5.get('stable_carry_start', -1) if has_sc5 else -1,
                'teacher_confidence': tl['confidence'] if tl else 0.0,
            })
            local_rows.append(row)
        # Missing step detection: any step ID in continuity span not processed → gap
        if not gap and cspan:
            missing_steps = cspan - processed_step_ids
            if missing_steps:
                gap = True
        return local_rows, gap, local_step

    # ── Main episode loop ──
    for ep in unique:
        records = ep['records']
        labels = teacher.label_trajectory(records)
        label_by_step = {l['step_idx']: l for l in labels}
        tier = ep.get('candidate_tier', '')
        parent_ep_id = ep.get('episode_id', '')
        split = ep.get('split', '')
        tb_stat = tier_b_status.get(parent_ep_id, {}).get('status', '')

        # Tier B not validated → exclude from corpus
        if tier == TIER_B and tb_stat != 'TIER_B_VALIDATED':
            post_dedup_disposition[tb_stat or 'TIER_B_UNVALIDATED'].append(
                {'episode_id': parent_ep_id, 'state_id': ep['state_id'],
                 'task': ep['task_name'],
                 'reason': tier_b_status.get(parent_ep_id, {}).get('reason', '')})
            continue

        # ── Tier C: event segmentation ──
        if tier == TIER_C:
            tier_c_stats['parent_episodes'] += 1
            seg_result = segmenter.segment(labels, records, K=K, guard=GUARD)

            valid_events = [e for e in seg_result['events'] if e.get('event_valid')]
            rejected = [e for e in seg_result['events'] if not e.get('event_valid')]

            # Write ALL events to manifest (valid and rejected)
            for evt in seg_result['events']:
                evt_sc5 = evt.get('sc5') or {}
                evt_rows.append({
                    'parent_episode_id': parent_ep_id,
                    'event_id': evt['event_id'],
                    'task': ep['task_name'], 'state_id': ep['state_id'],
                    'split': split, 'candidate_tier': tier,
                    'event_valid': evt.get('event_valid', False),
                    'event_start': evt['start_step'],
                    'event_end': evt['end_step'],
                    'phase_order_valid': evt.get('phase_order_valid', False),
                    'has_all_required_phases': evt.get('has_all_required_phases', False),
                    'has_stable_carry': evt.get('has_stable_carry', False),
                    'has_release': evt.get('has_release', False),
                    'object_verifiable': evt.get('object_verifiable', False),
                    'object_ok': evt.get('object_ok', False),
                    'sc5_valid': evt_sc5.get('valid', False),
                    'sc5_anchor': evt_sc5.get('anchor', -1),
                    'reject_reason': evt.get('reject_reason', ''),
                })

            # Parent-prefix adapter: carry causal history across sibling events
            parent_adapter = SC5StreamingFeatureAdapterV2()
            parent_schema_adapter = SC5SchemaAdapterV2()

            for evt in valid_events:
                evt_sc5 = evt.get('sc5', {})
                evt_corridor = None
                if evt_sc5.get('valid'):
                    evt_corridor = compute_sc5_valid_start_corridor(
                        labels, evt_sc5['anchor'], K=K)
                evt_rows_list, evt_gap, evt_steps = build_event_rows(
                    ep, labels, label_by_step, records,
                    evt_sc5, evt_corridor,
                    evt['start_step'], evt['end_step'],
                    (evt['start_step'], evt['end_step']),
                    evt['event_id'], parent_ep_id,
                    corpus_class='PRIMARY_SC5_POSITIVE',
                    adapter=parent_adapter, schema_adapter=parent_schema_adapter)

                if evt_gap:
                    tier_c_stats['event_timeline_gap'] += 1; continue
                if evt_steps == 0:
                    tier_c_stats['event_no_feature_rows'] += 1; continue

                tier_c_stats['valid_events'] += 1
                rows.extend(evt_rows_list)

            for evt in rejected:
                tier_c_stats[f'reject_{evt.get("reject_reason","unknown")}'] += 1

            if not valid_events:
                tier_c_stats['OOD_MULTI_STAGE_ABSTAIN'] += 1
                post_dedup_disposition['OOD_MULTI_STAGE_ABSTAIN'].append(
                    {'episode_id': parent_ep_id, 'state_id': ep['state_id'],
                     'task': ep['task_name'],
                     'reason': seg_result.get('abstain_reason', 'no_valid_event')})
            continue

        # ── Tier A+B: whole-episode SC5, full output span ──
        sc5 = find_sc5_anchor_v2(labels, K=K, guard=GUARD)
        corridor = compute_sc5_valid_start_corridor(labels, sc5['anchor'], K=K) if sc5['valid'] else None

        # Full episode output (all policy steps) — preserves approach/release negatives
        policy_steps = [int(r.get('step_idx', 0)) for r in records
                        if r.get('teacher_privileged_state_available')]
        out_start = min(policy_steps, default=0)
        out_end = max(policy_steps, default=0)

        # Continuity = output: every output step must be valid.
        # No silent compression of approach, post-release, or any timeline region.
        c_start, c_end = out_start, out_end

        corpus_class = 'PRIMARY_SC5_POSITIVE' if sc5['valid'] else 'NO_CORRIDOR_NEGATIVE'

        episode_rows, policy_step_gap, local_step = build_event_rows(
            ep, labels, label_by_step, records, sc5, corridor,
            out_start, out_end, (c_start, c_end),
            -1, parent_ep_id, corpus_class=corpus_class)

        if policy_step_gap:
            post_dedup_disposition['NONCONTIGUOUS_POLICY_TIMELINE'].append(
                {'episode_id': parent_ep_id, 'state_id': ep['state_id'],
                 'task': ep['task_name'], 'reason': 'gap_in_required_span',
                 'sc5_anchor': sc5.get('anchor', -1) if sc5 else -1})
        elif local_step == 0:
            post_dedup_disposition['NO_VALID_FEATURE_ROWS'].append(
                {'episode_id': parent_ep_id, 'state_id': ep['state_id'],
                 'task': ep['task_name'], 'reason': 'no_valid_policy_steps',
                 'sc5_anchor': sc5.get('anchor', -1) if sc5 else -1})
        else:
            post_dedup_disposition['INCLUDED'].append(
                {'episode_id': parent_ep_id, 'state_id': ep['state_id'],
                 'task': ep['task_name'], 'sc5_anchor': sc5.get('anchor', -1) if sc5 else -1})
            rows.extend(episode_rows)
            ep_rows.append({
                'episode_id': parent_ep_id, 'parent_episode_id': '',
                'event_id': -1,
                'run_id': ep['run_id'], 'task': ep['task_name'],
                'state_id': ep['state_id'], 'is_butter': ep['is_butter'],
                'is_held_out': ep['is_held_out'], 'split': ep['split'],
                'candidate_tier': tier,
                'initial_state_sha256': ep.get('initial_state_sha256', ''),
                'trajectory_content_sha256': ep.get('trajectory_content_sha256', ''),
                'n_steps': local_step, 'sc5_valid': sc5['valid'] and not policy_step_gap,
                'sc5_anchor': sc5['anchor'] if (sc5['valid'] and not policy_step_gap) else -1,
                'corridor_start': corridor['corridor_start'] if corridor and not policy_step_gap else -1,
                'corridor_end': corridor['corridor_end'] if corridor and not policy_step_gap else -1,
            })

    # Tier C statistics
    if tier_c_stats:
        print(f"\n  Tier C segmentation:")
        for k, v in sorted(tier_c_stats.items()):
            print(f"    {k}: {v}")

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
    if evt_rows:
        with open(os.path.join(args.output_dir, 'v2_sc5_canonical_event_manifest.csv'), 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(evt_rows[0].keys())); w.writeheader(); w.writerows(evt_rows)
    if field_audit:
        with open(os.path.join(args.output_dir, 'v2_sc5_field_source_audit.csv'), 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=['field_source','count']); w.writeheader()
            for fs, cnt in field_audit.most_common(): w.writerow({'field_source': fs, 'count': cnt})

    # Full Teacher config freeze (dataclass fields + calibration metadata)
    from dataclasses import asdict
    config_full = asdict(teacher.cfg)
    config_full['guard'] = GUARD; config_full['K'] = K
    config_full['calibration_tiers'] = 'Tier_A+B_train_only'
    config_full['tier_c_train_excluded'] = tier_c_count if 'tier_c_count' in dir() else 0
    config_full['n_calibration_paths'] = len(valid_paths)
    config_full['n_calibration_episodes'] = len(calib_eps)
    # Calibration sources: ONLY actually-used paths, with FILE CONTENT SHA
    calib_sources = []
    for jp in valid_paths:
        try:
            with open(jp, 'rb') as f:
                file_sha = hashlib.sha256(f.read()).hexdigest()
        except Exception:
            file_sha = 'UNREADABLE'
        calib_sources.append({'jsonl_path': jp, 'source_file_sha256': file_sha})
    config_full['calibration_sources'] = sorted(calib_sources,
                                                 key=lambda x: x['jsonl_path'])
    config_path = os.path.join(args.artifacts_dir, 'v2_sc5_teacher_config.json')
    with open(config_path, 'w') as f:
        json.dump(config_full, f, indent=2, default=str)
    config_sha = hashlib.sha256(json.dumps(config_full, sort_keys=True).encode()).hexdigest()
    with open(os.path.join(args.artifacts_dir, 'v2_sc5_teacher_config.sha256'), 'w') as f:
        f.write(f"{config_sha}  v2_sc5_teacher_config.json\n")
    print(f"  Teacher config frozen: {config_sha[:16]} ({len(valid_paths)} paths)")

    manifest = {'K': K, 'guard': GUARD, 'n_rows': len(rows), 'n_episodes': len(ep_rows),
                'n_events': len(evt_rows), 'teacher_sha': config_sha[:16]}
    with open(os.path.join(args.artifacts_dir, 'v2_sc5_canonical_manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=2)

    # Summary
    sc5_eps = [e for e in ep_rows if e['sc5_valid']]
    print(f"\nCorpus: {len(rows)} rows, {len(ep_rows)} episodes, {len(sc5_eps)} SC5-valid")
    for split_name in ['train', 'val', 'held_out']:
        eps_in = [e for e in ep_rows if e['split'] == split_name]
        print(f"  {split_name}: {len(eps_in)} episodes, "
              f"{len([e for e in eps_in if e['sc5_valid']])} SC5-valid")
    for e in sorted(ep_rows, key=lambda x: x.get('state_id',0)):
        if e['is_butter']:
            print(f"  butter_s{e['state_id']}: sc5={e['sc5_valid']} split={e['split']}")

    # Post-dedup gap report + disposition CSV
    print(f"\nPost-dedup disposition (314 unique → {len(ep_rows)} corpus):")
    for reason, eps_list in sorted(post_dedup_disposition.items()):
        print(f"  {reason}: {len(eps_list)}")
    gap = len(unique) - len(ep_rows)
    if gap > 0:
        print(f"  GAP: {gap} episodes excluded after dedup")

    # Write formal disposition CSV (consistent fieldnames across all types)
    DISP_FIELDS = ['disposition', 'episode_id', 'state_id', 'task', 'reason', 'sc5_anchor']
    disp_rows = []
    for reason, eps_list in post_dedup_disposition.items():
        for entry in eps_list:
            disp_rows.append({
                'disposition': reason,
                'episode_id': entry.get('episode_id', ''),
                'state_id': entry.get('state_id', ''),
                'task': entry.get('task', ''),
                'reason': entry.get('reason', ''),
                'sc5_anchor': entry.get('sc5_anchor', ''),
            })
    if disp_rows:
        disp_path = os.path.join(args.output_dir, 'v2_sc5_post_dedup_disposition.csv')
        with open(disp_path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=DISP_FIELDS)
            w.writeheader(); w.writerows(disp_rows)
        print(f"  Disposition CSV: {disp_path}")

    # Multi-split isolation audit: ALL 314 unique parents + events
    iso_result = validate_multi_split_isolation(
        unique, group_key='initial_state_sha256', split_key='split')
    print(f"\nMulti-split isolation (314 parents, all pairs):")
    print(f"  Valid: {'PASS' if iso_result['valid'] else 'FAIL'}")
    for pair, n_v in iso_result['violations_by_pair'].items():
        print(f"  {pair}: {n_v} violations")

    iso_audit = {
        'status': 'PASS' if iso_result['valid'] else 'FAIL',
        'group_key': 'initial_state_sha256',
        'split_values': iso_result['split_values'],
        'violations': iso_result['violations_by_pair'],
        'n_groups': iso_result['n_groups'],
    }
    iso_path = os.path.join(args.artifacts_dir, 'v2_sc5_split_isolation_audit.json')
    with open(iso_path, 'w') as f:
        json.dump(iso_audit, f, indent=2)
    print(f"  Split isolation audit: {iso_path}")

    if not iso_result['valid']:
        raise RuntimeError(
            f"SPLIT_ISOLATION_FAILED: cross-split leakage detected. "
            f"See {iso_path} for details. "
            f"Violations: {iso_result['violations_by_pair']}")


if __name__ == '__main__':
    main()
