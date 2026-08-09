"""T1: Placement Root-Cause Audit.

Read-only diagnosis of placement detection gap.
Classifies all 139 successful libero_object episodes.
Frozen 32-episode manual audit set.
NO label modification. NO model output reading.
"""
import json, os, sys, time, hashlib
from collections import defaultdict
import numpy as np

DIR = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(os.path.dirname(DIR), 'phase2_labels'))

from v22_production_v2 import (
    parse_sidecar, resolve_goal_target, resolve_manipulated_objects,
    get_object_slices_for_task, compute_grasp_state, compute_placement_state,
    compute_safe_release, compute_terminal_state, _slice_vector, _dist, _finite_vector,
    V22_CONFIG,
)

CS200_ROOT = '/mnt/sdc/dty_user/openvla_attack_evidence/c2g/c2g_cs200_official_v3_20260716/clean'
IDENTITY_MANIFEST = '/mnt/sdc/dty_user/openvla_attack_evidence/c2g/c2g_cs200_official_v3_20260716/ops/FACTORIZED_PHASE_B2_DETERMINISTIC_ALLOCATION_V3_804113EE_20260723/checkpoint_training_identity_manifest.json'
LABEL_ROOT = '/mnt/sdc/dty_user/openvla_attack_outputs/n5/phase2_labels/g4_label_production'
G6_SEAL_PATH = '/mnt/sdc/dty_user/openvla_attack_outputs/n5/phase3_student/g6_training_seal/G6_SEAL_V2.json'
T1_OUT = '/mnt/sdc/dty_user/openvla_attack_outputs/n5/phase3_student/t1_placement_audit'

os.makedirs(T1_OUT, exist_ok=True)

PLACEMENT_TOLERANCE = V22_CONFIG['safe_release']['placement_region_tolerance']
RELEASE_WIDTH = V22_CONFIG['safe_release']['release_width_open_threshold']


def load_bddl_info(suite, task_idx):
    """Parse BDDL for a task using the same code path as the V22 pilot."""
    bddl_info = get_object_slices_for_task(suite, task_idx)
    if bddl_info is None:
        return {
            'available': False,
            'error': 'get_object_slices_for_task returned None',
            'manipulated_objects': [], 'support_names': [], 'target_names': [],
            'n_object_slices': 0,
        }

    task_role = bddl_info.get('task_role', {})
    object_slices = bddl_info.get('object_slices', {})

    return {
        'available': True,
        'n_object_slices': len(object_slices),
        'manipulated_objects': task_role.get('manipulated_objects', []),
        'support_names': task_role.get('support_names', []),
        'target_names': task_role.get('target_names', []),
        'object_slices_keys': sorted(object_slices.keys()),
    }


def classify_episode_placement(sidecar, bddl_info, task_idx, suite, episode_summary):
    """Trace the V22 placement code path step by step (exact match to pilot V3).

    Returns dict with:
      - disposition: root cause category
      - per_step: key diagnostic counters
      - failure_details: detailed breakdown
    """
    steps = sidecar
    T = len(steps)
    task_idx_int = int(task_idx) if isinstance(task_idx, str) else task_idx

    # Exact code path from pilot V3:
    # get_object_slices_for_task → task_role dict
    bddl_full = get_object_slices_for_task(suite, task_idx_int)
    if bddl_full is None:
        return {
            'disposition': 'BDDL_PARSE_FAILED',
            'can_compute': False,
            'error': 'get_object_slices_for_task returned None',
        }

    task_role = bddl_full.get('task_role', {})
    object_slices = bddl_full.get('object_slices', {})
    manipulated_objects = task_role.get('manipulated_objects', [])
    support_names = task_role.get('support_names', [])
    target_names = task_role.get('target_names', [])

    # Compute grasp state: (steps, manipulated_objects, support_names)
    grasp_results = compute_grasp_state(steps, manipulated_objects, support_names)

    # Compute terminal state: (steps, episode_summary)
    terminal_results = compute_terminal_state(steps, episode_summary)

    # Compute placement state: (steps, grasp_results, manipulated_objects, object_slices, target_names)
    placement_results = compute_placement_state(
        steps, grasp_results, manipulated_objects, object_slices, target_names
    )

    # Compute safe_release
    safe_release_results = compute_safe_release(
        steps, grasp_results, terminal_results, placement_results
    )

    # Diagnostics
    n_steps = T
    n_grasp_known = sum(1 for g in grasp_results if g['grasp_known_mask'])
    n_grasp_established = sum(1 for g in grasp_results if g['grasp_established'])
    n_grasp_transitions = 0
    for t in range(1, T):
        if grasp_results[t-1]['grasp_established'] and not grasp_results[t]['grasp_established']:
            n_grasp_transitions += 1

    n_placement_known = sum(1 for p in placement_results if p['placement_known_mask'])
    n_object_placed = sum(1 for p in placement_results if p['object_placed'])
    n_safe_release = sum(1 for s in safe_release_results
                        if s.get('release_detected') or False)

    # Gripper opening events
    n_gripper_opening = 0
    prev_width = None
    for t in range(T):
        rec = steps[t]
        qpos = _finite_vector(rec.get('robot0_gripper_qpos'), 2)
        if qpos is not None:
            width = abs(qpos[0]) + abs(qpos[1])
            if prev_width is not None and (width - prev_width) > 0.02:
                n_gripper_opening += 1
            prev_width = width

    # Determine root cause
    can_compute = len(manipulated_objects) > 0 and len(object_slices) > 0
    has_grasp_known = n_grasp_known > 0
    has_grasp_established = n_grasp_established > 0
    has_grasp_transition = n_grasp_transitions > 0
    has_target_slices = len(target_names) > 0 and all(
        tname in object_slices for tname in target_names
    )

    # Check distance at transition points
    distance_checks = []
    for t in range(1, T):
        if grasp_results[t-1]['grasp_established'] and not grasp_results[t]['grasp_established']:
            for name in manipulated_objects:
                spec = object_slices.get(name)
                if spec is None: continue
                obj_pos = _slice_vector(steps[t].get('object_state', []), spec, 'pos')
                if obj_pos is None: continue
                for tname in target_names:
                    tspec = object_slices.get(tname)
                    if tspec is None: continue
                    tpos = _slice_vector(steps[t].get('object_state', []), tspec, 'pos')
                    if tpos is None: continue
                    d = _dist(obj_pos, tpos)
                    distance_checks.append({'step': t, 'object': name, 'target': tname,
                                           'distance': float(d),
                                           'passed': d < PLACEMENT_TOLERANCE})

    # Classification
    if not can_compute:
        if len(manipulated_objects) == 0:
            cause = 'BDDL_NO_MANIPULATED_OBJECTS'
        else:
            cause = 'BDDL_NO_OBJECT_SLICES'
    elif not has_grasp_known:
        cause = 'GRASP_NEVER_KNOWN'
    elif not has_grasp_established:
        cause = 'GRASP_NEVER_ESTABLISHED'
    elif not has_grasp_transition:
        cause = 'NO_GRASP_TO_RELEASE_TRANSITION'
    elif not has_target_slices:
        cause = 'TARGET_NOT_IN_OBJECT_SLICES'
    elif n_object_placed == 0 and len(distance_checks) > 0:
        # Check if distance was close
        min_dist = min(d['distance'] for d in distance_checks)
        if min_dist < PLACEMENT_TOLERANCE * 2:
            cause = 'DISTANCE_MARGINAL'
        else:
            cause = 'DISTANCE_TOO_LARGE'
    elif n_object_placed == 0:
        cause = 'NO_PLACEMENT_DETECTED_UNKNOWN_REASON'
    else:
        cause = 'PLACEMENT_DETECTED_BUT_NO_SAFE_RELEASE'

    return {
        'disposition': cause,
        'can_compute': can_compute,
        'has_grasp_known': has_grasp_known,
        'has_grasp_established': has_grasp_established,
        'has_grasp_transition': has_grasp_transition,
        'has_target_slices': has_target_slices,
        'n_steps': n_steps,
        'n_grasp_known': n_grasp_known,
        'n_grasp_established': n_grasp_established,
        'n_grasp_transitions': n_grasp_transitions,
        'n_placement_known': n_placement_known,
        'n_object_placed': n_object_placed,
        'n_safe_release': n_safe_release,
        'n_gripper_opening_events': n_gripper_opening,
        'n_manipulated_objects': len(manipulated_objects),
        'n_object_slices': len(object_slices),
        'n_target_names': len(target_names),
        'manipulated_objects': list(manipulated_objects)[:5],
        'target_names': list(target_names)[:5],
        'distance_checks': distance_checks[:10],  # first 10
        'placement_tolerance': PLACEMENT_TOLERANCE,
        'release_width_threshold': RELEASE_WIDTH,
    }


def main():
    print('=' * 60)
    print('T1: Placement Root-Cause Audit')
    print('=' * 60)
    print(f'Placement tolerance: {PLACEMENT_TOLERANCE}')
    print(f'Release width threshold: {RELEASE_WIDTH}')
    print()

    # ── 1. Load all 800 identities, find successful libero_object ──
    print('--- 1. Loading identities and BDDL info ---')

    with open(G6_SEAL_PATH) as f:
        seal = json.load(f)

    all_train = set()
    for k in ['train_identities', 'val_identities', 'cal_identities']:
        all_train.update(seal['split'].get(k, []))

    print(f'Total identities: {len(all_train)}')

    # Group by suite
    by_suite = defaultdict(list)
    for ident in sorted(all_train):
        suite = ident.split('/')[0]
        by_suite[suite].append(ident)

    # Load episode summaries to find successful ones
    successful_obj = []
    for ident in by_suite.get('libero_object', []):
        suite, task, state = ident.split('/')
        summary_path = os.path.join(CS200_ROOT, suite, task, state, 'episode_summary.json')
        if os.path.isfile(summary_path):
            with open(summary_path) as f:
                summary = json.load(f)
            if summary.get('success', False):
                successful_obj.append(ident)

    print(f'libero_object total: {len(by_suite.get("libero_object", []))}')
    print(f'libero_object successful: {len(successful_obj)}')

    # ── 2. BDDL analysis per task ──
    print('\n--- 2. Per-Task BDDL Analysis ---')
    task_bddl = {}
    for suite in ['libero_10', 'libero_goal', 'libero_object', 'libero_spatial']:
        for task_dir in sorted(os.listdir(os.path.join(CS200_ROOT, suite))):
            task_key = f'{suite}/{task_dir}'
            task_idx = int(task_dir.replace('task_', ''))
            info = load_bddl_info(suite, task_idx)
            task_bddl[task_key] = info
            n_manip = len(info.get('manipulated_objects', []))
            n_support = len(info.get('support_names', []))
            n_target = len(info.get('target_names', []))
            n_slices = info.get('n_object_slices', 0)
            has_error = not info.get('available', False)
            obj_keys = ', '.join(info.get('object_slices_keys', [])[:3])
            print(f'  {task_key}: manip={n_manip}, support={n_support}, '
                  f'target={n_target}, slices={n_slices}, keys=[{obj_keys}...]'
                  + (' BDDL_FAIL' if has_error else ''))

    # ── 3. Classify all 139 successful libero_object ──
    print(f'\n--- 3. Classifying {len(successful_obj)} successful libero_object episodes ---')
    dispositions = defaultdict(list)
    per_episode = {}

    for i, ident in enumerate(successful_obj):
        suite, task, state = ident.split('/')
        task_key = f'{suite}/{task}'
        task_idx = int(task.replace('task_', ''))

        # Load sidecar
        sidecar_path = os.path.join(CS200_ROOT, suite, task, state, 'privileged_teacher_sidecar.jsonl')
        if not os.path.isfile(sidecar_path):
            dispositions['SIDECAR_MISSING'].append(ident)
            per_episode[ident] = {'disposition': 'SIDECAR_MISSING'}
            continue

        parsed = parse_sidecar(sidecar_path)
        steps = parsed['steps']

        # Load episode summary
        summary_path = os.path.join(CS200_ROOT, suite, task, state, 'episode_summary.json')
        episode_summary = {}
        if os.path.isfile(summary_path):
            with open(summary_path) as f:
                episode_summary = json.load(f)

        bddl = task_bddl.get(task_key, {})
        result = classify_episode_placement(steps, bddl, task_idx, suite, episode_summary)
        dispositions[result['disposition']].append(ident)
        per_episode[ident] = result

        if (i + 1) % 30 == 0:
            print(f'  {i+1}/{len(successful_obj)} done')

    # ── 4. Disposition summary ──
    print(f'\n--- 4. Disposition Summary ---')
    total = len(successful_obj)
    for cause in sorted(dispositions.keys(), key=lambda c: -len(dispositions[c])):
        eps = dispositions[cause]
        pct = len(eps) / max(1, total) * 100
        print(f'  {cause}: {len(eps)} ({pct:.1f}%)')
        if len(eps) <= 5:
            for ep in eps:
                print(f'    {ep}')

    # ── 5. Per-task aggregate stats ──
    print(f'\n--- 5. Per-Task Aggregate ---')
    task_stats = defaultdict(lambda: {
        'n_episodes': 0, 'n_successful': 0,
        'n_grasp_known': 0, 'n_grasp_est': 0, 'n_grasp_trans': 0,
        'n_placement_known': 0, 'n_placed': 0, 'n_safe_release': 0,
        'n_gripper_open': 0, 'dispositions': defaultdict(int),
    })

    for ident in sorted(all_train):
        suite, task, state = ident.split('/')
        task_key = f'{suite}/{task}'

        summary_path = os.path.join(CS200_ROOT, suite, task, state, 'episode_summary.json')
        is_success = False
        if os.path.isfile(summary_path):
            with open(summary_path) as f:
                is_success = json.load(f).get('success', False)

        task_stats[task_key]['n_episodes'] += 1
        if is_success:
            task_stats[task_key]['n_successful'] += 1

        if ident in per_episode:
            ep = per_episode[ident]
            task_stats[task_key]['n_grasp_known'] += ep.get('n_grasp_known', 0)
            task_stats[task_key]['n_grasp_est'] += ep.get('n_grasp_established', 0)
            task_stats[task_key]['n_grasp_trans'] += ep.get('n_grasp_transitions', 0)
            task_stats[task_key]['n_placement_known'] += ep.get('n_placement_known', 0)
            task_stats[task_key]['n_placed'] += ep.get('n_object_placed', 0)
            task_stats[task_key]['n_safe_release'] += ep.get('n_safe_release', 0)
            task_stats[task_key]['n_gripper_open'] += ep.get('n_gripper_opening_events', 0)
            task_stats[task_key]['dispositions'][ep['disposition']] += 1

    print(f'{"Task":<22} {"Ep":>4} {"Succ":>5} {"GK":>5} {"GE":>5} {"GT":>5} {"PK":>5} {"Pl":>5} {"SR":>4} {"Open":>5}')
    print('-' * 80)
    for task_key in sorted(task_stats.keys()):
        s = task_stats[task_key]
        print(f'{task_key:<22} {s["n_episodes"]:>4} {s["n_successful"]:>5} '
              f'{s["n_grasp_known"]:>5} {s["n_grasp_est"]:>5} {s["n_grasp_trans"]:>5} '
              f'{s["n_placement_known"]:>5} {s["n_placed"]:>5} {s["n_safe_release"]:>4} '
              f'{s["n_gripper_open"]:>5}')

    # ── 6. Freeze 32-episode manual audit set ──
    print(f'\n--- 6. Freezing 32-Episode Manual Audit Set ---')
    rng = np.random.RandomState(19903 + 100)

    manual_set = []

    # 16 libero_object successful (stratified by disposition)
    obj_by_disp = defaultdict(list)
    for ident in successful_obj:
        disp = per_episode.get(ident, {}).get('disposition', 'UNKNOWN')
        obj_by_disp[disp].append(ident)

    n_obj_picked = 0
    for disp in sorted(obj_by_disp.keys(), key=lambda d: -len(obj_by_disp[d])):
        candidates = obj_by_disp[disp]
        rng.shuffle(candidates)
        n_pick = min(len(candidates), max(2, int(16 * len(candidates) / len(successful_obj))))
        for ident in candidates[:n_pick]:
            if n_obj_picked >= 16:
                break
            manual_set.append({'identity': ident, 'category': f'libero_object_{disp}',
                              'reason': 'OBJECT_SUCCESS_NO_PLACEMENT'})
            n_obj_picked += 1

    # 8 other-suite normal release (episodes with safe_release)
    sr_targets = []
    for suite in ['libero_10', 'libero_goal', 'libero_spatial']:
        for ident in by_suite.get(suite, []):
            suite_s, task_s, state_s = ident.split('/')
            label_path = os.path.join(LABEL_ROOT, suite_s, task_s, state_s, 'label_contract_v2.jsonl')
            if os.path.isfile(label_path):
                has_sr = False
                with open(label_path) as f:
                    for line in f:
                        if line.strip():
                            s = json.loads(line)
                            if s.get('safe_release', {}).get('valid_mask') and s.get('safe_release', {}).get('value'):
                                has_sr = True
                                break
                if has_sr:
                    sr_targets.append(ident)
    rng.shuffle(sr_targets)
    for ident in sr_targets[:8]:
        manual_set.append({'identity': ident, 'category': 'other_suite_safe_release',
                          'reason': 'NORMAL_RELEASE_REFERENCE'})

    # 4 pregrasp negative
    pregrasp_candidates = []
    for suite in ['libero_10', 'libero_goal', 'libero_object', 'libero_spatial']:
        for ident in by_suite.get(suite, []):
            suite_s, task_s, state_s = ident.split('/')
            sidecar_path = os.path.join(CS200_ROOT, suite_s, task_s, state_s, 'privileged_teacher_sidecar.jsonl')
            if os.path.isfile(sidecar_path):
                parsed = parse_sidecar(sidecar_path)
                if parsed['n_steps'] > 20:
                    # Check if first 10 steps have no grasp
                    steps = parsed['steps'][:10]
                    contacts = [s.get('contact_count', 0) for s in steps]
                    if max(contacts) == 0:
                        pregrasp_candidates.append(ident)
                        break
            if len(pregrasp_candidates) >= 4:
                break
    rng.shuffle(pregrasp_candidates)
    for ident in pregrasp_candidates[:4]:
        manual_set.append({'identity': ident, 'category': 'pregrasp_negative',
                          'reason': 'NO_GRASP_PREGRASP'})

    # 4 articulated unknown
    articulated_candidates = []
    for ident in by_suite.get('libero_goal', []):
        suite_s, task_s, state_s = ident.split('/')
        label_path = os.path.join(LABEL_ROOT, suite_s, task_s, state_s, 'label_contract_v2.jsonl')
        if os.path.isfile(label_path):
            n_unk = 0
            with open(label_path) as f:
                for line in f:
                    if line.strip():
                        s = json.loads(line)
                        if not s.get('physical_criticality', {}).get('valid_mask', False):
                            n_unk += 1
            if n_unk > 50:
                articulated_candidates.append(ident)
    rng.shuffle(articulated_candidates)
    for ident in articulated_candidates[:4]:
        manual_set.append({'identity': ident, 'category': 'articulated_unknown',
                          'reason': 'ARTICULATED_UNKNOWN_ABSTAIN'})

    print(f'Manual set: {len(manual_set)} episodes')
    cat_counts = defaultdict(int)
    for m in manual_set:
        cat_counts[m['category']] += 1
    for cat, n in sorted(cat_counts.items()):
        print(f'  {cat}: {n}')

    # ── 7. Per-stage miss counts ──
    print(f'\n--- 7. Per-Stage Miss Summary ---')
    stages = {
        'BDDL_NO_MANIPULATED_OBJECTS': 'Stage 1: BDDL has no manipulated_objects',
        'BDDL_NO_OBJECT_SLICES': 'Stage 1: BDDL has no object_slices',
        'GRASP_NEVER_KNOWN': 'Stage 2: grasp_known_mask never True',
        'GRASP_NEVER_ESTABLISHED': 'Stage 3: grasp_established never True',
        'NO_GRASP_TO_RELEASE_TRANSITION': 'Stage 4: grasp transition never detected',
        'TARGET_NOT_IN_OBJECT_SLICES': 'Stage 5: target_names not in object_slices',
        'DISTANCE_MARGINAL': 'Stage 6a: distance slightly above tolerance',
        'DISTANCE_TOO_LARGE': 'Stage 6b: object far from target at release',
        'NO_PLACEMENT_DETECTED_UNKNOWN_REASON': 'Stage ?: unknown failure',
        'PLACEMENT_DETECTED_BUT_NO_SAFE_RELEASE': 'Stage 7: placement ok but safe_release fails',
    }

    for cause, description in stages.items():
        n = len(dispositions.get(cause, []))
        if n > 0:
            print(f'  [{cause}]: {n} episodes — {description}')

    # ── 8. Write T1 report ──
    report = {
        'gate': 'T1_PLACEMENT_ROOT_CAUSE_AUDIT',
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'status': 'COMPLETE',
        'placement_tolerance': PLACEMENT_TOLERANCE,
        'release_width_threshold': RELEASE_WIDTH,

        'summary': {
            'n_total': len(all_train),
            'libero_object_total': len(by_suite.get('libero_object', [])),
            'libero_object_successful': len(successful_obj),
            'n_safe_release_all_suites': sum(1 for ident in all_train
                if os.path.isfile(os.path.join(LABEL_ROOT, *ident.split('/'), 'label_contract_v2.jsonl'))
                and any(json.loads(line).get('safe_release', {}).get('value')
                       for line in open(os.path.join(LABEL_ROOT, *ident.split('/'), 'label_contract_v2.jsonl')))
            ),
        },

        'per_task_bddl': {
            task: {k: v for k, v in info.items() if k != 'object_slices'}
            for task, info in task_bddl.items()
        },

        'dispositions': {
            cause: {'n': len(eps), 'pct': round(len(eps)/max(1,total)*100, 1),
                   'identities': eps}
            for cause, eps in dispositions.items()
        },

        'per_task_stats': {
            task: {
                k: (dict(v) if isinstance(v, defaultdict) else v)
                for k, v in stats.items()
            }
            for task, stats in task_stats.items()
        },

        'manual_audit_set': manual_set,

        'per_episode_details': {
            ident: {k: v for k, v in ep.items() if k != 'distance_checks'}
            for ident, ep in per_episode.items()
        },
    }

    report_path = os.path.join(T1_OUT, 'T1_REPORT.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    report_sha = hashlib.sha256(open(report_path, 'rb').read()).hexdigest()
    report['self_sha256'] = report_sha
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)

    print(f'\nT1 Report: {report_path}')
    print(f'SHA: {report_sha[:16]}...')

    # ── 9. Determine pass/fail ──
    all_classified = sum(len(v) for v in dispositions.values()) == len(successful_obj)
    if all_classified:
        print(f'\nT1: PASS — {len(successful_obj)}/{len(successful_obj)} episodes classified')
        sys.exit(0)
    else:
        print(f'\nT1: FAIL — {len(successful_obj) - all_classified} episodes unclassified')
        sys.exit(5)


if __name__ == '__main__':
    main()
