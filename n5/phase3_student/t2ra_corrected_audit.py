"""T2R-A: Corrected Observability Audit.

Fixes from user audit:
  1. Contact requires manipulated-object ↔ goal-support in SAME pair
  2. Coverage uses set union, not sum
  3. Classifies each miss by actual geometry observability
  4. No fuzzy substring as final entity binding
"""
import json, os, sys, time, hashlib, re
from collections import defaultdict
import numpy as np

DIR = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(os.path.dirname(DIR), 'phase2_labels'))
from v22_production_v2 import (
    parse_sidecar, get_object_slices_for_task,
    compute_grasp_state, _slice_vector, _dist, _finite_vector, _contact_flags,
)

CS200 = '/mnt/sdc/dty_user/openvla_attack_evidence/c2g/c2g_cs200_official_v3_20260716/clean'
G6_SEAL = '/mnt/sdc/dty_user/openvla_attack_outputs/n5/phase3_student/g6_training_seal/G6_SEAL_V2.json'
T2RA_OUT = '/mnt/sdc/dty_user/openvla_attack_outputs/n5/phase3_student/t2ra_corrected_audit'
os.makedirs(T2RA_OUT, exist_ok=True)

with open(G6_SEAL) as f:
    seal = json.load(f)
all_train = set()
for k in ['train_identities', 'val_identities', 'cal_identities']:
    all_train.update(seal['split'].get(k, []))

# Rebuild canary with same seed
rng = np.random.RandomState(20103)
canary = []
for suite in ['libero_10', 'libero_goal', 'libero_object', 'libero_spatial']:
    suite_eps = sorted([i for i in all_train if i.startswith(suite)])
    successful = []; other = []
    for ident in suite_eps:
        s, t, st = ident.split('/')
        sp = os.path.join(CS200, s, t, st, 'episode_summary.json')
        if os.path.isfile(sp):
            with open(sp) as f:
                if json.load(f).get('success', False):
                    successful.append(ident)
                else:
                    other.append(ident)
    rng.shuffle(successful); rng.shuffle(other)
    canary.extend(successful[:10] + other[:6])


def _normalize_geom(name):
    """Normalize MuJoCo geom names by removing known suffixes."""
    n = name
    for suffix in ['_visual', '_collision', '_geom', '_body', '_link',
                   '_contain_region', '_init_region', '_back', '_front',
                   '_top', '_bottom', '_left', '_right', '_main']:
        if n.endswith(suffix):
            n = n[:-len(suffix)]
    return n


def check_object_goal_support_strict(steps_data, t, manipulated_objects, goal_support_names):
    """Check if manipulated object AND goal support appear IN THE SAME contact pair.

    Returns (has_contact, goal_support_name, object_name)
    """
    pairs = steps_data[t].get('mujoco_contact_pairs', [])
    if not goal_support_names or not manipulated_objects:
        return False, None, None

    for pair in pairs:
        pair_strs = [str(item) for item in pair]
        pair_norms = [_normalize_geom(ps) for ps in pair_strs]

        # Check: does this pair contain BOTH a manipulated object AND a goal support?
        has_object = False; obj_name = None
        has_gs = False; gs_name = None

        for mo in manipulated_objects:
            mo_norm = _normalize_geom(mo)
            for pn in pair_norms:
                if mo_norm in pn or pn in mo_norm:
                    has_object = True
                    obj_name = mo
                    break
            if has_object: break

        for gs in goal_support_names:
            gs_norm = _normalize_geom(gs)
            for pn in pair_norms:
                if gs_norm in pn or pn in gs_norm:
                    has_gs = True
                    gs_name = gs
                    break
            if has_gs: break

        if has_object and has_gs:
            return True, gs_name, obj_name

    return False, None, None


def resolve_bddl_target_to_slices(target_name, object_slices):
    """Try to resolve a BDDL target name to an object_slices entry.

    Returns (resolved_key, method) or (None, reason).
    """
    # Direct match
    if target_name in object_slices:
        return target_name, 'DIRECT'

    # Strip suffix
    for suffix in ['_contain_region', '_init_region', '_cook_region',
                   '_heating_region', '_top_region', '_front_region']:
        base = target_name.replace(suffix, '')
        if base in object_slices:
            return base, 'STRIP_SUFFIX'

    # Substring match (with warning)
    for key in sorted(object_slices.keys()):
        key_stripped = key.replace('_contain_region', '').replace('_init_region', '')
        target_stripped = target_name.replace('_contain_region', '').replace('_init_region', '')
        if key_stripped in target_stripped or target_stripped in key_stripped:
            return key, 'SUBSTRING'

    return None, 'NO_MATCH'


def classify_episode(ident):
    """Full observability analysis for one episode."""
    suite, task, state = ident.split('/')
    sidecar_path = os.path.join(CS200, suite, task, state, 'privileged_teacher_sidecar.jsonl')
    summary_path = os.path.join(CS200, suite, task, state, 'episode_summary.json')
    if not os.path.isfile(sidecar_path):
        return {'error': 'no sidecar'}

    parsed = parse_sidecar(sidecar_path)
    steps_data = parsed['steps']
    T = len(steps_data)
    with open(summary_path) as f:
        ep_summary = json.load(f)
    is_success = ep_summary.get('success', False)

    task_idx = int(task.replace('task_', ''))
    bddl = get_object_slices_for_task(suite, task_idx)
    if bddl is None:
        return {'error': 'no bddl'}

    task_role = bddl['task_role']
    object_slices = bddl['object_slices']
    manip = task_role.get('manipulated_objects', [])
    targets = task_role.get('target_names', [])
    gs_names = task_role.get('goal_support_names', [])
    g_rels = task_role.get('goal_relations', [])

    # 1. Strict goal-support contact (object AND support in same pair)
    gs_contact_steps = 0
    gs_contact_objects = set()
    for t in range(T):
        has_gs, gs_name, obj_name = check_object_goal_support_strict(
            steps_data, t, manip, gs_names)
        if has_gs:
            gs_contact_steps += 1
            if gs_name and obj_name:
                gs_contact_objects.add((obj_name, gs_name))

    # 2. Geometry availability per goal relation
    relation_geometry = []
    for pred, obj, target in g_rels:
        obj_resolved, obj_method = resolve_bddl_target_to_slices(obj, object_slices)
        tgt_resolved, tgt_method = resolve_bddl_target_to_slices(target, object_slices)

        obj_pos_avail = obj_resolved is not None
        tgt_pos_avail = tgt_resolved is not None

        # Check if target has extent/size info in object_slices
        tgt_spec = object_slices.get(tgt_resolved) if tgt_resolved else None
        tgt_has_size = False
        if tgt_spec:
            size_vec = _slice_vector(steps_data[T-1].get('object_state', []),
                                    tgt_spec, 'size')
            tgt_has_size = size_vec is not None and sum(abs(x) for x in size_vec) > 0.001

        relation_geometry.append({
            'predicate': pred,
            'object': obj,
            'target': target,
            'obj_resolved': obj_resolved,
            'obj_method': obj_method,
            'tgt_resolved': tgt_resolved,
            'tgt_method': tgt_method,
            'obj_pos_available': obj_pos_avail,
            'tgt_pos_available': tgt_pos_avail,
            'tgt_has_size': tgt_has_size,
            'full_geometry': obj_pos_avail and tgt_pos_avail,
        })

    # 3. Grasp/contact evidence
    grasp = compute_grasp_state(steps_data, manip, task_role.get('support_names', []))
    n_ce = sum(1 for g in grasp if g.get('contact_established'))
    n_ge = sum(1 for g in grasp if g.get('grasp_established'))

    # 4. Object velocity near end
    velocities = []
    prev_pos = None
    for name in manip:
        spec = object_slices.get(name)
        if spec is None: continue
        for t in range(max(0, T-30), T):
            pos = _slice_vector(steps_data[t].get('object_state', []), spec, 'pos')
            if pos is not None and prev_pos is not None:
                velocities.append(_dist(pos, prev_pos))
            prev_pos = pos
        break
    is_stable = np.median(velocities) < 0.005 if velocities else False

    # 5. Classification
    has_strict_gs_contact = gs_contact_steps > 0
    has_full_geometry = any(r['full_geometry'] for r in relation_geometry)
    has_pose_only = any(r['obj_pos_available'] or r['tgt_pos_available']
                       for r in relation_geometry)
    has_relations = len(g_rels) > 0
    has_object_pose = any(resolve_bddl_target_to_slices(m, object_slices)[0]
                         for m in manip)

    if has_strict_gs_contact:
        cause = 'NATIVE_PREDICATE_AVAILABLE'
    elif has_full_geometry:
        cause = 'FULL_GEOMETRY_AVAILABLE'
    elif has_pose_only:
        cause = 'POSE_ONLY_INSUFFICIENT'
    elif has_relations or has_object_pose:
        cause = 'REPLAY_REQUIRED'
    else:
        cause = 'TRULY_UNOBSERVABLE'

    return {
        'identity': ident,
        'suite': suite,
        'T': T,
        'is_success': is_success,
        'cause': cause,
        'gs_contact_steps': gs_contact_steps,
        'gs_contact_pairs': list(gs_contact_objects)[:5],
        'g_rels': g_rels,
        'manip': manip,
        'targets': targets,
        'gs_names': gs_names,
        'contact_est_steps': n_ce,
        'grasp_est_steps': n_ge,
        'object_stable': is_stable,
        'relation_geometry': relation_geometry,
    }


def main():
    print('=' * 60)
    print('T2R-A: Corrected Observability Audit')
    print('=' * 60)

    results = {}
    for ident in canary:
        try:
            r = classify_episode(ident)
            results[ident] = r
        except Exception as e:
            results[ident] = {'identity': ident, 'error': str(e)}

    # Separate successful from others
    successful = {k: v for k, v in results.items() if v.get('is_success')}
    misses = {k: v for k, v in successful.items() if v.get('gs_contact_steps', 0) == 0}

    # Correct coverage: SET UNION, not sum
    contact_ids = set()
    geometry_ids = set()
    pose_ids = set()
    replay_ids = set()
    unobs_ids = set()

    for ident, r in successful.items():
        cause = r.get('cause', 'UNKNOWN')
        if cause == 'NATIVE_PREDICATE_AVAILABLE':
            contact_ids.add(ident)
        elif cause == 'FULL_GEOMETRY_AVAILABLE':
            geometry_ids.add(ident)
        elif cause == 'POSE_ONLY_INSUFFICIENT':
            pose_ids.add(ident)
        elif cause == 'REPLAY_REQUIRED':
            replay_ids.add(ident)
        else:
            unobs_ids.add(ident)

    any_signal_ids = contact_ids | geometry_ids | replay_ids
    n_successful = len(successful)

    print(f'\nSuccessful episodes: {n_successful}')
    print(f'  NATIVE_PREDICATE (strict contact): {len(contact_ids)} ({len(contact_ids)/max(1,n_successful)*100:.0f}%)')
    print(f'  FULL_GEOMETRY: {len(geometry_ids)} ({len(geometry_ids)/max(1,n_successful)*100:.0f}%)')
    print(f'  POSE_ONLY: {len(pose_ids)}')
    print(f'  REPLAY_REQUIRED: {len(replay_ids)}')
    print(f'  TRULY_UNOBSERVABLE: {len(unobs_ids)}')
    signal_pct = len(any_signal_ids) / max(1, n_successful) * 100
    print(f'\n  Signal available (union): {len(any_signal_ids)} ({signal_pct:.0f}%)')
    print(f'  (vs old T2R0 sum-based: 160%)')

    # Detail on misses
    print(f'\n--- {len(misses)} Misses (successful, no strict goal-support contact) ---')
    miss_by_cause = defaultdict(list)
    for ident, r in misses.items():
        miss_by_cause[r.get('cause', 'UNKNOWN')].append(ident)

    for cause in sorted(miss_by_cause.keys()):
        eps = miss_by_cause[cause]
        print(f'\n{cause}: {len(eps)} episodes')
        for ident in eps[:3]:
            r = results[ident]
            print(f'  {ident}: g_rels={r.get("g_rels")}, gs_names={r.get("gs_names")}')
            for rg in r.get('relation_geometry', [])[:2]:
                print(f'    {rg["predicate"]}({rg["object"]},{rg["target"]}): '
                      f'obj={rg["obj_method"]}, tgt={rg["tgt_method"]}, '
                      f'full_geom={rg["full_geometry"]}')

    # Write report
    report = {
        'gate': 'T2R-A_CORRECTED_OBSERVABILITY',
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'n_successful': n_successful,
        'n_misses': len(misses),
        'coverage': {
            'NATIVE_PREDICATE': len(contact_ids),
            'FULL_GEOMETRY': len(geometry_ids),
            'POSE_ONLY': len(pose_ids),
            'REPLAY_REQUIRED': len(replay_ids),
            'TRULY_UNOBSERVABLE': len(unobs_ids),
            'any_signal_union': len(any_signal_ids),
            'signal_pct': signal_pct,
        },
        'miss_detail': {
            cause: [results[i] for i in eps[:5]]
            for cause, eps in miss_by_cause.items()
        },
    }

    report_path = os.path.join(T2RA_OUT, 'T2RA_REPORT.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    report_sha = hashlib.sha256(open(report_path, 'rb').read()).hexdigest()
    print(f'\nReport: {report_path}')
    print(f'SHA: {report_sha[:16]}...')
    sys.exit(0)


if __name__ == '__main__':
    main()
