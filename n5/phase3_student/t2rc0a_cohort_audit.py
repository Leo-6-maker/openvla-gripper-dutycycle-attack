"""T2R-C0A: Confirmation Cohort Composition Audit.

Read-only. Classifies all 130 cohort identities by actual mechanism:
  supported placement (In/On/Stack)
  articulated unsupported
  multi-predicate placement
  failure-after-grasp
  true pregrasp/no-grasp
  normal safe-release
  unknown/parser gap

Reports per-relation, per-fixture, true pregrasp counts.
Does NOT run Teacher or compute metrics.
"""
import json, os, sys, time, hashlib
from collections import defaultdict
import numpy as np

DIR = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(os.path.dirname(DIR), 'phase2_labels'))
from v22_production_v2 import parse_sidecar, get_object_slices_for_task, _contact_flags

CS200 = '/mnt/sdc/dty_user/openvla_attack_evidence/c2g/c2g_cs200_official_v3_20260716/clean'
COHORT_MANIFEST = '/mnt/sdc/dty_user/openvla_attack_outputs/n5/phase3_student/t2rd_confirmation_cohort/T2RD_CONFIRM_MANIFEST_V1.json'
LABEL_ROOT = '/mnt/sdc/dty_user/openvla_attack_outputs/n5/phase2_labels/g4_label_production'
C0A_OUT = '/mnt/sdc/dty_user/openvla_attack_outputs/n5/phase3_student/t2rc0a_cohort_audit'
os.makedirs(C0A_OUT, exist_ok=True)


def classify_identity(ident):
    """Classify one episode by mechanism, relation type, and evidence availability."""
    suite, task, state = ident.split('/')
    result = {
        'identity': ident,
        'suite': suite,
        'task': task,
        'mechanism': 'UNKNOWN',
        'relations': [],
        'fixture_targets': [],
        'is_successful': False,
        'has_grasp_evidence': False,
        'is_pregrasp': False,
        'has_articulated_keywords': False,
    }

    # Load summary
    sp = os.path.join(CS200, suite, task, state, 'episode_summary.json')
    if os.path.isfile(sp):
        with open(sp) as f:
            result['is_successful'] = json.load(f).get('success', False)

    # Load sidecar for contact analysis
    sidecar_path = os.path.join(CS200, suite, task, state, 'privileged_teacher_sidecar.jsonl')
    if not os.path.isfile(sidecar_path):
        result['mechanism'] = 'SIDECAR_MISSING'
        return result

    parsed = parse_sidecar(sidecar_path)
    steps_data = parsed['steps']
    T = len(steps_data)

    # Parse BDDL
    task_idx = int(task.replace('task_', ''))
    bddl = get_object_slices_for_task(suite, task_idx)
    if bddl is None:
        result['mechanism'] = 'BDDL_MISSING'
        return result

    task_role = bddl['task_role']
    object_slices = bddl['object_slices']
    manip = task_role.get('manipulated_objects', [])
    g_rels = task_role.get('goal_relations', [])
    gs_names = task_role.get('goal_support_names', [])

    result['relations'] = [(r[0], r[1], r[2]) for r in g_rels]

    # Check for articulated keywords in instruction
    instruction = steps_data[0].get('task_language', '') if steps_data else ''
    articulated_kw = ['open', 'close', 'turn on', 'turn off', 'push', 'press', 'pull', 'rotate']
    result['has_articulated_keywords'] = any(kw in instruction.lower() for kw in articulated_kw)

    # Classify relation types present
    relation_types = set(r[0] for r in g_rels)
    has_in = 'In' in relation_types
    has_on = 'On' in relation_types
    has_stack = 'Stack' in relation_types

    # Check if any targets are fixtures (not in object_slices)
    for pred, obj, tgt in g_rels:
        tgt_base = tgt
        for suffix in ['_contain_region', '_init_region', '_cook_region',
                       '_heating_region', '_top_region', '_front_region',
                       '_back_contain_region']:
            tgt_base = tgt_base.replace(suffix, '')
        if tgt_base not in object_slices:
            result['fixture_targets'].append({'target': tgt, 'predicate': pred,
                                              'base_not_in_slices': tgt_base})

    # Check grasp/contact evidence
    contact_steps = 0
    grasp_detected = False
    for t in range(T):
        pairs = steps_data[t].get('mujoco_contact_pairs', [])
        for pair in pairs:
            pair_strs = [str(item) for item in pair]
            for mo in manip:
                mo_clean = mo.replace('_contain_region', '').replace('_init_region', '')
                for ps in pair_strs:
                    if mo_clean in ps.replace('_contain_region', '').replace('_init_region', ''):
                        contact_steps += 1
                        if contact_steps >= 5:
                            grasp_detected = True
                        break
                if grasp_detected: break
            if grasp_detected: break

    result['has_grasp_evidence'] = grasp_detected
    result['contact_steps_total'] = contact_steps

    # Classify mechanism
    if not g_rels:
        if result['has_articulated_keywords']:
            result['mechanism'] = 'ARTICULATED_UNSUPPORTED'
        elif 'task_07' in ident:
            result['mechanism'] = 'PARSER_GAP_NO_RELATION'
        else:
            result['mechanism'] = 'NO_GOAL_RELATION'
    elif result['has_articulated_keywords'] and not grasp_detected:
        result['mechanism'] = 'ARTICULATED_UNSUPPORTED'
    elif has_in or has_on or has_stack:
        if result['fixture_targets']:
            result['mechanism'] = 'SUPPORTED_PLACEMENT_FIXTURE_TARGET'
        else:
            result['mechanism'] = 'SUPPORTED_PLACEMENT_DIRECT_TARGET'
    else:
        result['mechanism'] = 'OTHER_RELATION'

    # Pregrasp: no grasp AND no contact AND first quarter of episode
    if not grasp_detected and contact_steps == 0:
        # True pregrasp: episode failed early
        if not result['is_successful']:
            result['is_pregrasp'] = True
            result['mechanism'] = 'PREGRASP_NO_GRASP'

    return result


def main():
    print('=' * 60)
    print('T2R-C0A: Cohort Composition Audit')
    print('=' * 60)

    with open(COHORT_MANIFEST) as f:
        cohort = json.load(f)
    print(f'Cohort SHA: {cohort["self_sha256"][:16]}...')
    print(f'Cohort size: {len(cohort["identities"])}')

    results = {}
    for i, ident in enumerate(cohort['identities']):
        try:
            results[ident] = classify_identity(ident)
        except Exception as e:
            results[ident] = {'identity': ident, 'error': str(e),
                             'mechanism': 'CLASSIFICATION_ERROR'}
        if (i + 1) % 30 == 0:
            print(f'  {i+1}/{len(cohort["identities"])} done')

    # Aggregate
    mech_counts = defaultdict(list)
    relation_counts = {'In': 0, 'On': 0, 'Stack': 0}
    fixture_counts = defaultdict(int)
    pregrasp_count = 0
    successful_supported = 0

    for ident, r in results.items():
        mech_counts[r.get('mechanism', 'UNKNOWN')].append(ident)
        for rel_type, _, _ in r.get('relations', []):
            if rel_type in relation_counts:
                relation_counts[rel_type] += 1
        for ft in r.get('fixture_targets', []):
            fixture_counts[ft['target']] += 1
        if r.get('is_pregrasp'):
            pregrasp_count += 1
        if r.get('is_successful') and 'SUPPORTED_PLACEMENT' in r.get('mechanism', ''):
            successful_supported += 1

    print(f'\n--- Mechanism Breakdown ---')
    for mech in sorted(mech_counts.keys(), key=lambda m: -len(mech_counts[m])):
        eps = mech_counts[mech]
        print(f'  {mech}: {len(eps)} episodes')
        if len(eps) <= 5:
            for ep in eps:
                print(f'    {ep}')

    print(f'\n--- Relation Coverage ---')
    for rel, count in sorted(relation_counts.items()):
        print(f'  {rel}: {count} episodes')

    print(f'\n--- Fixture Target Coverage ---')
    for target, count in sorted(fixture_counts.items(), key=lambda x: -x[1]):
        print(f'  {target}: {count}')

    print(f'\n--- Summary ---')
    n_cohort = len(cohort['identities'])
    print(f'Total cohort: {n_cohort}')
    print(f'Successful supported placement: {successful_supported}')
    print(f'True pregrasp: {pregrasp_count}')
    print(f'Successful (all): {sum(1 for _, r in results.items() if r.get("is_successful"))}')
    print(f'Failed (all): {sum(1 for _, r in results.items() if not r.get("is_successful"))}')

    # Check task_07
    task_07 = [ident for ident in cohort['identities'] if 'task_07' in ident]
    print(f'\ntask_07 in cohort: {len(task_07)}')
    for ep in task_07:
        r = results.get(ep, {})
        print(f'  {ep}: mech={r.get("mechanism","?")}, '
              f'relations={r.get("relations","?")}, '
              f'articulated={r.get("has_articulated_keywords","?")}')

    # Write report
    report = {
        'gate': 'T2R-C0A_COHORT_COMPOSITION_AUDIT',
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'cohort_sha': cohort['self_sha256'],
        'summary': {
            'total': n_cohort,
            'successful_supported_placement': successful_supported,
            'true_pregrasp': pregrasp_count,
            'per_mechanism': {m: len(e) for m, e in mech_counts.items()},
            'per_relation': dict(relation_counts),
            'per_fixture': dict(fixture_counts),
            'task_07_disposition': [
                {'identity': ep, 'mechanism': results.get(ep, {}).get('mechanism', '?')}
                for ep in task_07
            ],
        },
        'detail': {
            ident: {
                'mechanism': r.get('mechanism'),
                'relations': r.get('relations'),
                'fixture_targets': r.get('fixture_targets'),
                'is_successful': r.get('is_successful'),
                'is_pregrasp': r.get('is_pregrasp'),
                'has_grasp': r.get('has_grasp_evidence'),
            }
            for ident, r in sorted(results.items())
        },
    }

    report_path = os.path.join(C0A_OUT, 'C0A_AUDIT.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    report_sha = hashlib.sha256(open(report_path, 'rb').read()).hexdigest()
    print(f'\nReport: {report_path}')
    print(f'SHA: {report_sha[:16]}...')
    sys.exit(0)


if __name__ == '__main__':
    main()
