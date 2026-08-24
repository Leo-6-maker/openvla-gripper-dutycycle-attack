"""Safe-Release Teacher Root-Cause Audit.

Investigates:
  1. Per-suite: count successful & supported pick-place episodes.
  2. Count episodes with placement, support transition, gripper opening events.
  3. Generate candidate list for manual review (30-50 episodes).
  4. Check safe_release=1 => k10_feasible=0.
  5. Focus explanation: why is libero_object 0/200?

Reads CS200 privileged_teacher_sidecar.jsonl + episode_summary.json + label_contract_v2.jsonl.
No training required.
"""
import json, os, sys, hashlib, time
from collections import defaultdict
import numpy as np

DIR = os.path.dirname(__file__)
sys.path.insert(0, DIR)

CS200_ROOT = '/mnt/sdc/dty_user/openvla_attack_evidence/c2g/c2g_cs200_official_v3_20260716/clean'
LABEL_ROOT = '/mnt/sdc/dty_user/openvla_attack_outputs/n5/phase2_labels/g4_label_production'
IDENTITY_MANIFEST = '/mnt/sdc/dty_user/openvla_attack_evidence/c2g/c2g_cs200_official_v3_20260716/ops/FACTORIZED_PHASE_B2_DETERMINISTIC_ALLOCATION_V3_804113EE_20260723/checkpoint_training_identity_manifest.json'
G6_SEAL_PATH = '/mnt/sdc/dty_user/openvla_attack_outputs/n5/phase3_student/g6_training_seal/G6_SEAL_V2.json'
OUT_DIR = '/mnt/sdc/dty_user/openvla_attack_outputs/n5/phase3_student/safe_release_audit'

os.makedirs(OUT_DIR, exist_ok=True)


def load_sidecar(ident):
    """Load privileged teacher sidecar for one episode."""
    suite, task, state = ident.split('/')
    path = os.path.join(CS200_ROOT, suite, task, state, 'privileged_teacher_sidecar.jsonl')
    if not os.path.isfile(path):
        return None
    steps = []
    with open(path) as f:
        for line in f:
            if line.strip():
                steps.append(json.loads(line))
    return steps


def load_summary(ident):
    """Load episode summary."""
    suite, task, state = ident.split('/')
    path = os.path.join(CS200_ROOT, suite, task, state, 'episode_summary.json')
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        return json.load(f)


def load_labels(ident):
    """Load Label V2 steps."""
    suite, task, state = ident.split('/')
    path = os.path.join(LABEL_ROOT, suite, task, state, 'label_contract_v2.jsonl')
    if not os.path.isfile(path):
        return None
    steps = []
    with open(path) as f:
        for line in f:
            if line.strip():
                steps.append(json.loads(line))
    return steps


def compute_placement_events(sidecar_steps):
    """Detect placement-like events from sidecar: object Z stabilizes near support surface."""
    # Heuristic: detect when manipulated objects stop descending
    events = []
    for t, step in enumerate(sidecar_steps):
        obj_state = step.get('object_state', [])
        contact_count = step.get('contact_count', 0)
        qpos = step.get('robot0_gripper_qpos', [0, 0])
        gripper_open = abs(qpos[0]) + abs(qpos[1]) if len(qpos) >= 2 else 0

        events.append({
            'step': t,
            'object_state_len': len(obj_state),
            'contact_count': contact_count,
            'gripper_open_proxy': gripper_open,
        })
    return events


def compute_gripper_opening_events(sidecar_steps):
    """Detect gripper opening transitions."""
    openings = []
    prev_qpos = None
    for t, step in enumerate(sidecar_steps):
        qpos = step.get('robot0_gripper_qpos', [0, 0])
        if len(qpos) >= 2:
            proxy = abs(qpos[0]) + abs(qpos[1])
            if prev_qpos is not None:
                prev_proxy = abs(prev_qpos[0]) + abs(prev_qpos[1])
                delta = proxy - prev_proxy
                openings.append({'step': t, 'proxy': proxy, 'delta': float(delta),
                                 'opening': bool(delta > 0.02)})
            else:
                openings.append({'step': t, 'proxy': proxy, 'delta': 0, 'opening': False})
        prev_qpos = qpos
    return openings


def audit_all_episodes():
    """Full audit of all 800 training episodes."""
    # Load identities
    with open(G6_SEAL_PATH) as f:
        seal = json.load(f)

    all_train = set()
    for split_key in ['train_identities', 'val_identities', 'cal_identities']:
        all_train.update(seal['split'].get(split_key, []))

    identities = sorted(all_train)
    print(f'Total identities: {len(identities)}')

    results = {
        'per_suite': defaultdict(lambda: {
            'total': 0, 'successful': 0, 'any_gripper_opening': 0, 'any_contact': 0,
            'safe_release_pos': 0, 'safe_release_episodes': set(),
            'k10_feasible_when_safe_release': 0,
        }),
        'safe_release_episodes': [],  # episodes with at least 1 sr pos
        'candidates_for_manual_review': [],
        'libero_object_analysis': [],
    }

    for ident in identities:
        suite = ident.split('/')[0]
        results['per_suite'][suite]['total'] += 1

        summary = load_summary(ident)
        sidecar = load_sidecar(ident)
        labels = load_labels(ident)

        is_successful = summary.get('success', False) if summary else False

        # Gripper opening detection
        if sidecar:
            openings = compute_gripper_opening_events(sidecar)
            has_opening = any(o['opening'] for o in openings)
            max_contact = max((s.get('contact_count', 0) for s in sidecar), default=0)
            has_contact = max_contact > 0
            final_qpos = sidecar[-1].get('robot0_gripper_qpos', [0, 0]) if sidecar else [0, 0]
            final_open_proxy = abs(final_qpos[0]) + abs(final_qpos[1]) if len(final_qpos) >= 2 else 0
        else:
            has_opening = False
            has_contact = False
            final_open_proxy = 0

        if is_successful:
            results['per_suite'][suite]['successful'] += 1
        if has_opening:
            results['per_suite'][suite]['any_gripper_opening'] += 1
        if has_contact:
            results['per_suite'][suite]['any_contact'] += 1

        # Safe-release label check
        if labels:
            sr_steps = [s for s in labels if s.get('safe_release', {}).get('value')]
            sr_pos_steps = [s for s in labels
                           if s.get('safe_release', {}).get('valid_mask')
                           and s.get('safe_release', {}).get('value')]
            k10_steps = {s['step']: s.get('k10_feasible', {})
                        for s in labels if 'step' in s}

            if sr_pos_steps:
                results['per_suite'][suite]['safe_release_pos'] += len(sr_pos_steps)
                results['per_suite'][suite]['safe_release_episodes'].add(ident)
                results['safe_release_episodes'].append({
                    'identity': ident,
                    'n_sr_pos': len(sr_pos_steps),
                    'sr_steps': [(s['step'], s.get('safe_release', {}).get('reason'))
                                 for s in sr_pos_steps],
                })
                # Check k10 when safe_release
                for s in sr_pos_steps:
                    ks = k10_steps.get(s['step'], {})
                    if ks.get('valid_mask') and not ks.get('value'):
                        results['per_suite'][suite]['k10_feasible_when_safe_release'] += 1

            # Manual review candidates: successful episodes with gripper opening but NO safe_release
            if is_successful and has_opening and not sr_pos_steps:
                results['candidates_for_manual_review'].append({
                    'identity': ident,
                    'suite': suite,
                    'successful': True,
                    'has_gripper_opening': True,
                    'has_contact': has_contact,
                    'max_contact': max_contact if sidecar else 0,
                    'final_open_proxy': final_open_proxy,
                    'n_steps': len(sidecar) if sidecar else 0,
                    'priority': 'HIGH' if suite == 'libero_object' else 'MEDIUM',
                })

    # libero_object deep analysis
    obj_episodes = [ident for ident in identities if ident.startswith('libero_object')]
    for ident in obj_episodes:
        summary = load_summary(ident)
        labels = load_labels(ident)
        if summary and labels:
            is_success = summary.get('success', False)
            has_sr = any(s.get('safe_release', {}).get('value')
                        for s in labels
                        if s.get('safe_release', {}).get('valid_mask'))
            placement_labels = [s for s in labels
                              if s.get('safe_release', {}).get('reason', '').startswith('PLACEMENT')]
            results['libero_object_analysis'].append({
                'identity': ident,
                'success': is_success,
                'steps': summary.get('steps', 0),
                'safe_release_pos': bool(has_sr),
                'placement_reason_count': len(placement_labels),
            })

    return results


def main():
    print('=== Safe-Release Teacher Root-Cause Audit ===')
    print(f'CS200: {CS200_ROOT}')
    print(f'Labels: {LABEL_ROOT}')
    print()

    results = audit_all_episodes()

    # Report per-suite
    print('--- Per-Suite Summary ---')
    print(f'{"Suite":<18} {"Total":>6} {"Success":>8} {"Open":>6} {"Contact":>8} {"SR+":>6}')
    print('-' * 60)
    for suite in ['libero_10', 'libero_goal', 'libero_object', 'libero_spatial']:
        s = results['per_suite'][suite]
        n_sr_eps = len(s['safe_release_episodes'])
        print(f'{suite:<18} {s["total"]:>6} {s["successful"]:>8} '
              f'{s["any_gripper_opening"]:>6} {s["any_contact"]:>8} {n_sr_eps:>6}')

    # Safe-release episodes
    print(f'\n--- Safe-Release Positive Episodes ---')
    sr_eps = results['safe_release_episodes']
    print(f'Total episodes with safe_release: {len(sr_eps)}')
    for ep in sorted(sr_eps, key=lambda x: x['identity']):
        print(f'  {ep["identity"]}: {ep["n_sr_pos"]} pos steps')
        for step_num, reason in ep['sr_steps']:
            print(f'    step {step_num}: {reason}')

    # K10 relationship
    print(f'\n--- safe_release => k10_feasible Relationship ---')
    for suite in ['libero_10', 'libero_goal', 'libero_object', 'libero_spatial']:
        s = results['per_suite'][suite]
        print(f'  {suite}: k10=0 when sr=1: {s["k10_feasible_when_safe_release"]}')

    # Manual review candidates
    print(f'\n--- Manual Review Candidates (successful + gripper opening + NO safe_release) ---')
    candidates = sorted(results['candidates_for_manual_review'],
                       key=lambda x: (0 if x['priority'] == 'HIGH' else 1, x['identity']))
    print(f'Total candidates: {len(candidates)}')
    for c in candidates[:50]:
        print(f'  [{c["priority"]}] {c["identity"]}: {c["n_steps"]}s, '
              f'max_contact={c["max_contact"]}, final_open={c["final_open_proxy"]:.4f}')

    # libero_object deep dive
    print(f'\n--- libero_object Deep Analysis ---')
    obj = results['libero_object_analysis']
    n_success = sum(1 for e in obj if e['success'])
    n_placement = sum(1 for e in obj if e['placement_reason_count'] > 0)
    print(f'Total episodes: {len(obj)}')
    print(f'Successful: {n_success}')
    print(f'With placement reasons: {n_placement}')
    print(f'Safe-release positive: {sum(1 for e in obj if e["safe_release_pos"])}')
    print()
    if n_success > 0 and n_placement == 0:
        print('HYPOTHESIS: libero_object tasks may succeed without triggering placement detection.')
        print('  The V22 Teacher placement check requires object_placed AND placement_known.')
        print('  If placement_known is never True (e.g., target has no support surface in BDDL),')
        print('  safe_release can never fire regardless of physical state.')
        print()
        print('RECOMMENDATION: Inspect BDDL files for libero_object tasks — check if target objects')
        print('  have defined support surfaces (support_names). If not, placement detection is')
        print('  structurally impossible, and safe_release gap is a Teacher schema limitation,')
        print('  not a labeling defect.')

    # Write full report
    report = {
        'audit': 'SAFE_RELEASE_TEACHER_ROOT_CAUSE_V1',
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'results': {
            'per_suite': {
                suite: {
                    k: (sorted(list(v)) if isinstance(v, set) else v)
                    for k, v in data.items()
                }
                for suite, data in results['per_suite'].items()
            },
            'safe_release_episodes': results['safe_release_episodes'],
            'candidates_top50': candidates[:50],
            'libero_object_analysis': results['libero_object_analysis'][:50],
        },
    }

    report_path = os.path.join(OUT_DIR, 'SAFE_RELEASE_AUDIT.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    report_sha = hashlib.sha256(open(report_path, 'rb').read()).hexdigest()

    print(f'\nReport: {report_path}')
    print(f'SHA: {report_sha[:16]}...')


if __name__ == '__main__':
    main()
