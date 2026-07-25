"""Phase 1 Teacher Audit: Extract pre-gate Teacher factors from factorized_teacher_v1.jsonl.
Answers H1/H2/H3/H4 for Goal suite zero-coverage root cause.

Runs on CPU only. Reads factorized teacher labels, does NOT touch formal DB or GPU.

Teacher critical_raw (pre-gate) is constructed from physical factors:
  grasp_established, manipulation_active, gripper_contact_score, object_eef_comotion_score,
  lift_score, target_progress, release_or_instability

This is compared against:
  - candidate_close (the cc gate)
  - strict_k10_feasible (the K10 feasibility gate)
  - The implicit training label: K10_feasible AND candidate_close (if that's what was used)
"""
import json, os, sys, argparse
from collections import defaultdict
import numpy as np

LABEL_ROOT = '/mnt/sdc/dty_user/openvla_attack_evidence/c2g/c2g_cs200_official_v3_20260716/ops/FACTORIZED_TEACHER_STATES_35_49_20260725/labels'

def load_teacher_labels(suite, max_episodes=None):
    """Load all factorized_teacher_v1.jsonl files for a suite."""
    suite_dir = os.path.join(LABEL_ROOT, suite)
    if not os.path.isdir(suite_dir):
        return []

    episodes = []
    for task in sorted(os.listdir(suite_dir)):
        task_dir = os.path.join(suite_dir, task)
        if not os.path.isdir(task_dir):
            continue
        for state in sorted(os.listdir(task_dir)):
            state_dir = os.path.join(task_dir, state)
            if not os.path.isdir(state_dir):
                continue
            label_file = os.path.join(state_dir, 'factorized_teacher_v1.jsonl')
            if not os.path.isfile(label_file):
                continue
            with open(label_file) as f:
                steps = [json.loads(l) for l in f.read().splitlines() if l.strip()]
            if steps:
                episodes.append({
                    'suite': suite, 'task': task, 'state': state,
                    'canonical_key': steps[0].get('canonical_parent_key', f'{suite}/{task}/{state}'),
                    'steps': steps,
                })
            if max_episodes and len(episodes) >= max_episodes:
                break
        if max_episodes and len(episodes) >= max_episodes:
            break
    return episodes

def compute_teacher_critical_raw(step):
    """Reconstruct pre-gate teacher critical signal from physical factors.
    This is intentionally SEPARATE from candidate_close and k10_feasible.
    """
    grasp = step.get('grasp_established', False)
    manipulation = step.get('manipulation_active', False)
    contact = step.get('gripper_contact_score', 0) > 0.1
    comotion = step.get('object_eef_comotion_score', 0) > 0.1
    lift = step.get('lift_score', 0) > 0.1
    release = step.get('release_or_instability', False)
    target_done = step.get('target_progress', 0) >= 0.95
    target_relevant = step.get('target_relevant', False)

    # Physical engagement: any evidence of gripper-object interaction
    physically_engaged = grasp or manipulation or contact or comotion

    # Critical = physically engaged AND not releasing AND task not done
    critical_raw = physically_engaged and not release and not target_done

    # Release-safe = task done OR releasing
    release_safe = release or target_done

    # K10 window feasibility: horizon remaining AND known mask
    k10_feasible = step.get('strict_k10_feasible', False) and step.get('strict_k10_known_mask', False)

    return {
        'critical_raw': critical_raw,
        'physically_engaged': physically_engaged,
        'release_safe': release_safe,
        'grasp_established': grasp,
        'manipulation_active': manipulation,
        'contact': contact,
        'comotion': comotion,
        'lift': lift,
        'release_or_instability': release,
        'target_done': target_done,
        'target_relevant': target_relevant,
        'target_progress': step.get('target_progress', 0),
        'k10_feasible': k10_feasible,
        'candidate_close': step.get('candidate_close', False),
        'action_intent': step.get('action_intent', 'UNKNOWN'),
        'event_phase': step.get('event_phase', 'IDLE'),
        'mechanism_type': step.get('mechanism_type', 'unknown'),
    }

def audit_suite(suite):
    """Full audit of one suite's Teacher labels."""
    episodes = load_teacher_labels(suite)
    if not episodes:
        return {'error': f'No episodes found for {suite}', 'n_episodes': 0}

    total_steps = 0
    critical_steps = 0
    critical_cc_true = 0
    critical_cc_false = 0
    k10_feasible_steps = 0
    k10_feasible_cc_true = 0
    k10_feasible_cc_false = 0
    cc_true_steps = 0
    cc_false_steps = 0

    # Attackable = K10_feasible (the raw K10 label, before any cc AND)
    # Final label (if training uses cc AND K10) = K10_feasible AND candidate_close
    attackable_raw = 0  # K10_feasible
    attackable_final = 0  # K10_feasible AND candidate_close

    ep_stats = []
    longest_critical_streak = 0
    longest_k10_streak = 0

    # Per-mechanism breakdown
    mechanism_critical = defaultdict(int)
    mechanism_total = defaultdict(int)

    # Action intent breakdown
    action_intent_critical = defaultdict(int)

    for ep in episodes:
        ep_critical = 0
        ep_attackable_final = 0
        ep_cc_true = 0
        current_streak = 0
        current_k10_streak = 0
        max_streak = 0
        max_k10_streak = 0

        for step in ep['steps']:
            total_steps += 1
            t = compute_teacher_critical_raw(step)
            cc = t['candidate_close']
            k10 = t['k10_feasible']
            crit = t['critical_raw']

            if cc: cc_true_steps += 1
            else: cc_false_steps += 1

            if crit:
                critical_steps += 1
                ep_critical += 1
                if cc: critical_cc_true += 1
                else: critical_cc_false += 1
                current_streak += 1
                max_streak = max(max_streak, current_streak)
                action_intent_critical[t['action_intent']] += 1
            else:
                current_streak = 0

            if k10:
                k10_feasible_steps += 1
                if cc: k10_feasible_cc_true += 1
                else: k10_feasible_cc_false += 1
                current_k10_streak += 1
                max_k10_streak = max(max_k10_streak, current_k10_streak)
            else:
                current_k10_streak = 0

            mechanism_total[t['mechanism_type']] += 1
            if crit:
                mechanism_critical[t['mechanism_type']] += 1

            attackable_raw += 1 if k10 else 0
            attackable_final += 1 if (k10 and cc) else 0

        ep_stats.append({
            'key': ep['canonical_key'],
            'n_steps': len(ep['steps']),
            'n_critical': ep_critical,
            'n_attackable_final': ep_attackable_final,
            'n_cc_true': ep_cc_true,
            'max_critical_streak': max_streak,
            'max_k10_streak': max_k10_streak,
        })
        longest_critical_streak = max(longest_critical_streak, max_streak)
        longest_k10_streak = max(longest_k10_streak, max_k10_streak)

    n_episodes = len(episodes)
    eps_with_critical = sum(1 for e in ep_stats if e['n_critical'] > 0)
    eps_with_attackable = sum(1 for e in ep_stats if e['n_attackable_final'] > 0)
    eps_with_k10 = sum(1 for e in ep_stats if e['max_k10_streak'] > 0)

    # Student-valid: whether the step is valid for Student training
    student_valid_steps = sum(
        1 for ep in episodes for s in ep['steps'] if s.get('student_valid', False)
    )

    return {
        'suite': suite,
        'n_episodes': n_episodes,
        'n_total_steps': total_steps,
        'n_student_valid_steps': student_valid_steps,
        'cc_true_steps': cc_true_steps,
        'cc_false_steps': cc_false_steps,
        'P_cc_true': cc_true_steps / max(1, total_steps),

        # Pre-gate Teacher factors
        'critical_raw_steps': critical_steps,
        'critical_raw_cc_true': critical_cc_true,
        'critical_raw_cc_false': critical_cc_false,
        'P_critical_raw': critical_steps / max(1, total_steps),
        'P_critical_raw_given_cc_true': critical_cc_true / max(1, cc_true_steps),
        'P_critical_raw_given_cc_false': critical_cc_false / max(1, cc_false_steps),
        'eps_with_critical': eps_with_critical,
        'eps_critical_rate': eps_with_critical / max(1, n_episodes),
        'longest_critical_streak': longest_critical_streak,

        # K10 feasibility (pre-cc)
        'k10_feasible_steps': k10_feasible_steps,
        'k10_feasible_cc_true': k10_feasible_cc_true,
        'k10_feasible_cc_false': k10_feasible_cc_false,
        'P_k10_feasible': k10_feasible_steps / max(1, total_steps),
        'eps_with_k10': eps_with_k10,
        'longest_k10_streak': longest_k10_streak,

        # Attackable labels (with and without cc AND)
        'attackable_raw': attackable_raw,
        'attackable_final': attackable_final,
        'P_attackable_raw': attackable_raw / max(1, total_steps),
        'P_attackable_final': attackable_final / max(1, total_steps),
        'eps_with_attackable_final': eps_with_attackable,

        # Mechanism breakdown
        'mechanism_critical': dict(mechanism_critical),
        'mechanism_total': dict(mechanism_total),
        'action_intent_critical': dict(action_intent_critical),

        # Per-episode stats
        'episode_stats': ep_stats,
    }

def determine_hypothesis(audit):
    """Determine which hypothesis is supported by the audit data."""
    suite = audit['suite']
    has_critical = audit['critical_raw_steps'] > 0
    has_critical_cc_false = audit['critical_raw_cc_false'] > 0
    has_k10_cc_false = audit['k10_feasible_cc_false'] > 0
    attackable_rate = audit['P_attackable_final']
    critical_rate = audit['P_critical_raw']

    hypotheses = []

    if not has_critical:
        hypotheses.append({
            'hypothesis': 'H1_LIKELY',
            'verdict': 'Teacher has ZERO critical_raw windows on this suite',
            'implication': 'Suite may be legitimate negative control OR Teacher rules miss physical engagement',
            'confidence': 'HIGH' if audit['eps_with_critical'] == 0 else 'MEDIUM',
        })
    else:
        if has_critical_cc_false:
            frac_lost = audit['critical_raw_cc_false'] / max(1, audit['critical_raw_steps'])
            hypotheses.append({
                'hypothesis': 'H2_CONFIRMED',
                'verdict': f'Teacher HAS critical windows but {frac_lost:.1%} are on cc=False steps',
                'implication': 'Label contract MUST be fixed: critical should NOT depend on cc',
                'confidence': 'HIGH' if frac_lost > 0.5 else 'MEDIUM',
            })

        if has_k10_cc_false:
            hypotheses.append({
                'hypothesis': 'H2_K10_VARIANT',
                'verdict': f'K10 feasible steps exist on cc=False ({audit["k10_feasible_cc_false"]} steps)',
                'implication': 'K10 windows are lost to cc gate',
                'confidence': 'MEDIUM',
            })

        if attackable_rate == 0 and critical_rate > 0:
            hypotheses.append({
                'hypothesis': 'H3_LIKELY',
                'verdict': 'Teacher has critical windows but final attackable label is zero',
                'implication': 'Even after fixing labels, Student may need retraining for this suite',
                'confidence': 'MEDIUM',
            })

    # H4 check: is Teacher missing physical engagement that should be detected?
    if audit['mechanism_critical'].get('grasp_based', 0) == 0 and \
       audit['mechanism_critical'].get('contact_based', 0) == 0:
        hypotheses.append({
            'hypothesis': 'H4_CHECK_NEEDED',
            'verdict': 'Teacher mechanism breakdown shows no grasp/contact critical steps',
            'implication': 'Teacher physics rules may not detect goal-specific interaction patterns',
            'confidence': 'LOW',
        })

    return hypotheses

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--suites', nargs='*', default=['libero_goal', 'libero_10', 'libero_object', 'libero_spatial'])
    parser.add_argument('--output', default='reports/PHASE1_TEACHER_AUDIT_V1.json')
    args = parser.parse_args()

    results = {}
    for suite in args.suites:
        print(f'Auditing {suite}...')
        audit = audit_suite(suite)
        audit['hypotheses'] = determine_hypothesis(audit)
        results[suite] = audit

        print(f'  Episodes: {audit["n_episodes"]}')
        print(f'  Steps: {audit["n_total_steps"]}')
        print(f'  P(cc=True): {audit["P_cc_true"]:.4f}')
        print(f'  P(critical_raw): {audit["P_critical_raw"]:.4f}')
        print(f'  Critical steps: {audit["critical_raw_steps"]} (cc_true={audit["critical_raw_cc_true"]}, cc_false={audit["critical_raw_cc_false"]})')
        print(f'  P(critical_raw | cc=True): {audit["P_critical_raw_given_cc_true"]:.4f}')
        print(f'  P(critical_raw | cc=False): {audit["P_critical_raw_given_cc_false"]:.4f}')
        print(f'  Episodes with critical: {audit["eps_with_critical"]}/{audit["n_episodes"]}')
        print(f'  Longest critical streak: {audit["longest_critical_streak"]}')
        print(f'  K10 feasible steps: {audit["k10_feasible_steps"]} (cc_true={audit["k10_feasible_cc_true"]}, cc_false={audit["k10_feasible_cc_false"]})')
        print(f'  P(attackable_final): {audit["P_attackable_final"]:.4f}')
        print(f'  Action intent in critical: {audit["action_intent_critical"]}')
        print(f'  Mechanism critical: {audit["mechanism_critical"]}')
        for h in audit['hypotheses']:
            print(f'  *** {h["hypothesis"]}: {h["verdict"]}')

    # Cross-suite comparison
    print('\n=== CROSS-SUITE COMPARISON ===')
    header = f'{"Suite":20s} {"P(cc)":>8s} {"P(crit_raw)":>12s} {"Crit_ccT":>10s} {"Crit_ccF":>10s} {"Eps_crit":>10s} {"MaxStreak":>10s} {"P(atk_fin)":>11s}'
    print(header)
    print('-' * len(header))
    for suite, audit in results.items():
        if 'error' in audit:
            continue
        print(f'{suite:20s} {audit["P_cc_true"]:8.4f} {audit["P_critical_raw"]:12.4f} '
              f'{audit["critical_raw_cc_true"]:10d} {audit["critical_raw_cc_false"]:10d} '
              f'{audit["eps_with_critical"]}/{audit["n_episodes"]:5d} {audit["longest_critical_streak"]:10d} '
              f'{audit["P_attackable_final"]:11.4f}')

    # Goal-specific deep dive
    if 'libero_goal' in results:
        g = results['libero_goal']
        print('\n=== GOAL DEEP DIVE ===')
        print(f'  P(critical_raw): {g["P_critical_raw"]:.4f}')
        print(f'  Critical on cc=False: {g["critical_raw_cc_false"]} steps')
        print(f'  Critical on cc=True: {g["critical_raw_cc_true"]} steps')
        print(f'  K10 feasible on cc=False: {g["k10_feasible_cc_false"]} steps')
        if g['critical_raw_steps'] > 0:
            print(f'  Fraction of critical lost to cc gate: {g["critical_raw_cc_false"]/max(1,g["critical_raw_steps"]):.1%}')
        else:
            print(f'  ZERO critical_raw steps — checking if Teacher missed physical engagement...')
            # Check grasp/manipulation/contact
            for ep in g['episode_stats']:
                if ep['n_critical'] > 0:
                    print(f'    {ep["key"]}: {ep["n_critical"]} critical, streak={ep["max_critical_streak"]}')

        # H2 specific: how many attackable steps are lost
        lost_to_cc = g['k10_feasible_cc_false']
        total_k10 = g['k10_feasible_steps']
        if total_k10 > 0:
            print(f'  K10 steps lost to cc gate: {lost_to_cc}/{total_k10} ({lost_to_cc/max(1,total_k10):.1%})')

        # Decision
        print('\n=== GOAL VERDICT ===')
        for h in g['hypotheses']:
            print(f'  [{h["confidence"]}] {h["hypothesis"]}: {h["verdict"]}')
            print(f'         → {h["implication"]}')

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump({'analysis': 'PHASE1_TEACHER_AUDIT_V1', 'results': results}, f, indent=2, default=str)
    print(f'\nSaved to {args.output}')

if __name__ == '__main__':
    main()
