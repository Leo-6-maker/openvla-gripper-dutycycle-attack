"""Phase 1 Counterfactual Trigger Replay.
Runs 4 scheduler variants on existing steps.jsonl data from completed formal tasks.
No GPU required — pure offline computation on collected per-step records.

Schedulers:
  A: V4 hard gate — cc AND score>=τ, D_PERSIST=6
  B: score-only — score>=τ, D_PERSIST=6 (no cc gate)
  C: env-close gate — env_close AND score>=τ, D_PERSIST=6
  D: soft logit gate — score>=τ, close_mass as soft weight (no hard cc)

Outputs per-task comparison table and per-suite joint distributions.
"""
import json, os, sys, glob, argparse
import numpy as np
from collections import defaultdict

TAU = 0.855
D_PERSIST = 6

def run_scheduler(cal_probs, cc_flags, env_close_flags, close_mass_flags,
                  scheduler='v4_hard', tau=TAU, d_persist=D_PERSIST):
    """Run a scheduler variant over per-step data. Returns emit_step or None."""
    counter = 0
    for t in range(len(cal_probs)):
        score_ok = cal_probs[t] >= tau
        if scheduler == 'v4_hard':
            gate_ok = cc_flags[t]
        elif scheduler == 'score_only':
            gate_ok = True
        elif scheduler == 'env_close':
            gate_ok = env_close_flags[t]
        elif scheduler == 'soft_logit':
            gate_ok = True  # score drives, close_mass used for weight
        else:
            raise ValueError(f'Unknown scheduler: {scheduler}')

        if gate_ok and score_ok:
            counter += 1
        else:
            counter = 0

        if counter >= d_persist:
            return t - d_persist + 1  # first step of persistence window
    return None

def load_steps(steps_path):
    """Load steps.jsonl and extract relevant fields."""
    cal_probs = []
    cc_flags = []
    env_close_flags = []
    close_mass_flags = []
    raw_logits = []
    emit_v4 = None
    policy_steps = []

    with open(steps_path) as f:
        for line in f:
            d = json.loads(line)
            if not d.get('detector_updated'):
                continue

            cal_probs.append(float(d.get('calibrated_prob', 0)))
            cc_flags.append(bool(d.get('candidate_close', False)))
            # clean_env_gripper: env-1=OPENING, env+1=CLOSING
            env_g = float(d.get('clean_env_gripper', 0))
            env_close_flags.append(env_g > 0)
            close_mass_flags.append(0.0)  # not in steps.jsonl; placeholder
            raw_logits.append(float(d.get('raw_logit', 0)))
            policy_steps.append(d.get('policy_step'))

            if d.get('emitted_this_step'):
                emit_v4 = len(cal_probs) - 1

    return {
        'cal_probs': np.array(cal_probs),
        'cc_flags': np.array(cc_flags),
        'env_close_flags': np.array(env_close_flags),
        'close_mass_flags': np.array(close_mass_flags),
        'raw_logits': np.array(raw_logits),
        'policy_steps': policy_steps,
        'emit_v4': emit_v4,
        'n_steps': len(cal_probs),
        'n_cc_true': int(np.sum(cc_flags)),
        'n_env_close': int(np.sum(env_close_flags)),
        'max_cal': float(np.max(cal_probs)) if cal_probs else 0,
    }

def analyze_task(steps_data, task_id, suite):
    """Run all 4 schedulers on one task."""
    results = {'task_id': task_id, 'suite': suite}

    for name, scheduler in [
        ('V4_hard_gate', 'v4_hard'),
        ('score_only', 'score_only'),
        ('env_close_gate', 'env_close'),
        ('soft_logit', 'soft_logit'),
    ]:
        emit = run_scheduler(
            steps_data['cal_probs'].tolist(),
            steps_data['cc_flags'].tolist(),
            steps_data['env_close_flags'].tolist(),
            steps_data['close_mass_flags'].tolist(),
            scheduler=scheduler
        )
        results[f'emit_{name}'] = emit

    results.update({
        'n_policy_steps': steps_data['n_steps'],
        'n_cc_true': steps_data['n_cc_true'],
        'n_env_close': steps_data['n_env_close'],
        'max_cal': steps_data['max_cal'],
        'cc_rate': steps_data['n_cc_true'] / max(1, steps_data['n_steps']),
    })
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--attempts-root', required=True,
                        help='Path to formal_v2/attempts directory')
    parser.add_argument('--output', default='reports/PHASE1_COUNTERFACTUAL_REPLAY.json')
    args = parser.parse_args()

    all_results = []
    suite_summary = defaultdict(lambda: {'n_tasks': 0, 'v4_emit': 0, 'score_emit': 0,
                                          'env_emit': 0, 'soft_emit': 0,
                                          'cc_total': 0, 'env_close_total': 0,
                                          'n_steps_total': 0})

    for cell_dir in sorted(os.listdir(args.attempts_root)):
        cell_path = os.path.join(args.attempts_root, cell_dir)
        if not os.path.isdir(cell_path):
            continue

        suite = cell_dir.replace('formal_', '').rsplit('_task_', 1)[0]
        suite = suite.replace('_', '_')  # preserve libero_10 etc.

        # Find the first successful attempt
        for attempt_dir in sorted(os.listdir(cell_path)):
            attempt_path = os.path.join(cell_path, attempt_dir)
            steps_path = os.path.join(attempt_path, 'CLEAN', 'steps.jsonl')
            if os.path.isfile(steps_path):
                data = load_steps(steps_path)
                result = analyze_task(data, cell_dir, suite)
                all_results.append(result)

                ss = suite_summary[suite]
                ss['n_tasks'] += 1
                if result['emit_V4_hard_gate'] is not None: ss['v4_emit'] += 1
                if result['emit_score_only'] is not None: ss['score_emit'] += 1
                if result['emit_env_close_gate'] is not None: ss['env_emit'] += 1
                if result['emit_soft_logit'] is not None: ss['soft_emit'] += 1
                ss['cc_total'] += result['n_cc_true']
                ss['env_close_total'] += result['n_env_close']
                ss['n_steps_total'] += result['n_policy_steps']
                break  # first successful attempt only

    # Compute per-suite derived metrics
    for suite, ss in suite_summary.items():
        ss['cc_rate'] = ss['cc_total'] / max(1, ss['n_steps_total'])
        ss['env_close_rate'] = ss['env_close_total'] / max(1, ss['n_steps_total'])
        n = ss['n_tasks']
        ss['v4_emit_rate'] = ss['v4_emit'] / n if n else 0
        ss['score_emit_rate'] = ss['score_emit'] / n if n else 0
        ss['env_emit_rate'] = ss['env_emit'] / n if n else 0
        ss['soft_emit_rate'] = ss['soft_emit'] / n if n else 0

    output = {
        'analysis': 'PHASE1_COUNTERFACTUAL_TRIGGER_REPLAY_V1',
        'description': '4 scheduler variants on existing formal task CLEAN steps.jsonl',
        'note': 'close_mass not available in current steps.jsonl; soft_logit=score_only for now',
        'schedulers': {
            'V4_hard_gate': 'cc AND score>=tau, D_PERSIST=6',
            'score_only': 'score>=tau, D_PERSIST=6 (no cc gate)',
            'env_close_gate': 'env_close AND score>=tau, D_PERSIST=6',
            'soft_logit': 'score>=tau, close_mass as soft weight (no hard cc)',
        },
        'per_task': all_results,
        'per_suite': {k: dict(v) for k, v in suite_summary.items()},
        'key_findings': _summarize(all_results, suite_summary),
    }

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(output, f, indent=2)
    print(f'Wrote {len(all_results)} tasks to {args.output}')
    _print_summary(suite_summary)

def _summarize(all_results, suite_summary):
    findings = []
    for suite, ss in suite_summary.items():
        n = ss['n_tasks']
        if n == 0: continue
        findings.append({
            'suite': suite,
            'n_tasks': n,
            'v4_emit_rate': ss['v4_emit_rate'],
            'score_only_emit_rate': ss['score_emit_rate'],
            'env_close_emit_rate': ss['env_emit_rate'],
            'cc_rate': ss['cc_rate'],
            'env_close_rate': ss['env_close_rate'],
            'delta_score_vs_v4': ss['score_emit_rate'] - ss['v4_emit_rate'],
        })
    return findings

def _print_summary(suite_summary):
    print('\n=== Counterfactual Replay Summary ===')
    print(f'{"Suite":20s} {"N":>4s} {"V4_emit":>8s} {"Score_emit":>11s} {"Env_emit":>9s} {"CC_rate":>8s} {"EnvCl_rate":>10s}')
    print('-' * 75)
    for suite in sorted(suite_summary):
        ss = suite_summary[suite]
        n = ss['n_tasks']
        if n == 0: continue
        print(f'{suite:20s} {n:4d} {ss["v4_emit_rate"]:8.2f} {ss["score_emit_rate"]:11.2f} '
              f'{ss["env_emit_rate"]:9.2f} {ss["cc_rate"]:8.3f} {ss["env_close_rate"]:10.3f}')

if __name__ == '__main__':
    main()
