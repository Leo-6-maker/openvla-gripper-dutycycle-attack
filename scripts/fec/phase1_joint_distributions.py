"""Phase 1.2 Joint Distribution Analysis.
Computes per-suite joint distributions from per-step diagnostic data.
Can process both existing steps.jsonl and full diagnostic collector output.
"""
import json, os, sys, argparse
import numpy as np
from collections import defaultdict

def compute_joint_distributions(all_steps, suite_label):
    """Compute all Phase 1.2 joint distributions from per-step records."""
    n = len(all_steps)
    if n == 0:
        return {'error': 'no steps', 'n': 0}

    # Extract fields
    raw_close = np.array([s.get('raw_close', s.get('candidate_close', False)) for s in all_steps], dtype=bool)
    env_close = np.array([s.get('env_close', False) for s in all_steps], dtype=bool)

    # g9d fields (available in full diagnostic; fall back for sparse data)
    has_g9d = all(('g9d_close_mass' in s or 'close_mass' in s) for s in all_steps)

    if has_g9d:
        close_mass = np.array([s.get('g9d_close_mass', s.get('close_mass', 0)) for s in all_steps])
        open_mass = np.array([s.get('g9d_open_mass', s.get('open_mass', 0)) for s in all_steps])
        top1_is_close = np.array([s.get('g9d_top1_is_close', s.get('top1_is_close', False)) for s in all_steps], dtype=bool)
        close_gt_open = close_mass > open_mass
    else:
        close_mass = None
        open_mass = None
        top1_is_close = None
        close_gt_open = None

    # Physical state fields (available in full diagnostic)
    has_physical = 'physical_gripper_q7' in all_steps[0]
    if has_physical:
        q7 = np.array([s.get('physical_gripper_q7', 0) for s in all_steps])
        q8 = np.array([s.get('physical_gripper_q8', 0) for s in all_steps])
        physical_closing = np.diff(np.concatenate([[q7[0]+q8[0]], q7+q8])) < 0
        physical_closing = np.append(physical_closing, False)  # last step
    else:
        physical_closing = None

    # Detector fields
    has_detector = 'v4_calibrated_prob' in all_steps[0]
    if has_detector:
        cal_probs = np.array([s.get('v4_calibrated_prob', s.get('calibrated_prob', 0)) for s in all_steps])
        raw_logits = np.array([s.get('v4_raw_logit', s.get('raw_logit', 0)) for s in all_steps])
    else:
        cal_probs = None
        raw_logits = None

    # Teacher label fields
    has_teacher = 'teacher_critical' in all_steps[0]
    if has_teacher:
        teacher_critical = np.array([s.get('teacher_critical', False) for s in all_steps], dtype=bool)
        teacher_release = np.array([s.get('teacher_release_safe', False) for s in all_steps], dtype=bool)
        teacher_k10 = np.array([s.get('teacher_k10_feasible', False) for s in all_steps], dtype=bool)
    else:
        teacher_critical = None
        teacher_release = None
        teacher_k10 = None

    result = {
        'suite': suite_label,
        'n_steps': n,
        'n_episodes': None,  # filled by caller
    }

    # Basic rates
    result['P_raw_close'] = float(np.mean(raw_close))
    result['P_env_close'] = float(np.mean(env_close))

    if has_g9d:
        result['P_top1_close'] = float(np.mean(top1_is_close))
        result['P_close_mass_gt_open'] = float(np.mean(close_gt_open))

    # XOR analysis
    xor = raw_close != env_close
    result['P_raw_close_XOR_env_close'] = float(np.mean(xor))
    result['n_raw_close_XOR_env_close'] = int(np.sum(xor))

    # Conditional: P(raw_close | env_close) and vice versa
    n_env_close = int(np.sum(env_close))
    n_raw_close = int(np.sum(raw_close))
    result['P_raw_close_given_env_close'] = float(np.sum(raw_close & env_close) / max(1, n_env_close))
    result['P_env_close_given_raw_close'] = float(np.sum(raw_close & env_close) / max(1, n_raw_close))

    # Agreement
    agreement = raw_close == env_close
    result['P_agreement'] = float(np.mean(agreement))

    if physical_closing is not None:
        n_phys = int(np.sum(physical_closing))
        result['P_raw_close_given_physical_closing'] = float(np.sum(raw_close & physical_closing) / max(1, n_phys))
        result['P_env_close_given_physical_closing'] = float(np.sum(env_close & physical_closing) / max(1, n_phys))
        result['P_physical_closing'] = float(np.mean(physical_closing))
        result['n_physical_closing'] = n_phys

    if teacher_critical is not None:
        n_tc = int(np.sum(teacher_critical))
        result['n_teacher_critical'] = n_tc
        result['P_teacher_critical'] = float(np.mean(teacher_critical))
        result['P_teacher_critical_given_raw_close'] = float(np.sum(teacher_critical & raw_close) / max(1, n_raw_close))
        result['P_teacher_critical_given_not_raw_close'] = float(np.sum(teacher_critical & ~raw_close) / max(1, n - n_raw_close))
        result['P_raw_close_given_teacher_critical'] = float(np.sum(teacher_critical & raw_close) / max(1, n_tc))

    if cal_probs is not None:
        result['max_cal_prob'] = float(np.max(cal_probs))
        result['mean_cal_prob'] = float(np.mean(cal_probs))
        result['median_cal_prob'] = float(np.median(cal_probs))
        result['P_cal_above_tau'] = float(np.mean(cal_probs >= 0.855))
        result['P_cal_above_05'] = float(np.mean(cal_probs >= 0.5))
        result['mean_cal_when_cc_true'] = float(np.mean(cal_probs[raw_close])) if n_raw_close > 0 else 0.0
        result['mean_cal_when_cc_false'] = float(np.mean(cal_probs[~raw_close])) if n - n_raw_close > 0 else 0.0

    if close_mass is not None and cal_probs is not None:
        result['corr_close_mass_cal'] = float(np.corrcoef(close_mass, cal_probs)[0, 1]) if n > 1 else 0

    return result

def load_steps_jsonl(path):
    """Load steps from steps.jsonl (sparse format, from formal runner)."""
    steps = []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            if not d.get('detector_updated'):
                continue
            env_g = float(d.get('clean_env_gripper', 0))
            steps.append({
                'raw_close': bool(d.get('candidate_close', False)),
                'env_close': bool(env_g > 0),
                'v4_calibrated_prob': float(d.get('calibrated_prob', 0)),
                'v4_raw_logit': float(d.get('raw_logit', 0)),
            })
    return steps

def load_diagnostic_json(path):
    """Load steps from full diagnostic collector output."""
    with open(path) as f:
        data = json.load(f)
    all_steps = []
    for ep in data.get('episodes', []):
        all_steps.extend(ep['steps'])
    return all_steps

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', required=True,
                        help='Directory containing steps.jsonl files or diagnostic JSON files')
    parser.add_argument('--input-format', choices=['steps_jsonl', 'diagnostic_json'], default='steps_jsonl')
    parser.add_argument('--output', default='reports/PHASE1_JOINT_DISTRIBUTIONS.json')
    args = parser.parse_args()

    suite_steps = defaultdict(list)

    if args.input_format == 'steps_jsonl':
        import glob as g
        pattern = os.path.join(args.input_dir, '*', '*', 'CLEAN', 'steps.jsonl')
        for path in g.glob(pattern):
            suite = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(path))))
            suite = suite.replace('formal_', '').split('_task_')[0].replace('_', '_')
            steps = load_steps_jsonl(path)
            suite_steps[suite].extend(steps)
    else:
        for fname in os.listdir(args.input_dir):
            if fname.endswith('.json'):
                path = os.path.join(args.input_dir, fname)
                steps = load_diagnostic_json(path)
                # suite from first step
                suite = steps[0].get('suite', 'unknown') if steps else 'unknown'
                suite_steps[suite].extend(steps)

    results = {}
    for suite, steps in sorted(suite_steps.items()):
        print(f'{suite}: {len(steps)} steps')
        dist = compute_joint_distributions(steps, suite)
        dist['n_steps'] = len(steps)
        results[suite] = dist

    # Overall
    all_steps = []
    for steps in suite_steps.values():
        all_steps.extend(steps)
    results['ALL'] = compute_joint_distributions(all_steps, 'ALL')
    results['ALL']['n_steps'] = len(all_steps)

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump({
            'analysis': 'PHASE1_JOINT_DISTRIBUTIONS_V1',
            'input_format': args.input_format,
            'results': results,
        }, f, indent=2)

    # Print summary table
    print('\n=== Joint Distribution Summary ===')
    metrics = ['P_raw_close', 'P_env_close', 'P_raw_close_XOR_env_close',
               'P_agreement', 'max_cal_prob', 'P_cal_above_tau']
    header = f'{"Suite":20s}'
    for m in metrics:
        header += f' {m:>25s}'
    print(header)
    print('-' * (20 + 26 * len(metrics)))
    for suite, r in results.items():
        row = f'{suite:20s}'
        for m in metrics:
            v = r.get(m, 'N/A')
            if isinstance(v, float):
                row += f' {v:25.4f}'
            else:
                row += f' {str(v):>25s}'
        print(row)

if __name__ == '__main__':
    main()
