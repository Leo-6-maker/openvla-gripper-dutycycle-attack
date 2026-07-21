#!/usr/bin/env python3
"""V2 lexicographic candidate selection. Strict priority order, no manual override."""
import argparse, json, sys
from pathlib import Path
from statistics import mean
import numpy as np

PRIORITY_ORDER = [
    ('safety', ['unsupported_route_emit_rate', 'background_false_emit_rate', 'release_overlap_emit_rate']),
    ('release_auprc', ['release_auprc']),
    ('short_event_auprc', ['release_short_auprc']),
    ('first_later_gap', ['first_later_auprc_gap']),
    ('parameter_count', ['parameter_count']),
]

SAFETY_LIMITS = {
    'unsupported_route_emit_rate': {'max': 0.0, 'catastrophic': 'any_nonzero'},
    'background_false_emit_rate': {'max': 0.10, 'catastrophic': 0.30},
    'release_overlap_emit_rate': {'max': 0.05, 'catastrophic': 0.25},
}


def bootstrap_paired_diff(a_scores, b_scores, n_resamples=10000, seed=42):
    """Identity-paired bootstrap CI for (a - b) difference."""
    rng = np.random.RandomState(seed)
    diffs = []
    n = len(a_scores)
    for _ in range(n_resamples):
        idx = rng.randint(0, n, n)
        diffs.append(mean(a_scores[idx]) - mean(b_scores[idx]))
    diffs = np.array(diffs)
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def check_safety(candidate_metrics, candidate_name):
    """Priority 1: hard elimination on safety metrics. Uses worst-seed value."""
    failures = []
    for metric, limits in SAFETY_LIMITS.items():
        val = candidate_metrics.get(metric)
        if val is None:
            failures.append(f'{metric}: missing')
        elif limits.get('catastrophic') == 'any_nonzero' and val > 0:
            failures.append(f'{metric}: {val} > 0 (catastrophic)')
        elif val > limits.get('max', 999):
            failures.append(f'{metric}: {val} > {limits["max"]}')
    return len(failures) == 0, failures


def compare_on_metric(candidates, metric_key, higher_is_better=True):
    """Compare candidates on a metric with bootstrap CIs."""
    best = None
    best_name = None
    for name, metrics in candidates.items():
        val = metrics.get(metric_key)
        if val is None:
            continue
        if best is None or (higher_is_better and val > best) or (not higher_is_better and val < best):
            best = val
            best_name = name
    return best_name


def select_candidate(candidates_metrics, lr_baseline=None):
    """Run lexicographic selection. Returns (selected, trace, eliminated)."""
    remaining = dict(candidates_metrics)
    eliminated = {}
    trace = []

    for priority, metric_keys in PRIORITY_ORDER:
        if len(remaining) <= 1:
            break

        trace.append({'priority': priority, 'metrics': metric_keys, 'remaining': list(remaining.keys())})

        if priority == 'safety':
            new_remaining = {}
            for name, metrics in remaining.items():
                passed, failures = check_safety(metrics, name)
                if passed:
                    new_remaining[name] = metrics
                    trace.append({'candidate': name, 'safety': 'PASS'})
                else:
                    eliminated[name] = {'reason': 'safety', 'failures': failures}
                    trace.append({'candidate': name, 'safety': 'ELIMINATED', 'failures': failures})
            remaining = new_remaining
            if not remaining:
                trace.append({'result': 'ALL_ELIMINATED_BY_SAFETY'})
                return None, trace, eliminated
        else:
            best = compare_on_metric(remaining, metric_keys[0], higher_is_better=(priority != 'first_later_gap'))
            if best:
                new_remaining = {best: remaining[best]}
                for name in list(remaining.keys()):
                    if name != best:
                        eliminated[name] = {'reason': f'{priority}: {metric_keys[0]}'}
                remaining = new_remaining

    selected = list(remaining.keys())[0] if remaining else None
    trace.append({'result': 'SELECTED' if selected else 'ALL_ELIMINATED', 'selected': selected})

    # LR comparison
    if selected and lr_baseline:
        sel_auprc = candidates_metrics[selected].get('release_auprc')
        lr_auprc = lr_baseline.get('release_auprc')
        if sel_auprc is not None and lr_auprc is not None:
            if sel_auprc <= lr_auprc:
                trace.append({'lr_comparison': 'FAILED', 'candidate_auprc': sel_auprc, 'lr_auprc': lr_auprc})
            else:
                trace.append({'lr_comparison': 'PASSED', 'candidate_auprc': sel_auprc, 'lr_auprc': lr_auprc})

    return selected, trace, eliminated


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--metrics-file', type=Path, required=True,
                    help='JSON file with candidate metrics keyed by candidate name')
    ap.add_argument('--lr-baseline-file', type=Path, default=None)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()

    candidates = json.loads(args.metrics_file.read_text())
    lr_baseline = json.loads(args.lr_baseline_file.read_text()) if args.lr_baseline_file else None

    selected, trace, eliminated = select_candidate(candidates, lr_baseline)

    result = {
        'selected': selected,
        'eliminated': eliminated,
        'selection_trace': trace,
        'n_candidates_input': len(candidates),
        'n_eliminated': len(eliminated),
        'status': 'SELECTED' if selected else 'NO_CANDIDATE_SURVIVED',
    }

    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(f'Selected: {selected}' if selected else 'NO CANDIDATE SELECTED')
    for name, reason in eliminated.items():
        print(f'  Eliminated {name}: {reason["reason"]}')


if __name__ == '__main__':
    main()
