#!/usr/bin/env python3
"""V2 lexicographic candidate selection with bootstrap and LR gate.

Priority order (frozen):
  1. Safety hard elimination (worst-seed, worst-split)
  2. Release AUPRC (higher is better)
  3. Short-event AUPRC (higher is better)
  4. First/later recall gap (smaller absolute gap is better)
  5. Parameter count (fewer is better, tiebreaker only)

At each priority, keep all candidates not statistically significantly worse
than the current best (identity-paired bootstrap, 95% CI). Only eliminate
candidates that are significantly worse.
"""
import argparse, json, sys
from pathlib import Path
from statistics import mean
import numpy as np

BOOTSTRAP_SEED = 42
BOOTSTRAP_RESAMPLES = 10000
CI_LEVEL = 95

PRIORITY_ORDER = [
    ('safety', None),
    ('release_auprc', 'higher'),
    ('release_short_auprc', 'higher'),
    ('first_later_recall_gap', 'lower'),
    ('parameter_count', 'lower'),
]

SAFETY_LIMITS = {
    'unsupported_route_emit_rate': {'max': 0.0},
    'background_false_emit_rate': {'max': 0.10},
    'release_overlap_emit_rate': {'max': 0.05},
}


def paired_bootstrap_diff(scores_a, scores_b, n_resamples=BOOTSTRAP_RESAMPLES, seed=BOOTSTRAP_SEED):
    """Identity-paired bootstrap: is candidate A better than B?

    Returns (mean_diff, ci_low, ci_high, significant) where significant=True
    means A is significantly better (entire CI > 0 for 'higher' metrics).
    """
    a = np.asarray(scores_a, dtype=float)
    b = np.asarray(scores_b, dtype=float)
    n = len(a)
    rng = np.random.RandomState(seed)
    diffs = np.zeros(n_resamples)
    for i in range(n_resamples):
        idx = rng.randint(0, n, n)
        diffs[i] = mean(a[idx]) - mean(b[idx])
    lo = float(np.percentile(diffs, (100 - CI_LEVEL) / 2))
    hi = float(np.percentile(diffs, 100 - (100 - CI_LEVEL) / 2))
    mean_d = float(mean(diffs))
    return mean_d, lo, hi


def check_safety_elimination(candidates_metrics):
    """Priority 1: hard elimination. Uses worst value across all splits/seeds.

    Returns (surviving, eliminated_dict).
    """
    surviving = {}
    eliminated = {}
    for name, metrics in candidates_metrics.items():
        failures = []
        for metric, limits in SAFETY_LIMITS.items():
            val = metrics.get(metric)
            if val is None:
                failures.append(f'{metric}: missing')
            elif val > limits['max']:
                failures.append(f'{metric}: {val:.6f} > {limits["max"]}')
        if failures:
            eliminated[name] = {'priority': 'safety', 'failures': failures}
        else:
            surviving[name] = metrics
    return surviving, eliminated


def compare_and_filter(candidates, metric_key, direction='higher'):
    """Keep candidates not significantly worse than the best on this metric.

    Returns (surviving, eliminated).
    """
    if len(candidates) <= 1:
        return dict(candidates), {}

    # Find best
    if direction == 'higher':
        best_name = max(candidates, key=lambda n: candidates[n].get(metric_key, -1e9))
    else:
        best_name = min(candidates, key=lambda n: candidates[n].get(metric_key, 1e9))

    best_val = candidates[best_name].get(metric_key)
    if best_val is None:
        return dict(candidates), {}

    surviving = {best_name: candidates[best_name]}
    eliminated = {}

    for name, metrics in candidates.items():
        if name == best_name:
            continue
        val = metrics.get(metric_key)
        if val is None:
            eliminated[name] = {'priority': metric_key, 'reason': 'metric missing'}
            continue

        # Bootstrap comparison requires per-split scores
        best_scores = candidates[best_name].get(f'{metric_key}_per_split')
        curr_scores = metrics.get(f'{metric_key}_per_split')

        if best_scores is not None and curr_scores is not None and len(best_scores) == len(curr_scores):
            mean_diff, ci_lo, ci_hi = paired_bootstrap_diff(
                best_scores if direction == 'higher' else curr_scores,
                curr_scores if direction == 'higher' else best_scores)

            if direction == 'higher':
                significantly_worse = ci_hi < 0  # best - curr CI entirely negative → curr worse
            else:
                significantly_worse = ci_lo > 0  # best - curr CI entirely positive → best bigger gap

            if significantly_worse:
                eliminated[name] = {
                    'priority': metric_key,
                    'reason': f'significantly worse ({direction}): mean_diff={mean_diff:.6f} CI=[{ci_lo:.6f}, {ci_hi:.6f}]',
                }
            else:
                surviving[name] = metrics
        else:
            # No per-split scores — use point estimates
            if direction == 'higher' and val < best_val:
                eliminated[name] = {'priority': metric_key, 'reason': f'point estimate lower: {val:.6f} < {best_val:.6f}'}
            elif direction == 'lower' and val > best_val:
                eliminated[name] = {'priority': metric_key, 'reason': f'point estimate higher: {val:.6f} > {best_val:.6f}'}
            else:
                surviving[name] = metrics

    return surviving, eliminated


def select_candidate(candidates_metrics):
    """Run full lexicographic selection. Returns (selected_name, trace, eliminated)."""
    remaining = dict(candidates_metrics)
    all_eliminated = {}
    trace = []
    trace.append({'phase': 'start', 'n_candidates': len(remaining)})

    for priority, direction in PRIORITY_ORDER:
        if len(remaining) <= 1:
            break

        trace.append({'phase': priority, 'n_before': len(remaining), 'candidates': list(remaining.keys())})

        if priority == 'safety':
            surviving, eliminated = check_safety_elimination(remaining)
        else:
            surviving, eliminated = compare_and_filter(remaining, priority, direction)

        remaining = surviving
        all_eliminated.update(eliminated)
        trace.append({'phase': f'{priority}_result', 'n_surviving': len(remaining),
                      'n_eliminated': len(eliminated),
                      'eliminated': {k: v.get('reason', str(v)) for k, v in eliminated.items()}})

        if not remaining:
            trace.append({'result': 'ALL_ELIMINATED', 'phase': priority})
            return None, trace, all_eliminated

    selected = list(remaining.keys())[0] if remaining else None
    if selected and len(remaining) > 1:
        # Tie at last priority — use parameter_count
        by_params = sorted(remaining, key=lambda n: remaining[n].get('parameter_count', 1e9))
        selected = by_params[0]
        for n in by_params[1:]:
            all_eliminated[n] = {'priority': 'parameter_count_tiebreak', 'reason': f'more params than {selected}'}

    trace.append({'result': 'SELECTED' if selected else 'ALL_ELIMINATED', 'selected': selected})
    return selected, trace, all_eliminated


def check_lr_gate(selected, candidates, lr_baseline):
    """Verify selected candidate beats LR on both AUROC and AUPRC with bootstrap."""
    sel_m = candidates[selected]
    checks = []
    for metric in ['release_auroc', 'release_auprc']:
        sel_val = sel_m.get(metric)
        lr_val = lr_baseline.get(metric)
        if sel_val is None:
            return False, f'{metric} missing for candidate', checks
        if lr_val is None:
            return False, f'{metric} missing for LR baseline', checks
        if sel_val <= lr_val:
            return False, f'{metric}: candidate {sel_val:.4f} <= LR {lr_val:.4f}', checks
        checks.append({'metric': metric, 'candidate': sel_val, 'lr': lr_val, 'pass': True})

    # Bootstrap CI
    sel_auprc_scores = sel_m.get('release_auprc_per_split')
    lr_auprc_scores = lr_baseline.get('release_auprc_per_split')
    if sel_auprc_scores is not None and lr_auprc_scores is not None and len(sel_auprc_scores) == len(lr_auprc_scores):
        mean_diff, ci_lo, ci_hi = paired_bootstrap_diff(sel_auprc_scores, lr_auprc_scores)
        if ci_lo <= 0:
            return 'HOLD_STATISTICALLY_INCONCLUSIVE', \
                f'AUPRC 95% CI crosses zero: [{ci_lo:.6f}, {ci_hi:.6f}]', checks
    return True, 'candidate beats LR on both AUROC and AUPRC', checks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--metrics-file', type=Path, required=True)
    ap.add_argument('--lr-baseline-file', type=Path, default=None)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()

    candidates = json.loads(args.metrics_file.read_text())
    lr_baseline = json.loads(args.lr_baseline_file.read_text()) if args.lr_baseline_file else None

    selected, trace, eliminated = select_candidate(candidates)

    lr_status = None
    if selected and lr_baseline:
        lr_pass, lr_msg, lr_checks = check_lr_gate(selected, candidates, lr_baseline)
        lr_status = {'pass': lr_pass, 'message': lr_msg, 'checks': lr_checks}
        trace.append({'lr_gate': lr_status})

    if selected and lr_status and lr_status['pass'] is not True:
        result_status = lr_status['pass'] if isinstance(lr_status['pass'], str) else 'HOLD_LR_GATE_FAILED'
    elif selected:
        result_status = 'SELECTED'
    else:
        result_status = 'NO_CANDIDATE_SURVIVED'

    result = {
        'selected': selected,
        'status': result_status,
        'n_input': len(candidates),
        'n_eliminated': len(eliminated),
        'eliminated': eliminated,
        'selection_trace': trace,
        'lr_gate': lr_status,
    }

    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(f'Selected: {selected}' if selected else 'NO CANDIDATE')
    print(f'Status: {result_status}')
    for name, info in eliminated.items():
        print(f'  Eliminated {name}: {info.get("reason", info.get("failures", str(info)))}')


if __name__ == '__main__':
    main()
