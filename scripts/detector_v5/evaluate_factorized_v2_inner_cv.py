#!/usr/bin/env python3
"""V2 inner-CV evaluation runner. Supports single-run, pooled, and aggregated modes."""
import argparse, json, os, sys, uuid
from pathlib import Path
from collections import defaultdict
from statistics import mean, stdev

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent


def sha256_file(p):
    import hashlib
    d = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''): d.update(b)
    return d.hexdigest()


def _atomic_text(p, v):
    t = p.with_name(f'.{p.name}.{uuid.uuid4().hex}.tmp')
    with t.open('x') as f: f.write(v); f.flush(); os.fsync(f.fileno())
    os.replace(t, p)


def write_seal(root):
    excl = {'SHA256SUMS', 'SHA256SUMS.sha256'}
    fs = sorted((p for p in root.rglob('*') if p.is_file() and p.name not in excl),
                key=lambda p: p.relative_to(root).as_posix())
    c = ''.join(f'{sha256_file(p)}  {p.relative_to(root).as_posix()}\n' for p in fs)
    _atomic_text(root / 'SHA256SUMS', c)
    _atomic_text(root / 'SHA256SUMS.sha256', f'{sha256_file(root / "SHA256SUMS")}  SHA256SUMS\n')


THRESHOLD = 0.5
SUPPORTED_ROUTES = ['single_object_pick_place', 'multi_object_transfer']
DURATION_BUCKETS = [(0, 15), (15, 30), (30, 50), (50, 100), (100, 99999)]


from sklearn.metrics import roc_auc_score, average_precision_score

def compute_auroc(labels, scores):
    labels = np.array(labels, dtype=np.float64)
    scores = np.array(scores, dtype=np.float64)
    if len(np.unique(labels)) < 2:
        return None
    return float(roc_auc_score(labels, scores))


def compute_auprc(labels, scores):
    labels = np.array(labels)
    scores = np.array(scores)
    if labels.sum() == 0:
        return None
    return float(average_precision_score(labels, scores))


def load_events(pred_dir):
    f = pred_dir / 'heldout_event_predictions.jsonl'
    if not f.is_file():
        raise FileNotFoundError(str(f))
    return [json.loads(l) for l in f.read_text().splitlines() if l.strip()]


def load_steps(pred_dir):
    f = pred_dir / 'heldout_step_predictions.jsonl'
    if not f.is_file():
        raise FileNotFoundError(str(f))
    return [json.loads(l) for l in f.read_text().splitlines() if l.strip()]


def compute_metrics(events, steps, threshold=THRESHOLD):
    """Compute all metrics for a set of events."""
    m = {}

    for head in ['grasp', 'manipulation', 'release']:
        score_key = f'{head}_event_score'
        target_key = f'{head}_target'
        labels = [e[target_key] for e in events]
        scores = [e[score_key] for e in events]

        m[f'{head}_auroc'] = compute_auroc(labels, scores)
        m[f'{head}_auprc'] = compute_auprc(labels, scores)
        m[f'{head}_n_pos'] = int(sum(labels))
        m[f'{head}_n_neg'] = int(len(labels) - sum(labels))
        m[f'{head}_prevalence'] = sum(labels) / max(1, len(labels))

        # Recall@0.5
        pos_ev = [e for e in events if e[target_key]]
        if pos_ev:
            detected = sum(1 for e in pos_ev if e[score_key] >= threshold)
            m[f'{head}_recall_05'] = detected / len(pos_ev)
            m[f'{head}_precision_05'] = detected / max(1, sum(1 for e in events if e[score_key] >= threshold and not e.get(f'{head}_known_steps', 0) == 0))
            prec = m[f'{head}_precision_05']
            rec = m[f'{head}_recall_05']
            m[f'{head}_f1_05'] = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        else:
            m[f'{head}_recall_05'] = None

    # Per-route
    for route in SUPPORTED_ROUTES:
        route_ev = [e for e in events if e['mechanism_route'] == route]
        for head in ['release']:
            labels = [e[f'{head}_target'] for e in route_ev]
            scores = [e[f'{head}_event_score'] for e in route_ev]
            m[f'{route}_{head}_auroc'] = compute_auroc(labels, scores)
            m[f'{route}_{head}_auprc'] = compute_auprc(labels, scores)
            m[f'{route}_{head}_n_pos'] = int(sum(labels))

    # First/later event
    for ek, label in [('first', False), ('later', True)]:
        ek_ev = [e for e in events if e['is_later_event'] == label]
        for head in ['release']:
            labels = [e[f'{head}_target'] for e in ek_ev]
            scores = [e[f'{head}_event_score'] for e in ek_ev]
            m[f'{head}_{ek}_auprc'] = compute_auprc(labels, scores)
            pos_ek = [e for e in ek_ev if e[f'{head}_target']]
            if pos_ek:
                m[f'{head}_{ek}_recall_05'] = sum(1 for e in pos_ek if e[f'{head}_event_score'] >= threshold) / len(pos_ek)

    # Duration buckets
    for lo, hi in DURATION_BUCKETS:
        bucket_ev = [e for e in events if lo <= e.get('event_duration', 0) < hi]
        for head in ['release']:
            labels = [e[f'{head}_target'] for e in bucket_ev]
            scores = [e[f'{head}_event_score'] for e in bucket_ev]
            m[f'{head}_dur_{lo}_{hi}_auprc'] = compute_auprc(labels, scores)
            pos_b = [e for e in bucket_ev if e[f'{head}_target']]
            if pos_b:
                m[f'{head}_dur_{lo}_{hi}_recall_05'] = sum(1 for e in pos_b if e[f'{head}_event_score'] >= threshold) / len(pos_b)
                m[f'{head}_dur_{lo}_{hi}_n'] = len(pos_b)

    # Short event aggregate
    short_ev = [e for e in events if e.get('event_duration', 999) < 30]
    for head in ['release']:
        labels = [e[f'{head}_target'] for e in short_ev]
        scores = [e[f'{head}_event_score'] for e in short_ev]
        m[f'{head}_short_auprc'] = compute_auprc(labels, scores)

    # Safety metrics
    bg_known = 0; bg_emit = 0; bg_emit_grasp = 0; bg_emit_manip = 0; bg_emit_release = 0
    total_known = 0; release_overlap = 0
    unsupported_emit = 0; unsupported_total = 0

    for s in steps:
        if not s['route_supported']:
            unsupported_total += 1
            if s.get('grasp_prob', 0) > 1e-8 or s.get('manipulation_prob', 0) > 1e-8 or s.get('release_prob', 0) > 1e-8:
                unsupported_emit += 1
            continue

        g_km = s.get('grasp_known_mask', False)
        m_km = s.get('manipulation_known_mask', False)
        r_km = s.get('release_known_mask', False)
        g_p = s.get('grasp_prob', 0); m_p = s.get('manipulation_prob', 0); r_p = s.get('release_prob', 0)

        # Release overlap: both grasp and release emit on same step where both known
        if g_km and r_km and g_p >= threshold and r_p >= threshold:
            release_overlap += 1

        # Background false emit: event_id < 0 (IDLE/RELEASED background)
        if s.get('event_id', -1) < 0:
            any_known = g_km or m_km or r_km
            if any_known:
                bg_known += 1
                if (g_km and g_p >= threshold) or (m_km and m_p >= threshold) or (r_km and r_p >= threshold):
                    bg_emit += 1
                    if g_km and g_p >= threshold: bg_emit_grasp += 1
                    if m_km and m_p >= threshold: bg_emit_manip += 1
                    if r_km and r_p >= threshold: bg_emit_release += 1

        if g_km or m_km or r_km:
            total_known += 1

    m['release_overlap_emit_rate'] = release_overlap / max(1, total_known)
    m['background_false_emit_rate'] = bg_emit / max(1, bg_known) if bg_known > 0 else 0.0
    m['background_emit_grasp'] = bg_emit_grasp / max(1, bg_known) if bg_known > 0 else 0.0
    m['background_emit_manipulation'] = bg_emit_manip / max(1, bg_known) if bg_known > 0 else 0.0
    m['background_emit_release'] = bg_emit_release / max(1, bg_known) if bg_known > 0 else 0.0
    m['background_known_steps'] = bg_known
    m['unsupported_route_emit_rate'] = unsupported_emit / max(1, unsupported_total)
    m['unsupported_route_emit_count'] = unsupported_emit
    m['unsupported_route_total'] = unsupported_total

    # Window boundary diagnostics ONLY for windowed GRU
    encoder_type = steps[0].get('encoder_type', '') if steps else ''
    window_size = steps[0].get('window_size', 0) if steps else 0
    if encoder_type == 'windowed_gru' and window_size > 0:
        for pos in [0, 1, 2, window_size - 3, window_size - 2, window_size - 1]:
            pos_steps = [s for s in steps
                        if s.get('position_in_window', -1) == pos and s.get('route_supported')]
            if pos_steps:
                pos_known = sum(1 for s in pos_steps if s.get('release_known_mask'))
                pos_emit = sum(1 for s in pos_steps
                              if s.get('release_known_mask') and s.get('release_prob', 0) >= threshold)
                m[f'window_pos_{pos}_known'] = pos_known
                m[f'window_pos_{pos}_emit_rate'] = pos_emit / max(1, pos_known)

    # First/later gap
    if m.get('release_first_recall_05') is not None and m.get('release_later_recall_05') is not None:
        m['first_later_recall_gap'] = abs(m['release_first_recall_05'] - m['release_later_recall_05'])
    m['first_later_auprc_gap'] = abs(
        (m.get('release_first_auprc') or 0) - (m.get('release_later_auprc') or 0))

    return m


def aggregate_results(all_run_metrics, all_run_keys):
    """Aggregate metrics across runs: mean, stdev, worst-case."""
    agg = {}
    worst_case_keys = ['release_overlap_emit_rate', 'background_false_emit_rate',
                       'unsupported_route_emit_rate']

    metric_names = set()
    for metrics in all_run_metrics.values():
        metric_names.update(metrics.keys())

    for name in sorted(metric_names):
        vals = [m.get(name) for m in all_run_metrics.values() if m.get(name) is not None]
        if not vals:
            continue
        if name in worst_case_keys:
            agg[name] = {'worst': max(vals), 'mean': mean(vals), 'n': len(vals)}
        else:
            agg[name] = {'mean': mean(vals), 'stdev': stdev(vals) if len(vals) > 1 else 0, 'n': len(vals)}

    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--predictions-base', type=Path, required=True,
                    help='Directory with prediction shard subdirectories')
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--mode', choices=['single', 'aggregate'], default='aggregate')
    ap.add_argument('--candidate', type=str, default=None)
    ap.add_argument('--outer-fold', type=int, default=None)
    ap.add_argument('--inner-fold', type=int, default=None)
    ap.add_argument('--seed', type=int, default=None)
    args = ap.parse_args()

    pred_base = args.predictions_base.resolve()

    all_run_metrics = {}
    all_run_keys = []

    if args.mode == 'single':
        shard_dir = pred_base / f'{args.candidate}_outer{args.outer_fold}_inner{args.inner_fold}_seed{args.seed}'
        if not shard_dir.is_dir():
            raise SystemExit(f'Shard not found: {shard_dir}')
        events = load_events(shard_dir)
        steps = load_steps(shard_dir)
        metrics = compute_metrics(events, steps)
        all_run_metrics[f'{args.candidate}_o{args.outer_fold}_i{args.inner_fold}_s{args.seed}'] = metrics
        all_run_keys = list(all_run_metrics.keys())
    elif args.mode == 'aggregate':
        expected_runs = args.expected_runs if hasattr(args, 'expected_runs') else None
        if expected_runs and expected_runs.is_file():
            expected = json.loads(expected_runs.read_text())
            expected_keys = set(e['label'] for e in expected.get('jobs', []))
        else:
            expected_keys = None

        loaded = 0; missing = 0; invalid = 0; issues = []
        for d in sorted(pred_base.iterdir()):
            if not d.is_dir() or d.name.startswith('.'):
                continue
            try:
                events = load_events(d)
                steps = load_steps(d)
                metrics = compute_metrics(events, steps)
                all_run_metrics[d.name] = metrics
                all_run_keys.append(d.name)
                loaded += 1
            except Exception as e:
                invalid += 1
                issues.append(f'{d.name}: {e}')

        if expected_keys is not None:
            loaded_keys = set(all_run_metrics.keys())
            missing_keys = expected_keys - loaded_keys
            if missing_keys:
                missing = len(missing_keys)
                issues.append(f'Missing {missing} expected runs: {sorted(list(missing_keys))[:10]}...')

        if issues:
            print(f'EVALUATOR HOLD: {len(issues)} issues')
            for issue in issues[:20]:
                print(f'  {issue}')
            if missing > 0 or invalid > 0:
                raise SystemExit(f'HOLD: {missing} missing, {invalid} invalid')

        print(f'Loaded: {loaded}, Missing: {missing}, Invalid: {invalid}')
    else:
        raise SystemExit(f'Unknown mode: {args.mode}')

    print(f'Evaluated {len(all_run_metrics)} runs')

    agg = aggregate_results(all_run_metrics, all_run_keys)

    out = args.output.resolve()
    if out.exists():
        raise SystemExit(f'OUTPUT EXISTS: {out}')
    _atomic_text(out, json.dumps({
        'per_run': all_run_metrics,
        'aggregated': agg,
        'n_runs': len(all_run_metrics),
    }, indent=2) + '\n')

    # Print key results
    for mt in ['release_auroc', 'release_auprc', 'release_recall_05', 'release_short_auprc',
               'release_overlap_emit_rate', 'background_false_emit_rate']:
        if mt in agg:
            v = agg[mt]
            print(f'  {mt}: {json.dumps(v)}')

    print(f'Output: {out}')


if __name__ == '__main__':
    main()
