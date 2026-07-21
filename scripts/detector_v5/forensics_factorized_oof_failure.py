#!/usr/bin/env python3
"""Phase R1: Read-only OOF failure forensics.

Diagnoses release failure without modifying gates, thresholds, or aggregation.
Outputs: score distributions, event-duration analysis, later-event errors,
         threshold sensitivity (AUROC), per-event casebook sample.

Reads existing predictions — no re-inference needed for initial analysis.
"""
import hashlib, json, os, sys, uuid
from pathlib import Path
from collections import defaultdict
from statistics import mean, stdev, median

import argparse

OPS = Path('/mnt/sdc/dty_user/openvla_attack_evidence/c2g/c2g_cs200_official_v3_20260716/ops')
PRED_BASE = OPS / 'OFFICIAL_V3_FACTORIZED_STUDENT_OOF_PREDICTIONS_V1_20260721'
EVAL_ROOT = OPS / 'OFFICIAL_V3_FACTORIZED_STUDENT_OOF_EVALUATION_V1_20260721'
OUT_BASE = OPS / 'OFFICIAL_V3_FACTORIZED_STUDENT_OOF_FAILURE_FORENSICS_V1_20260721'

THRESHOLD = 0.5
SUPPORTED_ROUTES = ['single_object_pick_place', 'multi_object_transfer']


def sha256_file(p):
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


def load_event_preds(mt, fold, seed):
    f = PRED_BASE / f'predict_{mt}_fold{fold}_seed{seed}' / 'heldout_event_predictions.jsonl'
    return [json.loads(l) for l in f.read_text().splitlines() if l.strip()]


def load_step_preds(mt, fold, seed):
    f = PRED_BASE / f'predict_{mt}_fold{fold}_seed{seed}' / 'heldout_step_predictions.jsonl'
    return [json.loads(l) for l in f.read_text().splitlines() if l.strip()]


def compute_auroc(labels, scores):
    """Simple AUROC via Wilcoxon-Mann-Whitney."""
    if len(labels) == 0 or all(labels) or not any(labels):
        return None
    pos_scores = [s for s, l in zip(scores, labels) if l]
    neg_scores = [s for s, l in zip(scores, labels) if not l]
    if not pos_scores or not neg_scores:
        return None
    n_pos, n_neg = len(pos_scores), len(neg_scores)
    # Count pairs where pos > neg
    pos_sorted = sorted(pos_scores)
    neg_sorted = sorted(neg_scores)
    rank_sum = 0
    for ps in pos_scores:
        # Count negs below this pos
        lo, hi = 0, n_neg
        while lo < hi:
            mid = (lo + hi) // 2
            if neg_sorted[mid] < ps:
                lo = mid + 1
            else:
                hi = mid
        rank_sum += lo
        # Count negs equal
        eq = 0
        for ns in neg_scores:
            if ns == ps:
                eq += 1
        rank_sum += eq * 0.5
    return rank_sum / (n_pos * n_neg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output-root', type=Path, default=OUT_BASE)
    ap.add_argument('--sample-checkpoints', type=int, default=4,
                    help='Number of checkpoints to run full per-event analysis on')
    args = ap.parse_args()

    OUT = args.output_root.resolve()
    if OUT.exists():
        raise SystemExit(f'OUTPUT EXISTS: {OUT}')

    staging = OUT.with_name(f'.{OUT.name}.{uuid.uuid4().hex}.staging')
    staging.mkdir(parents=True)

    model_types = ['25D9D', '25D']
    folds = [0, 1, 2, 3]
    seeds = [42, 123, 456]

    # ── F1: Release score distribution analysis ──
    print('=== F1: Release score distributions ===')
    release_scores = {mt: {'positive': [], 'negative': [], 'by_route': {
        r: {'positive': [], 'negative': []} for r in SUPPORTED_ROUTES}} for mt in model_types}
    release_event_details = []  # per-event detail records

    for mt in model_types:
        for fold in folds:
            for seed in seeds:
                events = load_event_preds(mt, fold, seed)
                for e in events:
                    score = e['release_score_max']
                    target = e['release_target']
                    route = e['mechanism_route']

                    rec = {
                        'model_type': mt, 'fold_id': fold, 'seed': seed,
                        'canonical_parent_key': e['canonical_parent_key'],
                        'event_id': e['event_id'],
                        'event_ordinal': e['event_ordinal'],
                        'is_later_event': e['is_later_event'],
                        'mechanism_route': route,
                        'release_score_max': score,
                        'release_score_mean': e['release_score_mean'],
                        'release_target': target,
                        'release_emit': e['release_emit'],
                        'steps_in_event': e['steps_in_event'],
                        'known_steps_release': e['known_steps_release'],
                        'grasp_target': e['grasp_target'],
                        'manipulation_target': e['manipulation_target'],
                    }

                    if target:
                        release_scores[mt]['positive'].append(score)
                        if route in SUPPORTED_ROUTES:
                            release_scores[mt]['by_route'][route]['positive'].append(score)
                    else:
                        release_scores[mt]['negative'].append(score)
                        if route in SUPPORTED_ROUTES:
                            release_scores[mt]['by_route'][route]['negative'].append(score)

                    release_event_details.append(rec)

    # Build distribution stats
    dist_stats = {}
    for mt in model_types:
        dist_stats[mt] = {'overall': {}, 'by_route': {}}
        for label, key in [('positive', 'pos'), ('negative', 'neg')]:
            vals = release_scores[mt][label]
            if vals:
                dist_stats[mt]['overall'][key] = {
                    'count': len(vals),
                    'mean': mean(vals), 'median': median(vals),
                    'stdev': stdev(vals) if len(vals) > 1 else 0,
                    'min': min(vals), 'max': max(vals),
                    'p10': sorted(vals)[len(vals)//10],
                    'p25': sorted(vals)[len(vals)//4],
                    'p75': sorted(vals)[3*len(vals)//4],
                    'p90': sorted(vals)[9*len(vals)//10],
                    'frac_above_05': sum(1 for v in vals if v >= 0.5) / len(vals),
                }
            for route in SUPPORTED_ROUTES:
                rvals = release_scores[mt]['by_route'][route][label]
                if rvals:
                    dist_stats[mt]['by_route'][f'{route}_{key}'] = {
                        'count': len(rvals),
                        'mean': mean(rvals), 'median': median(rvals),
                        'frac_above_05': sum(1 for v in rvals if v >= 0.5) / len(rvals),
                    }
        print(f'  {mt} release pos mean={dist_stats[mt]["overall"]["pos"]["mean"]:.4f} median={dist_stats[mt]["overall"]["pos"]["median"]:.4f} frac>=0.5={dist_stats[mt]["overall"]["pos"]["frac_above_05"]:.4f}')
        print(f'  {mt} release neg mean={dist_stats[mt]["overall"]["neg"]["mean"]:.4f} median={dist_stats[mt]["overall"]["neg"]["median"]:.4f}')

    # ── F2: AUROC / threshold-free discrimination ──
    print('\n=== F2: Threshold-free discrimination ===')
    auroc_stats = {}
    for mt in model_types:
        auroc_stats[mt] = {}
        for head in ['grasp', 'manipulation', 'release']:
            all_scores = []
            all_labels = []
            for fold in folds:
                for seed in seeds:
                    events = load_event_preds(mt, fold, seed)
                    for e in events:
                        all_scores.append(e[f'{head}_score_max'])
                        all_labels.append(e[f'{head}_target'])

            auroc = compute_auroc(all_labels, all_scores)
            auroc_stats[mt][head] = auroc
            print(f'  {mt} {head}: AUROC={auroc:.4f}' if auroc else f'  {mt} {head}: AUROC=N/A')

        # Per-route AUROC
        for route in SUPPORTED_ROUTES:
            for head in ['release']:
                scores, labels = [], []
                for fold in folds:
                    for seed in seeds:
                        events = load_event_preds(mt, fold, seed)
                        for e in events:
                            if e['mechanism_route'] == route:
                                scores.append(e[f'{head}_score_max'])
                                labels.append(e[f'{head}_target'])
                auroc = compute_auroc(labels, scores)
                print(f'  {mt} {route} release: AUROC={auroc:.4f}' if auroc else f'  {mt} {route} release: AUROC=N/A')

    # ── F3: Event duration vs detection ──
    print('\n=== F3: Event duration vs detection ===')
    duration_analysis = {mt: {'detected': [], 'missed': [], 'by_route': {
        r: {'detected': [], 'missed': []} for r in SUPPORTED_ROUTES}} for mt in model_types}

    for rec in release_event_details:
        if not rec['release_target']:
            continue
        mt = rec['model_type']
        dur = rec['steps_in_event']
        if rec['release_emit']:
            duration_analysis[mt]['detected'].append(dur)
            if rec['mechanism_route'] in SUPPORTED_ROUTES:
                duration_analysis[mt]['by_route'][rec['mechanism_route']]['detected'].append(dur)
        else:
            duration_analysis[mt]['missed'].append(dur)
            if rec['mechanism_route'] in SUPPORTED_ROUTES:
                duration_analysis[mt]['by_route'][rec['mechanism_route']]['missed'].append(dur)

    dur_stats = {}
    for mt in model_types:
        dur_stats[mt] = {}
        for label in ['detected', 'missed']:
            vals = duration_analysis[mt][label]
            dur_stats[mt][label] = {
                'count': len(vals),
                'mean': mean(vals) if vals else 0,
                'median': median(vals) if vals else 0,
                'min': min(vals) if vals else 0,
                'max': max(vals) if vals else 0,
            }
        d_mean = dur_stats[mt]['detected']['mean']
        m_mean = dur_stats[mt]['missed']['mean']
        print(f'  {mt}: detected events mean duration={d_mean:.1f} steps, missed mean duration={m_mean:.1f} steps')

        for route in SUPPORTED_ROUTES:
            d_vals = duration_analysis[mt]['by_route'][route]['detected']
            m_vals = duration_analysis[mt]['by_route'][route]['missed']
            print(f'    {route}: detected={len(d_vals)} (dur={mean(d_vals):.1f}) missed={len(m_vals)} (dur={mean(m_vals):.1f})' if m_vals else '')

    # ── F4: Later-event error analysis ──
    print('\n=== F4: Later-event analysis ===')
    later_analysis = {mt: {'first': {'detected': 0, 'missed': 0, 'total': 0},
                            'later': {'detected': 0, 'missed': 0, 'total': 0}}
                      for mt in model_types}

    for rec in release_event_details:
        if not rec['release_target']:
            continue
        mt = rec['model_type']
        key = 'later' if rec['is_later_event'] else 'first'
        later_analysis[mt][key]['total'] += 1
        if rec['release_emit']:
            later_analysis[mt][key]['detected'] += 1
        else:
            later_analysis[mt][key]['missed'] += 1

    for mt in model_types:
        for ek in ['first', 'later']:
            a = later_analysis[mt][ek]
            rate = a['detected'] / max(1, a['total'])
            print(f'  {mt} {ek}_event: recall={rate:.4f} ({a["detected"]}/{a["total"]})')

    # ── F5: Missed-release casebook sample ──
    print('\n=== F5: Building release casebook ===')
    # Sample: 20 worst missed (lowest score) + 20 borderline missed (0.3-0.5)
    # + 20 detected (lowest detected score) for single route
    casebook_entries = []

    for mt in model_types:
        for route in SUPPORTED_ROUTES:
            route_events = [e for e in release_event_details
                           if e['mechanism_route'] == route and e['release_target'] and e['model_type'] == mt]

            missed = sorted([e for e in route_events if not e['release_emit']],
                           key=lambda e: e['release_score_max'])
            detected = sorted([e for e in route_events if e['release_emit']],
                             key=lambda e: e['release_score_max'])

            # Take samples
            samples = (missed[:10] + missed[-5:] +  # worst missed + borderline missed
                      [e for e in missed if 0.3 <= e['release_score_max'] < 0.5][:5] +
                      detected[:5] + detected[-5:])  # hardest detected + easiest detected

            for e in samples:
                casebook_entries.append({
                    'model_type': mt, 'route': route,
                    'identity': e['canonical_parent_key'],
                    'event_id': e['event_id'], 'event_ordinal': e['event_ordinal'],
                    'is_later_event': e['is_later_event'],
                    'fold_id': e['fold_id'], 'seed': e['seed'],
                    'release_score_max': e['release_score_max'],
                    'release_score_mean': e['release_score_mean'],
                    'release_emit': e['release_emit'],
                    'steps_in_event': e['steps_in_event'],
                    'known_steps_release': e['known_steps_release'],
                    'grasp_target': e['grasp_target'],
                    'manipulation_target': e['manipulation_target'],
                })

    # Deduplicate
    seen = set()
    unique_cases = []
    for c in casebook_entries:
        key = (c['identity'], c['event_id'], c['model_type'])
        if key not in seen:
            seen.add(key)
            unique_cases.append(c)

    print(f'  Casebook: {len(unique_cases)} unique event samples')

    # ── F6: Step-level release probe (for detailed casebook) ──
    print('\n=== F6: Step-level probe for casebook entries ===')
    step_casebook = []
    for case in unique_cases[:40]:  # top 40 cases get step-level detail
        mt = case['model_type']
        fold = case['fold_id']
        seed = case['seed']
        identity = case['identity']
        eid = case['event_id']

        steps = load_step_preds(mt, fold, seed)
        event_steps = [s for s in steps
                       if s['canonical_parent_key'] == identity and s['event_id'] == eid]

        step_detail = []
        for s in event_steps:
            step_detail.append({
                'step_index': s['step_index'],
                'grasp_prob': s['grasp_prob'], 'grasp_target': s['grasp_target'],
                'manipulation_prob': s['manipulation_prob'], 'manipulation_target': s['manipulation_target'],
                'release_prob': s['release_prob'], 'release_target': s['release_target'],
                'release_known_mask': s['release_known_mask'],
                'grasp_known_mask': s['grasp_known_mask'],
                'manipulation_known_mask': s['manipulation_known_mask'],
                'event_role': s['event_role'],
            })

        step_casebook.append({
            'case': case,
            'step_trace': step_detail,
        })

    # ── Write all artifacts ──
    print('\n=== Writing forensics artifacts ===')

    _atomic_text(staging / 'release_score_distributions.json',
                 json.dumps(dist_stats, indent=2))
    _atomic_text(staging / 'threshold_free_discrimination.json',
                 json.dumps(auroc_stats, indent=2))
    _atomic_text(staging / 'event_duration_analysis.json',
                 json.dumps(dur_stats, indent=2))
    _atomic_text(staging / 'later_event_error_analysis.json',
                 json.dumps(later_analysis, indent=2))
    _atomic_text(staging / 'release_casebook.json',
                 json.dumps({
                     'total_entries': len(unique_cases),
                     'entries': unique_cases,
                 }, indent=2))
    _atomic_text(staging / 'release_casebook_step_detail.json',
                 json.dumps({
                     'total_entries': len(step_casebook),
                     'entries': step_casebook,
                 }, indent=2))

    # Full per-event dump for offline analysis
    _atomic_text(staging / 'release_event_full_dump.json',
                 json.dumps(release_event_details, indent=2))

    # Summary
    summary = {
        'schema': 'DETECTOR_V5_FACTORIZED_OOF_FAILURE_FORENSICS_V1',
        'status': 'FORENSICS_COMPLETE',
        'key_findings': {
            'release_pos_mean': dist_stats['25D9D']['overall']['pos']['mean'],
            'release_pos_median': dist_stats['25D9D']['overall']['pos']['median'],
            'release_pos_frac_above_threshold': dist_stats['25D9D']['overall']['pos']['frac_above_05'],
            'release_auroc': auroc_stats['25D9D']['release'],
            'release_single_auroc': auroc_stats['25D9D'].get('release_single_object_pick_place',
                                    auroc_stats['25D9D'].get('single_object_pick_place_release')),
            'detected_event_mean_duration': dur_stats['25D9D']['detected']['mean'],
            'missed_event_mean_duration': dur_stats['25D9D']['missed']['mean'],
            'first_event_release_recall': later_analysis['25D9D']['first']['detected'] / max(1, later_analysis['25D9D']['first']['total']),
            'later_event_release_recall': later_analysis['25D9D']['later']['detected'] / max(1, later_analysis['25D9D']['later']['total']),
        },
    }
    _atomic_text(staging / 'forensics_summary.json', json.dumps(summary, indent=2))

    # Source binding
    _atomic_text(staging / 'source_binding.json', json.dumps({
        'predictions_base': str(PRED_BASE),
        'evaluation_root': str(EVAL_ROOT),
        'prediction_seal': sha256_file(PRED_BASE / 'predict_25D9D_fold0_seed42' / 'SHA256SUMS'),
    }, indent=2))

    write_seal(staging)
    os.replace(staging, OUT)

    print(f'\nForensics complete: {OUT}')
    print(f'Seal: {sha256_file(OUT / "SHA256SUMS")}')
    for k, v in summary['key_findings'].items():
        if isinstance(v, float):
            print(f'  {k}: {v:.4f}')
        else:
            print(f'  {k}: {v}')


if __name__ == '__main__':
    main()
