#!/usr/bin/env python3
"""S8 Phase 2 postprocess: compute oracle_normalized_qpos_area and VIS-RAND gap."""
import json, os, csv, numpy as np
from collections import defaultdict

def postprocess(summary_dir, oracle_ref_map, output_csv, output_report_md=None):
    summaries = []
    for fname in sorted(os.listdir(summary_dir)):
        if fname.startswith('summary_') and fname.endswith('.json'):
            with open(os.path.join(summary_dir, fname)) as f:
                summaries.append(json.load(f))

    # Group by logical_pair_key
    pairs = defaultdict(dict)
    for s in summaries:
        lp = s.get('pair_id', '')
        cond = 'VIS' if 'vis' in s.get('condition','') else 'RAND'
        pairs[lp][cond] = s

    results = []
    for lp in sorted(pairs):
        cd = pairs[lp]
        if len(cd) != 2: continue
        vis = cd.get('VIS', {}); rand = cd.get('RAND', {})

        # Find oracle reference from physical_pair_key
        # Pairs with window_start/end matching the original window
        ppk_short = '%s_s%d_w%d_%d_L10' % (vis.get('task',''), vis.get('state_id',0),
                                             vis.get('original_window_start', vis.get('window_start',0)),
                                             vis.get('original_window_end', vis.get('window_end',0)))
        oracle_ref = oracle_ref_map.get(ppk_short, {})
        oracle_pos_area = oracle_ref.get('qpos_pos_area', 0.0)

        vis_pos = vis.get('qpos_pos_area', 0); rand_pos = rand.get('qpos_pos_area', 0)
        vis_abs = vis.get('qpos_abs_area', 0); rand_abs = rand.get('qpos_abs_area', 0)
        vis_open = vis.get('decoded_open_count', 0); rand_open = rand.get('decoded_open_count', 0)
        vis_streak = vis.get('max_open_streak', 0); rand_streak = rand.get('max_open_streak', 0)
        vis_infra = vis.get('infra_status',''); rand_infra = rand.get('infra_status','')

        results.append({
            'logical_pair_key': lp,
            'task': vis.get('task',''),
            'window_start': vis.get('window_start',0),
            'window_end': vis.get('window_end',0),
            'length_mode': _get_length_mode(lp),
            'attack_seed': vis.get('attack_seed',0),
            'vis_open_count': vis_open, 'rand_open_count': rand_open,
            'vis_max_streak': vis_streak, 'rand_max_streak': rand_streak,
            'vis_qpos_pos_area': round(vis_pos, 8), 'rand_qpos_pos_area': round(rand_pos, 8),
            'vis_qpos_abs_area': round(vis_abs, 8), 'rand_qpos_abs_area': round(rand_abs, 8),
            'vis_minus_rand_pos_area': round(vis_pos - rand_pos, 8),
            'oracle_pos_area': round(oracle_pos_area, 8),
            'vis_oracle_normalized': round(vis_pos / oracle_pos_area, 4) if oracle_pos_area > 0 else 0,
            'rand_oracle_normalized': round(rand_pos / oracle_pos_area, 4) if oracle_pos_area > 0 else 0,
            'vis_infra': vis_infra, 'rand_infra': rand_infra,
        })

    # Write CSV
    cols = list(results[0].keys()) if results else []
    with open(output_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in results: w.writerow(r)

    # Report
    print('Pairs: %d' % len(results))
    n_infra_ok = sum(1 for r in results if r['vis_infra']=='ok' and r['rand_infra']=='ok')
    print('Infra OK: %d/%d' % (n_infra_ok, len(results)))

    if results:
        vis_pos_mean = np.mean([r['vis_qpos_pos_area'] for r in results])
        rand_pos_mean = np.mean([r['rand_qpos_pos_area'] for r in results])
        vis_norm_mean = np.mean([r['vis_oracle_normalized'] for r in results])
        rand_norm_mean = np.mean([r['rand_oracle_normalized'] for r in results])
        vis_open_mean = np.mean([r['vis_open_count'] for r in results])
        rand_open_mean = np.mean([r['rand_open_count'] for r in results])

        print()
        print('=== VIS vs RAND ===')
        print('VIS qpos_pos_area: %.6f' % vis_pos_mean)
        print('RAND qpos_pos_area: %.6f' % rand_pos_mean)
        print('VIS-RAND gap: %.6f' % (vis_pos_mean - rand_pos_mean))
        print('VIS oracle_norm: %.4f' % vis_norm_mean)
        print('RAND oracle_norm: %.4f' % rand_norm_mean)
        print('VIS open_count: %.2f' % vis_open_mean)
        print('RAND open_count: %.2f' % rand_open_mean)

        # Gate checks
        bridge = vis_pos_mean > rand_pos_mean
        vis_norm = vis_norm_mean >= 0.3
        cmd = vis_open_mean > rand_open_mean
        print()
        print('Bridge gate (VIS>RAND pos_area): %s' % ('PASS' if bridge else 'FAIL'))
        print('Normalized gate (VIS>=0.3 oracle): %s' % ('PASS' if vis_norm else 'FAIL'))
        print('Command gate (VIS>RAND open): %s' % ('PASS' if cmd else 'FAIL'))

        print()
        print('%-45s %4s %8s %8s %8s %8s %8s %8s %s' % ('Pair', 'seed', 'vis_pos', 'rand_pos', 'V-R', 'vis_norm', 'vis_op', 'rand_op', 'infra'))
        for r in sorted(results, key=lambda x: (x['task'], x['length_mode'], x['attack_seed'])):
            print('%-45s %4d %8.6f %8.6f %+8.6f %8.4f %8d %8d %s' % (
                r['logical_pair_key'][:45], r['attack_seed'],
                r['vis_qpos_pos_area'], r['rand_qpos_pos_area'],
                r['vis_minus_rand_pos_area'], r['vis_oracle_normalized'],
                r['vis_open_count'], r['rand_open_count'],
                'OK' if (r['vis_infra']=='ok' and r['rand_infra']=='ok') else 'FAIL'))

    return results

def _get_length_mode(lp):
    if 'extended20' in lp: return 'extended20'
    if '_short__' in lp or lp.endswith('_short'): return 'short'
    return 'short'

if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--summary_dir', required=True)
    ap.add_argument('--oracle_ref_csv', required=True)
    ap.add_argument('--output_csv', required=True)
    args = ap.parse_args()

    import csv as csv_module
    oracle_ref_map = {}
    with open(args.oracle_ref_csv) as f:
        for r in csv_module.DictReader(f):
            oracle_ref_map[r['physical_pair_key']] = {'qpos_pos_area': float(r['oracle_qpos_pos_area'])}

    postprocess(args.summary_dir, oracle_ref_map, args.output_csv)
