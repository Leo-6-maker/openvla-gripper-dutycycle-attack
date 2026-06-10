#!/usr/bin/env python3
"""S8 Phase 2 postprocess: oracle_normalized_qpos_area, VIS-RAND gap, trace audit."""
import json, os, csv, numpy as np, argparse
from collections import defaultdict

def postprocess(summary_dir, launch_csv_path, output_csv):
    # Load launch CSV for oracle_ref backup + metadata
    launch_map = {}
    if launch_csv_path and os.path.exists(launch_csv_path):
        with open(launch_csv_path) as f:
            for r in csv.DictReader(f):
                launch_map[r['logical_pair_key']] = r

    # Load summaries
    summaries = []
    trace_files = []
    for fname in sorted(os.listdir(summary_dir)):
        if fname.startswith('summary_') and fname.endswith('.json'):
            with open(os.path.join(summary_dir, fname)) as f:
                summaries.append(json.load(f))
        if fname.startswith('trace_') and fname.endswith('.csv'):
            trace_files.append(fname)

    # Group by pair_id (logical_pair_key)
    pairs = defaultdict(dict)
    for s in summaries:
        lp = s.get('pair_id', '')
        cond = 'VIS' if 'vis' in s.get('condition','') else 'RAND'
        pairs[lp][cond] = s

    results = []
    for lp in sorted(pairs):
        cd = pairs[lp]
        if len(cd) != 2:
            print('WARN: %s has %d conditions' % (lp, len(cd)))
            continue

        vis = cd.get('VIS', {}); rand = cd.get('RAND', {})

        # Oracle ref: priority = summary embedded > launch CSV
        oracle_pos = vis.get('oracle_ref_L10_pos_area', 0) or rand.get('oracle_ref_L10_pos_area', 0)
        if oracle_pos == 0 and lp in launch_map:
            oracle_pos = float(launch_map[lp].get('oracle_ref_L10_pos_area', 0))

        vis_pos = vis.get('qpos_pos_area', 0); rand_pos = rand.get('qpos_pos_area', 0)
        vis_abs = vis.get('qpos_abs_area', 0); rand_abs = rand.get('qpos_abs_area', 0)
        vis_open = vis.get('decoded_open_count', 0); rand_open = rand.get('decoded_open_count', 0)
        vis_streak = vis.get('max_open_streak', 0); rand_streak = rand.get('max_open_streak', 0)
        vis_infra = vis.get('infra_status',''); rand_infra = rand.get('infra_status','')
        vis_arm = vis.get('mean_arm_qpos_norm_pre', 0); rand_arm = rand.get('mean_arm_qpos_norm_pre', 0)

        results.append({
            'logical_pair_key': lp,
            'task': vis.get('task',''),
            'length_mode': vis.get('length_mode',''),
            'window_start': vis.get('window_start',0),
            'window_end': vis.get('window_end',0),
            'original_ws': vis.get('original_window_start',0),
            'original_we': vis.get('original_window_end',0),
            'attack_seed': vis.get('attack_seed',0),
            'vis_open_count': vis_open, 'rand_open_count': rand_open,
            'vis_max_streak': vis_streak, 'rand_max_streak': rand_streak,
            'vis_qpos_pos_area': round(vis_pos, 8), 'rand_qpos_pos_area': round(rand_pos, 8),
            'vis_qpos_abs_area': round(vis_abs, 8), 'rand_qpos_abs_area': round(rand_abs, 8),
            'vis_minus_rand_pos_area': round(vis_pos - rand_pos, 8),
            'oracle_pos_area': round(oracle_pos, 8),
            'vis_oracle_normalized': round(vis_pos / oracle_pos, 4) if oracle_pos > 0 else -1,
            'rand_oracle_normalized': round(rand_pos / oracle_pos, 4) if oracle_pos > 0 else -1,
            'vis_arm_norm': round(vis_arm, 8), 'rand_arm_norm': round(rand_arm, 8),
            'vis_infra': vis_infra, 'rand_infra': rand_infra,
            'trace_csv_exists': 'Y' if any((lp in t) or (lp.replace('__','_') in t.replace('__','_')) for t in trace_files) else 'N',
        })

    if not results:
        print('No results found')
        return []

    # Write CSV
    cols = list(results[0].keys())
    with open(output_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in results: w.writerow(r)
    print('Results: %s (%d rows)' % (output_csv, len(results)))

    # Report
    n_infra_ok = sum(1 for r in results if r['vis_infra']=='ok' and r['rand_infra']=='ok')
    n_trace_ok = sum(1 for r in results if r['trace_csv_exists'] == 'Y')
    print('Infra OK: %d/%d, Trace OK: %d/%d, Summaries: %d' % (
        n_infra_ok, len(results), n_trace_ok, len(results), len(summaries)))

    # By length_mode
    for lmode in ['short', 'extended20']:
        gr = [r for r in results if r['length_mode'] == lmode]
        if not gr: continue
        vis_pos = np.mean([r['vis_qpos_pos_area'] for r in gr])
        rand_pos = np.mean([r['rand_qpos_pos_area'] for r in gr])
        vis_norm = np.mean([r['vis_oracle_normalized'] for r in gr if r['vis_oracle_normalized'] >= 0])
        rand_norm = np.mean([r['rand_oracle_normalized'] for r in gr if r['rand_oracle_normalized'] >= 0])
        vis_op = np.mean([r['vis_open_count'] for r in gr])
        rand_op = np.mean([r['rand_open_count'] for r in gr])

        print()
        print('=== %s ===' % lmode)
        print('VIS-RAND pos_area: %.6f' % (vis_pos - rand_pos))
        print('VIS oracle_norm: %.4f  RAND oracle_norm: %.4f' % (vis_norm, rand_norm))
        print('VIS open: %.2f  RAND open: %.2f' % (vis_op, rand_op))
        bridge = vis_pos > rand_pos
        norm = vis_norm >= 0.3
        cmd = vis_op > rand_op
        print('Bridge: %s  Norm>=0.3: %s  Cmd_VIS>RAND: %s' % (
            'PASS' if bridge else 'FAIL', 'PASS' if norm else 'FAIL', 'PASS' if cmd else 'FAIL'))

    # Per-pair table
    print()
    print('%-45s %6s %4s %8s %8s %8s %8s %8s %8s %5s %5s' % (
        'Pair', 'Lmode', 'seed', 'vis_pos', 'rand_pos', 'V-R', 'vis_norm', 'vis_op', 'rand_op', 'infra', 'trc'))
    for r in sorted(results, key=lambda x: (x['length_mode'], x['attack_seed'])):
        print('%-45s %6s %4d %8.6f %8.6f %+8.6f %8.4f %8d %8d %5s %5s' % (
            r['logical_pair_key'][:45], r['length_mode'][:6], r['attack_seed'],
            r['vis_qpos_pos_area'], r['rand_qpos_pos_area'],
            r['vis_minus_rand_pos_area'], r['vis_oracle_normalized'],
            r['vis_open_count'], r['rand_open_count'],
            'OK' if (r['vis_infra']=='ok' and r['rand_infra']=='ok') else 'X',
            r['trace_csv_exists']))

    return results

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--summary_dir', required=True)
    ap.add_argument('--launch_csv', default='')
    ap.add_argument('--output_csv', required=True)
    args = ap.parse_args()
    postprocess(args.summary_dir, args.launch_csv, args.output_csv)
