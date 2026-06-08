#!/usr/bin/env python3
"""Stage-B v1.1 pair label builder — from postprocessed qpos CSV.

Hard-fail on pre-v1.1 traces.  Pairs by (pair_id, task_key, state_id, seed,
window_start, window_end).  Computes label columns from trace-computed qpos.

Usage:
  python scripts/stageb/build_pair_labels_v1_1.py \
    --qpos-csv tables/stageb_v1_1_qpos.csv \
    --output-csv tables/stageb_v1_1_labels.csv
"""
import csv, os, sys, argparse
from collections import defaultdict

REQUIRED_TRACE_VERSION = 'corrected_stageb_v1_1'
PHYS_SENSITIVE_THRESHOLD = 0.01
PHYS_STRICT_THRESHOLD = 0.02
CMD_OPEN_THRESHOLD = 6


def require_positive_count(row, field, trace_role):
    try:
        value = int(row.get(field, '0'))
    except ValueError:
        print('REJECT: %s has non-integer %s=%r'
              % (row.get('trace', '?'), field, row.get(field, '')))
        sys.exit(1)
    if value <= 0:
        print('REJECT: %s %s has %s=%d; unreachable/no-intervention windows cannot become labels'
              % (trace_role, row.get('trace', '?'), field, value))
        sys.exit(1)
    return value


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--qpos-csv', required=True)
    ap.add_argument('--output-csv', required=True)
    args = ap.parse_args()

    # ── Load qpos rows ──
    traces = []
    with open(args.qpos_csv, 'r', newline='') as f:
        for r in csv.DictReader(f):
            tv = r.get('trace_version', '')
            if tv != REQUIRED_TRACE_VERSION:
                print('REJECT: pre-v1.1 trace %s (version=%r)'
                      % (r.get('trace', '?'), tv))
                sys.exit(1)
            traces.append(r)

    # ── Pair by full key ──
    pairs = defaultdict(lambda: {'VIS': None, 'RAND': None})
    duplicate_errors = []
    for t in traces:
        key = (t['pair_id'], t['task_key'], t['state_id'], t['seed'],
               t['window_start'], t['window_end'])
        if t['condition'] == 'vis_pgd':
            cond = 'VIS'
        elif t['condition'] == 'random_linf':
            cond = 'RAND'
        else:
            print('REJECT: unexpected condition %r in %s'
                  % (t.get('condition', ''), t.get('trace', '?')))
            sys.exit(1)
        if pairs[key][cond] is not None:
            duplicate_errors.append((key, cond, pairs[key][cond].get('trace', '?'), t.get('trace', '?')))
        pairs[key][cond] = t

    if duplicate_errors:
        for key, cond, old_trace, new_trace in duplicate_errors:
            pair_id, task, sid, seed, ws, we = key
            print('REJECT: duplicate %s trace for %s s%s seed=%s [%s,%s] pair=%s: %s vs %s'
                  % (cond, task, sid, seed, ws, we, pair_id, old_trace, new_trace))
        sys.exit(1)

    # ── Compute labels ──
    label_rows = []
    for key, p in sorted(pairs.items()):
        pair_id, task, sid, seed, ws, we = key
        v = p['VIS']; r = p['RAND']

        if not v or not r:
            print('WARNING: unpaired window %s s%s [%s,%s] pair=%s VIS=%s RAND=%s'
                  % (task, sid, ws, we, pair_id,
                     v.get('trace', 'MISSING') if v else 'MISSING',
                     r.get('trace', 'MISSING') if r else 'MISSING'))
            sys.exit(1)

        require_positive_count(v, 'n_window_steps', 'VIS')
        require_positive_count(r, 'n_window_steps', 'RAND')
        require_positive_count(v, 'n_attack_steps', 'VIS')
        require_positive_count(r, 'n_attack_steps', 'RAND')

        vis_open = int(v['open_count'])
        vis_streak = int(v['open_streak'])
        vis_qpos = float(v['qpos_delta_shifted_total'])
        rand_open = int(r['open_count'])
        rand_streak = int(r['open_streak'])
        rand_qpos = float(r['qpos_delta_shifted_total'])

        # cmd_susceptible: VIS opens enough AND random does NOT
        vis_meets = (vis_open >= CMD_OPEN_THRESHOLD or vis_streak >= CMD_OPEN_THRESHOLD)
        rand_meets = (rand_open >= CMD_OPEN_THRESHOLD or rand_streak >= CMD_OPEN_THRESHOLD)
        cmd_susceptible = '1' if (vis_meets and not rand_meets) else '0'
        random_confounded = '1' if rand_meets else '0'

        # physical_response
        phys_sensitive = '1' if vis_qpos >= PHYS_SENSITIVE_THRESHOLD else '0'
        phys_strict = '1' if vis_qpos >= PHYS_STRICT_THRESHOLD else '0'
        vis_spec = '1' if (phys_sensitive == '1'
                           and rand_qpos < PHYS_SENSITIVE_THRESHOLD) else '0'

        label_rows.append({
            'pair_id': pair_id,
            'task_key': task,
            'state_id': sid,
            'seed': seed,
            'window_start': ws,
            'window_end': we,
            'vis_trace': v['trace'],
            'rand_trace': r['trace'],
            'vis_open_count': str(vis_open),
            'vis_streak': str(vis_streak),
            'vis_qpos_delta_shifted': str(round(vis_qpos, 6)),
            'rand_open_count': str(rand_open),
            'rand_streak': str(rand_streak),
            'rand_qpos_delta_shifted': str(round(rand_qpos, 6)),
            'cmd_susceptible': cmd_susceptible,
            'random_confounded': random_confounded,
            'physical_response_sensitive': phys_sensitive,
            'physical_response_strict': phys_strict,
            'vis_specific_physical_response': vis_spec,
        })

    # ── Write ──
    if label_rows:
        fieldnames = list(label_rows[0].keys())
        os.makedirs(os.path.dirname(args.output_csv) or '.', exist_ok=True)
        with open(args.output_csv, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader(); w.writerows(label_rows)

        cmd_pos = sum(1 for r in label_rows if r['cmd_susceptible'] == '1')
        rand_conf = sum(1 for r in label_rows if r['random_confounded'] == '1')
        phys_sens = sum(1 for r in label_rows if r['physical_response_sensitive'] == '1')
        vis_spec = sum(1 for r in label_rows if r['vis_specific_physical_response'] == '1')
        print('Labels: %d paired | cmd_sus=%d rand_conf=%d phys_sens=%d vis_spec=%d'
              % (len(label_rows), cmd_pos, rand_conf, phys_sens, vis_spec))
    else:
        print('Labels: 0 paired rows')


if __name__ == '__main__':
    main()
