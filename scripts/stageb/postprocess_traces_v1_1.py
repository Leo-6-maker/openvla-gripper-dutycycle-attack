#!/usr/bin/env python3
"""Stage-B v1.1 trace postprocessor — recompute qpos from trace CSVs.

Hard-fail on old trace_version.  Reads trace CSV (never summary qpos_delta).
Uses pair_id + task_key + state_id + seed + window_start + window_end for pairing.

Usage:
  python scripts/stageb/postprocess_traces_v1_1.py \
    --input-dir /path/to/outputs --output-csv tables/stageb_v1_1_qpos.csv
"""
import csv, os, sys, argparse, json, glob

REQUIRED_TRACE_VERSION = 'corrected_stageb_v1_1'
VALID_OPEN_CONVENTION = 'env_action_6_lt_neg_0p5_means_OPEN'


def is_open_env(ea6):
    return float(ea6) < -0.5


def abs_sum(q0, q1):
    return abs(float(q0)) + abs(float(q1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input-dir', required=True)
    ap.add_argument('--output-csv', required=True)
    ap.add_argument('--max-traces', type=int, default=0)
    args = ap.parse_args()

    trace_files = sorted(glob.glob(os.path.join(args.input_dir, 'trace_*.csv')))
    if args.max_traces > 0:
        trace_files = trace_files[:args.max_traces]

    qpos_rows = []
    rejected = 0

    for tf in trace_files:
        with open(tf, 'r', newline='') as f:
            reader = csv.DictReader(f)
            cols = list(reader.fieldnames)
            rows = list(reader)

        # ── Hard-fail checks ──
        tv = rows[0].get('trace_version', '') if rows else ''
        if str(tv) != REQUIRED_TRACE_VERSION:
            print('REJECT %s: trace_version=%r (require %s)'
                  % (os.path.basename(tf), tv, REQUIRED_TRACE_VERSION))
            rejected += 1
            continue

        oc = rows[0].get('open_convention', '')
        if oc != VALID_OPEN_CONVENTION:
            print('REJECT %s: open_convention=%r (require %s)'
                  % (os.path.basename(tf), oc, VALID_OPEN_CONVENTION))
            rejected += 1
            continue

        # ── Metadata from trace (not filename) ──
        pair_id = rows[0].get('pair_id', '')
        condition = rows[0].get('condition', '')
        task_key = rows[0].get('task_key', '')
        state_id = rows[0].get('state_id', '')
        seed = rows[0].get('seed', '')
        ws = rows[0].get('window_start', '')
        we = rows[0].get('window_end', '')

        # ── Build step_dict ──
        step_dict = {}
        for r in rows:
            s = int(r.get('step', -1))
            if s >= 0:
                step_dict[s] = r

        # ── Find attack steps ──
        in_win = [r for r in rows if r.get('in_window') == '1']
        att_steps = sorted(int(r['step']) for r in in_win
                           if r.get('attack_this_step') == '1')

        open_count = sum(1 for r in in_win if is_open_env(r.get('env_action_6', '99')))
        streak = 0; max_streak = 0
        for r in in_win:
            if is_open_env(r.get('env_action_6', '99')):
                streak += 1; max_streak = max(max_streak, streak)
            else:
                streak = 0

        # ── Shifted qpos deltas ──
        qpos_deltas = []
        for s in att_steps:
            if s not in step_dict or s + 1 not in step_dict:
                continue
            r_b = step_dict[s]; r_a = step_dict[s + 1]
            q0b = float(r_b.get('obs_gripper_qpos_0', 0))
            q1b = float(r_b.get('obs_gripper_qpos_1', 0))
            q0a = float(r_a.get('obs_gripper_qpos_0', 0))
            q1a = float(r_a.get('obs_gripper_qpos_1', 0))
            abs_b = abs_sum(q0b, q1b)
            abs_a = abs_sum(q0a, q1a)
            qpos_deltas.append(abs_a - abs_b)

        total_delta = sum(qpos_deltas) if qpos_deltas else 0.0
        max_delta = max(qpos_deltas) if qpos_deltas else 0.0

        qpos_rows.append({
            'trace': os.path.basename(tf),
            'pair_id': pair_id,
            'condition': condition,
            'task_key': task_key,
            'state_id': state_id,
            'seed': seed,
            'window_start': ws,
            'window_end': we,
            'trace_version': tv,
            'open_count': str(open_count),
            'open_streak': str(max_streak),
            'qpos_delta_shifted_total': str(round(total_delta, 6)),
            'qpos_delta_shifted_max': str(round(max_delta, 6)),
            'n_attack_steps': str(len(att_steps)),
            'n_window_steps': str(len(in_win)),
            'n_shifted_pairs': str(len(qpos_deltas)),
        })

    # ── Write output ──
    if qpos_rows:
        fieldnames = list(qpos_rows[0].keys())
        os.makedirs(os.path.dirname(args.output_csv) or '.', exist_ok=True)
        with open(args.output_csv, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader(); w.writerows(qpos_rows)
        print('Qpos: %d traces, %d rejected, wrote %s'
              % (len(qpos_rows) + rejected, rejected, args.output_csv))
    else:
        print('Qpos: no valid traces, %d rejected' % rejected)

    if rejected > 0:
        print('HARD_FAIL: %d old-format traces rejected' % rejected)
        sys.exit(1)


if __name__ == '__main__':
    main()
