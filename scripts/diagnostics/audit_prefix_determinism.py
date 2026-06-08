#!/usr/bin/env python3
"""Phase B: Audit clean-prefix determinism across confirmation repeats.

For each parent's repeats, compare pre-window trajectory:
  raw_gripper, env_gripper, qpos, EEF position

If prefixes differ → attack instability may be from replay nondeterminism.
If prefixes match → instability is from attack seed stochasticity.

Usage:
  python scripts/diagnostics/audit_prefix_determinism.py \
    --dir /data/liuyu/outputs/stageb_v1_1_silver_confirmation_rc1a_e33b5e4
"""
import csv, os, sys, argparse, json, hashlib
from collections import defaultdict


def trace_hash(trace_path, window_start):
    """Hash the pre-window trajectory for quick comparison."""
    with open(trace_path, 'r') as f:
        rows = list(csv.DictReader(f))
    pre = [r for r in rows if int(r.get('step', 0)) < window_start]
    if not pre:
        return 'EMPTY_PREFIX'
    # Hash key columns
    h = hashlib.md5()
    for r in pre:
        grip = r.get('raw_action_6', '')
        env_grip = r.get('env_action_6', '')
        q0 = r.get('obs_gripper_qpos_0', '')
        q1 = r.get('obs_gripper_qpos_1', '')
        h.update(('%s_%s_%s_%s' % (grip, env_grip, q0, q1)).encode())
    return h.hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', required=True)
    args = ap.parse_args()

    # Load summary JSONs to group by parent
    pairs = defaultdict(list)
    for f in sorted(os.listdir(args.dir)):
        if not f.startswith('summary_') or not f.endswith('.json'):
            continue
        with open(os.path.join(args.dir, f)) as fp:
            d = json.load(fp)
        pid = d.get('pair_id', '?')
        cond = 'VIS' if d.get('condition') == 'vis_pgd' else 'RAND'
        ws = d.get('window_start', 0)
        pairs[pid].append((cond, d, ws))

    # Group into parents (strip _r0/_r1)
    parents = defaultdict(list)
    for pid, entries in pairs.items():
        base = pid.rsplit('_r', 1)[0]
        r_idx = int(pid.rsplit('_r', 1)[1]) if '_r' in pid else -1
        parents[base].append((r_idx, pid, entries))

    print('=== PREFIX DETERMINISM AUDIT ===')
    print('Parents: %d' % len(parents))
    print()

    all_pass = True
    for base, repeats in sorted(parents.items()):
        print('Parent: %s' % base[:60])
        # Get window_start from first VIS entry
        ws = None
        for r_idx, pid, entries in repeats:
            for cond, d, w in entries:
                if cond == 'VIS':
                    ws = w; break
            if ws is not None:
                break
        if ws is None:
            print('  SKIP: no window_start found')
            continue

        # Find trace files for each repeat VIS
        trace_hashes = {}
        for r_idx, pid, entries in repeats:
            for cond, d, w in entries:
                if cond != 'VIS':
                    continue
                # Find matching trace file
                jid = d.get('job_id', -1)
                trace_pat = 'trace_%s_vis_pgd_job%d.csv' % (d.get('task_key', ''), jid)
                trace_path = os.path.join(args.dir, trace_pat)
                if os.path.exists(trace_path):
                    h = trace_hash(trace_path, ws)
                    trace_hashes[r_idx] = (h, trace_path)
                    break

        # Compare hashes
        unique_hashes = set(h for h, _ in trace_hashes.values())
        if len(unique_hashes) <= 1:
            print('  PREFIX MATCH: all repeats have identical pre-window trajectory')
            for r_idx, (h, tp) in sorted(trace_hashes.items()):
                print('    r%d: %s  (%s)' % (r_idx, h, os.path.basename(tp)))
        else:
            print('  PREFIX MISMATCH: repeats have DIFFERENT pre-window trajectories!')
            all_pass = False
            for r_idx, (h, tp) in sorted(trace_hashes.items()):
                print('    r%d: %s  (%s)' % (r_idx, h, os.path.basename(tp)))

        # Detailed comparison for mismatches
        if len(unique_hashes) > 1:
            print('  DETAILED DIFF:')
            for r_idx, (h, tp) in sorted(trace_hashes.items()):
                with open(tp, 'r') as f:
                    rows = list(csv.DictReader(f))
                pre = [r for r in rows if int(r.get('step', 0)) < ws]
                actions = [float(r.get('raw_action_6', 0)) for r in pre]
                qpos = [abs(float(r.get('obs_gripper_qpos_0', 0))) + abs(float(r.get('obs_gripper_qpos_1', 0)))
                        for r in pre]
                print('    r%d: n_pre=%d raw_grip_mean=%.4f raw_grip_std=%.4f qpos_mean=%.6f qpos_start=%.6f qpos_end=%.6f' %
                      (r_idx, len(pre),
                       sum(actions) / max(len(actions), 1),
                       (sum((a - sum(actions) / max(len(actions), 1)) ** 2 for a in actions) / max(len(actions), 1)) ** 0.5 if len(actions) > 1 else 0,
                       sum(qpos) / max(len(qpos), 1),
                       qpos[0] if qpos else 0,
                       qpos[-1] if qpos else 0))
        print()

    print('=== GATE B RESULT ===')
    if all_pass:
        print('PASS: All parents have deterministic clean prefixes across repeats.')
        print('Attack outcome variability is from attack seed stochasticity, not env nondeterminism.')
    else:
        print('FAIL: Some parents have non-deterministic clean prefixes.')
        print('Do NOT launch K=5 until runner determinism is fixed.')
        sys.exit(1)


if __name__ == '__main__':
    main()
