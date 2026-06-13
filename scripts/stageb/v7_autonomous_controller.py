#!/usr/bin/env python3
"""V7 Autonomous Experiment Controller.

Monitors 4-parent mini, auto-advances strong candidates to formal confirmation.
Runs continuously until all GPU pairs complete.
"""

import json, os, sys, time, glob, subprocess, argparse
from collections import defaultdict

# ── Frozen config ──
PARENTS = {
    'butter_s2': ('butter', 2),
    'cream_cheese_s2': ('cream_cheese', 2),
    'bbq_sauce_s0': ('bbq_sauce', 0),
    'chocolate_pudding_s2': ('chocolate_pudding', 2),
}

GPU_PAIRS = {
    'GPU10': {'cuda': '0,1', 'render': 1},
    'GPU26': {'cuda': '2,6', 'render': 2},
    'GPU45': {'cuda': '4,5', 'render': 4},
}

SEEDS_701_703 = [701, 702, 703]
FORMAL_SEEDS = [711, 712, 713, 714, 715, 716, 717, 718, 719, 720, 721, 722]


def load_all_summaries(output_root):
    """Load all summaries from GPU pair directories."""
    summaries = []
    for pair in GPU_PAIRS:
        pair_dir = os.path.join(output_root, pair)
        if not os.path.isdir(pair_dir):
            continue
        for sf in glob.glob(os.path.join(pair_dir, 'summary_*.json')):
            with open(sf) as f:
                s = json.load(f)
            s['_pair'] = pair
            s['_path'] = sf
            summaries.append(s)
    return summaries


def audit_mini(summaries):
    """Audit mini results. Returns per-parent metrics and decision."""
    # Group by parent and condition
    by_parent = defaultdict(lambda: {'clean': [], 'rand': [], 'vis': []})
    for s in summaries:
        pid = '%s_s%d' % (s['task'], s['state_id'])
        cond = s['condition']
        if cond == 'clean_observer':
            by_parent[pid]['clean'].append(s)
        elif cond == 'online_random_linf':
            by_parent[pid]['rand'].append(s)
        elif cond == 'online_vis_pgd':
            by_parent[pid]['vis'].append(s)

    results = {}
    for pid in sorted(by_parent.keys()):
        if pid not in PARENTS:
            continue
        d = by_parent[pid]
        n_clean = len(d['clean'])
        n_rand = len(d['rand'])
        n_vis = len(d['vis'])

        # Only evaluate if all expected rollouts are done
        expected_rand = 3  # seeds 701-703
        expected_vis = 3
        expected_clean = 2

        if n_rand < expected_rand or n_vis < expected_vis or n_clean < expected_clean:
            results[pid] = {
                'status': 'INCOMPLETE',
                'clean': n_clean, 'rand': n_rand, 'vis': n_vis,
            }
            continue

        # Command: open_count_B3 per episode
        vis_open_b3 = [s.get('open_count_B3', 0) for s in d['vis']]
        rand_open_b3 = [s.get('open_count_B3', 0) for s in d['rand']]

        # Paired seeds: match by seed
        vis_by_seed = {s['attack_seed']: s for s in d['vis']}
        rand_by_seed = {s['attack_seed']: s for s in d['rand']}
        paired_wins = 0
        common_seeds = set(vis_by_seed.keys()) & set(rand_by_seed.keys())
        for seed in common_seeds:
            v = vis_by_seed[seed].get('open_count_B3', 0)
            r = rand_by_seed[seed].get('open_count_B3', 0)
            if v > r:
                paired_wins += 1

        sum_vis_b3 = sum(vis_open_b3)
        sum_rand_b3 = sum(rand_open_b3)

        # Command gate
        cmd_gate = paired_wins >= 2 and (sum_vis_b3 - sum_rand_b3) >= 2

        # Physical: qpos_abs_peak_delta
        clean_peaks = [s.get('qpos_abs_peak_delta', 0) or 0 for s in d['clean']]
        rand_peaks = [s.get('qpos_abs_peak_delta', 0) or 0 for s in d['rand']]
        vis_peaks = [s.get('qpos_abs_peak_delta', 0) or 0 for s in d['vis']]
        baseline_peaks = clean_peaks + rand_peaks
        if baseline_peaks and len(baseline_peaks) >= 3:
            import statistics
            median_baseline = statistics.median(baseline_peaks)
            # MAD
            mad = statistics.median([abs(p - median_baseline) for p in baseline_peaks])
            T_phys = median_baseline + max(3 * mad, 0.0001)  # noise floor
        else:
            T_phys = 0.001

        vis_phys_pos = sum(1 for p in vis_peaks if p > T_phys)
        rand_phys_pos = sum(1 for p in rand_peaks if p > T_phys)
        phys_gate = vis_phys_pos >= 2 and rand_phys_pos <= 1

        # Task
        rand_success = sum(1 for s in d['rand'] if s.get('success_primary', False))
        vis_success = sum(1 for s in d['vis'] if s.get('success_primary', False))
        clean_success = sum(1 for s in d['clean'] if s.get('success_primary', False))
        task_gate = (clean_success >= 2 and rand_success >= 2 and vis_success <= 1)

        # Decision
        if cmd_gate and (phys_gate or task_gate):
            decision = 'STRONG_CANDIDATE'
        elif cmd_gate:
            decision = 'COMMAND_ONLY'
        else:
            decision = 'NO_COMMAND_SPECIFICITY'

        results[pid] = {
            'status': 'COMPLETE',
            'clean': n_clean, 'rand': n_rand, 'vis': n_vis,
            'vis_open_b3': vis_open_b3,
            'rand_open_b3': rand_open_b3,
            'paired_wins': paired_wins,
            'sum_vis_b3': sum_vis_b3,
            'sum_rand_b3': sum_rand_b3,
            'cmd_gate': cmd_gate,
            'T_phys': T_phys,
            'vis_phys_pos': vis_phys_pos,
            'rand_phys_pos': rand_phys_pos,
            'phys_gate': phys_gate,
            'rand_success': rand_success,
            'vis_success': vis_success,
            'task_gate': task_gate,
            'decision': decision,
        }

    return results


def launch_formal(parent_pid, gpu_pair, output_root, repo, model):
    """Launch 12-seed formal confirmation for one parent on one GPU pair."""
    task, sid = PARENTS[parent_pid]
    cfg = GPU_PAIRS[gpu_pair]
    py = '/data/aviary/envs/openvla_official_libero_20260525/bin/python'
    runner = os.path.join(repo, 'scripts', 'stageb',
                          'run_s20d_v6_online_trigger_l3_runner.py')
    formal_dir = os.path.join(output_root, 'formal_%s_%s' % (parent_pid, gpu_pair))
    os.makedirs(formal_dir, exist_ok=True)

    jobs = []
    # 12 RAND + 12 VIS + 6 clean, counterbalanced
    for i, seed in enumerate(FORMAL_SEEDS):
        if i % 2 == 0:
            jobs.append(('rand', seed))
            jobs.append(('vis', seed))
        else:
            jobs.append(('vis', seed))
            jobs.append(('rand', seed))

    # Add 6 clean (evenly spaced)
    clean_inserted = 0
    for i in range(0, len(jobs), 4):
        if clean_inserted >= 6:
            break
        jobs.insert(i, ('clean', 0))
        clean_inserted += 1

    print('[FORMAL] Launching %s on %s (%d jobs)' % (parent_pid, gpu_pair, len(jobs)))
    for cond, seed in jobs:
        job_id = 'v7_formal_%s_%s_s%d_%s' % (parent_pid, cond, seed, gpu_pair)
        condition = ('clean_observer' if cond == 'clean'
                     else 'online_random_linf' if cond == 'rand'
                     else 'online_vis_pgd')
        cmd = [py, '-u', runner,
               '--task', task, '--state_id', str(sid),
               '--condition', condition,
               '--max_steps_override', '280',
               '--success_metric', 'check_success', '--num_steps_wait', '10',
               '--model_path', model,
               '--render_gpu_device_id', str(cfg['render']),
               '--model_gpu_device_id', '-1',
               '--output_dir', formal_dir,
               '--job_id', job_id, '--seed', '0',
               '--attack_seed', str(seed),
               '--eps_raw_pixels', '6', '--pgd_steps', '20']
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = cfg['cuda']
        env['OPENVLA_ATTN_IMPLEMENTATION'] = 'eager'
        env['OPENVLA_CUDA_MAX_MEMORY'] = '10000MiB'
        log_path = os.path.join(formal_dir, 'log_%s.txt' % job_id)
        with open(log_path, 'w') as log:
            log.write('CMD: %s\nSTART: %s\n\n' % (
                ' '.join(cmd), time.strftime('%H:%M:%S')))
            log.flush()
            r = subprocess.run(cmd, env=env, capture_output=True, text=True,
                             timeout=900)
            log.write(r.stdout)
            if r.stderr:
                log.write('\nSTDERR:\n%s\n' % r.stderr[-3000:])
            log.write('\nRC=%d\nEND: %s\n' % (r.returncode,
                     time.strftime('%H:%M:%S')))
            if r.returncode != 0:
                print('[FORMAL] FAIL %s RC=%d, continuing...' % (job_id, r.returncode))
    print('[FORMAL] %s on %s DONE' % (parent_pid, gpu_pair))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', required=True)
    parser.add_argument('--output-root', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--poll-interval', type=int, default=120)
    args = parser.parse_args()

    launched_formal = set()

    while True:
        summaries = load_all_summaries(args.output_root)
        n_total = len(summaries)

        # Count per pair
        pair_counts = defaultdict(int)
        for s in summaries:
            pair_counts[s['_pair']] += 1

        results = audit_mini(summaries)

        # Check for completed pairs that can be reassigned
        active_pairs = set()
        for pair in GPU_PAIRS:
            pair_dir = os.path.join(args.output_root, pair)
            # Check if launcher is still running
            launcher_done = not os.path.exists(
                os.path.join(args.output_root, 'launcher.log'))
            # Or check if all expected jobs have summaries
            n = pair_counts.get(pair, 0)
            expected = {'GPU10': 10, 'GPU26': 11, 'GPU45': 11}
            if n >= expected.get(pair, 0):
                # This pair is complete — launch formal if strong candidate
                for pid, r in results.items():
                    if r['status'] == 'COMPLETE' and r['decision'] == 'STRONG_CANDIDATE':
                        key = (pid, pair)
                        if key not in launched_formal:
                            print('\n*** STRONG CANDIDATE: %s ***' % pid)
                            print('  cmd=%s phys=%s task=%s' % (
                                r['cmd_gate'], r['phys_gate'], r['task_gate']))
                            launch_formal(pid, pair, args.output_root,
                                        args.repo, args.model)
                            launched_formal.add(key)

        # Print status
        completed = sum(1 for r in results.values()
                       if r['status'] == 'COMPLETE')
        print('[%s] summaries=%d pairs=%s completed=%d/%d' % (
            time.strftime('%H:%M:%S'), n_total,
            dict(pair_counts), completed, len(PARENTS)))

        for pid, r in sorted(results.items()):
            if r['status'] == 'COMPLETE':
                print('  %s: %s (cmd=%s phys=%s task=%s b3_v=%s b3_r=%s wins=%d)' % (
                    pid, r['decision'], r['cmd_gate'], r['phys_gate'],
                    r['task_gate'], r['vis_open_b3'], r['rand_open_b3'],
                    r['paired_wins']))
            else:
                print('  %s: INCOMPLETE (c=%d r=%d v=%d)' % (
                    pid, r['clean'], r['rand'], r['vis']))

        # Stop condition
        if completed >= len(PARENTS) and not launched_formal:
            # Check if any strong candidate without formal launched
            any_strong = any(
                r['status'] == 'COMPLETE' and r['decision'] == 'STRONG_CANDIDATE'
                for r in results.values())
            if not any_strong:
                print('\nAll parents complete. No strong candidates.')
                print('Final audit complete.')
                break

        time.sleep(args.poll_interval)


if __name__ == '__main__':
    main()
