#!/usr/bin/env python3
"""V7 4-parent harm-sensitive mini launcher.

Parents: butter_s2, cream_cheese_s2, bbq_sauce_s0, chocolate_pudding_s2
Seeds: 701, 702, 703
Total: 32 rollouts across GPU10/26/45
"""

import argparse, csv, os, subprocess, sys, threading, time

def run_job(gpu_pair, cuda_vis, render_gpu, job_id, task, state_id, condition,
            attack_seed, output_dir, repo, model_path, py):
    runner = os.path.join(repo, 'scripts', 'stageb',
                          'run_s20d_v6_online_trigger_l3_runner.py')
    cmd = [py, '-u', runner,
        '--task', task, '--state_id', str(state_id),
        '--condition', condition,
        '--max_steps_override', '280',
        '--success_metric', 'check_success', '--num_steps_wait', '10',
        '--model_path', model_path,
        '--render_gpu_device_id', str(render_gpu),
        '--model_gpu_device_id', '-1',
        '--output_dir', output_dir,
        '--job_id', job_id, '--seed', '0',
        '--attack_seed', str(attack_seed),
        '--eps_raw_pixels', '6', '--pgd_steps', '20']
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = cuda_vis
    env['OPENVLA_ATTN_IMPLEMENTATION'] = 'eager'
    env['OPENVLA_CUDA_MAX_MEMORY'] = '10000MiB'

    log_path = os.path.join(output_dir, 'log_%s.txt' % job_id)
    with open(log_path, 'w') as log:
        log.write('CMD: %s\nCUDA=%s\nSTART: %s\n\n' % (
            ' '.join(cmd), cuda_vis, time.strftime('%H:%M:%S')))
        log.flush()
        r = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=900)
        log.write(r.stdout)
        if r.stderr:
            log.write('\nSTDERR:\n%s\n' % r.stderr[-3000:])
        log.write('\nRC=%d\nEND: %s\n' % (r.returncode, time.strftime('%H:%M:%S')))
    return job_id, r.returncode, log_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', required=True)
    parser.add_argument('--output-root', required=True)
    parser.add_argument('--model', required=True)
    args = parser.parse_args()
    py = '/data/aviary/envs/openvla_official_libero_20260525/bin/python'

    pairs = {
        'GPU10': {'cuda': '0,1', 'render': 1},
        'GPU26': {'cuda': '2,6', 'render': 2},
        'GPU45': {'cuda': '4,5', 'render': 4},
    }

    parents = [
        ('butter', 2),
        ('cream_cheese', 2),
        ('bbq_sauce', 0),
        ('chocolate_pudding', 2),
    ]

    # GPU10: RAND701+VIS701 for all 4 parents + cream_cheese clean0 + chocolate clean1
    # GPU26: RAND702+VIS702 for all 4 parents + butter clean1 + cream clean1 + bbq clean0
    # GPU45: RAND703+VIS703 for all 4 parents + butter clean0 + bbq clean1 + chocolate clean0
    # Order within pair alternates: butter(R→V), cream(V→R), bbq(R→V), choc(V→R)

    gpu_jobs = {}

    # GPU10: seeds 701
    gpu_jobs['GPU10'] = [
        ('v7_butter_rand701', 'butter', 2, 'online_random_linf', 701),
        ('v7_butter_vis701', 'butter', 2, 'online_vis_pgd', 701),
        ('v7_cream_vis701', 'cream_cheese', 2, 'online_vis_pgd', 701),
        ('v7_cream_rand701', 'cream_cheese', 2, 'online_random_linf', 701),
        ('v7_bbq_rand701', 'bbq_sauce', 0, 'online_random_linf', 701),
        ('v7_bbq_vis701', 'bbq_sauce', 0, 'online_vis_pgd', 701),
        ('v7_choc_vis701', 'chocolate_pudding', 2, 'online_vis_pgd', 701),
        ('v7_choc_rand701', 'chocolate_pudding', 2, 'online_random_linf', 701),
        ('v7_cream_clean0', 'cream_cheese', 2, 'clean_observer', 0),
        ('v7_choc_clean1', 'chocolate_pudding', 2, 'clean_observer', 0),
    ]

    # GPU26: seeds 702
    gpu_jobs['GPU26'] = [
        ('v7_butter_vis702', 'butter', 2, 'online_vis_pgd', 702),
        ('v7_butter_rand702', 'butter', 2, 'online_random_linf', 702),
        ('v7_cream_rand702', 'cream_cheese', 2, 'online_random_linf', 702),
        ('v7_cream_vis702', 'cream_cheese', 2, 'online_vis_pgd', 702),
        ('v7_bbq_vis702', 'bbq_sauce', 0, 'online_vis_pgd', 702),
        ('v7_bbq_rand702', 'bbq_sauce', 0, 'online_random_linf', 702),
        ('v7_choc_rand702', 'chocolate_pudding', 2, 'online_random_linf', 702),
        ('v7_choc_vis702', 'chocolate_pudding', 2, 'online_vis_pgd', 702),
        ('v7_butter_clean1', 'butter', 2, 'clean_observer', 0),
        ('v7_cream_clean1', 'cream_cheese', 2, 'clean_observer', 0),
        ('v7_bbq_clean0', 'bbq_sauce', 0, 'clean_observer', 0),
    ]

    # GPU45: seeds 703
    gpu_jobs['GPU45'] = [
        ('v7_butter_rand703', 'butter', 2, 'online_random_linf', 703),
        ('v7_butter_vis703', 'butter', 2, 'online_vis_pgd', 703),
        ('v7_cream_vis703', 'cream_cheese', 2, 'online_vis_pgd', 703),
        ('v7_cream_rand703', 'cream_cheese', 2, 'online_random_linf', 703),
        ('v7_bbq_rand703', 'bbq_sauce', 0, 'online_random_linf', 703),
        ('v7_bbq_vis703', 'bbq_sauce', 0, 'online_vis_pgd', 703),
        ('v7_choc_vis703', 'chocolate_pudding', 2, 'online_vis_pgd', 703),
        ('v7_choc_rand703', 'chocolate_pudding', 2, 'online_random_linf', 703),
        ('v7_butter_clean0', 'butter', 2, 'clean_observer', 0),
        ('v7_bbq_clean1', 'bbq_sauce', 0, 'clean_observer', 0),
        ('v7_choc_clean0', 'chocolate_pudding', 2, 'clean_observer', 0),
    ]

    os.makedirs(args.output_root, exist_ok=True)
    for pair_name in pairs:
        os.makedirs(os.path.join(args.output_root, pair_name), exist_ok=True)

    results = {}
    lock = threading.Lock()

    def run_pair(pair_name):
        cfg = pairs[pair_name]
        pair_out = os.path.join(args.output_root, pair_name)
        for job_id, task, sid, cond, seed in gpu_jobs[pair_name]:
            jid, rc, log = run_job(
                pair_name, cfg['cuda'], cfg['render'],
                job_id, task, sid, cond, seed,
                pair_out, args.repo, args.model, py)
            with lock:
                results[jid] = rc
            if rc != 0:
                print('FAIL %s RC=%d' % (jid, rc))
                break
            print('PASS %s' % jid)

    threads = []
    for pn in ['GPU10', 'GPU26', 'GPU45']:
        t = threading.Thread(target=run_pair, args=(pn,))
        t.start(); threads.append(t)
        print('Started %s (%d jobs)' % (pn, len(gpu_jobs[pn])))

    for t in threads:
        t.join()

    n_pass = sum(1 for rc in results.values() if rc == 0)
    n_fail = sum(1 for rc in results.values() if rc != 0)
    print('\n=== MINI DONE: %d PASS, %d FAIL ===' % (n_pass, n_fail))
    for jid, rc in sorted(results.items()):
        print('  %s: RC=%d' % (jid, rc))


if __name__ == '__main__':
    main()
