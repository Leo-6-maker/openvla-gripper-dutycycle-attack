#!/usr/bin/env python3
"""Rebuild retry audit CSV from static manifest + retry summaries."""
import csv, json, glob, os

DIR = '/data/liuyu/outputs/stageb_v1_1_k5c_targeted_expansion_rc1a_ca3a97e'
REPO = '/data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607'

MANIFEST = {
    520060: {'pair_id': 'k5c_cmd_milk_neg', 'condition': 'vis_pgd', 'attack_seed': 0,
             'task': 'milk', 'state_id': 0, 'window_start': 235, 'window_end': 245,
             'original_n_steps': 116, 'original_error': 'CUDA illegal memory access',
             'original_gpu_pair': '2,6', 'retry_gpu_pair': '1,0'},
    520065: {'pair_id': 'k5c_cmd_milk_neg', 'condition': 'random_linf', 'attack_seed': 2,
             'task': 'milk', 'state_id': 0, 'window_start': 235, 'window_end': 245,
             'original_n_steps': 123, 'original_error': 'CUDA illegal memory access',
             'original_gpu_pair': '2,6', 'retry_gpu_pair': '4,5'},
    520090: {'pair_id': 'k5c_cmd_alpha', 'condition': 'vis_pgd', 'attack_seed': 0,
             'task': 'alphabet_soup', 'state_id': 0, 'window_start': 65, 'window_end': 75,
             'original_n_steps': 70, 'original_error': 'CUDA illegal memory access',
             'original_gpu_pair': '2,6', 'retry_gpu_pair': '4,5'},
}

retry_data = {}
for f in glob.glob(os.path.join(DIR, 'summary_*.json')):
    with open(f) as fh:
        j = json.load(fh)
    jid = j.get('job_id')
    if jid in MANIFEST and j.get('infra_status') == 'ok':
        retry_data[jid] = j

path = os.path.join(REPO, 'tables/stageb_v1_1_k5c_retry_audit_rc1a_ca3a97e.csv')
with open(path, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['job_id','pair_id','condition','attack_seed','task','window',
                'original_gpu_pair','original_n_steps','original_error',
                'retry_gpu_pair','retry_n_steps','retry_open_count','retry_infra','retry_valid'])
    for jid in sorted(MANIFEST):
        m = MANIFEST[jid]
        rj = retry_data.get(jid, {})
        retry_n = rj.get('n_total_steps', '?')
        retry_open = rj.get('decoded_open_count', '?')
        retry_infra = rj.get('infra_status', '?')
        we = m['window_end']
        retry_valid = (retry_infra == 'ok' and isinstance(retry_n, int) and retry_n >= we + 5)
        win = '{}_{}_{}'.format(m['window_start'], m['window_end'], m['state_id'])
        w.writerow([jid, m['pair_id'], m['condition'], m['attack_seed'],
                    m['task'], win,
                    m['original_gpu_pair'], m['original_n_steps'], m['original_error'],
                    m['retry_gpu_pair'], retry_n, retry_open, retry_infra, retry_valid])

print('Retry audit rebuilt:')
with open(path) as f:
    for line in f:
        print(line.strip())
