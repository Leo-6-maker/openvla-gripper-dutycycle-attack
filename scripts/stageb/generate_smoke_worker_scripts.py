#!/usr/bin/env python3
"""Generate 3-worker shell scripts for 6-parent / 12-job smoke."""
import os

# ── Config ──
PY = '/home/liuyu/.conda/envs/openvla_official_libero_20260525/bin/python'
SCRIPT = '/data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607/scripts/run_stageb_vis_labeling.py'
OUT_DIR = '/data/liuyu/outputs/stageb_v1_1_targeted_expansion_smoke_rc1a_d4a3827'

WORKERS = {
    'worker_10': {'gpu': '1,0', 'pair': '0,1'},
    'worker_26': {'gpu': '2,6', 'pair': '0,1'},
    'worker_45': {'gpu': '4,5', 'pair': '0,1'},
}

# 6 smoke parents (VIS + RAND each = 12 jobs)
# Format: (task, state_id, seed, window_start, window_end, max_steps, category)
SMOKE = [
    ('alphabet_soup', 1, 1, 50, 60, 400, 'cmd_expansion'),
    ('bbq_sauce', 1, 1, 55, 65, 400, 'cmd_expansion'),
    ('cream_cheese', 2, 2, 50, 60, 400, 'phys_enrichment'),
    ('orange_juice', 2, 2, 20, 30, 400, 'phys_enrichment'),
    ('bbq_sauce', 2, 2, 100, 110, 400, 'hard_negative'),
    ('tomato_sauce', 2, 2, 90, 100, 400, 'rand_abstain'),
]

# Assign 2 parents per worker (4 jobs: VIS+RAND for each)
worker_parents = {'worker_10': SMOKE[0:2], 'worker_26': SMOKE[2:4], 'worker_45': SMOKE[4:6]}

os.makedirs(OUT_DIR, exist_ok=True)  # mkdir on server? No, this is local. Let me create dir via SSH.

job_id_base = 200000  # distinct range for smoke

for wname, wcfg in WORKERS.items():
    parents = worker_parents[wname]
    lines = []
    lines.append('#!/bin/bash')
    lines.append('# Smoke worker: %s GPU=%s' % (wname, wcfg['gpu']))
    lines.append('# %d parents, %d jobs' % (len(parents), len(parents) * 2))
    lines.append('set +e')
    lines.append('')
    lines.append('export CUDA_VISIBLE_DEVICES=%s' % wcfg['gpu'])
    lines.append('')
    lines.append('echo "[$(date +%%H:%%M:%%S)] %s SMOKE START: %d parents"' % (wname, len(parents)))
    lines.append('')

    for i, (task, sid, seed, ws, we, max_s, cat) in enumerate(parents):
        pair_id = 'smoke_%s_%s_s%d_w%d_%d_seed%d' % (cat, task, sid, ws, we, seed)
        vis_jid = job_id_base + 2 * i + 1
        rand_jid = job_id_base + 2 * i + 2

        lines.append('echo "=== VIS %d: %s s%d [%d,%d] seed=%d %s ==="' % (vis_jid, task, sid, ws, we, seed, cat))
        lines.append('%s -u %s \\' % (PY, SCRIPT))
        lines.append('  --gpu_pair %s \\' % wcfg['pair'])
        lines.append('  --task %s --state-id %d --window_start %d --window_end %d \\' % (task, sid, ws, we))
        lines.append('  --condition vis_pgd --pgd_steps 20 --eps_raw_pixels 6 \\')
        lines.append('  --max_steps %d --seed %d \\' % (max_s, seed))
        lines.append('  --job_id %d --pair_id %s \\' % (vis_jid, pair_id))
        lines.append('  --output_dir %s \\' % OUT_DIR)
        lines.append('  --image_preprocess official_rot180 \\')
        lines.append('  || echo "VIS_FAIL %d %s"' % (vis_jid, pair_id))
        lines.append('')

        lines.append('echo "=== RAND %d: %s s%d [%d,%d] seed=%d %s ==="' % (rand_jid, task, sid, ws, we, seed, cat))
        lines.append('%s -u %s \\' % (PY, SCRIPT))
        lines.append('  --gpu_pair %s \\' % wcfg['pair'])
        lines.append('  --task %s --state-id %d --window_start %d --window_end %d \\' % (task, sid, ws, we))
        lines.append('  --condition random_linf --eps_raw_pixels 6 \\')
        lines.append('  --max_steps %d --seed %d \\' % (max_s, seed))
        lines.append('  --job_id %d --pair_id %s \\' % (rand_jid, pair_id))
        lines.append('  --output_dir %s \\' % OUT_DIR)
        lines.append('  --image_preprocess official_rot180 \\')
        lines.append('  || echo "RAND_FAIL %d %s"' % (rand_jid, pair_id))
        lines.append('')

        job_id_base += 2

    lines.append('echo "[$(date +%%H:%%M:%%S)] %s SMOKE DONE"' % wname)
    lines.append('')

    script_path = 'scripts/stageb/run_smoke_%s.sh' % wname
    with open(script_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('Generated %s (%d parents -> %d jobs)' % (script_path, len(parents), len(parents)*2))

print('\nOutput dir: %s' % OUT_DIR)
print('\nUpload + launch:')
for wname in ['worker_10', 'worker_26', 'worker_45']:
    local = 'scripts/stageb/run_smoke_%s.sh' % wname
    remote = '/data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607/%s' % local
    print('  cat %s | ssh vla "cat > %s"' % (local, remote))
    print('  ssh vla "mkdir -p %s && nohup bash %s > %s/%s.log 2>&1 &"' % (OUT_DIR, remote, OUT_DIR, wname))
