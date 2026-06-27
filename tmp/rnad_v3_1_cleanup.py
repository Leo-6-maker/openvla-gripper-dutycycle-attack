#!/usr/bin/env python3
"""rNAD v3.1 cleanup: delete old ACTION_STATS_SHA256.txt, generate input manifest."""
import os, json, csv, hashlib
from pathlib import Path

BASE = Path('/mnt/sdc/dty_user/openvla_attack/evidence/phase7_object/metric_refresh_v2')
OUT = Path('/mnt/sdc/dty_user/openvla_attack/reports/phase7_table1/rnad_v3')

# 1. Delete old conflicting checksum file
old_sha = OUT / 'ACTION_STATS_SHA256.txt'
if old_sha.exists():
    old_sha.unlink()
    print('Deleted old ACTION_STATS_SHA256.txt')

# 2. Generate input manifest
rows = []
for cond in sorted(os.listdir(str(BASE))):
    cp = BASE / cond
    if not cp.is_dir(): continue
    for run_dir in sorted(os.listdir(str(cp))):
        rp = cp / run_dir
        tele_path = rp / 'step_telemetry.csv'
        summ_path = rp / 'episode_summary.json'
        video_path = rp / 'rollout_raw.mp4'
        comp_path = rp / 'COMPLETE.json'

        tele_sha = hashlib.sha256(tele_path.read_bytes()).hexdigest() if tele_path.is_file() else 'MISSING'
        summ_sha = hashlib.sha256(summ_path.read_bytes()).hexdigest() if summ_path.is_file() else 'MISSING'
        video_sha = hashlib.sha256(video_path.read_bytes()).hexdigest() if video_path.is_file() else 'MISSING'
        comp_sha = hashlib.sha256(comp_path.read_bytes()).hexdigest() if comp_path.is_file() else 'MISSING'

        with open(summ_path) as f: s = json.load(f)
        rows.append({
            'condition': cond, 'run_dir': run_dir,
            'task_idx': s.get('task_idx', ''), 'state_id': s.get('state_id', ''),
            'perturbation_seed': s.get('perturbation_seed', ''),
            'step_telemetry_sha256': tele_sha,
            'episode_summary_sha256': summ_sha,
            'video_sha256': video_sha,
            'COMPLETE_sha256': comp_sha,
        })

manifest_path = OUT / 'RNAD_V3_INPUT_MANIFEST.csv'
with open(manifest_path, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)

# Add input manifest SHA to output SHA256SUMS
sha_out = OUT / 'RNAD_V3_OUTPUT_SHA256SUMS.txt'
manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
with open(sha_out, 'a') as f:
    f.write('{}  RNAD_V3_INPUT_MANIFEST.csv\n'.format(manifest_sha))
# Also add the script SHA
script_sha = hashlib.sha256(Path('/tmp/rnad_v3_1_final.py').read_bytes()).hexdigest()
with open(sha_out, 'a') as f:
    f.write('{}  rnad_v3_1_final.py\n'.format(script_sha))

print('Input manifest: {} rows'.format(len(rows)))
print('Manifest SHA: {}'.format(manifest_sha))
print('Done.')
