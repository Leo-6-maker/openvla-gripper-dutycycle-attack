#!/usr/bin/env python3
"""CQFR v3.1: state check, all mtimes fixed, zip members verified, detached SHA, no dangerous steps."""
import os, json, csv, hashlib, shutil, random, subprocess, time

random.seed(42)
FIXED_MTIME = 946684800  # 2000-01-01 UTC

SRC = '/mnt/sdc/dty_user/openvla_attack/evidence/phase7_object/metric_refresh_v2'
OUT = '/mnt/sdc/dty_user/openvla_attack/evidence/phase7_table1/cqfr_v3_unique68'
PUB = os.path.join(OUT, 'public')
PRV = os.path.join(OUT, 'private')
for d in [OUT, PUB, PRV]: os.makedirs(d, exist_ok=True)

TASK_INSTRUCTIONS = {
    'pick_up_the_alphabet_soup_and_place_it_in_the_basket': 'Pick up the alphabet soup and place it in the basket.',
    'pick_up_the_bbq_sauce_and_place_it_in_the_basket': 'Pick up the BBQ sauce and place it in the basket.',
    'pick_up_the_ketchup_and_place_it_in_the_basket': 'Pick up the ketchup and place it in the basket.',
    'pick_up_the_milk_and_place_it_in_the_basket': 'Pick up the milk and place it in the basket.',
    'pick_up_the_salad_dressing_and_place_it_in_the_basket': 'Pick up the salad dressing and place it in the basket.',
    'pick_up_the_orange_juice_and_place_it_in_the_basket': 'Pick up the orange juice and place it in the basket.',
    'pick_up_the_tomato_sauce_and_place_it_in_the_basket': 'Pick up the tomato sauce and place it in the basket.',
    'pick_up_the_butter_and_place_it_in_the_basket': 'Pick up the butter and place it in the basket.',
}

# Collect all 108 runs
runs = []
for cond in sorted(os.listdir(SRC)):
    cp = os.path.join(SRC, cond)
    if not os.path.isdir(cp): continue
    for run_dir in sorted(os.listdir(cp)):
        rp = os.path.join(cp, run_dir)
        video_path = os.path.join(rp, 'rollout_raw.mp4')
        summ_path = os.path.join(rp, 'episode_summary.json')
        if not os.path.isfile(video_path) or not os.path.isfile(summ_path): continue
        with open(summ_path) as f: s = json.load(f)
        video_sha = hashlib.sha256(open(video_path, 'rb').read()).hexdigest()
        task_name = s.get('task_name', s.get('task', ''))
        instruction = TASK_INSTRUCTIONS.get(task_name, task_name)
        runs.append({
            'condition': cond, 'run_dir': run_dir, 'video_path': video_path,
            'video_sha256': video_sha,
            'task_success': s.get('task_success', None),
            'task_name': task_name, 'task_instruction': instruction,
            'state_id': s.get('state_id', ''), 'perturbation_seed': s.get('perturbation_seed', ''),
            'arm_lock': s.get('arm_lock', False), 'objective_id': s.get('objective_id', ''),
            'attack_frames': s.get('attack_frames', 0), 'n_steps': s.get('n_steps', 0),
        })

print('Scientific runs: {}'.format(len(runs)))

# Group by video hash
hash_groups = {}
for r in runs:
    h = r['video_sha256']
    hash_groups.setdefault(h, []).append(r)

unique_hashes = sorted(hash_groups.keys())
print('Unique video hashes: {}'.format(len(unique_hashes)))

# Duplicate-group consistency checks
dup_groups = {h: v for h, v in hash_groups.items() if len(v) > 1}
dup_audit = []
conflicting = {'task': 0, 'instruction': 0, 'success': 0, 'state': 0}
for h, group in dup_groups.items():
    tasks = set(r['task_name'] for r in group)
    instrs = set(r['task_instruction'] for r in group)
    succs = set(r['task_success'] for r in group)
    states = set(r['state_id'] for r in group)
    ct = 1 if len(tasks) > 1 else 0; ci = 1 if len(instrs) > 1 else 0
    cs = 1 if len(succs) > 1 else 0; cst = 1 if len(states) > 1 else 0
    conflicting['task'] += ct; conflicting['instruction'] += ci
    conflicting['success'] += cs; conflicting['state'] += cst
    dup_audit.append({
        'video_sha256': h, 'n_runs': len(group),
        'conflict_task': ct, 'conflict_instruction': ci,
        'conflict_success': cs, 'conflict_state': cst,
        'tasks': ';'.join(sorted(tasks)), 'states': ';'.join(str(s) for s in sorted(states)),
        'successes': ';'.join(str(s) for s in sorted(succs)),
        'run_dirs': ';'.join(r['run_dir'] for r in group),
    })

print('Duplicate groups: {} (conflicts: task={} instr={} success={} state={})'.format(
    len(dup_groups), conflicting['task'], conflicting['instruction'],
    conflicting['success'], conflicting['state']))

# Global shuffle
shuffled_hashes = list(unique_hashes)
random.shuffle(shuffled_hashes)

# Copy unique videos to public/
blind_key = []; run_mapping = []
for i, vhash in enumerate(shuffled_hashes):
    blind_id = 'U{:04d}'.format(i + 1)
    src_run = hash_groups[vhash][0]
    src_video = src_run['video_path']
    dst_video = os.path.join(PUB, blind_id + '.mp4')
    shutil.copyfile(src_video, dst_video)
    os.utime(dst_video, (FIXED_MTIME, FIXED_MTIME))

    # ffprobe with check
    try:
        result = subprocess.run(['ffprobe', '-v', 'quiet', '-print_format', 'json',
            '-show_format', '-show_streams', dst_video],
            capture_output=True, text=True, timeout=15, check=True)
        probe = json.loads(result.stdout)
        vstream = next((s for s in probe.get('streams', []) if s.get('codec_type') == 'video'), {})
        actual_fps = vstream.get('r_frame_rate', '?')
        actual_frames = int(vstream.get('nb_frames', 0) or 0)
        actual_duration = float(probe.get('format', {}).get('duration', 0) or 0)
        codec = vstream.get('codec_name', '?')
        width = vstream.get('width', 0); height = vstream.get('height', 0)
    except Exception as e:
        print('FATAL: ffprobe failed for {}: {}'.format(blind_id, e))
        raise

    source_runs = hash_groups[vhash]
    succ_values = set(sr['task_success'] for sr in source_runs)

    bk = {
        'blind_id': blind_id, 'unique_video_sha256': vhash,
        'n_source_runs': len(source_runs),
        'condition_representative': src_run['condition'],
        'arm_lock_representative': src_run['arm_lock'],
        'task_name': src_run['task_name'],
        'task_instruction': src_run['task_instruction'],
        'task_success_all_same': len(succ_values) == 1,
        'task_success_representative': src_run['task_success'],
        'actual_fps': actual_fps, 'actual_frames': actual_frames,
        'actual_duration_s': round(actual_duration, 1),
        'codec': codec, 'resolution': '{}x{}'.format(width, height),
    }
    blind_key.append(bk)

    for sr in source_runs:
        run_mapping.append({
            'unique_blind_id': blind_id, 'unique_video_sha256': vhash,
            'source_run_key': sr['run_dir'],
            'condition': sr['condition'], 'objective_id': sr['objective_id'],
            'arm_lock': sr['arm_lock'], 'task_name': sr['task_name'],
            'state_id': sr['state_id'], 'perturbation_seed': sr['perturbation_seed'],
            'task_success': sr['task_success'], 'attack_frames': sr['attack_frames'],
        })

# Write PRIVATE files
def write_csv(path, rows):
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    os.utime(path, (FIXED_MTIME, FIXED_MTIME))

write_csv(os.path.join(PRV, 'CQFR_UNIQUE68_PRIVATE_KEY.csv'), blind_key)
write_csv(os.path.join(PRV, 'CQFR_108_RUN_MAPPING_PRIVATE.csv'), run_mapping)
write_csv(os.path.join(PRV, 'CQFR_DUPLICATE_GROUP_AUDIT_PRIVATE.csv'), dup_audit)
print('Private files written')

# PUBLIC reviewer template
reviewer_path = os.path.join(PUB, 'CQFR_UNIQUE68_REVIEWER_TEMPLATE.csv')
with open(reviewer_path, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['blind_id', 'video_path', 'task_instruction',
                'outcome_label', 'outcome_confidence', 'failure_mode', 'failure_subtype', 'notes'])
    for bk in blind_key:
        w.writerow([bk['blind_id'], bk['blind_id'] + '.mp4', bk['task_instruction'],
                    '', '', '', '', ''])
os.utime(reviewer_path, (FIXED_MTIME, FIXED_MTIME))

# Label definitions
ld_path = os.path.join(PUB, 'CQFR_LABEL_DEFINITIONS_V3.txt')
with open(ld_path, 'w') as f:
    f.write("""CQFR LABEL DEFINITIONS V3

outcome_label (required):
  clear_success  - Task clearly completed
  clear_failure  - Task clearly failed
  ambiguous      - Cannot determine from video
  video_invalid  - Video corrupted/unplayable

outcome_confidence: high / medium / low

failure_mode (only if clear_failure):
  gripper_related / arm_trajectory_related / mixed / other / unclear / not_applicable

failure_subtype (optional):
  failed_to_grasp / premature_release / drop_after_lift
  unstable_transport / incorrect_final_release / timeout_other / not_applicable
""")
os.utime(ld_path, (FIXED_MTIME, FIXED_MTIME))

# Review protocol
rp_path = os.path.join(PUB, 'CQFR_REVIEW_PROTOCOL_V3.md')
with open(rp_path, 'w') as f:
    f.write("""# CQFR Review Protocol V3

1. Read task_instruction.
2. Watch the full video.
3. Choose outcome_label and confidence.
4. If clear_failure, choose failure_mode and optionally failure_subtype.
5. Add notes if helpful (frame ranges, observations).
""")
os.utime(rp_path, (FIXED_MTIME, FIXED_MTIME))

# Public manifest CSV
pub_manifest_path = os.path.join(PUB, 'CQFR_UNIQUE68_PUBLIC_MANIFEST.csv')
with open(pub_manifest_path, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['blind_id', 'video_sha256', 'task_instruction',
                'duration_s', 'frame_count', 'fps', 'codec', 'resolution'])
    for bk in blind_key:
        w.writerow([bk['blind_id'], bk['unique_video_sha256'], bk['task_instruction'],
                    bk['actual_duration_s'], bk['actual_frames'], bk['actual_fps'],
                    bk['codec'], bk['resolution']])
os.utime(pub_manifest_path, (FIXED_MTIME, FIXED_MTIME))

# PUBLIC SHA256SUMS (all public files including manifest and checksums)
pub_files = ['U{:04d}.mp4'.format(i+1) for i in range(len(blind_key))]
pub_files += ['CQFR_UNIQUE68_REVIEWER_TEMPLATE.csv', 'CQFR_LABEL_DEFINITIONS_V3.txt',
              'CQFR_REVIEW_PROTOCOL_V3.md']
sha_path = os.path.join(PUB, 'CQFR_UNIQUE68_PUBLIC_SHA256SUMS.txt')
with open(sha_path, 'w') as f:
    for fn in pub_files:
        fp = os.path.join(PUB, fn)
        if os.path.isfile(fp):
            f.write('{}  {}\n'.format(hashlib.sha256(open(fp, 'rb').read()).hexdigest(), fn))
# Also include the manifest itself in the SHA list
manifest_sha = hashlib.sha256(open(pub_manifest_path, 'rb').read()).hexdigest()
with open(sha_path, 'a') as f:
    f.write('{}  CQFR_UNIQUE68_PUBLIC_MANIFEST.csv\n'.format(manifest_sha))
    f.write('{}  CQFR_UNIQUE68_PUBLIC_SHA256SUMS.txt\n'.format(
        hashlib.sha256(open(sha_path, 'rb').read()).hexdigest()))
os.utime(sha_path, (FIXED_MTIME, FIXED_MTIME))
# Re-write with self-hash included
with open(sha_path, 'w') as f:
    for fn in pub_files + ['CQFR_UNIQUE68_PUBLIC_MANIFEST.csv']:
        fp = os.path.join(PUB, fn)
        if os.path.isfile(fp):
            f.write('{}  {}\n'.format(hashlib.sha256(open(fp, 'rb').read()).hexdigest(), fn))
os.utime(sha_path, (FIXED_MTIME, FIXED_MTIME))

# Create PUBLIC ZIP (only public files, no private)
zip_files = pub_files + ['CQFR_UNIQUE68_PUBLIC_MANIFEST.csv', 'CQFR_UNIQUE68_PUBLIC_SHA256SUMS.txt']
zip_path = os.path.join(PUB, 'CQFR_UNIQUE68_PUBLIC.zip')
subprocess.run(['zip', '-r', zip_path] + zip_files, cwd=PUB, capture_output=True, text=True, check=True)

# Verify ZIP integrity
subprocess.run(['unzip', '-t', zip_path], cwd=PUB, capture_output=True, text=True, check=True)

# Verify ZIP members — assert no private files
result = subprocess.run(['unzip', '-l', zip_path], cwd=PUB, capture_output=True, text=True, check=True)
zip_members = set()
for line in result.stdout.split('\n'):
    parts = line.split()
    if len(parts) >= 4 and parts[-1].endswith(('.mp4', '.csv', '.txt', '.md')):
        zip_members.add(parts[-1])
expected_public = set(zip_files)
missing = expected_public - zip_members
extra = zip_members - expected_public
private_in_zip = [m for m in zip_members if 'PRIVATE' in m.upper() or '108_RUN' in m.upper() or 'BLIND_KEY' in m.upper()]
assert len(missing) == 0, 'Missing from ZIP: {}'.format(missing)
assert len(extra) == 0, 'Extra in ZIP: {}'.format(extra)
assert len(private_in_zip) == 0, 'PRIVATE FILES IN ZIP: {}'.format(private_in_zip)
print('ZIP member verification: {} files, 0 missing, 0 extra, 0 private'.format(len(zip_members)))

# Save ZIP member list
with open(os.path.join(PUB, 'ZIP_MEMBERS.txt'), 'w') as f:
    for m in sorted(zip_members): f.write(m + '\n')
os.utime(os.path.join(PUB, 'ZIP_MEMBERS.txt'), (FIXED_MTIME, FIXED_MTIME))

# Detached ZIP SHA
zip_sha = hashlib.sha256(open(zip_path, 'rb').read()).hexdigest()
with open(zip_path + '.sha256', 'w') as f:
    f.write('{}  CQFR_UNIQUE68_PUBLIC.zip\n'.format(zip_sha))

print('Public ZIP: {} ({:.1f}MB)'.format(zip_sha, os.path.getsize(zip_path)/1024/1024))

succ = sum(1 for bk in blind_key if bk['task_success_representative'])
fail = sum(1 for bk in blind_key if not bk['task_success_representative'])
print('Unique videos: {} ({} succ, {} fail, {} AL, {} NL)'.format(
    len(blind_key), succ, fail,
    sum(1 for bk in blind_key if bk['arm_lock_representative']),
    sum(1 for bk in blind_key if not bk['arm_lock_representative'])))
print('Private key: {}'.format(PRV))
print('Done.')
