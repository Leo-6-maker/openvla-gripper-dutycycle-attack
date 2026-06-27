#!/usr/bin/env python3
"""CQFR v3: 108-row mapping, consistency checks, mtime scrub, full metadata, complete SHAs."""
import os, json, csv, hashlib, shutil, random, subprocess

random.seed(42)
FIXED_MTIME = 946684800  # 2000-01-01 UTC

SRC = '/mnt/sdc/dty_user/openvla_attack/evidence/phase7_object/metric_refresh_v2'
OUT = '/mnt/sdc/dty_user/openvla_attack/evidence/phase7_table1/cqfr_v3_unique68'
os.makedirs(OUT, exist_ok=True)

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

# ---- Duplicate-group consistency checks ----
conflicting_task = 0; conflicting_instruction = 0; conflicting_success = 0
dup_groups = {h: v for h, v in hash_groups.items() if len(v) > 1}
for h, group in dup_groups.items():
    tasks = set(r['task_name'] for r in group)
    instrs = set(r['task_instruction'] for r in group)
    succs = set(r['task_success'] for r in group)
    if len(tasks) > 1: conflicting_task += 1
    if len(instrs) > 1: conflicting_instruction += 1
    if len(succs) > 1: conflicting_success += 1

print('Duplicate groups: {} (conflict: task={} instr={} succ={})'.format(
    len(dup_groups), conflicting_task, conflicting_instruction, conflicting_success))

# Global shuffle
shuffled_hashes = list(unique_hashes)
random.shuffle(shuffled_hashes)

# Copy unique videos
blind_key = []
run_mapping = []  # 108-row private mapping

for i, vhash in enumerate(shuffled_hashes):
    blind_id = 'U{:04d}'.format(i + 1)
    src_run = hash_groups[vhash][0]
    src_video = src_run['video_path']
    dst_video = os.path.join(OUT, blind_id + '.mp4')

    # Copy without preserving mtime
    shutil.copyfile(src_video, dst_video)
    os.utime(dst_video, (FIXED_MTIME, FIXED_MTIME))

    # Get actual video metadata via ffprobe
    try:
        result = subprocess.run([
            'ffprobe', '-v', 'quiet', '-print_format', 'json',
            '-show_format', '-show_streams', dst_video
        ], capture_output=True, text=True, timeout=10)
        probe = json.loads(result.stdout)
        vstream = next((s for s in probe.get('streams', []) if s.get('codec_type') == 'video'), {})
        actual_fps = vstream.get('r_frame_rate', '?')
        actual_frames = int(vstream.get('nb_frames', 0))
        actual_duration = float(probe.get('format', {}).get('duration', 0))
        codec = vstream.get('codec_name', '?')
        width = vstream.get('width', 0)
        height = vstream.get('height', 0)
    except:
        actual_fps = '?'; actual_frames = 0; actual_duration = 0.0
        codec = '?'; width = 0; height = 0

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
        'actual_fps': str(actual_fps), 'actual_frames': actual_frames,
        'actual_duration_s': round(actual_duration, 1),
        'codec': codec, 'resolution': '{}x{}'.format(width, height),
    }
    blind_key.append(bk)

    # 108-row mapping: one row per scientific run
    for sr in source_runs:
        run_mapping.append({
            'unique_blind_id': blind_id,
            'unique_video_sha256': vhash,
            'source_run_key': sr['run_dir'],
            'condition': sr['condition'],
            'objective_id': sr['objective_id'],
            'arm_lock': sr['arm_lock'],
            'task_name': sr['task_name'],
            'state_id': sr['state_id'],
            'perturbation_seed': sr['perturbation_seed'],
            'task_success': sr['task_success'],
            'attack_frames': sr['attack_frames'],
        })

# PRIVATE key (68 rows, representative)
key_path = os.path.join(OUT, 'CQFR_UNIQUE68_PRIVATE_KEY.csv')
with open(key_path, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(blind_key[0].keys()))
    w.writeheader(); w.writerows(blind_key)

# PRIVATE 108-row mapping
map_path = os.path.join(OUT, 'CQFR_108_RUN_MAPPING_PRIVATE.csv')
with open(map_path, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(run_mapping[0].keys()))
    w.writeheader(); w.writerows(run_mapping)
print('108-row mapping: {} rows'.format(len(run_mapping)))

# PUBLIC reviewer template
reviewer_path = os.path.join(OUT, 'CQFR_UNIQUE68_REVIEWER_TEMPLATE.csv')
with open(reviewer_path, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['blind_id', 'video_path', 'task_instruction',
                'outcome_label', 'outcome_confidence',
                'failure_mode', 'failure_subtype', 'notes'])
    for bk in blind_key:
        w.writerow([bk['blind_id'], bk['blind_id'] + '.mp4', bk['task_instruction'],
                    '', '', '', '', ''])

# Label definitions
with open(os.path.join(OUT, 'CQFR_LABEL_DEFINITIONS_V3.txt'), 'w') as f:
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

# Review protocol
with open(os.path.join(OUT, 'CQFR_REVIEW_PROTOCOL_V3.md'), 'w') as f:
    f.write("""# CQFR Review Protocol V3

1. Read task_instruction.
2. Watch the full video.
3. Choose outcome_label and confidence.
4. If clear_failure, choose failure_mode and optionally failure_subtype.
5. Add notes if helpful (frame ranges, observations).
""")

# PUBLIC SHA256 manifest (all public files)
pub_files = ['U{:04d}.mp4'.format(i+1) for i in range(len(blind_key))]
pub_files += ['CQFR_UNIQUE68_REVIEWER_TEMPLATE.csv',
              'CQFR_LABEL_DEFINITIONS_V3.txt',
              'CQFR_REVIEW_PROTOCOL_V3.md']

sha_path = os.path.join(OUT, 'CQFR_UNIQUE68_PUBLIC_SHA256SUMS.txt')
with open(sha_path, 'w') as f:
    for fn in pub_files:
        fp = os.path.join(OUT, fn)
        if os.path.isfile(fp):
            sha = hashlib.sha256(open(fp, 'rb').read()).hexdigest()
            f.write('{}  {}\n'.format(sha, fn))

# PUBLIC manifest CSV
pub_manifest_path = os.path.join(OUT, 'CQFR_UNIQUE68_PUBLIC_MANIFEST.csv')
with open(pub_manifest_path, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['blind_id', 'video_sha256', 'task_instruction',
                'duration_s', 'frame_count', 'fps', 'codec', 'resolution'])
    for bk in blind_key:
        w.writerow([bk['blind_id'], bk['unique_video_sha256'], bk['task_instruction'],
                    bk['actual_duration_s'], bk['actual_frames'],
                    bk['actual_fps'], bk['codec'], bk['resolution']])

# Create PUBLIC ZIP with fixed timestamps
zip_path = os.path.join(OUT, 'CQFR_UNIQUE68_PUBLIC.zip')
zip_items = pub_files
# Use --mtime to set fixed timestamp on all entries
subprocess.run(['zip', '-r', '-X', '-j', zip_path, OUT] + zip_items,
               cwd=OUT, check=True)
# -j junks paths, so we need to work from the directory itself
# Actually let's use a different approach
subprocess.run(['rm', '-f', zip_path], check=False)
cmd = ['zip', '-r', zip_path] + zip_items
result = subprocess.run(cmd, cwd=OUT, capture_output=True, text=True, check=True)

if os.path.isfile(zip_path):
    zip_sha = hashlib.sha256(open(zip_path, 'rb').read()).hexdigest()
    # Verify zip integrity
    subprocess.run(['unzip', '-t', zip_path], cwd=OUT, check=True)
    print('Public ZIP: {} ({:.1f}MB) verified'.format(zip_sha, os.path.getsize(zip_path)/1024/1024))

succ = sum(1 for bk in blind_key if bk['task_success_representative'])
fail = sum(1 for bk in blind_key if not bk['task_success_representative'])
al = sum(1 for bk in blind_key if bk['arm_lock_representative'])
nl = sum(1 for bk in blind_key if not bk['arm_lock_representative'])
print('Unique videos: {} ({} succ, {} fail, {} AL, {} NL)'.format(len(blind_key), succ, fail, al, nl))
for cond in sorted(set(bk['condition_representative'] for bk in blind_key)):
    c = sum(1 for bk in blind_key if bk['condition_representative'] == cond)
    print('  {}: {}'.format(cond, c))
print('Private key: {}'.format(key_path))
print('108 mapping: {}'.format(map_path))
