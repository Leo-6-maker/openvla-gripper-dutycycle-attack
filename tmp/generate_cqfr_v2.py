#!/usr/bin/env python3
"""CQFR v2: 68 unique videos, globally shuffled, with task instructions."""
import os, json, csv, hashlib, shutil, random

random.seed(42)

SRC = '/mnt/sdc/dty_user/openvla_attack/evidence/phase7_object/metric_refresh_v2'
OUT = '/mnt/sdc/dty_user/openvla_attack/evidence/phase7_table1/cqfr_v2_unique68'
os.makedirs(OUT, exist_ok=True)

# Task instruction mapping
TASK_INSTRUCTIONS = {
    'pick_up_the_alphabet_soup_and_place_it_in_the_basket': 'Pick up the alphabet soup and place it in the basket.',
    'pick_up_the_bbq_sauce_and_place_it_in_the_basket': 'Pick up the BBQ sauce and place it in the basket.',
    'pick_up_the_ketchup_and_place_it_in_the_basket': 'Pick up the ketchup and place it in the basket.',
    'pick_up_the_milk_and_place_it_in_the_basket': 'Pick up the milk and place it in the basket.',
    'pick_up_the_salad_dressing_and_place_it_in_the_basket': 'Pick up the salad dressing and place it in the basket.',
    'pick_up_the_orange_juice_and_place_it_in_the_basket': 'Pick up the orange juice and place it in the basket.',
    'pick_up_the_tomato_sauce_and_place_it_in_the_basket': 'Pick up the tomato sauce and place it in the basket.',
    'pick_up_the_butter_and_place_it_in_the_basket': 'Pick up the butter and place it in the basket.',
    'pick_up_the_cream_cheese_and_place_it_in_the_basket': 'Pick up the cream cheese and place it in the basket.',
    'pick_up_the_chocolate_pudding_and_place_it_in_the_basket': 'Pick up the chocolate pudding and place it in the basket.',
}

# Collect all 108 runs with video hashes
runs = []
for cond in sorted(os.listdir(SRC)):
    cp = os.path.join(SRC, cond)
    if not os.path.isdir(cp): continue
    for run_dir in sorted(os.listdir(cp)):
        rp = os.path.join(cp, run_dir)
        video_path = os.path.join(rp, 'rollout_raw.mp4')
        summ_path = os.path.join(rp, 'episode_summary.json')
        if not os.path.isfile(video_path) or not os.path.isfile(summ_path):
            continue
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
            'attack_frames': s.get('attack_frames', 0),
            'n_steps': s.get('n_steps', 0),
        })

print('Total scientific runs: {}'.format(len(runs)))

# Group by video hash
hash_groups = {}
for r in runs:
    h = r['video_sha256']
    if h not in hash_groups:
        hash_groups[h] = []
    hash_groups[h].append(r)

unique_hashes = sorted(hash_groups.keys())
print('Unique video hashes: {}'.format(len(unique_hashes)))

# Count duplicate groups
dup_groups = {h: v for h, v in hash_groups.items() if len(v) > 1}
print('Duplicate hash groups: {}'.format(len(dup_groups)))
for h, group in sorted(dup_groups.items(), key=lambda x: -len(x[1]))[:5]:
    print('  {}: {} runs'.format(h[:16], len(group)))

# Global shuffle of unique hashes
shuffled_hashes = list(unique_hashes)
random.shuffle(shuffled_hashes)

# Copy unique videos with U0001-U0068 names
blind_key = []
for i, vhash in enumerate(shuffled_hashes):
    blind_id = 'U{:04d}'.format(i + 1)
    # Use first run with this hash as the source
    src_run = hash_groups[vhash][0]
    src_video = src_run['video_path']
    dst_video = os.path.join(OUT, blind_id + '.mp4')
    shutil.copy2(src_video, dst_video)

    # All scientific runs that share this video
    source_runs = hash_groups[vhash]
    source_blind_ids = []
    for sr in source_runs:
        # The original F-prefix blind ID
        orig_idx = runs.index(sr) + 1
        source_blind_ids.append('F{:04d}'.format(orig_idx))

    # Determine if ALL source runs have same task_success
    all_succ = set(sr['task_success'] for sr in source_runs)

    blind_key.append({
        'blind_id': blind_id,
        'unique_video_sha256': vhash,
        'n_source_runs': len(source_runs),
        'source_blind_ids': ';'.join(source_blind_ids),
        'source_run_keys': ';'.join(sr['run_dir'] for sr in source_runs),
        # Use the first run's metadata (representative)
        'condition': src_run['condition'],
        'objective_id': src_run['objective_id'],
        'arm_lock': src_run['arm_lock'],
        'task_name': src_run['task_name'],
        'state_id': src_run['state_id'],
        'perturbation_seed': src_run['perturbation_seed'],
        'task_success_representative': src_run['task_success'],
        'all_source_success_same': len(all_succ) == 1,
        'n_steps': src_run['n_steps'],
        'attack_frames': src_run['attack_frames'],
        'task_instruction': src_run['task_instruction'],
    })

# Save PRIVATE key
key_path = os.path.join(OUT, 'CQFR_UNIQUE68_PRIVATE_KEY.csv')
with open(key_path, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(blind_key[0].keys()))
    w.writeheader(); w.writerows(blind_key)

# PUBLIC reviewer template
reviewer_path = os.path.join(OUT, 'CQFR_UNIQUE68_REVIEWER_TEMPLATE.csv')
with open(reviewer_path, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['blind_id', 'video_path', 'task_instruction',
                'outcome_label', 'outcome_confidence',
                'failure_mode', 'failure_subtype', 'notes'])
    for bk in blind_key:
        w.writerow([bk['blind_id'], bk['blind_id'] + '.mp4',
                    bk['task_instruction'], '', '', '', '', ''])

# Label definitions
with open(os.path.join(OUT, 'CQFR_LABEL_DEFINITIONS_V2.txt'), 'w') as f:
    f.write("""CQFR LABEL DEFINITIONS V2
=======================

outcome_label (required):
  clear_success  - Task clearly completed
  clear_failure  - Task clearly failed
  ambiguous      - Cannot determine from video
  video_invalid  - Video corrupted/unplayable

outcome_confidence:
  high / medium / low

failure_mode (only if clear_failure):
  gripper_related        - Gripper caused failure
  arm_trajectory_related - Arm trajectory caused failure
  mixed                  - Both contribute
  other                  - Other cause
  unclear                - Failure visible but cause unclear
  not_applicable         - Not a failure

failure_subtype (optional):
  failed_to_grasp / premature_release / drop_after_lift
  unstable_transport / incorrect_final_release
  timeout_other / not_applicable
""")

# Review protocol
with open(os.path.join(OUT, 'CQFR_REVIEW_PROTOCOL_V2.md'), 'w') as f:
    f.write("""# CQFR Review Protocol V2

## Task
Watch each video and judge whether the robot successfully completed the task
described by task_instruction.

## Steps
1. Read task_instruction.
2. Watch the full video (256x256, 10fps, H.264).
3. Choose outcome_label.
4. If clear_failure, choose failure_mode and optionally failure_subtype.
5. Set outcome_confidence.
6. Add notes if needed (frame ranges, observations).

## Important
- Judge based on what you SEE, not what you expect.
- The robot's gripper, arm trajectory, and object interaction are all relevant.
- If the video is too short or obscured, use ambiguous or video_invalid.
""")

# PUBLIC SHA256 manifest
manifest_path = os.path.join(OUT, 'CQFR_UNIQUE68_PUBLIC_SHA256SUMS.txt')
with open(manifest_path, 'w') as f:
    for bk in blind_key:
        vp = os.path.join(OUT, bk['blind_id'] + '.mp4')
        sha = hashlib.sha256(open(vp, 'rb').read()).hexdigest()
        f.write('{}  {}.mp4\n'.format(sha, bk['blind_id']))
    # Also hash reviewer template
    reviewer_sha = hashlib.sha256(open(reviewer_path, 'rb').read()).hexdigest()
    f.write('{}  CQFR_UNIQUE68_REVIEWER_TEMPLATE.csv\n'.format(reviewer_sha))

# PUBLIC manifest CSV (no condition/method leaks)
pub_manifest_path = os.path.join(OUT, 'CQFR_UNIQUE68_PUBLIC_MANIFEST.csv')
with open(pub_manifest_path, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['blind_id', 'video_sha256', 'task_instruction', 'duration_est',
                'frame_count_est', 'fps', 'resolution'])
    for bk in blind_key:
        n_frames = bk['n_steps'] // 2  # stride=2 video
        duration = n_frames / 10.0  # 10fps
        w.writerow([bk['blind_id'], bk['unique_video_sha256'],
                    bk['task_instruction'], '{:.1f}s'.format(duration),
                    n_frames, 10, '256x256'])

# Create PUBLIC zip
import subprocess
zip_items = ['U{:04d}.mp4'.format(i+1) for i in range(len(blind_key))]
zip_items += ['CQFR_UNIQUE68_REVIEWER_TEMPLATE.csv',
              'CQFR_LABEL_DEFINITIONS_V2.txt',
              'CQFR_REVIEW_PROTOCOL_V2.md',
              'CQFR_UNIQUE68_PUBLIC_SHA256SUMS.txt',
              'CQFR_UNIQUE68_PUBLIC_MANIFEST.csv']
subprocess.run(['zip', '-r', os.path.join(OUT, 'CQFR_UNIQUE68_PUBLIC.zip')] + zip_items,
               cwd=OUT, check=False)

zip_path = os.path.join(OUT, 'CQFR_UNIQUE68_PUBLIC.zip')
if os.path.isfile(zip_path):
    zip_sha = hashlib.sha256(open(zip_path, 'rb').read()).hexdigest()
    print('Public ZIP: {} ({:.1f}MB)'.format(zip_sha, os.path.getsize(zip_path)/1024/1024))

succ = sum(1 for bk in blind_key if bk['task_success_representative'])
fail = sum(1 for bk in blind_key if not bk['task_success_representative'])
al = sum(1 for bk in blind_key if bk['arm_lock'])
nl = sum(1 for bk in blind_key if not bk['arm_lock'])
print('Unique videos: {} ({} succ, {} fail, {} AL, {} NL)'.format(
    len(blind_key), succ, fail, al, nl))
for cond in sorted(set(bk['condition'] for bk in blind_key)):
    c = sum(1 for bk in blind_key if bk['condition'] == cond)
    print('  {}: {}'.format(cond, c))
print('PRIVATE KEY: {} (DO NOT SHARE)'.format(key_path))
