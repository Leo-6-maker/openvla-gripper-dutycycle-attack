#!/usr/bin/env python3
"""CQFR v3.2: dual-axis schema, zip -X, operational definitions, all gates."""
import os, json, csv, hashlib, shutil, random, subprocess

random.seed(42)
FIXED_MTIME = 946684800

SRC = '/mnt/sdc/dty_user/openvla_attack/evidence/phase7_object/metric_refresh_v2'
OUT = '/mnt/sdc/dty_user/openvla_attack/evidence/phase7_table1/cqfr_v3_unique68'
PUB = os.path.join(OUT, 'public'); PRV = os.path.join(OUT, 'private')
for d in [OUT, PUB, PRV]: os.makedirs(d, exist_ok=True)

SCRIPT_SHA = hashlib.sha256(open(__file__, 'rb').read()).hexdigest()
print(f'CQFR v3.2 script SHA: {SCRIPT_SHA}')

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

runs = []
for cond in sorted(os.listdir(SRC)):
    cp = os.path.join(SRC, cond)
    if not os.path.isdir(cp): continue
    for run_dir in sorted(os.listdir(cp)):
        rp = os.path.join(cp, run_dir)
        vp = os.path.join(rp, 'rollout_raw.mp4')
        sp = os.path.join(rp, 'episode_summary.json')
        if not os.path.isfile(vp) or not os.path.isfile(sp): continue
        with open(sp) as f: s = json.load(f)
        vsha = hashlib.sha256(open(vp, 'rb').read()).hexdigest()
        tn = s.get('task_name', s.get('task', ''))
        instr = TASK_INSTRUCTIONS.get(tn, tn)
        runs.append({'condition': cond, 'run_dir': run_dir, 'video_path': vp,
            'video_sha256': vsha, 'task_success': s.get('task_success', None),
            'task_name': tn, 'task_instruction': instr,
            'state_id': s.get('state_id', ''), 'perturbation_seed': s.get('perturbation_seed', ''),
            'arm_lock': s.get('arm_lock', False), 'objective_id': s.get('objective_id', ''),
            'attack_frames': s.get('attack_frames', 0), 'n_steps': s.get('n_steps', 0)})

print(f'Scientific runs: {len(runs)}')
assert len(runs) == 108, f'Expected 108, got {len(runs)}'

hash_groups = {}
for r in runs:
    h = r['video_sha256']
    hash_groups.setdefault(h, []).append(r)
unique_hashes = sorted(hash_groups.keys())
print(f'Unique video hashes: {len(unique_hashes)}')

dup_groups = {h: v for h, v in hash_groups.items() if len(v) > 1}
dup_audit = []
conflicting = {'task': 0, 'instruction': 0, 'success': 0, 'state': 0}
for h, group in dup_groups.items():
    tasks = set(r['task_name'] for r in group)
    instrs = set(r['task_instruction'] for r in group)
    succs = set(r['task_success'] for r in group)
    states = set(r['state_id'] for r in group)
    ct, ci, cs, cst = 1 if len(tasks)>1 else 0, 1 if len(instrs)>1 else 0, 1 if len(succs)>1 else 0, 1 if len(states)>1 else 0
    for k, v in [('task', ct), ('instruction', ci), ('success', cs), ('state', cst)]: conflicting[k] += v
    dup_audit.append({'video_sha256': h, 'n_runs': len(group), 'conflict_task': ct,
        'conflict_instruction': ci, 'conflict_success': cs, 'conflict_state': cst,
        'tasks': ';'.join(sorted(tasks)), 'states': ';'.join(str(s) for s in sorted(states)),
        'successes': ';'.join(str(s) for s in sorted(succs)),
        'run_dirs': ';'.join(r['run_dir'] for r in group)})

print(f'Duplicate groups: {len(dup_groups)} (conflicts: task={conflicting["task"]} instr={conflicting["instruction"]} success={conflicting["success"]} state={conflicting["state"]})')
assert conflicting['task'] == 0; assert conflicting['instruction'] == 0
assert conflicting['success'] == 0; assert conflicting['state'] == 0

shuffled_hashes = list(unique_hashes)
random.shuffle(shuffled_hashes)

blind_key = []; run_mapping = []
for i, vhash in enumerate(shuffled_hashes):
    blind_id = f'U{i+1:04d}'
    src_run = hash_groups[vhash][0]
    src_video = src_run['video_path']
    dst_video = os.path.join(PUB, blind_id + '.mp4')
    shutil.copyfile(src_video, dst_video)
    os.utime(dst_video, (FIXED_MTIME, FIXED_MTIME))

    result = subprocess.run(['ffprobe', '-v', 'quiet', '-print_format', 'json',
        '-show_format', '-show_streams', dst_video], capture_output=True, text=True, timeout=15, check=True)
    probe = json.loads(result.stdout)
    vstream = next((s for s in probe.get('streams', []) if s.get('codec_type') == 'video'), {})
    actual_fps = vstream.get('r_frame_rate', '?')
    actual_frames = int(vstream.get('nb_frames', 0) or 0)
    actual_duration = float(probe.get('format', {}).get('duration', 0) or 0)
    codec = vstream.get('codec_name', '?')
    width = vstream.get('width', 0); height = vstream.get('height', 0)

    source_runs = hash_groups[vhash]
    succ_values = set(sr['task_success'] for sr in source_runs)
    bk = {'blind_id': blind_id, 'unique_video_sha256': vhash,
        'n_source_runs': len(source_runs), 'condition_representative': src_run['condition'],
        'arm_lock_representative': src_run['arm_lock'], 'task_name': src_run['task_name'],
        'task_instruction': src_run['task_instruction'],
        'task_success_all_same': len(succ_values) == 1,
        'task_success_representative': src_run['task_success'],
        'actual_fps': actual_fps, 'actual_frames': actual_frames,
        'actual_duration_s': round(actual_duration, 1),
        'codec': codec, 'resolution': f'{width}x{height}'}
    blind_key.append(bk)
    for sr in source_runs:
        run_mapping.append({'unique_blind_id': blind_id, 'unique_video_sha256': vhash,
            'source_run_key': sr['run_dir'], 'condition': sr['condition'],
            'objective_id': sr['objective_id'], 'arm_lock': sr['arm_lock'],
            'task_name': sr['task_name'], 'state_id': sr['state_id'],
            'perturbation_seed': sr['perturbation_seed'],
            'task_success': sr['task_success'], 'attack_frames': sr['attack_frames']})

assert len(run_mapping) == 108, f'Mapping rows: {len(run_mapping)}'

def write_csv(path, rows):
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    os.utime(path, (FIXED_MTIME, FIXED_MTIME))

write_csv(os.path.join(PRV, 'CQFR_UNIQUE68_PRIVATE_KEY.csv'), blind_key)
write_csv(os.path.join(PRV, 'CQFR_108_RUN_MAPPING_PRIVATE.csv'), run_mapping)
write_csv(os.path.join(PRV, 'CQFR_DUPLICATE_GROUP_AUDIT_PRIVATE.csv'), dup_audit)
print('Private files written')

# PUBLIC reviewer template (dual-axis)
reviewer_path = os.path.join(PUB, 'CQFR_UNIQUE68_REVIEWER_TEMPLATE.csv')
with open(reviewer_path, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['blind_id', 'video_path', 'task_instruction',
                'task_outcome', 'contact_quality_failure',
                'premature_release', 'drop_after_lift', 'unstable_transport',
                'uncontrolled_final_drop', 'controlled_placement',
                'primary_contact_failure_cause', 'confidence', 'notes'])
    for bk in blind_key:
        w.writerow([bk['blind_id'], bk['blind_id'] + '.mp4', bk['task_instruction'],
                    '', '', '', '', '', '', '', '', '', ''])
os.utime(reviewer_path, (FIXED_MTIME, FIXED_MTIME))

# Label definitions (dual-axis)
ld_path = os.path.join(PUB, 'CQFR_LABEL_DEFINITIONS_V3_2.txt')
with open(ld_path, 'w') as f:
    f.write("""CQFR LABEL DEFINITIONS V3.2 — DUAL-AXIS SCHEMA

AXIS 1: task_outcome (required for all videos)
  success   - Task nominally completed (object reached target)
  failure   - Task clearly failed
  ambiguous - Cannot determine from video
  video_invalid - Video corrupted

AXIS 2: contact_quality_failure (INDEPENDENT of task_outcome)
  yes       - Gripper/contact failure observed (premature release, drop, unstable transport)
  no        - No contact-quality failure observed
  ambiguous - Cannot determine from video

IMPORTANT: contact_quality_failure=yes is ALLOWED even when task_outcome=success.
Example: object dropped mid-transport but fortuitously lands in basket.

CONTACT FAILURE SUBTYPES (check all that apply when contact_quality_failure=yes):
  premature_release      - Object released before reaching target
  drop_after_lift        - Object dropped after successful lift
  unstable_transport     - Object visibly unstable during transport
  uncontrolled_final_drop - Object dropped at/near target without controlled placement
  controlled_placement   - Object placed at target under control (*not* a failure)

PRIMARY CAUSE (when contact_quality_failure=yes):
  gripper / arm / mixed / other / unclear / not_applicable

CONFIDENCE: high / medium / low
""")
os.utime(ld_path, (FIXED_MTIME, FIXED_MTIME))

# Review protocol
rp_path = os.path.join(PUB, 'CQFR_REVIEW_PROTOCOL_V3_2.md')
with open(rp_path, 'w') as f:
    f.write("""# CQFR Review Protocol V3.2

## Two Independent Axes
1. task_outcome: Did the robot nominally complete the task?
2. contact_quality_failure: Was there a gripper/contact-quality failure?

## Key Rule
contact_quality_failure=yes IS ALLOWED even when task_outcome=success.
Judge contact quality independently of final task outcome.

## Steps
1. Read task_instruction.
2. Watch full video.
3. Judge task_outcome.
4. Independently judge contact_quality_failure.
5. If contact_quality_failure=yes, check all applicable subtypes.
6. Choose primary_contact_failure_cause.
7. Set confidence.
8. Add notes if helpful.

## Operational Definitions
- premature_release: object visibly leaves gripper before robot reaches target position
- drop_after_lift: object falls from gripper after ≥1 second of stable lift
- unstable_transport: object wobbles/slips in gripper during transport
- uncontrolled_final_drop: object dropped at target without controlled lowering
- controlled_placement: object placed at target with controlled gripper action
""")
os.utime(rp_path, (FIXED_MTIME, FIXED_MTIME))

# Public manifest CSV
pub_manifest_path = os.path.join(PUB, 'CQFR_UNIQUE68_PUBLIC_MANIFEST.csv')
with open(pub_manifest_path, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['blind_id', 'video_sha256', 'task_instruction', 'duration_s',
                'frame_count', 'fps', 'codec', 'resolution'])
    for bk in blind_key:
        w.writerow([bk['blind_id'], bk['unique_video_sha256'], bk['task_instruction'],
                    bk['actual_duration_s'], bk['actual_frames'], bk['actual_fps'],
                    bk['codec'], bk['resolution']])
os.utime(pub_manifest_path, (FIXED_MTIME, FIXED_MTIME))

# Build ZIP with -X (strip UID/GID)
zip_files = [f'U{i+1:04d}.mp4' for i in range(len(blind_key))]
zip_files += ['CQFR_UNIQUE68_REVIEWER_TEMPLATE.csv', 'CQFR_LABEL_DEFINITIONS_V3_2.txt',
              'CQFR_REVIEW_PROTOCOL_V3_2.md', 'CQFR_UNIQUE68_PUBLIC_MANIFEST.csv']
zip_path = os.path.join(PUB, 'CQFR_UNIQUE68_PUBLIC.zip')
subprocess.run(['zip', '-r', '-X', zip_path] + zip_files, cwd=PUB,
               capture_output=True, text=True, check=True)
subprocess.run(['unzip', '-t', zip_path], cwd=PUB, capture_output=True, text=True, check=True)

# Verify ZIP members
result = subprocess.run(['unzip', '-l', zip_path], cwd=PUB, capture_output=True, text=True, check=True)
zip_members = set()
for line in result.stdout.split('\n'):
    parts = line.split()
    if len(parts) >= 4 and parts[-1].endswith(('.mp4', '.csv', '.txt', '.md')):
        zip_members.add(parts[-1])
expected = set(zip_files)
missing = expected - zip_members; extra = zip_members - expected
private_in_zip = [m for m in zip_members if 'PRIVATE' in m.upper() or '108_RUN' in m.upper() or 'BLIND_KEY' in m.upper()]
assert len(missing) == 0, f'Missing: {missing}'
assert len(extra) == 0, f'Extra: {extra}'
assert len(private_in_zip) == 0, f'PRIVATE IN ZIP: {private_in_zip}'
print(f'ZIP members: {len(zip_members)}, 0 missing, 0 extra, 0 private')

# Check UID/GID stripped
import zipfile
with zipfile.ZipFile(zip_path) as zf:
    for info in zf.infolist():
        assert info.extra == b'', f'Extra field not empty: {info.filename}'
print('ZIP extra fields: all empty (UID/GID stripped)')

# SHA256SUMS
sha_path = os.path.join(PUB, 'CQFR_UNIQUE68_PUBLIC_SHA256SUMS.txt')
all_public = zip_files + ['CQFR_UNIQUE68_PUBLIC_SHA256SUMS.txt']
with open(sha_path, 'w') as f:
    for fn in all_public[:-1]:
        fp = os.path.join(PUB, fn)
        if os.path.isfile(fp):
            f.write(f'{hashlib.sha256(open(fp, "rb").read()).hexdigest()}  {fn}\n')
os.utime(sha_path, (FIXED_MTIME, FIXED_MTIME))
# Rewrite with self-hash
with open(sha_path, 'w') as f:
    for fn in all_public:
        fp = os.path.join(PUB, fn)
        if os.path.isfile(fp):
            f.write(f'{hashlib.sha256(open(fp, "rb").read()).hexdigest()}  {fn}\n')
os.utime(sha_path, (FIXED_MTIME, FIXED_MTIME))

# Rebuild ZIP with checksums included
zip_files.append('CQFR_UNIQUE68_PUBLIC_SHA256SUMS.txt')
os.remove(zip_path)
subprocess.run(['zip', '-r', '-X', zip_path] + zip_files, cwd=PUB,
               capture_output=True, text=True, check=True)
subprocess.run(['unzip', '-t', zip_path], cwd=PUB, capture_output=True, text=True, check=True)

with zipfile.ZipFile(zip_path) as zf:
    for info in zf.infolist():
        assert info.extra == b'', f'Extra field after rebuild: {info.filename}'

# Detached SHA
zip_sha = hashlib.sha256(open(zip_path, 'rb').read()).hexdigest()
with open(zip_path + '.sha256', 'w') as f:
    f.write(f'{zip_sha}  CQFR_UNIQUE68_PUBLIC.zip\n')

# ZIP_MEMBERS
with open(os.path.join(PUB, 'ZIP_MEMBERS.txt'), 'w') as f:
    for m in sorted(zip_files): f.write(m + '\n')

print(f'Public ZIP: {zip_sha} ({os.path.getsize(zip_path)/1024/1024:.1f}MB)')
succ = sum(1 for bk in blind_key if bk['task_success_representative'])
fail = sum(1 for bk in blind_key if not bk['task_success_representative'])
al = sum(1 for bk in blind_key if bk['arm_lock_representative'])
print(f'Unique videos: {len(blind_key)} ({succ} succ, {fail} fail, {al} AL, {len(blind_key)-al} NL)')
for cond in sorted(set(bk['condition_representative'] for bk in blind_key)):
    c = sum(1 for bk in blind_key if bk['condition_representative'] == cond)
    print(f'  {cond}: {c}')
print('Done. All gates passed.')
