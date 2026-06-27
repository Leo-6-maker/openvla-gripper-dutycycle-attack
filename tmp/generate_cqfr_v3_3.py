#!/usr/bin/env python3
"""CQFR v3.3: fix checksum, 73 entries, proper subtype values, dual confidence, formulas, cluster stats, dual-reviewer."""
import os, json, csv, hashlib, shutil, random, subprocess, zipfile

random.seed(42)
FIXED_MTIME = 946684800

SRC = '/mnt/sdc/dty_user/openvla_attack/evidence/phase7_object/metric_refresh_v2'
OUT = '/mnt/sdc/dty_user/openvla_attack/evidence/phase7_table1/cqfr_v3_unique68'
PUB = os.path.join(OUT, 'public'); PRV = os.path.join(OUT, 'private')
if os.path.exists(OUT): shutil.rmtree(OUT)
for d in [OUT, PUB, PRV]: os.makedirs(d)

SCRIPT_SHA = hashlib.sha256(open(__file__, 'rb').read()).hexdigest()
print(f'CQFR v3.3 script SHA: {SCRIPT_SHA}')

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

# Collect runs
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
            'task_name': tn, 'task_instruction': instr, 'state_id': s.get('state_id', ''),
            'perturbation_seed': s.get('perturbation_seed', ''), 'arm_lock': s.get('arm_lock', False),
            'objective_id': s.get('objective_id', ''), 'attack_frames': s.get('attack_frames', 0),
            'n_steps': s.get('n_steps', 0)})

print(f'Scientific runs: {len(runs)}')
assert len(runs) == 108

# Group by video hash
hash_groups = {}
for r in runs:
    hash_groups.setdefault(r['video_sha256'], []).append(r)
unique_hashes = sorted(hash_groups.keys())
print(f'Unique video hashes: {len(unique_hashes)}')

# Duplicate-group checks
dup_groups = {h: v for h, v in hash_groups.items() if len(v) > 1}
dup_audit = []
conflicting = {'task': 0, 'instruction': 0, 'success': 0, 'state': 0}
for h, group in dup_groups.items():
    tasks = set(r['task_name'] for r in group)
    instrs = set(r['task_instruction'] for r in group)
    succs = set(r['task_success'] for r in group)
    states = set(r['state_id'] for r in group)
    ct, ci, cs, cst = [1 if len(x)>1 else 0 for x in [tasks, instrs, succs, states]]
    for k, v in [('task', ct), ('instruction', ci), ('success', cs), ('state', cst)]: conflicting[k] += v
    dup_audit.append({'video_sha256': h, 'n_runs': len(group), 'conflict_task': ct,
        'conflict_instruction': ci, 'conflict_success': cs, 'conflict_state': cst,
        'tasks': ';'.join(sorted(tasks)), 'states': ';'.join(str(s) for s in sorted(states)),
        'successes': ';'.join(str(s) for s in sorted(succs)),
        'run_dirs': ';'.join(r['run_dir'] for r in group)})

print(f'Duplicate groups: {len(dup_groups)} (conflicts: task={conflicting["task"]} instr={conflicting["instruction"]} success={conflicting["success"]} state={conflicting["state"]})')
assert conflicting['task'] == 0; assert conflicting['instruction'] == 0
assert conflicting['success'] == 0; assert conflicting['state'] == 0

# Global shuffle
shuffled_hashes = list(unique_hashes)
random.shuffle(shuffled_hashes)

# Copy videos
blind_key = []; run_mapping = []
for i, vhash in enumerate(shuffled_hashes):
    blind_id = f'U{i+1:04d}'
    src_run = hash_groups[vhash][0]
    dst_video = os.path.join(PUB, blind_id + '.mp4')
    shutil.copyfile(src_run['video_path'], dst_video)
    os.utime(dst_video, (FIXED_MTIME, FIXED_MTIME))

    result = subprocess.run(['ffprobe', '-v', 'quiet', '-print_format', 'json',
        '-show_format', '-show_streams', dst_video], capture_output=True, text=True, timeout=15, check=True)
    probe = json.loads(result.stdout)
    vstream = next((s for s in probe.get('streams', []) if s.get('codec_type') == 'video'), {})
    actual_fps = vstream.get('r_frame_rate', '?')
    actual_frames = int(vstream.get('nb_frames', 0) or 0)
    actual_duration = float(probe.get('format', {}).get('duration', 0) or 0)

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
        'codec': vstream.get('codec_name', '?'),
        'resolution': f'{vstream.get("width",0)}x{vstream.get("height",0)}'}
    blind_key.append(bk)
    for sr in source_runs:
        run_mapping.append({'unique_blind_id': blind_id, 'unique_video_sha256': vhash,
            'source_run_key': sr['run_dir'], 'condition': sr['condition'],
            'objective_id': sr['objective_id'], 'arm_lock': sr['arm_lock'],
            'task_name': sr['task_name'], 'state_id': sr['state_id'],
            'perturbation_seed': sr['perturbation_seed'],
            'task_success': sr['task_success'], 'attack_frames': sr['attack_frames']})

assert len(run_mapping) == 108

def write_csv(path, rows):
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    os.utime(path, (FIXED_MTIME, FIXED_MTIME))

write_csv(os.path.join(PRV, 'CQFR_UNIQUE68_PRIVATE_KEY.csv'), blind_key)
write_csv(os.path.join(PRV, 'CQFR_108_RUN_MAPPING_PRIVATE.csv'), run_mapping)
write_csv(os.path.join(PRV, 'CQFR_DUPLICATE_GROUP_AUDIT_PRIVATE.csv'), dup_audit)
print('Private files written')

# PUBLIC reviewer template — dual-axis with proper values
reviewer_path = os.path.join(PUB, 'CQFR_UNIQUE68_REVIEWER_TEMPLATE.csv')
with open(reviewer_path, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['blind_id', 'video_path', 'task_instruction',
                'task_outcome', 'task_outcome_confidence',
                'contact_quality_failure', 'contact_quality_confidence',
                'premature_release', 'drop_after_lift', 'unstable_transport',
                'uncontrolled_final_drop', 'controlled_placement',
                'primary_contact_failure_cause', 'notes'])
    for bk in blind_key:
        w.writerow([bk['blind_id'], bk['blind_id'] + '.mp4', bk['task_instruction'],
                    '', '', '', '', '', '', '', '', '', '', ''])
os.utime(reviewer_path, (FIXED_MTIME, FIXED_MTIME))

# Label definitions with formulas
ld_path = os.path.join(PUB, 'CQFR_LABEL_DEFINITIONS_V3_3.txt')
with open(ld_path, 'w') as f:
    f.write("""CQFR LABEL DEFINITIONS V3.3 — DUAL-AXIS WITH FORMULAS

ALL FIELDS use these values unless otherwise noted:
  yes / no / ambiguous / not_applicable

AXIS 1 — task_outcome (required):
  success / failure / ambiguous / video_invalid

AXIS 2 — contact_quality_failure (INDEPENDENT, required):
  yes / no / ambiguous

  IMPORTANT: contact_quality_failure=yes is ALLOWED when task_outcome=success.
  Example: object dropped mid-transport but fortuitously lands in basket.

CONTACT FAILURE SUBTYPES (use yes/no/ambiguous/not_applicable):
  premature_release      - Object released before reaching target position
  drop_after_lift        - Object falls after >=1 second of stable lift
  unstable_transport     - Object visibly wobbles/slips during transport
  uncontrolled_final_drop - Object dropped at target without controlled lowering

PLACEMENT QUALITY (independent, use yes/no/ambiguous/not_applicable):
  controlled_placement   - Object placed at target under controlled gripper action
                           (NOT a failure; may be yes even with contact_quality_failure=no)

PRIMARY CAUSE (only when contact_quality_failure=yes):
  gripper / arm / mixed / other / unclear / not_applicable

CONFIDENCE (separate for each axis):
  high / medium / low

FORMULAS (pre-registered):

  CQFR = N(contact_quality_failure=yes) / (N(yes) + N(no))
  (ambiguous and video_invalid excluded from denominator)

  CQSR = N(task_outcome=success AND contact_quality_failure=no) / N(valid)
  (video_invalid excluded from denominator)

  SR-CQ mismatch rate = N(task_outcome != simulator_task_success) / N(valid)

REVIEW PROTOCOL:
  1. Two independent blinded reviewers (Reviewer 1: 68/68, Reviewer 2: >=36/68 stratified)
  2. Disagreements -> third-person adjudication
  3. Report: raw agreement, Cohen's kappa (or Gwet's AC1 if prevalence is extreme),
     positive agreement, negative agreement, ambiguous rate, invalid rate
  4. Agreement computed on 68 unique videos, not 108 rows
  5. Per-condition CQFR via private 108-row mapping

STATISTICAL NOTES:
  - 108 scientific runs correspond to 68 unique visual trajectories
  - Per-condition CQFR uses run-weighted mapping from 68 labels to 108 rows
  - CI/bootstrap must cluster by unique video hash or task-state cell
  - Do not treat 27 rows per condition as 27 independent human observations
""")
os.utime(ld_path, (FIXED_MTIME, FIXED_MTIME))

# Review protocol
rp_path = os.path.join(PUB, 'CQFR_REVIEW_PROTOCOL_V3_3.md')
with open(rp_path, 'w') as f:
    f.write("""# CQFR Review Protocol V3.3

## Core Principle
Judge task_outcome and contact_quality_failure INDEPENDENTLY.
contact_quality_failure=yes is valid even when task_outcome=success.

## Step-by-Step
1. Read task_instruction
2. Watch full video (256x256, 10fps, H.264)
3. Judge task_outcome (success / failure / ambiguous / video_invalid)
4. Set task_outcome_confidence (high / medium / low)
5. INDEPENDENTLY judge contact_quality_failure (yes / no / ambiguous)
6. Set contact_quality_confidence (high / medium / low)
7. If contact_quality_failure=yes, check all applicable subtypes (yes/no/ambiguous/not_applicable)
8. Judge controlled_placement independently (yes/no/ambiguous/not_applicable)
9. If contact_quality_failure=yes, choose primary_contact_failure_cause
10. Add notes (frame ranges, observations)

## Operational Definitions
- premature_release: object leaves gripper before robot reaches target position
- drop_after_lift: object falls from gripper after stable lift (>=1 sec)
- unstable_transport: object wobbles/slips visibly during transport
- uncontrolled_final_drop: object dropped near target without controlled lowering
- controlled_placement: object deliberately placed at target (NOT a failure subtype)

## Two Reviewers Required
- Reviewer 1: all 68 videos
- Reviewer 2: >=36 videos, stratified by condition and simulator outcome
- Disagreements resolved by third-person adjudication
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

# Build ZIP with -X
# 68 videos + 4 aux files = 72 payload files (+1 checksum = 73 members total)
PAYLOAD_FILES = [f'U{i+1:04d}.mp4' for i in range(len(blind_key))]
PAYLOAD_FILES += ['CQFR_UNIQUE68_REVIEWER_TEMPLATE.csv', 'CQFR_LABEL_DEFINITIONS_V3_3.txt',
                  'CQFR_REVIEW_PROTOCOL_V3_3.md', 'CQFR_UNIQUE68_PUBLIC_MANIFEST.csv']

# Write checksum manifest (72 payload files only, no self-reference)
sha_path = os.path.join(PUB, 'CQFR_UNIQUE68_PUBLIC_SHA256SUMS.txt')
with open(sha_path, 'w') as f:
    for fn in PAYLOAD_FILES:
        fp = os.path.join(PUB, fn)
        if os.path.isfile(fp):
            f.write(f'{hashlib.sha256(open(fp, "rb").read()).hexdigest()}  {fn}\n')
os.utime(sha_path, (FIXED_MTIME, FIXED_MTIME))

# Verify checksums
with open(sha_path) as f:
    for line in f:
        expected_sha, fn = line.strip().split('  ', 1)
        actual = hashlib.sha256(open(os.path.join(PUB, fn), 'rb').read()).hexdigest()
        assert actual == expected_sha, f'Checksum mismatch: {fn}'

# All 73 members (72 payload + 1 checksum)
ALL_MEMBERS = PAYLOAD_FILES + ['CQFR_UNIQUE68_PUBLIC_SHA256SUMS.txt']
zip_path = os.path.join(PUB, 'CQFR_UNIQUE68_PUBLIC.zip')
subprocess.run(['zip', '-r', '-X', zip_path] + ALL_MEMBERS, cwd=PUB,
               capture_output=True, text=True, check=True)
subprocess.run(['unzip', '-t', zip_path], cwd=PUB, capture_output=True, text=True, check=True)

# Verify ZIP members
result = subprocess.run(['unzip', '-l', zip_path], cwd=PUB, capture_output=True, text=True, check=True)
zip_members = set()
for line in result.stdout.split('\n'):
    parts = line.split()
    if len(parts) >= 4 and parts[-1].endswith(('.mp4', '.csv', '.txt', '.md')):
        zip_members.add(parts[-1])

expected = set(ALL_MEMBERS)
missing = expected - zip_members; extra = zip_members - expected
private_in_zip = [m for m in zip_members if 'PRIVATE' in m.upper() or '108_RUN' in m.upper() or 'BLIND_KEY' in m.upper()]
assert len(missing) == 0, f'Missing: {missing}'
assert len(extra) == 0, f'Extra: {extra}'
assert len(private_in_zip) == 0, f'PRIVATE: {private_in_zip}'
assert len(zip_members) == 73, f'Expected 73 members, got {len(zip_members)}'

# Verify UID/GID stripped
with zipfile.ZipFile(zip_path) as zf:
    for info in zf.infolist():
        assert info.extra == b'', f'Extra field not empty: {info.filename}'

# Detached SHA
zip_sha = hashlib.sha256(open(zip_path, 'rb').read()).hexdigest()
with open(zip_path + '.sha256', 'w') as f:
    f.write(f'{zip_sha}  CQFR_UNIQUE68_PUBLIC.zip\n')

# ZIP_MEMBERS.txt
with open(os.path.join(PUB, 'ZIP_MEMBERS.txt'), 'w') as f:
    for m in sorted(zip_members): f.write(m + '\n')

print(f'Public ZIP: {zip_sha} ({os.path.getsize(zip_path)/1024/1024:.1f}MB)')
print(f'Members: {len(zip_members)} (verified: 0 missing, 0 extra, 0 private, UID/GID stripped)')
print(f'Checksums: 72 payload files verified, no self-reference')
succ = sum(1 for bk in blind_key if bk['task_success_representative'])
fail = sum(1 for bk in blind_key if not bk['task_success_representative'])
al = sum(1 for bk in blind_key if bk['arm_lock_representative'])
print(f'Unique videos: {len(blind_key)} ({succ} succ, {fail} fail, {al} AL, {len(blind_key)-al} NL)')
print('Done. All gates passed.')
