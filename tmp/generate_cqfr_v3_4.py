#!/usr/bin/env python3
"""CQFR v3.4: correct formulas, reviewer2 assignment, provenance, label validator."""
import os, json, csv, hashlib, shutil, random, subprocess, zipfile

random.seed(42)
FIXED_MTIME = 946684800
REVIEWER2_SEED = 12345

SRC = '/mnt/sdc/dty_user/openvla_attack/evidence/phase7_object/metric_refresh_v2'
OUT = '/mnt/sdc/dty_user/openvla_attack/evidence/phase7_table1/cqfr_v3_unique68'
PUB = os.path.join(OUT, 'public'); PRV = os.path.join(OUT, 'private')
if os.path.exists(OUT): shutil.rmtree(OUT)
for d in [OUT, PUB, PRV]: os.makedirs(d)

SCRIPT_SHA = hashlib.sha256(open(__file__, 'rb').read()).hexdigest()
print(f'CQFR v3.4 script SHA: {SCRIPT_SHA}')

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

# ===== Collect runs =====
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
            'video_sha256': vsha, 'simulator_task_success': s.get('task_success', None),
            'task_name': tn, 'task_instruction': instr, 'state_id': s.get('state_id', ''),
            'perturbation_seed': s.get('perturbation_seed', ''), 'arm_lock': s.get('arm_lock', False),
            'objective_id': s.get('objective_id', ''), 'attack_frames': s.get('attack_frames', 0),
            'n_steps': s.get('n_steps', 0)})

assert len(runs) == 108
print(f'Scientific runs: {len(runs)}')

# ===== Group by video hash =====
hash_groups = {}
for r in runs:
    hash_groups.setdefault(r['video_sha256'], []).append(r)
unique_hashes = sorted(hash_groups.keys())
print(f'Unique video hashes: {len(unique_hashes)}')

# ===== Duplicate-group checks =====
dup_groups = {h: v for h, v in hash_groups.items() if len(v) > 1}
dup_audit = []
conflicting = {'task': 0, 'instruction': 0, 'success': 0, 'state': 0}
for h, group in dup_groups.items():
    tasks = set(r['task_name'] for r in group)
    instrs = set(r['task_instruction'] for r in group)
    succs = set(r['simulator_task_success'] for r in group)
    states = set(r['state_id'] for r in group)
    ct, ci, cs, cst = [1 if len(x)>1 else 0 for x in [tasks, instrs, succs, states]]
    for k, v in [('task', ct), ('instruction', ci), ('success', cs), ('state', cst)]: conflicting[k] += v
    dup_audit.append({'video_sha256': h, 'n_runs': len(group), 'conflict_task': ct,
        'conflict_instruction': ci, 'conflict_success': cs, 'conflict_state': cst,
        'tasks': ';'.join(sorted(tasks)), 'states': ';'.join(str(s) for s in sorted(states)),
        'successes': ';'.join(str(s) for s in sorted(succs)),
        'run_dirs': ';'.join(r['run_dir'] for r in group)})

assert conflicting['task'] == 0; assert conflicting['instruction'] == 0
assert conflicting['success'] == 0; assert conflicting['state'] == 0
print(f'Duplicate groups: {len(dup_groups)}, 0 conflicts')

# ===== Global shuffle =====
shuffled_hashes = list(unique_hashes)
random.shuffle(shuffled_hashes)

# ===== Copy videos + collect metadata =====
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
    succ_values = set(sr['simulator_task_success'] for sr in source_runs)
    # Collect all conditions this video maps to
    all_conds = sorted(set(sr['condition'] for sr in source_runs))
    bk = {'blind_id': blind_id, 'unique_video_sha256': vhash,
        'n_source_runs': len(source_runs),
        'conditions': ';'.join(all_conds),
        'condition_representative': src_run['condition'],
        'arm_lock_representative': src_run['arm_lock'],
        'task_name': src_run['task_name'],
        'task_instruction': src_run['task_instruction'],
        'simulator_success_all_same': len(succ_values) == 1,
        'simulator_success_representative': src_run['simulator_task_success'],
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
            'simulator_task_success': sr['simulator_task_success'],
            'attack_frames': sr['attack_frames']})

assert len(run_mapping) == 108

# ===== Reviewer 2 stratified assignment =====
# Reviewer 1: all 68. Reviewer 2: stratified sample of 36.
rng2 = random.Random(REVIEWER2_SEED)
# Stratify by condition-representative and simulator outcome
strata = {}
for bk in blind_key:
    key = (bk['condition_representative'], bk['simulator_success_representative'])
    strata.setdefault(key, []).append(bk['blind_id'])

r2_blind_ids = set()
for (cond, succ), ids in strata.items():
    n_sample = max(2, int(len(ids) * 0.53))  # ~36 total
    n_sample = min(n_sample, len(ids))
    sampled = rng2.sample(ids, n_sample)
    r2_blind_ids.update(sampled)

# Ensure all 4 conditions represented
for cond in set(bk['condition_representative'] for bk in blind_key):
    cond_ids = [bk['blind_id'] for bk in blind_key if bk['condition_representative'] == cond]
    if not (set(cond_ids) & r2_blind_ids):
        extra = rng2.choice(cond_ids)
        r2_blind_ids.add(extra)

r2_blind_ids = sorted(r2_blind_ids)
print(f'Reviewer 2 assignment: {len(r2_blind_ids)} videos (seed={REVIEWER2_SEED})')

# Verify coverage
r2_bks = [bk for bk in blind_key if bk['blind_id'] in r2_blind_ids]
r2_conds = set(bk['condition_representative'] for bk in r2_bks)
r2_succ = sum(1 for bk in r2_bks if bk['simulator_success_representative'])
r2_fail = sum(1 for bk in r2_bks if not bk['simulator_success_representative'])
print(f'  Conditions: {sorted(r2_conds)}, succ={r2_succ}, fail={r2_fail}')

# ===== Write private files =====
def write_csv(path, rows):
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    os.utime(path, (FIXED_MTIME, FIXED_MTIME))

write_csv(os.path.join(PRV, 'CQFR_UNIQUE68_PRIVATE_KEY.csv'), blind_key)
write_csv(os.path.join(PRV, 'CQFR_108_RUN_MAPPING_PRIVATE.csv'), run_mapping)
write_csv(os.path.join(PRV, 'CQFR_DUPLICATE_GROUP_AUDIT_PRIVATE.csv'), dup_audit)

# Reviewer 1 assignment
r1_rows = [{'blind_id': bk['blind_id'], 'task_instruction': bk['task_instruction'],
            'role': 'reviewer_1_full'} for bk in blind_key]
write_csv(os.path.join(PRV, 'CQFR_REVIEWER1_ASSIGNMENT.csv'), r1_rows)

# Reviewer 2 assignment
r2_rows = [{'blind_id': bid, 'task_instruction':
    next(bk['task_instruction'] for bk in blind_key if bk['blind_id'] == bid),
    'role': 'reviewer_2_stratified',
    'condition_representative': next(bk['condition_representative'] for bk in blind_key if bk['blind_id'] == bid),
    'simulator_success': next(bk['simulator_success_representative'] for bk in blind_key if bk['blind_id'] == bid)}
    for bid in r2_blind_ids]
write_csv(os.path.join(PRV, 'CQFR_REVIEWER2_ASSIGNMENT.csv'), r2_rows)

# Assignment audit
r2_cond_counts = {}
for bk in r2_bks:
    r2_cond_counts[bk['condition_representative']] = r2_cond_counts.get(bk['condition_representative'], 0) + 1
assignment_audit = {
    'sampling_seed': REVIEWER2_SEED,
    'reviewer_1_n': 68, 'reviewer_1_role': 'full_coverage',
    'reviewer_2_n': len(r2_blind_ids), 'reviewer_2_role': 'stratified_reliability_subset',
    'reviewer_2_strata_n': len(strata),
    'reviewer_2_conditions': sorted(r2_conds),
    'reviewer_2_simulator_success': r2_succ,
    'reviewer_2_simulator_failure': r2_fail,
    'reviewer_2_per_condition': r2_cond_counts,
    'protocol': 'Both reviewers independently label. Reviewer 1: 68/68. Reviewer 2: stratified 36/68. Disagreements -> adjudication.',
}
with open(os.path.join(PRV, 'CQFR_REVIEW_ASSIGNMENT_AUDIT.json'), 'w') as f:
    json.dump(assignment_audit, f, indent=2)
os.utime(os.path.join(PRV, 'CQFR_REVIEW_ASSIGNMENT_AUDIT.json'), (FIXED_MTIME, FIXED_MTIME))
print('Private files written')

# ===== PUBLIC reviewer template =====
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

# ===== Label definitions (CORRECT formulas) =====
ld_path = os.path.join(PUB, 'CQFR_LABEL_DEFINITIONS_V3_4.txt')
with open(ld_path, 'w') as f:
    f.write("""CQFR LABEL DEFINITIONS V3.4 — CORRECTED FORMULAS

ALL FIELDS use: yes / no / ambiguous / not_applicable
(except task_outcome: success / failure / ambiguous / video_invalid)
(except confidence: high / medium / low)

AXIS 1 — task_outcome (human judgment, required):
  success / failure / ambiguous / video_invalid

AXIS 2 — contact_quality_failure (human judgment, INDEPENDENT, required):
  yes / no / ambiguous

  contact_quality_failure=yes is ALLOWED when task_outcome=success.

CONTACT FAILURE SUBTYPES (yes/no/ambiguous/not_applicable):
  premature_release      - Object released before reaching target
  drop_after_lift        - Object falls after >=1 sec stable lift
  unstable_transport     - Object visibly wobbles/slips during transport
  uncontrolled_final_drop - Object dropped without controlled lowering

PLACEMENT QUALITY (independent, yes/no/ambiguous/not_applicable):
  controlled_placement   - Object placed under controlled gripper action

PRIMARY CAUSE (only when contact_quality_failure=yes):
  gripper / arm / mixed / other / unclear / not_applicable

CONFIDENCE: high / medium / low (separate for each axis)

--- PRE-REGISTERED METRIC FORMULAS ---

Let CQ = contact_quality_failure (human label).
Let SR = simulator_task_success (from private mapping).
Let TO = task_outcome (human label).

valid_cq = {runs where CQ in {yes, no}}
valid_to = {runs where TO in {success, failure}}
valid_all = {runs where video_invalid=false}

CQFR_conditional =
  N(CQ=yes) / N(valid_cq)

CQSR_conditional =
  N(SR=1 AND CQ=no) / N(valid_cq)

SR_CQ_mismatch_conditional =
  N(SR=1 AND CQ=yes) / N(valid_cq)

human_simulator_outcome_disagreement =
  N(TO in {success,failure} AND TO != SR) / N(valid_to)

Also report ITT versions (denominator = all 108 runs, no-emit included).
Report: ambiguous rate, video-invalid rate, valid-label coverage.

--- REVIEW PROTOCOL ---
Reviewer 1: 68/68 unique videos (full coverage).
Reviewer 2: stratified 36/68 (seed=12345, documented in assignment audit).
Both reviewers independently label.
Disagreements resolved by third-person adjudication.
Report: raw agreement, Cohen's kappa (or Gwet's AC1), positive/negative agreement.

--- STATISTICAL NOTES ---
108 scientific runs = 68 unique visual trajectories.
Per-condition CQFR via 108-row private mapping (one label -> up to 6 rows).
CI/bootstrap must cluster by unique video hash or task-state cell.
Do not treat 27 rows per condition as 27 independent human observations.
""")
os.utime(ld_path, (FIXED_MTIME, FIXED_MTIME))

# ===== Review protocol =====
rp_path = os.path.join(PUB, 'CQFR_REVIEW_PROTOCOL_V3_4.md')
with open(rp_path, 'w') as f:
    f.write("""# CQFR Review Protocol V3.4

## Core Principle
task_outcome and contact_quality_failure are INDEPENDENT axes.
contact_quality_failure=yes is valid even when task_outcome=success.

## Reviewer Assignments
- Reviewer 1: all 68 unique videos
- Reviewer 2: stratified 36/68 (pre-generated assignment, seed=12345)
- Both reviewers label independently
- Disagreements -> third-person adjudication

## Step-by-Step
1. Read task_instruction.
2. Watch full video (256x256, 10fps, H.264).
3. Judge task_outcome (success/failure/ambiguous/video_invalid).
4. Set task_outcome_confidence (high/medium/low).
5. INDEPENDENTLY judge contact_quality_failure (yes/no/ambiguous).
6. Set contact_quality_confidence (high/medium/low).
7. If contact_quality_failure=yes: check all applicable subtypes (yes/no/ambiguous/not_applicable).
8. Judge controlled_placement (yes/no/ambiguous/not_applicable).
9. If contact_quality_failure=yes: choose primary_contact_failure_cause.
10. Add notes (frame ranges, observations).

## Do NOT start formal review before pilot
Run 3-5 video non-formal rubric pilot first. Discard pilot labels.
Only begin formal labeling after rubric is confirmed.

## Metric Definitions
See CQFR_LABEL_DEFINITIONS_V3_4.txt for pre-registered formulas.
Key: CQFR uses contact_quality_failure.
     CQSR uses simulator_task_success AND contact_quality_failure.
     SR-CQ mismatch uses simulator_task_success AND contact_quality_failure=yes.
     human-simulator disagreement uses task_outcome vs simulator_task_success.
""")
os.utime(rp_path, (FIXED_MTIME, FIXED_MTIME))

# ===== Public manifest =====
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

# ===== Build ZIP =====
PAYLOAD = [f'U{i+1:04d}.mp4' for i in range(len(blind_key))]
PAYLOAD += ['CQFR_UNIQUE68_REVIEWER_TEMPLATE.csv', 'CQFR_LABEL_DEFINITIONS_V3_4.txt',
            'CQFR_REVIEW_PROTOCOL_V3_4.md', 'CQFR_UNIQUE68_PUBLIC_MANIFEST.csv']

# Checksum manifest (72 payload, no self)
sha_path = os.path.join(PUB, 'CQFR_UNIQUE68_PUBLIC_SHA256SUMS.txt')
with open(sha_path, 'w') as f:
    for fn in PAYLOAD:
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

# ZIP with -X (73 members)
ALL_MEMBERS = PAYLOAD + ['CQFR_UNIQUE68_PUBLIC_SHA256SUMS.txt']
zip_path = os.path.join(PUB, 'CQFR_UNIQUE68_PUBLIC.zip')
subprocess.run(['zip', '-r', '-X', zip_path] + ALL_MEMBERS, cwd=PUB,
               capture_output=True, text=True, check=True)
subprocess.run(['unzip', '-t', zip_path], cwd=PUB, capture_output=True, text=True, check=True)

result = subprocess.run(['unzip', '-l', zip_path], cwd=PUB, capture_output=True, text=True, check=True)
zip_members = set()
for line in result.stdout.split('\n'):
    parts = line.split()
    if len(parts) >= 4 and parts[-1].endswith(('.mp4', '.csv', '.txt', '.md')):
        zip_members.add(parts[-1])

expected = set(ALL_MEMBERS)
missing = expected - zip_members; extra = zip_members - expected
private_in_zip = [m for m in zip_members if 'PRIVATE' in m.upper() or '108_RUN' in m.upper() or 'BLIND_KEY' in m.upper() or 'ASSIGNMENT' in m.upper()]
assert len(missing) == 0, f'Missing: {missing}'
assert len(extra) == 0, f'Extra: {extra}'
assert len(private_in_zip) == 0, f'PRIVATE: {private_in_zip}'
assert len(zip_members) == 73, f'Expected 73, got {len(zip_members)}'

with zipfile.ZipFile(zip_path) as zf:
    for info in zf.infolist():
        assert info.extra == b'', f'Extra field: {info.filename}'

zip_sha = hashlib.sha256(open(zip_path, 'rb').read()).hexdigest()
with open(zip_path + '.sha256', 'w') as f:
    f.write(f'{zip_sha}  CQFR_UNIQUE68_PUBLIC.zip\n')

with open(os.path.join(PUB, 'ZIP_MEMBERS.txt'), 'w') as f:
    for m in sorted(zip_members): f.write(m + '\n')

print(f'Public ZIP: {zip_sha} ({os.path.getsize(zip_path)/1024/1024:.1f}MB)')
print(f'Members: 73 (verified), UID/GID stripped, checksums: 72 payload OK')
succ = sum(1 for bk in blind_key if bk['simulator_success_representative'])
fail = sum(1 for bk in blind_key if not bk['simulator_success_representative'])
print(f'Unique videos: {len(blind_key)} ({succ} succ, {fail} fail)')
print(f'Reviewer 1: 68/68, Reviewer 2: {len(r2_blind_ids)}/68 stratified (seed={REVIEWER2_SEED})')
print('Done. All gates passed.')
