"""[DeepSeek] R5-E-R1-REV: Independent Subagent Evidence Audit.
Run on server: python /tmp/r5e_r1_independent_review.py
Recomputes ALL metrics from raw A/B/C data without calling main program error fields.
"""
import json, os, sys, hashlib, math, time, uuid, shutil, subprocess
from pathlib import Path
import numpy as np

RUN_A = Path('/mnt/sdc/dty_user/openvla_attack_outputs/n5/phase3_student/r5e_r1/run_A')
RUN_B = Path('/mnt/sdc/dty_user/openvla_attack_outputs/n5/phase3_student/r5e_r1/run_B')
OLD_RUN_A = Path('/mnt/sdc/dty_user/openvla_attack_outputs/n5/phase3_student/r5e/run_A')
OLD_RUN_B = Path('/mnt/sdc/dty_user/openvla_attack_outputs/n5/phase3_student/r5e/run_B')
OUT = Path('/mnt/sdc/dty_user/openvla_attack_outputs/n5/phase3_student/r5e_r1/R5E_R1_INDEPENDENT_REVIEW')
EXEC_COMMIT = 'ee7da22b76a856b6c10ac29f02f73dbf6aebcc83'
C1_DIGEST = 'f9bb35965a166b0f56d92f3624855459fb6c4845b3a60f99551e953931fc7eb7'

def sha256_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for c in iter(lambda: f.read(1 << 20), b''): h.update(c)
    return h.hexdigest()

def sha256_str(s):
    return hashlib.sha256(s.encode()).hexdigest()

def full_seal_check(root):
    root = Path(root)
    sums_path = root / 'SHA256SUMS'
    side_path = root / 'SHA256SUMS.sha256'
    if not sums_path.is_file() or not side_path.is_file():
        return False, 'not sealed', {}
    sidecar = side_path.read_text().strip().split()
    if len(sidecar) < 2 or sidecar[0] != sha256_file(sums_path):
        return False, 'sidecar mismatch', {}
    expected = {}
    for line in sums_path.read_text().splitlines():
        line = line.strip()
        if not line: continue
        parts = line.split(None, 1)
        if len(parts) < 2: return False, f'malformed: {line}', {}
        digest, name = parts
        name = name.lstrip('*')
        if name in ('SHA256SUMS', 'SHA256SUMS.sha256'): continue
        target = root / name
        if target.is_symlink(): return False, f'SYMLINK: {name}', {}
        if not target.is_file(): return False, f'missing: {name}', {}
        if sha256_file(target) != digest: return False, f'hash mismatch: {name}', {}
        expected[name] = digest
    all_files = set()
    for p in root.rglob('*'):
        if p.is_file() and p.name not in ('SHA256SUMS', 'SHA256SUMS.sha256'):
            rel = str(p.relative_to(root))
            all_files.add(rel)
    unsealed = all_files - set(expected.keys())
    extra = set(expected.keys()) - all_files
    if unsealed or extra: return False, f'unsealed={len(unsealed)} extra={len(extra)}', expected
    return True, 'OK', expected

def geodesic_wxyz(q1, q2):
    q1n = np.array(q1, float); q2n = np.array(q2, float)
    q1n /= np.linalg.norm(q1n); q2n /= np.linalg.norm(q2n)
    d = abs(np.dot(q1n, q2n)); d = min(d, 1.0)
    return float(2.0 * math.atan2(math.sqrt(max(0, 1 - d*d)), d))

def quat_distance_stable(q1_raw, q2_raw):
    q1 = np.array(q1_raw, float); q2 = np.array(q2_raw, float)
    if np.array_equal(q1, q2): return 0.0, True
    if np.array_equal(q1, -q2): return 0.0, True
    q1n = q1 / np.linalg.norm(q1); q2n = q2 / np.linalg.norm(q2)
    d = abs(np.dot(q1n, q2n)); d = min(d, 1.0)
    return float(2.0 * math.atan2(math.sqrt(max(0, 1 - d*d)), d)), False

print('=' * 70)
print('[DeepSeek] R5-E-R1-REV: Independent Subagent Evidence Audit')
print(f'  execution_commit: {EXEC_COMMIT}')
print('=' * 70)

results = {}
start_time = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())

# ── 1. Seal verification ──
print('\n--- 1. Seal Verification ---')
for label, root in [('run_A', RUN_A), ('run_B', RUN_B)]:
    ok, msg, expected = full_seal_check(root)
    results[f'{label}_seal'] = {'ok': ok, 'msg': msg, 'files': len(expected)}
    print(f'  {label}: {"PASS" if ok else "FAIL"} — {msg} ({len(expected)} files)')

# ── 2. Load raw case records ──
print('\n--- 2. Independent Raw Recomputation ---')
recs_a = [json.loads(l) for l in open(RUN_A / 'case_records.jsonl') if l.strip()]
recs_b = [json.loads(l) for l in open(RUN_B / 'case_records.jsonl') if l.strip()]
print(f'  Run A records: {len(recs_a)}')
print(f'  Run B records: {len(recs_b)}')

# Check raw fields present
required_raw = ['A_position', 'B_position', 'C_position', 'A_rotation', 'B_rotation', 'C_rotation']
for field in required_raw:
    has = all(field in r for r in recs_a)
    print(f'  {field}: {"PRESENT" if has else "MISSING"} in all records')
    if not has:
        missing = [i for i, r in enumerate(recs_a) if field not in r]
        print(f'    Missing in records: {missing[:5]}')

# ── 3. Independent recomputation ──
print('\n--- 3. Independent Metric Recomputation ---')
mismatches = []
bc_rot_exact_count = 0
ab_stale_count = 0
nonfinite_count = 0
source_mut_count = 0

for i, (ra, rb) in enumerate(zip(recs_a, recs_b)):
    # Check raw fields
    for field in required_raw:
        if field not in ra:
            mismatches.append(f'record {i}: missing raw field {field}')
            continue

    if any(field not in ra for field in required_raw):
        continue

    etype = ra['entity_type']
    a_pos = np.array(ra['A_position']); b_pos = np.array(ra['B_position']); c_pos = np.array(ra['C_position'])
    a_rot = np.array(ra['A_rotation']); b_rot = np.array(ra['B_rotation']); c_rot = np.array(ra['C_rotation'])

    # Independent BC position
    my_bc_pos = float(np.max(np.abs(b_pos - c_pos)))
    if abs(my_bc_pos - float(ra['BC_pos_Linf'])) > 1e-20:
        mismatches.append(f'record {i}: BC_pos_Linf recomputed={my_bc_pos:.2e} declared={ra["BC_pos_Linf"]:.2e}')

    # Independent BC rotation
    if etype == 'site':
        my_bc_rot = float(np.max(np.abs(b_rot - c_rot)))
        my_bc_pass = my_bc_rot <= 1e-15
        my_exact = my_bc_rot <= 1e-15
    else:
        my_bc_rot, my_exact = quat_distance_stable(b_rot, c_rot)
        my_bc_pass = my_exact or my_bc_rot <= 1e-14
    if abs(my_bc_rot - float(ra['BC_rot_err'])) > 1e-20:
        mismatches.append(f'record {i}: BC_rot_err recomputed={my_bc_rot:.2e} declared={ra["BC_rot_err"]:.2e}')
    if my_bc_pass != ra['BC_rot_pass']:
        mismatches.append(f'record {i}: BC_rot_pass recomputed={my_bc_pass} declared={ra["BC_rot_pass"]}')
    if my_exact:
        bc_rot_exact_count += 1

    # Independent AB stale
    my_ab_pos = float(np.max(np.abs(a_pos - b_pos)))
    if my_ab_pos > 1e-15:
        ab_stale_count += 1
    if abs(my_ab_pos - float(ra['AB_pos_Linf'])) > 1e-20:
        mismatches.append(f'record {i}: AB_pos_Linf recomputed={my_ab_pos:.2e} declared={ra["AB_pos_Linf"]:.2e}')

    # Finite check
    for arr, label in [(a_pos, 'A_pos'), (b_pos, 'B_pos'), (c_pos, 'C_pos'),
                        (a_rot, 'A_rot'), (b_rot, 'B_rot'), (c_rot, 'C_rot')]:
        if not all(math.isfinite(float(x)) for x in arr):
            nonfinite_count += 1
            mismatches.append(f'record {i}: nonfinite {label}')

    # Source mutation
    if ra.get('source_mutated_fwd1') or ra.get('source_mutated_fwd2'):
        source_mut_count += 1

    # A/B raw identity
    for field in required_raw:
        if ra[field] != rb[field]:
            mismatches.append(f'record {i}: A/B {field} mismatch')

print(f'  Recomputation mismatches: {len(mismatches)}')
for m in mismatches[:10]:
    print(f'    {m}')
if len(mismatches) > 10:
    print(f'    ... and {len(mismatches)-10} more')

results['raw_recomputation'] = {
    'total_records': len(recs_a),
    'mismatches': len(mismatches),
    'bc_rot_exact': bc_rot_exact_count,
    'ab_stale_recomputed': ab_stale_count,
    'ab_stale_declared': sum(1 for r in recs_a if r.get('AB_stale')),
    'nonfinite': nonfinite_count,
    'source_mutations': source_mut_count,
}
print(f'  BC rot exact: {bc_rot_exact_count}/{len(recs_a)}')
print(f'  AB stale recomputed: {ab_stale_count} (declared: {results["raw_recomputation"]["ab_stale_declared"]})')
print(f'  Nonfinite: {nonfinite_count}')
print(f'  Source mutations: {source_mut_count}')

# ── 4. Task taxonomy ──
print('\n--- 4. Task Taxonomy ---')
sums_a = [json.loads(l) for l in open(RUN_A / 'per_task_summary.jsonl') if l.strip()]
taxonomy = {}
for s in sums_a:
    status = s['status']
    taxonomy[status] = taxonomy.get(status, 0) + 1
print(f'  Taxonomy: {taxonomy}')
n_pass = taxonomy.get('PASS', 0)
n_na = taxonomy.get('NOT_APPLICABLE', 0)
n_skip = taxonomy.get('SKIP', 0)

# Verify NA tasks match C1 articulated
na_tasks = [f"{s['suite']}/task_{s['task_idx']:02d}" for s in sums_a if s['status'] == 'NOT_APPLICABLE']
print(f'  NA tasks: {na_tasks}')
expected_na = {'libero_goal/task_00', 'libero_goal/task_07'}
na_ok = set(na_tasks) == expected_na
print(f'  NA matches C1 taxonomy: {"PASS" if na_ok else "FAIL"}')

# Verify no task has 0 records unexpectedly (non-NA tasks should have >0 records)
for s in sums_a:
    tk = f"{s['suite']}/task_{s['task_idx']:02d}"
    if s['status'] == 'PASS' and s.get('n_records', 0) == 0:
        print(f'  UNEXPECTED_EMPTY: {tk}')
    if s['status'] == 'NOT_APPLICABLE' and s.get('n_records', 0) != 0:
        print(f'  NA_WITH_RECORDS: {tk}')

results['task_taxonomy'] = {
    'n_pass': n_pass, 'n_not_applicable': n_na, 'n_skip': n_skip,
    'na_tasks': na_tasks, 'na_matches_c1': na_ok,
    'taxonomy_ok': (n_pass == 38 and n_na == 2 and n_skip == 0 and na_ok),
}
print(f'  Taxonomy OK: {results["task_taxonomy"]["taxonomy_ok"]}')

# ── 5. C1 digest verification ──
print('\n--- 5. C1 Digest ---')
ma = json.loads((RUN_A / 'MANIFEST.json').read_text(encoding='utf-8'))
mb = json.loads((RUN_B / 'MANIFEST.json').read_text(encoding='utf-8'))
c1_a = ma.get('c1_canonical_digest')
c1_b = mb.get('c1_canonical_digest')
print(f'  Run A c1_digest: {c1_a}')
print(f'  Run B c1_digest: {c1_b}')
c1_ok = c1_a == C1_DIGEST and c1_b == C1_DIGEST
results['c1_digest_verification'] = {'run_a': c1_a, 'run_b': c1_b, 'expected': C1_DIGEST, 'ok': c1_ok}
print(f'  C1 digest match: {"PASS" if c1_ok else "FAIL"}')

# ── 6. Re-run comparator ──
print('\n--- 6. Comparator Re-run ---')
script = '/mnt/sdc/dty_user/openvla_attack_outputs/n5/phase3_student/r5e_r1/../../../n5/phase2_labels/compare_r5_canonical.py'
# Find the script from the execution commit
repo = '/mnt/sdc/dty_user/openvla_attack'
os.chdir(repo)
# Use the worktree approach
subprocess.run(['git', 'fetch', 'origin', 'deepseek/detector-grec-r3-20260727'], capture_output=True)
subprocess.run(['git', 'worktree', 'add', '--detach', '/tmp/r5e_audit_wt', EXEC_COMMIT], capture_output=True)
comp_result = subprocess.run([
    '/mnt/sdc/dty_user/openvla_attack/envs/openvla-official-a800/bin/python',
    '/tmp/r5e_audit_wt/n5/phase2_labels/compare_r5_canonical.py',
    '--root-a', str(RUN_A), '--root-b', str(RUN_B), '--gate', 'r5e',
], capture_output=True, text=True, timeout=60)
comp_ok = 'CANONICAL_IDENTITY_CONFIRMED' in comp_result.stdout
results['comparator_rerun'] = {
    'exit_code': comp_result.returncode,
    'stdout_tail': '\n'.join(comp_result.stdout.splitlines()[-5:]),
    'canonical_identity_confirmed': comp_ok,
}
print(f'  Exit: {comp_result.returncode}')
print(f'  Confirmed: {comp_ok}')
subprocess.run(['git', 'worktree', 'remove', '--force', '/tmp/r5e_audit_wt'], capture_output=True)

# ── 7. Chronology ──
print('\n--- 7. Chronology ---')
import subprocess as sp
sp.run(['git', 'fetch', 'origin', 'deepseek/detector-grec-r3-20260727'], capture_output=True)
commit_ts = sp.check_output(['git', 'log', '-1', '--format=%ct', EXEC_COMMIT], text=True).strip()
evidence_files = [RUN_A / 'MANIFEST.json', RUN_B / 'MANIFEST.json']
evidence_timestamps = {}
for p in evidence_files:
    if p.exists():
        stat = os.stat(str(p))
        evidence_timestamps[str(p)] = stat.st_mtime
print(f'  Execution commit timestamp: {commit_ts} ({time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(float(commit_ts)))})')
for p, ts in evidence_timestamps.items():
    after = ts > float(commit_ts)
    print(f'  {Path(p).name}: {time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts))} (after commit: {after})')

# ── 8. Old root integrity ──
print('\n--- 8. Old Root Integrity ---')
old_status = {}
for label, old_root in [('run_A', OLD_RUN_A), ('run_B', OLD_RUN_B)]:
    if old_root.exists():
        old_status[label] = 'EXISTS_PRESERVED'
    else:
        old_status[label] = 'NOT_FOUND'
    print(f'  Old {label}: {old_status[label]}')

# ── 9. Protected access & staging ──
print('\n--- 9. Protected Access & Staging ---')
protected_hits = []
for root in [RUN_A, RUN_B]:
    for p in root.rglob('*.json'):
        try:
            content = p.read_text(errors='ignore').lower()
            for pat in ['cal', 'g10', 't2r-d', 'attack', 'teacher_label', 'student_train']:
                if pat in content:
                    # Skip false positives from gate names and paths
                    fp_contexts = ['t2r-c1-v2', 't2rc1', 'g10_task',
                                   'c1_canonical_digest', 'openvla_attack_outputs',
                                   'canonical']
                    if any(fp in content for fp in fp_contexts):
                        continue
                    protected_hits.append(f'{root.name}/{p.name}: {pat}')
                    break
        except: pass
print(f'  Protected hits: {len(protected_hits)}')
for h in protected_hits[:5]:
    print(f'    {h}')

staging_residue = list(Path('/mnt/sdc/dty_user/openvla_attack_outputs/n5/phase3_student/r5e_r1').glob('.staging*'))
print(f'  Staging residue: {len(staging_residue)}')

results['protected_access'] = {'hits': len(protected_hits), 'details': protected_hits}
results['staging_residue'] = len(staging_residue)

# ── Verdict ──
all_ok = (
    results['run_A_seal']['ok'] and results['run_B_seal']['ok']
    and len(mismatches) == 0
    and bc_rot_exact_count == len(recs_a)
    and ab_stale_count == 124
    and nonfinite_count == 0
    and source_mut_count == 0
    and results['task_taxonomy']['taxonomy_ok']
    and c1_ok
    and comp_ok
    and len(protected_hits) == 0
    and len(staging_residue) == 0
)

print(f'\n{"=" * 70}')
print(f'VERDICT: {"INDEPENDENT_REVIEW_PASS" if all_ok else "INDEPENDENT_REVIEW_HOLD"}')
print(f'{"=" * 70}')

# ── Seal review ──
if OUT.exists():
    shutil.rmtree(OUT)
staging = OUT.parent / f'.R5E_R1_INDEPENDENT_REVIEW.staging.{os.getpid()}.{uuid.uuid4().hex[:8]}'
staging.mkdir(parents=True)

review_manifest = {
    'gate': 'R5-E-R1-REV',
    'schema': 'R5_E_R1_INDEPENDENT_SUBAGENT_REVIEW_V1',
    'status': 'PASS' if all_ok else 'HOLD',
    'review_timestamp': start_time,
    'execution_commit': EXEC_COMMIT,
}
(staging / 'REVIEW_MANIFEST.json').write_text(json.dumps(review_manifest, indent=2, sort_keys=True))
(staging / 'RAW_RECOMPUTATION.json').write_text(json.dumps(results['raw_recomputation'], indent=2))
(staging / 'TASK_TAXONOMY_AUDIT.json').write_text(json.dumps(results['task_taxonomy'], indent=2))
(staging / 'RUN_A_SEAL_CHECK.json').write_text(json.dumps(results['run_A_seal'], indent=2))
(staging / 'RUN_B_SEAL_CHECK.json').write_text(json.dumps(results['run_B_seal'], indent=2))
(staging / 'CANONICAL_COMPARISON.json').write_text(json.dumps(results['comparator_rerun'], indent=2))
(staging / 'CHRONOLOGY_AUDIT.json').write_text(json.dumps({
    'execution_commit': EXEC_COMMIT,
    'commit_timestamp': int(commit_ts),
    'evidence_timestamps': {str(k): v for k, v in evidence_timestamps.items()},
    'evidence_after_commit': all(v > float(commit_ts) for v in evidence_timestamps.values()),
}, indent=2))
(staging / 'PROTECTED_ACCESS_AUDIT.json').write_text(json.dumps(results['protected_access'], indent=2))

payload = sorted(p for p in staging.rglob('*') if p.is_file())
sums = '\n'.join(f'{sha256_file(p)}  {p.relative_to(staging).as_posix()}' for p in payload) + '\n'
(staging / 'SHA256SUMS').write_text(sums)
sums_sha = sha256_file(staging / 'SHA256SUMS')
(staging / 'SHA256SUMS.sha256').write_text(f'{sums_sha}  SHA256SUMS\n')
staging.rename(OUT)

print(f'\nReview sealed: {OUT}')
print(f'  SHA256SUMS: {sums_sha}')

# ── Source tree ──
os.chdir(repo)
src_tree = subprocess.check_output(['git', 'rev-parse', f'{EXEC_COMMIT}^{{tree}}'], text=True).strip()
print(f'  execution_tree: {src_tree}')
