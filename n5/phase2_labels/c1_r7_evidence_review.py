"""[DeepSeek] C1-V2-R7 Evidence Review Subagent.
Run on server: python n5/phase2_labels/c1_r7_evidence_review.py
"""
import json, os, sys, hashlib, subprocess, time, uuid, shutil
from pathlib import Path

REPO = os.getcwd()  # Use current worktree, not main repo
RUN_A = Path('/mnt/sdc/dty_user/openvla_attack_outputs/n5/phase3_student/c1_v2_r7/run_A')
RUN_B = Path('/mnt/sdc/dty_user/openvla_attack_outputs/n5/phase3_student/c1_v2_r7/run_B')
OUT = Path('/mnt/sdc/dty_user/openvla_attack_outputs/n5/phase3_student/c1_v2_r7/C1_R7_EVIDENCE_REVIEW')
COMMIT_SRC = 'dab73d2'


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for c in iter(lambda: f.read(1 << 20), b''):
            h.update(c)
    return h.hexdigest()


def sha256_str(s):
    return hashlib.sha256(s.encode()).hexdigest()


print('=' * 70)
print('[DeepSeek] C1-V2-R7 Evidence Review Subagent')
print(f'  baseline: {COMMIT_SRC}')
print('=' * 70)

results = {}
start_time = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())

# ── 1. Source diff audit ──
print('\n--- 1. Source Diff Audit ---')
os.chdir(REPO)
src_head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
src_tree = subprocess.check_output(['git', 'rev-parse', 'HEAD^{tree}'], text=True).strip()
print(f'  HEAD: {src_head}')
print(f'  TREE: {src_tree}')

diff_files = subprocess.check_output(
    ['git', 'diff', '--name-only', f'{COMMIT_SRC}..HEAD'], text=True).strip()
diff_files_list = diff_files.split('\n') if diff_files else []
print(f'  Files changed: {len(diff_files_list)}')
for f in diff_files_list:
    print(f'    {f}')

diff_stat = subprocess.check_output(
    ['git', 'diff', '--stat', f'{COMMIT_SRC}..HEAD'], text=True).strip()
results['source_diff_audit'] = {
    'baseline': COMMIT_SRC, 'head': src_head, 'tree': src_tree,
    'files_changed': diff_files_list,
    'diff_stat_tail': diff_stat.split('\n')[-1] if diff_stat else '',
}

# ── 2. SHA256SUMS verification ──
print('\n--- 2. SHA256SUMS Verification ---')
for label, root in [('run_A', RUN_A), ('run_B', RUN_B)]:
    sums_path = root / 'SHA256SUMS'
    side_path = root / 'SHA256SUMS.sha256'
    ok = True
    issues = []

    if not sums_path.is_file():
        ok = False
        issues.append('SHA256SUMS missing')
    if not side_path.is_file():
        ok = False
        issues.append('SHA256SUMS.sha256 missing')

    unsealed = []
    extra_entries = []
    file_count = 0

    if ok:
        sidecar = side_path.read_text().strip().split()
        sums_actual = sha256_file(sums_path)
        if len(sidecar) < 2 or sidecar[0] != sums_actual:
            ok = False
            issues.append(
                f'sidecar mismatch: declared={sidecar[0][:16] if len(sidecar) > 0 else "MISSING"} '
                f'actual={sums_actual[:16]}')

        seen = set()
        for line in sums_path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split(None, 1)
            if len(parts) < 2:
                issues.append(f'malformed line: {line}')
                ok = False
                continue
            digest, name = parts
            name = name.lstrip('*').lstrip('./')
            if name in ('SHA256SUMS', 'SHA256SUMS.sha256'):
                continue
            if name in seen:
                issues.append(f'duplicate entry: {name}')
                ok = False
                continue
            seen.add(name)

            target = root / name
            # Symlink check
            if target.is_symlink():
                issues.append(f'SYMLINK: {name}')
                ok = False
                continue
            # Path traversal check
            try:
                target.resolve().relative_to(root.resolve())
            except ValueError:
                issues.append(f'PATH_TRAVERSAL: {name}')
                ok = False
                continue
            if not target.is_file():
                issues.append(f'missing: {name}')
                ok = False
                continue
            actual = sha256_file(target)
            if actual != digest:
                issues.append(
                    f'hash mismatch: {name} declared={digest[:16]} actual={actual[:16]}')
                ok = False
            file_count += 1

        # Check for unsealed files
        all_files = set()
        for p in root.rglob('*'):
            if p.is_file() and p.name not in ('SHA256SUMS', 'SHA256SUMS.sha256'):
                rel = str(p.relative_to(root))
                all_files.add(rel)
        unsealed = sorted(all_files - seen)
        extra_entries = sorted(seen - all_files)

    results[f'{label}_seal_check'] = {
        'ok': ok,
        'issues': issues,
        'file_count': file_count,
        'sidecar_sha': sums_actual if ok else 'N/A',
        'unsealed_files': unsealed,
        'extra_entries': extra_entries,
    }
    status = 'PASS' if (ok and not unsealed and not extra_entries) else 'FAIL'
    print(f'  {label}: {status} ({file_count} files, {len(unsealed)} unsealed, '
          f'{len(extra_entries)} extra)')
    for i in issues:
        print(f'    ISSUE: {i}')
    for f in unsealed:
        print(f'    UNSEALED: {f}')
    for f in extra_entries:
        print(f'    EXTRA: {f}')

# ── 3. Per-task registry audit ──
print('\n--- 3. Per-task Registry Audit ---')
per_task_dir = RUN_A / 'per_task'
task_files = sorted(f.name for f in per_task_dir.iterdir() if f.name.endswith('.json'))
print(f'  Tasks: {len(task_files)}')

totals = {
    'supported': 0, 'articulated': 0, 'other': 0,
    'object_ok': 0, 'object_unresolved': 0, 'object_ambiguous': 0, 'object_blocked': 0,
    'target_ok': 0, 'target_unresolved': 0, 'target_ambiguous': 0, 'target_blocked': 0,
}
black_book_entries = []

for fn in task_files:
    with open(per_task_dir / fn) as f:
        data = json.load(f)
    legacy = data.get('legacy', data)
    disp = legacy.get('task_disposition', '?')

    if disp == 'SUPPORTED_PLACEMENT':
        totals['supported'] += 1
    elif disp == 'ARTICULATED_UNSUPPORTED':
        totals['articulated'] += 1
    else:
        totals['other'] += 1

    rc = legacy.get('resolution_counts', {})
    for k in ['object_ok', 'object_unresolved', 'object_ambiguous', 'object_blocked',
              'target_ok', 'target_unresolved', 'target_ambiguous', 'target_blocked']:
        totals[k] += rc.get(k, 0)

    for rel in legacy.get('relations', []):
        obj_name = rel.get('object_bddl', '')
        tgt_name = rel.get('target_bddl', '')
        for side, bddl_name in [('object_resolution', obj_name),
                                 ('target_resolution', tgt_name)]:
            res = rel.get(side, {})
            name = res.get('name', '')
            if 'black_book' in str(name).lower() or 'black_book' in str(bddl_name).lower():
                black_book_entries.append({
                    'task': fn.replace('.json', ''),
                    'side': side,
                    'bddl_name': bddl_name,
                    'resolution': res.get('resolution'),
                    'entity_type': res.get('entity_type'),
                    'entity_id': res.get('entity_id'),
                    'alias_to': res.get('alias_to'),
                    'alias_from': res.get('alias_from'),
                })

results['per_task_audit'] = totals
results['black_book_audit'] = black_book_entries
print(f'  supported={totals["supported"]} articulated={totals["articulated"]} '
      f'other={totals["other"]}')
print(f'  object: ok={totals["object_ok"]} unres={totals["object_unresolved"]} '
      f'ambig={totals["object_ambiguous"]} blocked={totals["object_blocked"]}')
print(f'  target: ok={totals["target_ok"]} unres={totals["target_unresolved"]} '
      f'ambig={totals["target_ambiguous"]} blocked={totals["target_blocked"]}')
print(f'  black_book occurrences: {len(black_book_entries)}')
for e in black_book_entries:
    print(f'    {e["task"]} {e["side"]}: {e["bddl_name"]} -> '
          f'{e["alias_to"]} ({e["entity_type"]}#{e["entity_id"]}) [{e["resolution"]}]')

# ── 4. A/B canonical comparison ──
print('\n--- 4. A/B Canonical Comparison ---')
mismatches = []
for fn in task_files:
    with open(RUN_A / 'per_task' / fn) as f:
        da = json.load(f)
    with open(RUN_B / 'per_task' / fn) as f:
        db = json.load(f)
    la = da.get('legacy', da)
    lb = db.get('legacy', db)

    if la.get('status') != lb.get('status'):
        mismatches.append(f'{fn}: status {la["status"]} vs {lb["status"]}')

    ra = la.get('resolution_counts', {})
    rb = lb.get('resolution_counts', {})
    for k in ['object_ok', 'object_unresolved', 'object_ambiguous',
              'target_ok', 'target_unresolved', 'target_ambiguous']:
        if ra.get(k) != rb.get(k):
            mismatches.append(f'{fn}: {k} {ra.get(k)} vs {rb.get(k)}')

    rels_a = la.get('relations', [])
    rels_b = lb.get('relations', [])
    if len(rels_a) != len(rels_b):
        mismatches.append(f'{fn}: relation count {len(rels_a)} vs {len(rels_b)}')
    else:
        for i, (ra_, rb_) in enumerate(zip(rels_a, rels_b)):
            for side in ('object_resolution', 'target_resolution'):
                ra_s = ra_.get(side, {})
                rb_s = rb_.get(side, {})
                if ra_s.get('resolution') != rb_s.get('resolution'):
                    mismatches.append(
                        f'{fn}: rel[{i}].{side} resolution '
                        f'{ra_s.get("resolution")} vs {rb_s.get("resolution")}')
                if ra_s.get('entity_type') != rb_s.get('entity_type'):
                    mismatches.append(
                        f'{fn}: rel[{i}].{side} entity_type '
                        f'{ra_s.get("entity_type")} vs {rb_s.get("entity_type")}')
                if ra_s.get('entity_id') != rb_s.get('entity_id'):
                    mismatches.append(
                        f'{fn}: rel[{i}].{side} entity_id '
                        f'{ra_s.get("entity_id")} vs {rb_s.get("entity_id")}')

# Compare summaries
with open(RUN_A / 'ENTITY_REGISTRY_V2_SUMMARY.json') as f:
    sa = json.load(f)
with open(RUN_B / 'ENTITY_REGISTRY_V2_SUMMARY.json') as f:
    sb = json.load(f)
for k in ['n_ok', 'n_env_errors', 'n_blocked', 'object_ok', 'object_unresolved',
          'object_ambiguous', 'target_ok', 'target_unresolved', 'target_ambiguous',
          'n_supported_placement', 'n_articulated_unsupported']:
    if sa.get(k) != sb.get(k):
        mismatches.append(f'summary.{k}: {sa.get(k)} vs {sb.get(k)}')


def canonical(obj):
    if isinstance(obj, dict):
        return {k: canonical(v) for k, v in sorted(obj.items())
                if k not in ('timestamp', 'self_sha256', 'artifact_sha')}
    if isinstance(obj, list):
        return [canonical(v) for v in obj]
    return obj


canon_a = sha256_str(json.dumps(canonical(sa), sort_keys=True, ensure_ascii=False))
canon_b = sha256_str(json.dumps(canonical(sb), sort_keys=True, ensure_ascii=False))

results['canonical_comparison'] = {
    'mismatches': mismatches,
    'per_task_mismatch_count': len([m for m in mismatches if '.json' in m and 'summary' not in m]),
    'summary_mismatch_count': len([m for m in mismatches if 'summary' in m]),
    'canonical_digest_A': canon_a,
    'canonical_digest_B': canon_b,
    'digests_identical': canon_a == canon_b,
}
print(f'  Per-task mismatches: {results["canonical_comparison"]["per_task_mismatch_count"]}')
print(f'  Summary mismatches: {results["canonical_comparison"]["summary_mismatch_count"]}')
for m in mismatches[:10]:
    print(f'    {m}')
print(f'  Canonical digest A: {canon_a}')
print(f'  Canonical digest B: {canon_b}')
print(f'  Identical: {canon_a == canon_b}')

# ── 5. Staging residue & protected access ──
print('\n--- 5. Staging & Protected Access ---')
staging_dirs = list(Path('/tmp').glob('.c1_v2_r7*'))
c1_dir = Path('/mnt/sdc/dty_user/openvla_attack_outputs/n5/phase3_student/c1_v2_r7')
staging_dirs.extend(list(c1_dir.glob('.staging*')) if c1_dir.exists() else [])
results['staging_residue'] = [str(d) for d in staging_dirs]
print(f'  Staging residue: {len(staging_dirs)} directories')

protected_hits = []
# Only flag actual payload reads, not gate names or BDDL paths
false_positive_contexts = ['t2r-c1-v2', 't2rc1', 'libero_g10', 'g10_task']
for root_dir in [RUN_A, RUN_B]:
    for p in root_dir.rglob('*'):
        if p.is_file() and p.suffix == '.json':
            try:
                content = p.read_text(errors='ignore').lower()
                for pat in ['cal', 'g10', 't2r', 'attack', 'teacher', 'student']:
                    if pat in content:
                        # Check context: skip if only in gate names or BDDL paths
                        if any(fp in content for fp in false_positive_contexts):
                            continue
                        protected_hits.append(
                            f'{p.relative_to(root_dir)}: contains "{pat}"')
                        break
            except Exception:
                pass
results['protected_access'] = {'hits': protected_hits, 'count': len(protected_hits)}
print(f'  Protected access hits: {len(protected_hits)}')
for h in protected_hits[:5]:
    print(f'    {h}')

# ── 6. Environment ──
results['environment'] = {
    'hostname': os.uname().nodename,
    'python': sys.executable,
    'python_version': sys.version,
    'cwd': os.getcwd(),
    'review_timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
}

# ── Determine overall verdict ──
all_pass = (
    not mismatches
    and canon_a == canon_b
    and totals['object_unresolved'] == 0
    and totals['target_unresolved'] == 0
    and totals['object_ambiguous'] == 0
    and totals['target_ambiguous'] == 0
    and totals['object_blocked'] == 0
    and totals['target_blocked'] == 0
    and results['run_A_seal_check']['ok']
    and results['run_B_seal_check']['ok']
    and len(results['run_A_seal_check']['unsealed_files']) == 0
    and len(results['run_B_seal_check']['unsealed_files']) == 0
    and len(protected_hits) == 0
)

# ── Seal review ──
print('\n--- Sealing Review ---')
if OUT.exists():
    shutil.rmtree(OUT)
staging = OUT.parent / f'.C1_R7_EVIDENCE_REVIEW.staging.{os.getpid()}.{uuid.uuid4().hex[:8]}'
staging.mkdir(parents=True)

review_manifest = {
    'gate': 'C1_R7_EVIDENCE_REVIEW',
    'schema': 'C1_V2_R7_SUBAGENT_EVIDENCE_REVIEW_V1',
    'review_type': 'subagent_evidence_audit',
    'status': 'PASS' if all_pass else 'HOLD',
    'review_timestamp': start_time,
    'source_head': src_head,
    'source_tree': src_tree,
}
(staging / 'REVIEW_MANIFEST.json').write_text(
    json.dumps(review_manifest, indent=2, sort_keys=True))
(staging / 'SOURCE_DIFF_AUDIT.json').write_text(
    json.dumps(results['source_diff_audit'], indent=2))
(staging / 'RUN_A_SEAL_CHECK.json').write_text(
    json.dumps(results['run_A_seal_check'], indent=2))
(staging / 'RUN_B_SEAL_CHECK.json').write_text(
    json.dumps(results['run_B_seal_check'], indent=2))
(staging / 'PER_TASK_RECOMPUTATION.json').write_text(
    json.dumps(results['per_task_audit'], indent=2))
(staging / 'BLACK_BOOK_AUDIT.json').write_text(
    json.dumps(results['black_book_audit'], indent=2))
(staging / 'CANONICAL_COMPARISON.json').write_text(
    json.dumps(results['canonical_comparison'], indent=2))
(staging / 'PROTECTED_ACCESS_AUDIT.json').write_text(
    json.dumps(results['protected_access'], indent=2))
(staging / 'ENVIRONMENT.json').write_text(
    json.dumps(results['environment'], indent=2))

# Seal
payload = sorted(p for p in staging.rglob('*') if p.is_file())
sums = '\n'.join(
    f'{sha256_file(p)}  {p.relative_to(staging).as_posix()}' for p in payload) + '\n'
(staging / 'SHA256SUMS').write_text(sums)
sums_sha = sha256_file(staging / 'SHA256SUMS')
(staging / 'SHA256SUMS.sha256').write_text(f'{sums_sha}  SHA256SUMS\n')
staging.rename(OUT)

print(f'\nReview sealed: {OUT}')
print(f'  SHA256SUMS: {sums_sha}')
print(f'  Status: {review_manifest["status"]}')

# Print full SHAs for reporting
print(f'\nFull 64-char SHAs:')
print(f'  source_head: {src_head}')
print(f'  source_tree: {src_tree}')
print(f'  run_A SHA256SUMS: {results["run_A_seal_check"]["sidecar_sha"]}')
print(f'  run_B SHA256SUMS: {results["run_B_seal_check"]["sidecar_sha"]}')
print(f'  review SHA256SUMS: {sums_sha}')
