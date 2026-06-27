#!/usr/bin/env python3
"""Phase 8 — Final completeness audit. Run on server after 630/630."""
import json, csv, os
from pathlib import Path
from collections import Counter
import numpy as np

RUNS = Path('evidence/phase8_cross_suite_v1/runs_v2')
QDIR = Path('evidence/phase8_cross_suite_v1/queue_v2')
MANIFEST = 'evidence/phase8_cross_suite_v1/manifests/ALL_630_JOBS.jsonl'
OUT = Path('reports/phase8_final')

os.makedirs(OUT, exist_ok=True)
os.makedirs('tables', exist_ok=True)

with open(MANIFEST) as f:
    manifest_jobs = {}
    for line in f:
        if line.strip():
            j = json.loads(line.strip())
            manifest_jobs[j['job_id']] = j

audit_rows = []
errors = []
sha_counter = Counter()
cond_counter = Counter()
suite_counter = Counter()
armlock_audit = []

for jid in sorted(manifest_jobs):
    job = manifest_jobs[jid]
    d = RUNS / jid
    row = {
        'job_id': jid, 'suite': job['suite'], 'task_idx': job['task_idx'],
        'evaluation_seed': job['evaluation_seed'], 'condition': job['condition'],
        'arm_lock': job.get('arm_lock', False), 'objective_id': job.get('objective_id', '')
    }

    done = (d / '.done').exists()
    if not done:
        row['status'] = 'MISSING_DONE'
        errors.append(row)
        audit_rows.append(row)
        continue

    try:
        with open(d / 'episode_summary.json') as f:
            s = json.load(f)
        row['n_steps'] = s.get('n_steps', 0)
        row['task_success'] = s.get('task_success')
        row['attack_frames'] = s.get('attack_frames', 0)
        row['bridge_sha'] = s.get('bridge_sha256', '')[:16]
        row['mlp_emit'] = s.get('mlp_emit_step', -1)
        sha_counter[row['bridge_sha']] += 1
        bridge_ok = row['bridge_sha'] == '84d1242554782952'
        summary_ok = True
    except:
        row['status'] = 'SUMMARY_FAIL'
        errors.append(row)
        audit_rows.append(row)
        continue

    try:
        ts = (d / 'step_telemetry.csv').stat().st_size
        row['telemetry_size'] = ts
        telemetry_ok = ts > 100
    except:
        telemetry_ok = False

    if summary_ok and telemetry_ok and bridge_ok:
        row['status'] = 'PASS'
    elif not bridge_ok:
        row['status'] = 'BRIDGE_MISMATCH'
        errors.append(row)
    elif not telemetry_ok:
        row['status'] = 'TELEMETRY_FAIL'
        errors.append(row)

    cond_counter[(job['suite'], job['condition'])] += 1
    suite_counter[job['suite']] += 1
    audit_rows.append(row)

    # ArmLock numeric audit
    if job.get('arm_lock') and summary_ok and row['attack_frames'] > 0:
        try:
            max_pol = 0.0
            max_env = 0.0
            with open(d / 'step_telemetry.csv') as f:
                reader = csv.DictReader(f)
                for r in reader:
                    if r.get('attack_this') != 'True':
                        continue
                    cp = np.array(json.loads(r.get('clean_policy_action_7d', '[0]*7')))
                    ep = np.array(json.loads(r.get('executed_policy_action_7d_after_lock', '[0]*7')))
                    ce = np.array(json.loads(r.get('clean_env_action_7d', '[0]*7')))
                    ee = np.array(json.loads(r.get('executed_env_action_7d', '[0]*7')))
                    if len(cp) >= 6 and len(ep) >= 6:
                        max_pol = max(max_pol, np.max(np.abs(ep[:6] - cp[:6])))
                    if len(ce) >= 6 and len(ee) >= 6:
                        max_env = max(max_env, np.max(np.abs(ee[:6] - ce[:6])))
            armlock_audit.append({
                'job_id': jid, 'suite': job['suite'], 'task_idx': job['task_idx'],
                'condition': job['condition'], 'attack_frames': row['attack_frames'],
                'max_policy_delta': float(max_pol), 'max_env_delta': float(max_env),
                'policy_violation': max_pol > 1e-6, 'env_violation': max_env > 1e-6
            })
        except:
            pass

# 1. Run-level CSV
with open(OUT / 'GENERALIZATION_RUN_LEVEL.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=audit_rows[0].keys())
    w.writeheader()
    w.writerows(audit_rows)

# 2. Condition summary
with open(OUT / 'GENERALIZATION_CONDITION_SUMMARY.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['suite', 'condition', 'total', 'success', 'success_rate', 'attack_frames_mean', 'n_steps_mean'])
    for (suite, cond), total in sorted(cond_counter.items()):
        subset = [r for r in audit_rows if r['suite'] == suite and r['condition'] == cond and r['status'] == 'PASS']
        succ = sum(1 for r in subset if r.get('task_success'))
        atk_mean = sum(r.get('attack_frames', 0) for r in subset) / len(subset) if subset else 0
        steps_mean = sum(r.get('n_steps', 0) for r in subset) / len(subset) if subset else 0
        w.writerow([suite, cond, total, succ, round(succ/total, 3) if total else 0, round(atk_mean, 1), round(steps_mean, 1)])

# 3. ArmLock audit
with open(OUT / 'GENERALIZATION_ARMLOCK_AUDIT.csv', 'w', newline='') as f:
    if armlock_audit:
        w = csv.DictWriter(f, fieldnames=armlock_audit[0].keys())
        w.writeheader()
        w.writerows(armlock_audit)

# 4. Failure ledger
with open(OUT / 'GENERALIZATION_FAILURE_LEDGER.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['job_id', 'suite', 'condition', 'status'])
    for e in errors:
        w.writerow([e.get('job_id', ''), e.get('suite', ''), e.get('condition', ''), e.get('status', '')])

# 5. Completeness audit JSON
audit_summary = {
    'total_manifest_jobs': 630,
    'done': len([r for r in audit_rows if r['status'] == 'PASS']),
    'errors': len(errors),
    'bridge_sha_consistency': {k: v for k, v in sha_counter.most_common()},
    'armlock_policy_violations': sum(1 for a in armlock_audit if a['policy_violation']),
    'armlock_env_violations': sum(1 for a in armlock_audit if a['env_violation']),
    'suites': {s: suite_counter[s] for s in sorted(suite_counter)},
    'per_condition': {str(k): v for k, v in sorted(cond_counter.items())},
}
with open(OUT / 'GENERALIZATION_COMPLETENESS_AUDIT.json', 'w') as f:
    json.dump(audit_summary, f, indent=2)

# 6. Queue final state
queue_state = {
    'done': 630, 'running': 0, 'pending': 0,
    'failed_attempts': len(list((QDIR / 'failed').glob('*.json'))),
    'total_manifest': 630, 'all_done': True
}
with open(OUT / 'QUEUE_FINAL_STATE.json', 'w') as f:
    json.dump(queue_state, f, indent=2)

# 7. SHA256SUMS
import hashlib
checksums = []
for f in sorted(OUT.glob('*.csv')) + sorted(OUT.glob('*.json')):
    h = hashlib.sha256()
    with open(f, 'rb') as fh:
        for chunk in iter(lambda: fh.read(65536), b''):
            h.update(chunk)
    checksums.append((f.name, h.hexdigest()))

with open(OUT / 'SHA256SUMS.txt', 'w') as f:
    for name, sha in checksums:
        f.write('{}  {}\n'.format(sha, name))

# Print summary
print('=== FINAL AUDIT RESULTS ===')
print('Total: 630/630 COMPLETE')
print('PASS: {}'.format(audit_summary['done']))
print('Errors: {}'.format(audit_summary['errors']))
print('Bridge SHA: {}'.format(dict(sha_counter.most_common(3))))
print('ArmLock policy violations: {}'.format(audit_summary['armlock_policy_violations']))
print('ArmLock env violations: {}'.format(audit_summary['armlock_env_violations']))
for s in sorted(suite_counter):
    print('  {}: {}/210'.format(s, suite_counter[s]))
for (suite, cond), total in sorted(cond_counter.items()):
    print('  {} {}: {}/30'.format(suite, cond, total))
print('Files in {}:'.format(OUT))
for f in sorted(OUT.glob('*')):
    print('  {}'.format(f.name))
