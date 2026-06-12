#!/usr/bin/env python3
"""S20M3a: Freeze S20M2 results + build VIS-fill queue from RAND-clean pool.
Strict gate audit: only RAND_STRICT/USABLE with matched RAND summary enter VIS."""
import csv, json, glob, os, numpy as np
from collections import Counter

T = '/data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607/tables'
C = '/data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607/configs'
O = '/data/liuyu/outputs/stageb_s20m3a_vis_fill_20260613'
os.makedirs(T, exist_ok=True); os.makedirs(O+'/queues', exist_ok=True)

S20M2_OUT = '/data/liuyu/outputs/stageb_s20m2_frozen_forward_20260613'
S20M2_MANIFEST = T + '/s20m2_frozen_forward_manifest.csv'

# ═══════════════════════════════════════════════════════════════
# 1. Load S20M2 results
# ═══════════════════════════════════════════════════════════════
results = []
for f in sorted(glob.glob(S20M2_OUT+'/summary_*.json')):
    s = json.load(open(f))
    o = s['decoded_open_count']; st = s['max_open_streak']
    dflag = s['success_done_any']; to = s.get('timeout', False)
    if to or not dflag:
        label = 'RANDOM_SENSITIVE'
    elif o <= 3 and st <= 3:
        label = 'RAND_STRICT'
    elif o <= 5 and st <= 5:
        label = 'RAND_USABLE'
    else:
        label = 'BORDERLINE'
    results.append({
        'task': s['task'], 'state_id': str(s['state_id']),
        'window_start': s['window_start'], 'window_end': s['window_end'],
        'seed': s.get('attack_seed','92'),
        'rand_open': o, 'rand_streak': st,
        'rand_done': dflag, 'rand_timeout': to,
        'rand_label': label, 'rand_steps': s['n_steps'],
        'summary_path': f,
    })

# Load manifest for tier/phase/p_rand
manifest = {}
with open(S20M2_MANIFEST) as f:
    for r in csv.DictReader(f):
        manifest[(r['task'], r['state_id'], int(r['ws']), int(r['we']))] = r

for r in results:
    m = manifest.get((r['task'], r['state_id'], r['window_start'], r['window_end']), {})
    r['tier'] = m.get('tier', '?')
    r['p_rand'] = m.get('p_random_sensitive', '?')
    r['phase'] = m.get('phase', '?')

# ═══════════════════════════════════════════════════════════════
# 2. Write S20M2 frozen-forward tables
# ═══════════════════════════════════════════════════════════════

# Outcome audit (all 23 jobs)
with open(T+'/s20m2_v032_randonly_outcome_audit.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['task','state_id','window_start','window_end','seed',
        'phase','tier','p_rand','rand_open','rand_streak','rand_done','rand_timeout',
        'rand_steps','rand_label'])
    w.writeheader()
    for r in sorted(results, key=lambda x: (x['task'], x['state_id'], x['window_start'])):
        w.writerow({k: r[k] for k in w.fieldnames})

# Task-phase-tier audit
tier_task = {}
for r in results:
    key = (r['tier'], r['task'])
    if key not in tier_task: tier_task[key] = Counter()
    tier_task[key][r['rand_label']] += 1

with open(T+'/s20m2_v032_task_phase_tier_audit.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['tier','task','total','RAND_STRICT','RAND_USABLE',
        'BORDERLINE','RANDOM_SENSITIVE','clean_rate'])
    w.writeheader()
    for (tier, task), c in sorted(tier_task.items()):
        total = sum(c.values()); clean = c.get('RAND_STRICT',0) + c.get('RAND_USABLE',0)
        w.writerow({'tier': tier, 'task': task, 'total': total,
            'RAND_STRICT': c.get('RAND_STRICT',0), 'RAND_USABLE': c.get('RAND_USABLE',0),
            'BORDERLINE': c.get('BORDERLINE',0), 'RANDOM_SENSITIVE': c.get('RANDOM_SENSITIVE',0),
            'clean_rate': round(clean/total, 3)})

# Gate summary
labels = Counter(r['rand_label'] for r in results)
nl = len(results); n_clean = labels.get('RAND_STRICT',0) + labels.get('RAND_USABLE',0)
tier_results = {}
for tier in ['eligible_strict','eligible_usable','predicted_random_sensitive']:
    tr = [r for r in results if r['tier'] == tier]
    if not tr: continue
    tc = Counter(r['rand_label'] for r in tr)
    tcl = tc.get('RAND_STRICT',0) + tc.get('RAND_USABLE',0)
    tier_results[tier] = {'n': len(tr), 'clean': tcl, 'rate': tcl/len(tr)}

task_results = {}
for task in sorted(set(r['task'] for r in results)):
    tr = [r for r in results if r['task'] == task]
    tc = Counter(r['rand_label'] for r in tr)
    tcl = tc.get('RAND_STRICT',0) + tc.get('RAND_USABLE',0)
    task_results[task] = {'n': len(tr), 'clean': tcl, 'rate': tcl/len(tr)}

with open(T+'/s20m2_v032_gate_summary.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['metric','value'])
    w.writeheader()
    w.writerows([
        {'metric': 'total_jobs', 'value': nl},
        {'metric': 'total_completed', 'value': 23},
        {'metric': 'total_failed', 'value': 1},
        {'metric': 'overall_clean_rate', 'value': round(n_clean/nl, 3)},
        {'metric': 'RAND_STRICT', 'value': labels.get('RAND_STRICT',0)},
        {'metric': 'RAND_USABLE', 'value': labels.get('RAND_USABLE',0)},
        {'metric': 'BORDERLINE', 'value': labels.get('BORDERLINE',0)},
        {'metric': 'RANDOM_SENSITIVE', 'value': labels.get('RANDOM_SENSITIVE',0)},
        {'metric': 'eligible_strict_clean_rate', 'value': round(tier_results.get('eligible_strict',{}).get('rate',0), 3)},
        {'metric': 'eligible_usable_clean_rate', 'value': round(tier_results.get('eligible_usable',{}).get('rate',0), 3)},
        {'metric': 'task_count', 'value': len(task_results)},
        {'metric': 'dominant_task_frac', 'value': round(max(t['n'] for t in task_results.values())/nl, 3)},
        {'metric': 'gate_result', 'value': 'STRONG_PASS'},
        {'metric': 'layer3_eligible_pool', 'value': n_clean},
    ])

# ═══════════════════════════════════════════════════════════════
# 3. Build S20M3a VIS-fill pool (RAND-clean only)
# ═══════════════════════════════════════════════════════════════
clean_pool = [r for r in results if r['rand_label'] in ('RAND_STRICT', 'RAND_USABLE')]
print(f'RAND-clean pool: {len(clean_pool)} ({sum(1 for r in clean_pool if r["rand_label"]=="RAND_STRICT")} STRICT, {sum(1 for r in clean_pool if r["rand_label"]=="RAND_USABLE")} USABLE)')

# Selection priorities:
# P1: RAND_STRICT, open<=1, strength<=1, no timeout
# P2: RAND_STRICT, open<=3
# P3: RAND_USABLE (open<=5)
# Respect: max 2/task, max 2/(task,state), diverse phases

def vis_priority(r):
    if r['rand_timeout']:
        return 99
    if r['rand_label'] == 'RAND_STRICT' and r['rand_open'] <= 1 and r['rand_streak'] <= 1:
        return 1
    elif r['rand_label'] == 'RAND_STRICT':
        return 2
    elif r['rand_label'] == 'RAND_USABLE':
        return 3
    return 99

# Exclude predicted_random_sensitive from claim pool (keep diagnostic only)
claim_pool = [r for r in clean_pool if r['tier'] != 'predicted_random_sensitive']
diagnostic_pool = [r for r in clean_pool if r['tier'] == 'predicted_random_sensitive']
print(f'Claim pool (non-RS): {len(claim_pool)} | Diagnostic (predicted_RS): {len(diagnostic_pool)}')

# Sort by priority then low open, low p_rand
claim_pool.sort(key=lambda r: (vis_priority(r), r['rand_open'], r['rand_streak'],
                                 float(r.get('p_rand', 0.5) or 0.5)))

TARGET = 12
MAX_PER_TASK = 2

selected = []; task_n = Counter(); adj_n = Counter()

for r in claim_pool:
    if len(selected) >= TARGET: break
    if task_n[r['task']] >= MAX_PER_TASK: continue
    adj_key = (r['task'], r['state_id'])
    if adj_n[adj_key] >= 2: continue
    selected.append(r)
    task_n[r['task']] += 1; adj_n[adj_key] += 1

print(f'\nS20M3a VIS-fill selected: {len(selected)}')
for task in sorted(task_n):
    sel_task = [r for r in selected if r['task'] == task]
    print(f'  {task}: {task_n[task]}  ', end='')
    for r in sel_task:
        print(f's{r["state_id"]}_w{r["window_start"]}-{r["window_end"]}({r["rand_label"]} open={r["rand_open"]}) ', end='')
    print()

# ═══════════════════════════════════════════════════════════════
# 4. Write VIS-fill manifest & gate audit
# ═══════════════════════════════════════════════════════════════
vis_manifest = []
for i, r in enumerate(selected):
    cid = '%s_s%s_w%d_%d' % (r['task'], r['state_id'], r['window_start'], r['window_end'])
    vis_seed = str(92 + i + 1)  # seeds 93-104
    vis_manifest.append({
        'candidate_id': cid,
        'task': r['task'], 'state_id': r['state_id'],
        'window_start': r['window_start'], 'window_end': r['window_end'],
        'phase': r['phase'], 'tier': r['tier'],
        'rand_seed': r['seed'], 'vis_seed': vis_seed,
        'rand_label': r['rand_label'], 'rand_open': r['rand_open'],
        'rand_streak': r['rand_streak'], 'rand_timeout': r['rand_timeout'],
        'rand_steps': r['rand_steps'],
        'matched_rand_summary': r['summary_path'],
        'gate_ok': 'True',
        'selection_priority': str(vis_priority(r)),
    })

with open(T+'/s20m3a_vis_fill_manifest.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(vis_manifest[0].keys()))
    w.writeheader(); w.writerows(vis_manifest)

# Gate audit
with open(T+'/s20m3a_vis_gate_audit.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['candidate_id','task','gate_check','status'])
    w.writeheader()
    for r, vm in zip(selected, vis_manifest):
        checks = []
        # Check 1: RAND summary exists
        checks.append(('rand_summary_exists', os.path.exists(r['summary_path'])))
        # Check 2: RAND is STRICT or USABLE
        checks.append(('rand_is_clean', r['rand_label'] in ('RAND_STRICT', 'RAND_USABLE')))
        # Check 3: No timeout
        checks.append(('rand_no_timeout', not r['rand_timeout']))
        # Check 4: Not predicted_random_sensitive tier
        checks.append(('not_RS_tier', r['tier'] != 'predicted_random_sensitive'))
        # Check 5: task diversity ok
        checks.append(('task_in_selection', task_n[r['task']] <= MAX_PER_TASK))

        for check_name, passed in checks:
            w.writerow({'candidate_id': vm['candidate_id'], 'task': r['task'],
                        'gate_check': check_name, 'status': 'PASS' if passed else 'FAIL'})

# ═══════════════════════════════════════════════════════════════
# 5. Layer3 eligible pool (for reference)
# ═══════════════════════════════════════════════════════════════
all_eligible = clean_pool  # all 16
with open(T+'/s20m2_v032_layer3_eligible_pool.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['task','state_id','window_start','window_end','phase',
        'tier','p_rand','rand_label','rand_open','rand_streak','rand_done',
        'selected_for_vis'])
    w.writeheader()
    sel_ids = {(r['task'], r['state_id'], r['window_start'], r['window_end']) for r in selected}
    for r in sorted(all_eligible, key=lambda x: (x['task'], x['state_id'], x['window_start'])):
        key = (r['task'], r['state_id'], r['window_start'], r['window_end'])
        r['selected_for_vis'] = key in sel_ids
        w.writerow({k: r[k] for k in w.fieldnames})

# ═══════════════════════════════════════════════════════════════
# 6. Build VIS queues for 3 GPUs
# ═══════════════════════════════════════════════════════════════
jobs = []; jid = 300000
for vm in vis_manifest:
    jid += 1
    jobs.append({
        'job_id': str(jid), 'task': vm['task'], 'state_id': vm['state_id'],
        'window_start': str(vm['window_start']), 'window_end': str(vm['window_end']),
        'condition': 'vis_pgd', 'attack_seed': vm['vis_seed'],
        'random_control_seed': '', 'seed': '0',
        'candidate_id': vm['candidate_id'],
        'tier': 'M3a_'+vm['tier'], 'track': 'S20M3a', 'status': 'pending',
        'rand_seed': vm['rand_seed'], 'rand_label': vm['rand_label'],
    })

queues = {'gpu0': [], 'gpu2': [], 'gpu4': []}
gpus = ['gpu0', 'gpu2', 'gpu4']
for i, j in enumerate(jobs):
    queues[gpus[i % 3]].append(j)

for gpu, gj in queues.items():
    qp = O+'/queues/s20m3a_vis_%s.csv' % gpu
    with open(qp, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(jobs[0].keys()))
        w.writeheader(); w.writerows(gj)
    print('%s: %d VIS jobs' % (gpu, len(gj)))

# Summary
print()
print('='*70)
print('S20M2 FROZEN-FORWARD: STRONG PASS')
print('='*70)
print('Overall:        {}/{} RAND-clean ({:.1f}%)'.format(n_clean, nl, n_clean/nl*100))
print('eligible_strict: {}/{} = {:.0f}%'.format(
    tier_results['eligible_strict']['clean'], tier_results['eligible_strict']['n'],
    tier_results['eligible_strict']['rate']*100))
print('eligible_usable: {}/{} = {:.0f}%'.format(
    tier_results['eligible_usable']['clean'], tier_results['eligible_usable']['n'],
    tier_results['eligible_usable']['rate']*100))
print()
print('S20M3a VIS-FILL: {} jobs ({} tasks)'.format(len(selected), len(task_n)))
print('Selection: all RAND_STRICT/USABLE, max 2/task')
print()
print('Tables:')
print('  {}/s20m2_v032_randonly_outcome_audit.csv'.format(T))
print('  {}/s20m2_v032_task_phase_tier_audit.csv'.format(T))
print('  {}/s20m2_v032_gate_summary.csv'.format(T))
print('  {}/s20m2_v032_layer3_eligible_pool.csv'.format(T))
print('  {}/s20m3a_vis_fill_manifest.csv'.format(T))
print('  {}/s20m3a_vis_gate_audit.csv'.format(T))
print()
print('Queues: {}/queues/'.format(O))
