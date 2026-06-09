#!/usr/bin/env python3
"""Action-Dynamics v0.2: provenance audit, mode labels, interaction readout. CPU-only."""
import json, glob, csv, os, re, hashlib, numpy as np
from collections import defaultdict, Counter
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

SOURCE_DIRS = [
    '/data/liuyu/outputs/stageb_v1_1_k5_repeat_stability_rc1a_a20379f',
    '/data/liuyu/outputs/stageb_v1_1_k5b_targeted_stability_rc1a_0e3428f',
    '/data/liuyu/outputs/stageb_v1_1_k5c_targeted_expansion_rc1a_ca3a97e',
    '/data/liuyu/outputs/stageb_v1_1_k5c_targeted_expansion_rc1a_ca3a97e/pipeline_v0_3_confirmation',
    '/data/liuyu/outputs/stageb_v1_1_k5c_targeted_expansion_rc1a_ca3a97e/pipeline_v0_3_robustness_seed78',
]
STABLE = '/data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607/tables/stageb_v1_1_stable_parent_pool_k5_k5b_k5c_rc1a_ca3a97e.csv'
KNOWN = ['alphabet_soup','bbq_sauce','butter','cream_cheese','milk','orange_juice','salad_dressing','tomato_sauce']
REPO = '/data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607'

stable = {}
with open(STABLE) as f:
    for r in csv.DictReader(f): stable[r['parent']] = r

def parse(pk):
    task=sid=ws=we=None
    for tk in KNOWN:
        if tk in pk: task=tk; break
    m_s=re.search(r'_s(\d+)',pk)
    if m_s: sid=m_s.group(1)
    m_w=re.search(r'_w(\d+)_(\d+)',pk)
    if m_w: ws=m_w.group(1); we=m_w.group(2)
    return task,sid,ws,we

def get_win(pk, pr):
    task,sid,ws,we=parse(pk)
    if not task: task=pr.get('task','?')
    if not ws:
        win=pr.get('window','')
        parts=win.replace('_env0','').replace('_env1','').replace('_env2','').split('_')
        if len(parts)>=2: ws,we=parts[0],parts[1]
    if not sid:
        win=pr.get('window','')
        sid=win.split('_env')[1] if '_env' in win else '0'
    return task,sid,ws,we

def classify_mode(r):
    """Classify pre-window gripper action mode."""
    o = r['open_rate']; t = r['transitions']; s = r['raw_slope']
    cs = r['close_streak']; os_st = r['open_streak']; n = r['n_pre']
    if o > 0.85 and t <= 3:
        return 'static_open'
    if o < 0.15 and t <= 3:
        return 'static_close'
    if s < -0.005 and cs >= 3:
        return 'closing_transition'
    if s > 0.005 and os_st >= 10:
        return 'opening_transition'
    if t >= 5:
        return 'mixed_transition'
    if o > 0.85:
        return 'static_open'
    if o < 0.15:
        return 'static_close'
    return 'other'

# ── 1. Extract features + trace provenance ──
results = []; prov_rows = []
found = 0; missing = 0
for pk, pr in stable.items():
    task, sid, ws, we = get_win(pk, pr)
    if not all([task, ws, we]): missing += 1; continue
    ws_i = int(ws); we_i = int(we)

    trace_path = source_cond = source_jid = source_dir = None
    for sd in SOURCE_DIRS:
        for sf in glob.glob(os.path.join(sd, 'summary_*.json')):
            with open(sf) as fh: j = json.load(fh)
            if (j.get('task_key')==task and str(j.get('state_id',''))==str(sid) and
                j.get('window_start')==ws_i and j.get('window_end')==we_i and
                j.get('infra_status')=='ok'):
                jid = j.get('job_id'); cond = j.get('condition','?')
                for tf in glob.glob(os.path.join(sd, 'trace_*job%d.csv' % jid)):
                    trace_path = tf; source_cond = cond; source_jid = jid
                    source_dir = sd; break
                if trace_path: break
        if trace_path: break
    if not trace_path: missing += 1; continue

    with open(trace_path) as f:
        rows = list(csv.DictReader(f))
    pre = [r for r in rows if int(r.get('step',0)) < ws_i]
    if len(pre) < 5: missing += 1; continue

    def g(field): return np.array([float(r.get(field,0)) for r in pre])
    raw_g = g('raw_action_6'); env_g = g('env_action_6')
    dist = np.abs(raw_g - 0.5); is_open = (env_g < 0).astype(float)
    q0 = g('obs_gripper_qpos_0'); q1 = g('obs_gripper_qpos_1')
    arm = g('arm_l2')

    so = sc = co = cc = transitions = 0; prev = None
    for v in env_g:
        if v < 0: co += 1; cc = 0
        else: cc += 1; co = 0
        so = max(so, co); sc = max(sc, cc)
        if prev is not None and prev != v: transitions += 1
        prev = v

    t = np.arange(len(raw_g))
    slope = np.polyfit(t, raw_g, 1)[0] if len(t) > 1 else 0
    last_transition = 0
    for i in range(len(env_g)-1, 0, -1):
        if env_g[i] != env_g[i-1]: last_transition = len(env_g) - i; break

    # Prefix hash
    pre_raw_str = ','.join(str(r.get('raw_action_6','0')) for r in pre)
    pre_hash = hashlib.md5(pre_raw_str.encode()).hexdigest()[:8]

    r = {
        'parent': pk, 'task': task, 'ws': ws_i, 'we': we_i,
        'cmd_label': pr.get('cmd_label','?'), 'risk_rand': float(pr.get('risk_rand',0)),
        'pV': float(pr.get('pV_cmd',0)), 'pR': float(pr.get('pR_cmd',0)),
        'yield': float(pr.get('yield_cmd',0)),
        'is_rand': 1 if 'rand_sensitive' in pr.get('cmd_label','') else 0,
        'is_cmd': 1 if 'cmd_specific' in pr.get('cmd_label','') else 0,
        'n_pre': len(pre),
        'raw_mean': round(np.mean(raw_g),4), 'raw_std': round(np.std(raw_g),4),
        'raw_last': round(raw_g[-1],4), 'raw_slope': round(slope,6),
        'dist_mean': round(np.mean(dist),4), 'dist_last': round(dist[-1],4),
        'open_rate': round(np.mean(is_open),4),
        'open_streak': so, 'close_streak': sc,
        'transitions': transitions, 'last_transition': last_transition,
        'q0_mean': round(np.mean(q0),6), 'q0_std': round(np.std(q0),6),
        'q1_mean': round(np.mean(q1),6), 'q1_std': round(np.std(q1),6),
        'arm_mean': round(np.mean(arm),6), 'arm_max': round(np.max(arm),6),
        'mode': '',
        'source_trace': os.path.basename(trace_path),
        'source_condition': source_cond, 'source_job_id': source_jid,
        'feature_safe': True,  # all features from step < ws, verified
        'pre_hash': pre_hash,
    }
    results.append(r); found += 1

    prov_rows.append({
        'parent': pk, 'task': task, 'ws': ws_i, 'we': we_i,
        'source_trace': os.path.basename(trace_path),
        'source_condition': source_cond, 'source_job_id': source_jid,
        'source_dir': os.path.basename(source_dir) if source_dir else '?',
        'n_pre': len(pre), 'pre_step_range': f'0-{ws_i-1}',
        'pre_hash': pre_hash, 'feature_safe': True,
    })

# Classify modes
for r in results:
    r['mode'] = classify_mode(r)

# ── 2. Write provenance audit ──
with open(os.path.join(REPO, 'tables/action_dynamics_prewindow_source_audit_4b3aabb.csv'), 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['parent','task','ws','we','source_trace','source_condition','source_job_id',
                'source_dir','n_pre','pre_step_range','pre_hash','feature_safe'])
    for p in prov_rows:
        w.writerow([p['parent'],p['task'],p['ws'],p['we'],p['source_trace'],p['source_condition'],
                    p['source_job_id'],p['source_dir'],p['n_pre'],p['pre_step_range'],
                    p['pre_hash'],p['feature_safe']])

# ── 3. Mode audit ──
mode_dist = Counter(r['mode'] for r in results)
print('=== Action Mode Distribution ===')
for m, c in mode_dist.most_common():
    m_rows = [r for r in results if r['mode']==m]
    cmd_n = sum(1 for r in m_rows if r['is_cmd'])
    rand_n = sum(1 for r in m_rows if r['is_rand'])
    print('  %-20s n=%2d cmd=%d rand=%d' % (m, c, cmd_n, rand_n))

print()
print('=== FP/FN Mode ===')
for r in results:
    if 'k5b_contrast_tomato_far' in r['parent']:
        print('  FP tomato[55,65]: mode=%s open_rate=%.2f raw_std=%.3f raw_slope=%.4f transitions=%d' % (
            r['mode'], r['open_rate'], r['raw_std'], r['raw_slope'], r['transitions']))
    if 'k5b_strict_phys_salad' in r['parent']:
        print('  FN salad[70,80]:  mode=%s open_rate=%.2f raw_std=%.3f raw_slope=%.4f transitions=%d' % (
            r['mode'], r['open_rate'], r['raw_std'], r['raw_slope'], r['transitions']))

# ── 4. Interaction readout ──
print()
print('=== Interaction Readout ===')
n = len(results)

X_prio = np.column_stack([
    [r['raw_mean'] for r in results], [r['raw_std'] for r in results],
    [r['open_rate'] for r in results], [r['q0_mean'] for r in results],
    [r['q1_mean'] for r in results], [r['arm_mean'] for r in results],
])
X_dyn = np.column_stack([
    [r['raw_slope'] for r in results],
    [r['open_streak'] for r in results], [r['close_streak'] for r in results],
    [r['transitions'] for r in results], [r['last_transition'] for r in results],
])
X_inter = np.column_stack([
    [r['q0_mean'] * r['open_streak'] for r in results],
    [r['q0_mean'] * r['open_rate'] for r in results],
    [r['q0_mean'] * (1.0 - r['open_rate']) for r in results],  # close interaction
    [r['raw_std'] * r['q0_mean'] for r in results],
    [r['raw_slope'] * r['q0_mean'] for r in results],
    [r['transitions'] * r['q0_mean'] for r in results],
    [r['open_rate'] * r['transitions'] for r in results],
    [r['close_streak'] * r['raw_mean'] for r in results],
])

ws_a = np.array([r['ws'] for r in results])
we_a = np.array([r['we'] for r in results])
wc_a = (ws_a + we_a) / 2.0
X_timing = np.column_stack([wc_a, wc_a / 300.0])

# Mode one-hot
modes_list = sorted(set(r['mode'] for r in results))
mode_oh = np.array([[1 if r['mode']==m else 0 for m in modes_list] for r in results])

y_rand = np.array([r['is_rand'] for r in results])
y_cmd = np.array([r['is_cmd'] for r in results])
groups = np.array([r['task'] for r in results])

nsp = min(3, len(set(groups)))
gkf = GroupKFold(n_splits=nsp)

configs = {
    'CleanProprio': np.column_stack([X_prio, X_timing]),
    '+ActionDynamics': np.column_stack([X_prio, X_dyn, X_timing]),
    '+Interactions': np.column_stack([X_prio, X_dyn, X_inter, X_timing]),
    'ModeAware': np.column_stack([X_prio, mode_oh, X_timing]),
    'ModeAware+Inter': np.column_stack([X_prio, X_dyn, X_inter, mode_oh, X_timing]),
}

for name, X_feat in configs.items():
    oof_rand = np.zeros(n)
    for ti, tei in gkf.split(X_feat, y_rand, groups=groups):
        ss = StandardScaler(); Xt = ss.fit_transform(X_feat[ti]); Xe = ss.transform(X_feat[tei])
        m = LogisticRegression(max_iter=3000, class_weight='balanced', random_state=42, C=0.5)
        m.fit(Xt, y_rand[ti]); oof_rand[tei] = m.predict_proba(Xe)[:, 1]

    auc_r = roc_auc_score(y_rand, oof_rand) if len(set(y_rand)) > 1 else 0
    fp_s = fn_s = 0
    for i, r in enumerate(results):
        if 'k5b_contrast_tomato_far' in r['parent']: fp_s = oof_rand[i]
        if 'k5b_strict_phys_salad' in r['parent']: fn_s = oof_rand[i]

    # Selector simulation: CleanRand abstain + random rank
    thresh = np.percentile(oof_rand, 50)
    keep = oof_rand <= thresh
    cmd_hit = np.mean([r['is_cmd'] for i, r in enumerate(results) if keep[i]]) if sum(keep) > 0 else 0
    rand_hit = np.mean([r['is_rand'] for i, r in enumerate(results) if keep[i]]) if sum(keep) > 0 else 0
    yld = np.mean([r['yield'] for i, r in enumerate(results) if keep[i]]) if sum(keep) > 0 else 0

    print('%-22s Rand_AUC=%.3f FP=%.4f FN=%.4f n_pass=%d cmd_hit=%.2f rand_hit=%.2f yield=%+.2f' % (
        name, auc_r, fp_s, fn_s, sum(keep), cmd_hit, rand_hit, yld))

print()
print('Mode legend:', dict(enumerate(modes_list)))
print('Saved provenance + features.')
