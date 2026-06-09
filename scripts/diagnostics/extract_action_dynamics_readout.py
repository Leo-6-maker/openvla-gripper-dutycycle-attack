#!/usr/bin/env python3
"""Extract pre-window action-dynamics features from existing traces + readout."""
import json, glob, csv, os, re, numpy as np
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
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

def extract_prewindow_features(trace_path, ws):
    try:
        with open(trace_path) as f:
            reader = csv.DictReader(f)
            rows = [r for r in reader]
    except:
        return None
    pre = [r for r in rows if int(r.get('step',0)) < ws]
    if len(pre) < 5: return None

    def g(field): return np.array([float(r.get(field,0)) for r in pre])
    raw_g = g('raw_action_6'); env_g = g('env_action_6')
    dist = np.abs(raw_g - 0.5); is_open = (env_g < 0).astype(float)
    q0 = g('obs_gripper_qpos_0'); q1 = g('obs_gripper_qpos_1'); arm = g('arm_l2')

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

    return {
        'n_pre': len(pre),
        'raw_mean': round(np.mean(raw_g),4), 'raw_std': round(np.std(raw_g),4),
        'raw_last': round(raw_g[-1],4), 'raw_slope': round(slope,6),
        'raw_min': round(np.min(raw_g),4), 'raw_max': round(np.max(raw_g),4),
        'dist_mean': round(np.mean(dist),4), 'dist_last': round(dist[-1],4),
        'open_rate': round(np.mean(is_open),4),
        'open_streak': so, 'close_streak': sc, 'transitions': transitions,
        'last_transition': last_transition,
        'q0_mean': round(np.mean(q0),6), 'q0_std': round(np.std(q0),6),
        'q1_mean': round(np.mean(q1),6), 'q1_std': round(np.std(q1),6),
        'arm_mean': round(np.mean(arm),6), 'arm_max': round(np.max(arm),6),
    }

results = []; found = 0; missing = 0
for pk, pr in stable.items():
    task, sid, ws, we = get_win(pk, pr)
    if not all([task, ws, we]): missing += 1; continue
    ws_i = int(ws); we_i = int(we)

    trace_path = None
    for sd in SOURCE_DIRS:
        for sf in glob.glob(os.path.join(sd, 'summary_*.json')):
            with open(sf) as fh: j = json.load(fh)
            if (j.get('task_key')==task and str(j.get('state_id',''))==str(sid) and
                j.get('window_start')==ws_i and j.get('window_end')==we_i and
                j.get('infra_status')=='ok'):
                jid = j.get('job_id')
                for tf in glob.glob(os.path.join(sd, 'trace_*job%d.csv' % jid)):
                    trace_path = tf; break
                if trace_path: break
        if trace_path: break
    if not trace_path: missing += 1; continue

    feats = extract_prewindow_features(trace_path, ws_i)
    if not feats: missing += 1; continue

    feats['parent'] = pk; feats['task'] = task; feats['ws'] = ws_i; feats['we'] = we_i
    feats['cmd_label'] = pr.get('cmd_label','?')
    feats['risk_rand'] = float(pr.get('risk_rand',0))
    feats['pV'] = float(pr.get('pV_cmd',0)); feats['pR'] = float(pr.get('pR_cmd',0))
    feats['yield'] = float(pr.get('yield_cmd',0))
    feats['is_rand'] = 1 if 'rand_sensitive' in pr.get('cmd_label','') else 0
    feats['is_cmd'] = 1 if 'cmd_specific' in pr.get('cmd_label','') else 0
    results.append(feats); found += 1

print('Extracted: %d parents, missing: %d' % (found, missing))

# Write features
out_csv = os.path.join(REPO, 'tables/action_dynamics_prewindow_features_stable_pool_v2.csv')
with open(out_csv, 'w', newline='') as f:
    w = csv.writer(f)
    cols = ['parent','task','ws','we','cmd_label','is_rand','is_cmd','pV','pR','yield','risk_rand',
            'n_pre','raw_mean','raw_std','raw_last','raw_slope','raw_min','raw_max',
            'dist_mean','dist_last','open_rate','open_streak','close_streak','transitions','last_transition',
            'q0_mean','q0_std','q1_mean','q1_std','arm_mean','arm_max']
    w.writerow(cols)
    for r in results:
        w.writerow([r.get(c,'') for c in cols])

# Readout
print()
print('=== Readout: CleanProprio vs +ActionDynamics ===')
n = len(results)

X_proprio = np.column_stack([
    [r['raw_mean'] for r in results], [r['raw_std'] for r in results],
    [r['dist_mean'] for r in results], [r['dist_last'] for r in results],
    [r['q0_mean'] for r in results], [r['q1_mean'] for r in results],
])
X_dyn = np.column_stack([
    [r['raw_slope'] for r in results],
    [r['open_streak'] for r in results], [r['close_streak'] for r in results],
    [r['transitions'] for r in results], [r['last_transition'] for r in results],
    [r['open_rate'] for r in results],
])
ws_a = np.array([r['ws'] for r in results]); we_a = np.array([r['we'] for r in results])
wc_a = (ws_a + we_a) / 2.0
X_timing = np.column_stack([wc_a, wc_a / 300.0])

X_prio = np.column_stack([X_proprio, X_timing])
X_both = np.column_stack([X_proprio, X_dyn, X_timing])

y_rand = np.array([r['is_rand'] for r in results])
y_cmd = np.array([r['is_cmd'] for r in results])
groups = np.array([r['task'] for r in results])

nsp = min(3, len(set(groups)))
gkf = GroupKFold(n_splits=nsp)

for name, X_feat in [('CleanProprio', X_prio), ('+ActionDynamics', X_both)]:
    oof_rand = np.zeros(n); oof_cmd = np.zeros(n)
    for ti, tei in gkf.split(X_feat, y_rand, groups=groups):
        ss = StandardScaler(); Xt = ss.fit_transform(X_feat[ti]); Xe = ss.transform(X_feat[tei])
        mr = LogisticRegression(max_iter=2000, class_weight='balanced', random_state=42)
        mr.fit(Xt, y_rand[ti]); oof_rand[tei] = mr.predict_proba(Xe)[:, 1]
        mc = LogisticRegression(max_iter=2000, class_weight='balanced', random_state=42)
        mc.fit(Xt, y_cmd[ti]); oof_cmd[tei] = mc.predict_proba(Xe)[:, 1]

    auc_r = roc_auc_score(y_rand, oof_rand)
    auc_c = roc_auc_score(y_cmd, oof_cmd)

    # FP/FN scores
    fp_score = fn_score = 0
    for i, r in enumerate(results):
        if 'k5b_contrast_tomato_far' in r['parent']: fp_score = oof_rand[i]
        if 'k5b_strict_phys_salad' in r['parent']: fn_score = oof_rand[i]

    print('%s: Rand_AUROC=%.3f Cmd_AUROC=%.3f FP_tomato=%.4f FN_salad=%.4f' % (
        name, auc_r, auc_c, fp_score, fn_score))

print()
print('Saved:', out_csv)
