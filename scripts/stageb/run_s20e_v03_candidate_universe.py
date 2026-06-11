#!/usr/bin/env python3
"""S20e Phase 3-4: Build v0.3-compatible candidate universe from S20d clean traces,
then score with frozen detector_v0.3_rc1a.
Window convention: half-open [ws, we), window_len=10, stride=5."""
import csv, numpy as np, os, re
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

LABELS_72 = '/data/liuyu/outputs/stageb_v1_1_detector_v0_rc1a_20260608/all_labels_rc1a_14cfabe_72pairs.csv'
STABLE = '/data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607/tables/stageb_v1_1_stable_parent_pool_k5_k5b_k5c_rc1a_ca3a97e.csv'
OUT_UNIVERSE = '/data/liuyu/outputs/stageb_s20e_v03_mainline_20260611/s20e_v03_candidate_universe.csv'
OUT_CANDIDATES = '/data/liuyu/outputs/stageb_s20e_v03_mainline_20260611/s20e_detector_v03_window_candidates.csv'

WINDOW_LEN = 10
STRIDE = 5
OUT_DIR = '/data/liuyu/outputs/stageb_s20e_v03_mainline_20260611'
os.makedirs(OUT_DIR, exist_ok=True)

TRACES = {
    ('ketchup', '1'): '/data/liuyu/outputs/stageb_s20e_mainline_official_closure_20260611/clean/trace_ketchup_s1_w0_10_s20d_clean_seed0_job960100.csv',
    ('tomato_sauce', '3'): '/data/liuyu/outputs/stageb_s20e_mainline_official_closure_20260611/clean/trace_tomato_sauce_s3_w0_10_s20d_clean_seed0_job960101.csv',
    ('tomato_sauce', '5'): '/data/liuyu/outputs/stageb_s20e_mainline_official_closure_20260611/clean/trace_tomato_sauce_s5_w0_10_s20d_clean_seed0_job960102.csv',
}

# ── Load 72 labeled pairs and stable pool ──
labels = {}
with open(LABELS_72) as f:
    for r in csv.DictReader(f):
        key = (r['task_key'], r['state_id'], r.get('seed', '0'),
               r['window_start'], r['window_end'])
        labels[key] = r

KNOWN_TASKS = ['alphabet_soup', 'bbq_sauce', 'butter', 'cream_cheese',
               'milk', 'orange_juice', 'salad_dressing', 'tomato_sauce']

with open(STABLE) as f:
    stable = {r['parent']: r for r in csv.DictReader(f)}

# ── Train frozen v0.3 on full 72-pair set ──
train_rows = []
for pk, pr in stable.items():
    if pr['cmd_label'] == 'unstable_or_unknown':
        continue
    task = None
    for tk in KNOWN_TASKS:
        if tk in pk: task = tk; break
    m_s = re.search(r'_s(\d+)', pk)
    sid = m_s.group(1) if m_s else '0'
    m_w = re.search(r'_w(\d+)_(\d+)', pk)
    if not m_w: continue
    ws_s, we_s = m_w.group(1), m_w.group(2)
    found = None
    for s in ['0', '1', '2']:
        key = (task, str(sid), s, ws_s, we_s)
        if key in labels: found = labels[key]; break
    if not found: continue

    def ff(field, d=0.0):
        try: return float(found.get(field, d) or d)
        except: return d

    train_rows.append({
        'clean_open_count': ff('clean_open_count'),
        'clean_open_frac': ff('clean_open_frac'),
        'raw_gripper_mean': ff('raw_gripper_mean'),
        'raw_gripper_max': ff('raw_gripper_max'),
        'qpos_pre': ff('qpos_pre'),
        'qpos_mean': ff('qpos_mean'),
        'wc': (int(ws_s) + int(we_s)) / 2.0,
        'rel_timing': ((int(ws_s) + int(we_s)) / 2.0) / max(int(found.get('actual_max_step', 299) or 299), 1),
        'is_rand': 1 if 'rand_sensitive' in pr['cmd_label'] else 0,
        'is_cmd': 1 if 'cmd_specific' in pr['cmd_label'] else 0,
    })

print('Training set: %d (cmd=%d rand=%d)' % (
    len(train_rows), sum(r['is_cmd'] for r in train_rows),
    sum(r['is_rand'] for r in train_rows)))

X_tr = np.column_stack([
    [r['clean_open_count'] for r in train_rows],
    [r['clean_open_frac'] for r in train_rows],
    [r['raw_gripper_mean'] for r in train_rows],
    [r['raw_gripper_max'] for r in train_rows],
    [r['qpos_pre'] for r in train_rows],
    [r['qpos_mean'] for r in train_rows],
    [r['wc'] for r in train_rows],
    [r['rel_timing'] for r in train_rows],
])
y_rand = np.array([r['is_rand'] for r in train_rows])
y_cmd = np.array([r['is_cmd'] for r in train_rows])

ss = StandardScaler(); X_tr_s = ss.fit_transform(X_tr)
m_rand = LogisticRegression(max_iter=2000, class_weight='balanced', random_state=42)
m_rand.fit(X_tr_s, y_rand)
m_cmd = LogisticRegression(max_iter=2000, class_weight='balanced', random_state=42)
m_cmd.fit(X_tr_s, y_cmd)

p_rand_train = m_rand.predict_proba(X_tr_s)[:, 1]
abstain_threshold = np.percentile(p_rand_train, 50)
print('Abstain threshold (p_rand 50pct): %.4f' % abstain_threshold)

# ── Build candidate universe from S20d clean traces ──
def compute_features(trace_path, ws, we):
    with open(trace_path) as f:
        rows = list(csv.DictReader(f))

    window_rows = [r for r in rows if ws <= int(r['step']) < we]
    pre_rows = [r for r in rows if int(r['step']) < ws]

    def g(row, key, default=0.0):
        try: return float(row.get(key, default) or default)
        except: return default

    clean_open_count = sum(1 for r in window_rows if g(r, 'decoded_open_bool') == 1)
    clean_open_frac = clean_open_count / max(len(window_rows), 1)
    gripper_vals = [g(r, 'clean_gripper_env') for r in window_rows]
    raw_gripper_mean = float(np.mean(gripper_vals)) if gripper_vals else 0.0
    raw_gripper_max = float(np.max(gripper_vals)) if gripper_vals else 0.0
    qpos_pre_vals = [g(r, 'gripper_qpos_before') for r in pre_rows[-5:]] if pre_rows else [0.0]
    qpos_pre = float(np.median(qpos_pre_vals)) if qpos_pre_vals else 0.0
    qpos_vals = [g(r, 'gripper_qpos_before') for r in window_rows]
    qpos_mean = float(np.mean(qpos_vals)) if qpos_vals else 0.0
    qpos_max = float(np.max(qpos_vals)) if qpos_vals else 0.0
    # qpos_slope
    if len(qpos_vals) >= 2:
        xs = np.arange(len(qpos_vals))
        slope = float(np.polyfit(xs, qpos_vals, 1)[0])
    else:
        slope = 0.0
    # eef displacement during window
    eef_z_vals = [g(r, 'eef_z') for r in window_rows]
    eef_disp = float(np.max(eef_z_vals) - np.min(eef_z_vals)) if len(eef_z_vals) >= 2 else 0.0

    actual_max_step = max(int(r['step']) for r in rows)
    wc = (ws + we) / 2.0
    rel_timing = wc / max(actual_max_step, 1)

    return {
        'clean_open_count': clean_open_count, 'clean_open_frac': clean_open_frac,
        'raw_gripper_mean': raw_gripper_mean, 'raw_gripper_max': raw_gripper_max,
        'qpos_pre': qpos_pre, 'qpos_mean': qpos_mean,
        'qpos_max': qpos_max, 'qpos_slope': slope, 'eef_disp': eef_disp,
        'wc': wc, 'rel_timing': rel_timing,
        'actual_max_step': actual_max_step,
    }

# Generate sliding windows
universe = []
for (task, sid), tpath in TRACES.items():
    with open(tpath) as f:
        rows = list(csv.DictReader(f))
    max_step = max(int(r['step']) for r in rows)
    clean_success = any(r.get('success_primary') == '1' for r in rows)
    has_obj_pose = any(r.get('obj_z', '') not in (None, '') for r in rows)
    has_eef_pose = any(r.get('eef_z', '') not in (None, '') for r in rows)

    for ws in range(0, max_step - WINDOW_LEN + 1, STRIDE):
        we = ws + WINDOW_LEN
        cid = '%s_s%s_w%d_%d' % (task, sid, ws, we)
        feats = compute_features(tpath, ws, we)
        universe.append({
            'candidate_id': cid, 'task': task, 'state_id': sid,
            'window_start': ws, 'window_end': we, 'window_len': WINDOW_LEN,
            'window_indexing': 'half_open', 'actual_max_step': max_step,
            **feats,
            'clean_success_done': clean_success, 'clean_success_check': clean_success,
            'has_obj_pose': has_obj_pose, 'has_eef_pose': has_eef_pose,
            'source_trace': tpath, 'source_runner': 's20d_v4_fixed_window_l3',
        })

print('Candidate universe: %d windows from %d states' % (len(universe), len(TRACES)))

# ── Score with frozen v0.3 ──
for u in universe:
    X_test = np.array([[
        u['clean_open_count'], u['clean_open_frac'],
        u['raw_gripper_mean'], u['raw_gripper_max'],
        u['qpos_pre'], u['qpos_mean'],
        u['wc'], u['rel_timing'],
    ]])
    X_test_s = ss.transform(X_test)
    u['p_cmd_specific'] = float(m_cmd.predict_proba(X_test_s)[0, 1])
    u['p_random_sensitive'] = float(m_rand.predict_proba(X_test_s)[0, 1])
    u['abstain'] = u['p_random_sensitive'] > abstain_threshold
    u['abstain_threshold'] = abstain_threshold
    u['attack_score'] = u['p_cmd_specific']  # primary selection score
    u['detector_version'] = 'detector_v0.3_rc1a'
    u['feature_group'] = 'CleanNoTaskNoTiming'
    u['model'] = 'LogisticRegression_trained_on_22_pairs'
    u['pre_registered_before_rand_vis'] = True

# Sort by attack_score descending, abstain last
universe.sort(key=lambda u: (u['abstain'], -u['attack_score']))

# Assign ranks
for i, u in enumerate(universe):
    u['rank_global'] = i + 1

# Per-state rank
for task_sid in sorted(set((u['task'], u['state_id']) for u in universe)):
    state_windows = sorted(
        [u for u in universe if (u['task'], u['state_id']) == task_sid],
        key=lambda u: (u['abstain'], -u['attack_score']))
    for j, u in enumerate(state_windows):
        u['rank_within_state'] = j + 1

# Write universe
u_fields = ['candidate_id','task','state_id','window_start','window_end','window_len',
            'window_indexing','actual_max_step',
            'clean_open_count','clean_open_frac','raw_gripper_mean','raw_gripper_max',
            'qpos_pre','qpos_mean','qpos_max','qpos_slope','eef_disp',
            'wc','rel_timing','clean_success_done','clean_success_check',
            'has_obj_pose','has_eef_pose','source_trace','source_runner']
with open(OUT_UNIVERSE, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=u_fields, extrasaction='ignore')
    w.writeheader(); w.writerows(universe)

# Write ranked candidates
c_fields = ['rank_global','rank_within_state','candidate_id','task','state_id',
            'window_start','window_end','window_len','window_indexing',
            'detector_version','feature_group','model',
            'p_cmd_specific','p_random_sensitive','attack_score',
            'abstain','abstain_threshold',
            'clean_success_done','clean_success_check',
            'pre_registered_before_rand_vis','source_trace']
with open(OUT_CANDIDATES, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=c_fields, extrasaction='ignore')
    w.writeheader(); w.writerows(universe)

# Print top candidates
print('\n=== TOP v0.3 CANDIDATES ===')
for u in universe[:10]:
    status = 'ABSTAIN' if u['abstain'] else 'CANDIDATE'
    print('rank=%3d %-12s %-40s p_cmd=%.4f p_rand=%.4f %s' % (
        u['rank_global'], status, u['candidate_id'],
        u['p_cmd_specific'], u['p_random_sensitive'],
        'ABSTAIN(p_rand=%.4f>%.4f)' % (u['p_random_sensitive'], abstain_threshold) if u['abstain'] else ''))

print('\nUniverse: %s' % OUT_UNIVERSE)
print('Candidates: %s' % OUT_CANDIDATES)
print('Non-abstain: %d / %d' % (
    sum(1 for u in universe if not u['abstain']), len(universe)))
