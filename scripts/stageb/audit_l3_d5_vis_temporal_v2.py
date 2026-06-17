#!/usr/bin/env python3
"""Auditor v2: Hard-gate G1-G5 classification with fail-closed on missing cells."""
import csv, json, os, sys, numpy as np
from pathlib import Path

OUT_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('tables')
RPT_DIR = Path(sys.argv[2]) if len(sys.argv) > 2 else Path('reports')
ART_DIR = Path(sys.argv[3]) if len(sys.argv) > 3 else Path('artifacts')
BASE = '/data/liuyu/outputs/l3_d5_vis_temporal_r1'

for d in [OUT_DIR, RPT_DIR, ART_DIR]: d.mkdir(parents=True, exist_ok=True)

EXPECTED_CELLS = {
    ('seed81', 'CLEAN_D5'), ('seed81', 'TRUE_SINGLE'), ('seed81', 'TRUE_TEMPORAL_K10'),
    ('seed81', 'RAND_TEMPORAL_K10'), ('seed81', 'SHUFFLED_TEMPORAL_K10'),
    ('seed82', 'CLEAN_D5'), ('seed82', 'TRUE_SINGLE'), ('seed82', 'TRUE_TEMPORAL_K10'),
    ('seed82', 'RAND_TEMPORAL_K10'), ('seed82', 'SHUFFLED_TEMPORAL_K10'),
}

# ── Load all episodes ──
episodes = {}
missing = []
for (seed, cond) in sorted(EXPECTED_CELLS):
    path = os.path.join(BASE, seed, cond)
    tele = os.path.join(path, 'step_telemetry.csv')
    summ = os.path.join(path, 'episode_summary.json')
    if not os.path.isfile(tele) or not os.path.isfile(summ):
        missing.append('{}/{}'.format(seed, cond))
        continue
    rows = list(csv.DictReader(open(tele)))
    summary = json.load(open(summ))
    episodes[(seed, cond)] = {'rows': rows, 'summary': summary}

if missing:
    print('AUDIT_BLOCKED: {} missing cells: {}'.format(len(missing), missing))
    # Continue for partial audit but mark gate
else:
    print('10/10 cells present')

if len(episodes) == 0:
    print('No data.')
    sys.exit(1)

# ── Per-episode metrics ──
def safe_float(v, default=float('nan')):
    try: return float(v)
    except: return default

results = []
for (seed, cond), ep in sorted(episodes.items()):
    rows = ep['rows']; sm = ep['summary']
    n_steps = len(rows)
    d5_emit = int(sm.get('d5_emit_step', -1))
    n_atk = int(sm.get('attack_frames', 0))

    # Attack frames
    atk_rows = [r for r in rows if r.get('attack_this') in ('True', 'true', '1')]
    atk_steps = [int(r['step']) for r in atk_rows]
    n_open_tok = sum(1 for r in atk_rows if safe_float(r.get('adv_token',''), 0) == 31744)
    n_arm_ok = sum(1 for r in atk_rows if safe_float(r.get('adv_arm',''), 0) >= 5)
    n_env_open = sum(1 for r in atk_rows if safe_float(r.get('env_gripper',''), 1) < 0)
    arm_vals = [int(safe_float(r.get('adv_arm',''), 0)) for r in atk_rows if r.get('adv_arm','') not in ('','nan')]
    min_arm = min(arm_vals) if arm_vals else None

    # prev_delta flags
    prev_flags = sm.get('prev_delta_flags', [])
    if isinstance(prev_flags, str):
        prev_flags = json.loads(prev_flags) if prev_flags else []

    # Physical
    pre_steps = [r for r in rows if max(0, d5_emit-5) <= int(r['step']) < d5_emit]
    post_steps = [r for r in rows if d5_emit <= int(r['step']) < d5_emit + 20]
    pre_q = [safe_float(r['qpos_sum']) for r in pre_steps if r.get('qpos_sum','') not in ('nan','')]
    post_q = [(int(r['step']), safe_float(r['qpos_sum'])) for r in post_steps if r.get('qpos_sum','') not in ('nan','')]
    baseline = np.mean(pre_q) if pre_q else float('nan')
    post_vals = [q for _,q in post_q]
    peak_q = min(post_vals) if post_vals else float('nan')
    mean_post = np.mean(post_vals) if post_vals else float('nan')
    peak_delta = baseline - peak_q if not np.isnan(baseline) and not np.isnan(peak_q) else float('nan')

    # AUC: sum of positive opening beyond baseline
    auc = sum(max(0, baseline - q) for q in post_vals) if post_vals and not np.isnan(baseline) else 0

    # Latency: first step with qpos responding (q < baseline - 0.0005)
    latency = None
    for s, q in post_q:
        if not np.isnan(baseline) and q < baseline - 0.0005:
            latency = s - d5_emit; break

    # Sustain: consecutive steps where qpos stays below baseline, starting from first response
    sustain = 0
    started = False
    for s, q in post_q:
        if not started:
            if not np.isnan(baseline) and q < baseline - 0.0005:
                started = True; sustain = 1
        else:
            if not np.isnan(baseline) and q < baseline:
                sustain += 1
            else:
                break

    # Q7/Q8 consistency
    q7_post = [safe_float(r['q7']) for r in post_steps if r.get('q7','') not in ('nan','')]
    q8_post = [safe_float(r['q8']) for r in post_steps if r.get('q8','') not in ('nan','')]
    q7_mean = np.mean(q7_post) if q7_post else float('nan')
    q8_mean = np.mean(q8_post) if q8_post else float('nan')

    success = bool(sm.get('task_success', False))

    results.append({
        'seed': seed, 'condition': cond, 'n_steps': n_steps, 'd5_emit': d5_emit,
        'n_atk': n_atk, 'atk_steps': str(atk_steps[:5]) + ('...' if len(atk_steps)>5 else ''),
        'n_open_token': n_open_tok, 'n_env_open': n_env_open, 'n_arm_ok': n_arm_ok,
        'min_arm': min_arm, 'token_duty': round(n_open_tok/n_atk,3) if n_atk>0 else 0,
        'env_duty': round(n_env_open/n_atk,3) if n_atk>0 else 0,
        'arm_duty': round(n_arm_ok/n_atk,3) if n_atk>0 else 0,
        'prev_flags_ok': (prev_flags[:1]==[False] and all(prev_flags[1:])) if len(prev_flags)>=2 else False,
        'baseline_q': round(baseline,6) if not np.isnan(baseline) else None,
        'peak_q': round(peak_q,6) if not np.isnan(peak_q) else None,
        'mean_post_q': round(mean_post,6) if not np.isnan(mean_post) else None,
        'peak_delta': round(peak_delta,6) if not np.isnan(peak_delta) else None,
        'auc': round(auc,6), 'latency': latency, 'sustain': sustain,
        'q7_mean': round(q7_mean,6) if not np.isnan(q7_mean) else None,
        'q8_mean': round(q8_mean,6) if not np.isnan(q8_mean) else None,
        'success': success,
    })

# ── Write episode table ──
with open(OUT_DIR / 'l3_temporal_episode_results.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(results[0].keys())); w.writeheader(); w.writerows(results)

print('Episodes: {}/10 present ({} missing)'.format(len(results), len(missing)))

# ── Gate classification ──
def gv(seed, cond, key):
    for r in results:
        if r['seed'] == seed and r['condition'] == cond:
            return r.get(key)
    return None

AUDIT_BLOCKED = len(missing) > 0
gates = []

for seed in ['seed81', 'seed82']:
    # G1: D5 emit = 60, trigger count = 1, window steps correct
    d5_ok = all(gv(seed, c, 'd5_emit') == 60 for c in ['CLEAN_D5','TRUE_SINGLE','TRUE_TEMPORAL_K10','RAND_TEMPORAL_K10','SHUFFLED_TEMPORAL_K10'] if gv(seed, c, 'd5_emit') is not None)
    single_steps = gv(seed, 'TRUE_SINGLE', 'atk_steps') or ''
    single_step_ok = str(60) in str(single_steps)
    t10_steps = gv(seed, 'TRUE_TEMPORAL_K10', 'atk_steps') or ''
    t10_count = gv(seed, 'TRUE_TEMPORAL_K10', 'n_atk') or 0
    t10_steps_ok = (t10_count == 10)
    g1 = d5_ok and single_step_ok and t10_steps_ok

    # G2: Token duty >= 0.80, arm duty >= 0.80, min_arm >= 4, selectivity
    true_tok = gv(seed, 'TRUE_TEMPORAL_K10', 'token_duty') or 0
    true_arm = gv(seed, 'TRUE_TEMPORAL_K10', 'arm_duty') or 0
    true_min_arm = gv(seed, 'TRUE_TEMPORAL_K10', 'min_arm') or 0
    rand_tok = gv(seed, 'RAND_TEMPORAL_K10', 'token_duty') or 0
    shuf_tok = gv(seed, 'SHUFFLED_TEMPORAL_K10', 'token_duty') or 0
    g2 = (true_tok >= 0.80 and true_arm >= 0.80 and (true_min_arm is not None and true_min_arm >= 4)
          and (true_tok - max(rand_tok, shuf_tok) >= 0.50))

    # G3: Env OPEN duty >= 0.80, selectivity
    true_env = gv(seed, 'TRUE_TEMPORAL_K10', 'env_duty') or 0
    rand_env = gv(seed, 'RAND_TEMPORAL_K10', 'env_duty') or 0
    shuf_env = gv(seed, 'SHUFFLED_TEMPORAL_K10', 'env_duty') or 0
    g3 = (true_env >= 0.80 and (true_env - max(rand_env, shuf_env) >= 0.50))

    # G4: Physical: peak_delta > max(controls), latency <= 5, sustain >= 2, AUC > all controls
    true_pd = gv(seed, 'TRUE_TEMPORAL_K10', 'peak_delta') or 0
    true_auc = gv(seed, 'TRUE_TEMPORAL_K10', 'auc') or 0
    true_lat = gv(seed, 'TRUE_TEMPORAL_K10', 'latency')
    true_sus = gv(seed, 'TRUE_TEMPORAL_K10', 'sustain') or 0
    clean_pd = gv(seed, 'CLEAN_D5', 'peak_delta') or 0
    single_pd = gv(seed, 'TRUE_SINGLE', 'peak_delta') or 0
    rand_pd = gv(seed, 'RAND_TEMPORAL_K10', 'peak_delta') or 0
    shuf_pd = gv(seed, 'SHUFFLED_TEMPORAL_K10', 'peak_delta') or 0
    clean_auc = gv(seed, 'CLEAN_D5', 'auc') or 0
    single_auc = gv(seed, 'TRUE_SINGLE', 'auc') or 0
    rand_auc = gv(seed, 'RAND_TEMPORAL_K10', 'auc') or 0
    shuf_auc = gv(seed, 'SHUFFLED_TEMPORAL_K10', 'auc') or 0
    max_ctrl_pd = max(clean_pd, single_pd, rand_pd, shuf_pd)
    max_ctrl_auc = max(clean_auc, single_auc, rand_auc, shuf_auc)
    g4 = (true_pd > max_ctrl_pd and true_auc > max_ctrl_auc
          and (true_lat is not None and true_lat <= 5) and true_sus >= 2)

    # Controls clean
    rand_open = gv(seed, 'RAND_TEMPORAL_K10', 'n_env_open') or 0
    shuf_open = gv(seed, 'SHUFFLED_TEMPORAL_K10', 'n_env_open') or 0
    specificity_ok = (rand_open <= 3 and shuf_open <= 3)

    # G5A: Contact — NOT AUDITABLE from current telemetry
    g5a = 'NOT_AUDITABLE'

    # G5B: Task failure
    clean_succ = gv(seed, 'CLEAN_D5', 'success')
    rand_succ = gv(seed, 'RAND_TEMPORAL_K10', 'success')
    shuf_succ = gv(seed, 'SHUFFLED_TEMPORAL_K10', 'success')
    true_succ = gv(seed, 'TRUE_TEMPORAL_K10', 'success')
    single_succ = gv(seed, 'TRUE_SINGLE', 'success')
    g5b = (clean_succ and rand_succ and shuf_succ and (not true_succ))

    gates.append({
        'seed': seed, 'G1_emit_window': g1, 'G2_token_arm': g2, 'G3_env_duty': g3,
        'G4_physical': g4, 'G5A_contact': g5a, 'G5B_task': g5b,
        'specificity_ok': specificity_ok,
        'true_tok': true_tok, 'true_env': true_env, 'true_pd': true_pd,
        'max_ctrl_pd': max_ctrl_pd, 'true_auc': true_auc, 'max_ctrl_auc': max_ctrl_auc,
        'true_lat': true_lat, 'true_sus': true_sus,
    })

# ── Overall classification ──
all_specific = all(g['specificity_ok'] for g in gates)
all_g1 = all(g['G1_emit_window'] for g in gates)
all_g2 = all(g['G2_token_arm'] for g in gates)
all_g3 = all(g['G3_env_duty'] for g in gates)
all_g4 = all(g['G4_physical'] for g in gates)
all_g5b = all(g['G5B_task'] for g in gates)

if AUDIT_BLOCKED:
    classification = 'AUDIT_BLOCKED_MISSING_CELLS'
elif not all_specific:
    classification = 'SPECIFICITY_BLOCKED'
elif all_g1 and all_g2 and all_g3 and all_g4 and all_g5b:
    classification = 'L3_D5_TRIGGERED_VIS_TEMPORAL_TASK_FAILURE_PASS'
elif all_g1 and all_g2 and all_g3 and all_g4:
    classification = 'L3_D5_TRIGGERED_VIS_TEMPORAL_PHYSICAL_BRIDGE_PASS_TASK_EFFECT_NOT_PROVEN'
elif all_g1 and all_g2 and all_g3:
    classification = 'VIS_TEMPORAL_SEMANTIC_COMMAND_DUTY_PASS_PHYSICAL_NOT_ESTABLISHED'
else:
    classification = 'VIS_TEMPORAL_INCOMPLETE'

print('\n=== GATE ===')
for g in gates:
    print('  {}: G1={} G2={}(tok={:.1f}/arm={:.1f}) G3={}(env={:.1f}) G4={}(pd={:.5f}>{:.5f} auc={:.5f}>{:.5f} lat={} sus={}) G5B={} spec={}'.format(
        g['seed'], g['G1_emit_window'], g['G2_token_arm'], g['true_tok'], g['true_env'],
        g['G3_env_duty'], g['true_env'] if isinstance(g.get('true_env'), float) else 0,
        g['G4_physical'], g['true_pd'], g['max_ctrl_pd'], g['true_auc'], g['max_ctrl_auc'],
        g['true_lat'], g['true_sus'], g['G5B_task'], g['specificity_ok']))
print('Classification: {}'.format(classification))

# Write outputs
with open(OUT_DIR / 'l3_temporal_final_gates.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(gates[0].keys())); w.writeheader(); w.writerows(gates)

gate_json = {'classification': classification, 'missing_cells': missing,
             'audit_blocked': AUDIT_BLOCKED, 'gates': gates, 'n_episodes': len(results)}
with open(ART_DIR / 'l3_temporal_final_gate.json', 'w') as f:
    json.dump(gate_json, f, indent=2, default=str)

print('Done: {} -> {}'.format(OUT_DIR, ART_DIR))
