#!/usr/bin/env python3
"""Independent D5 VIS temporal auditor — recomputes all gates from raw telemetry."""
import csv, json, os, sys, numpy as np
from collections import defaultdict
from pathlib import Path

OUT_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('/data/liuyu/worktrees/l3_d5_vis_temporal/tables')
RPT_DIR = Path(sys.argv[2]) if len(sys.argv) > 2 else Path('/data/liuyu/worktrees/l3_d5_vis_temporal/reports')
ART_DIR = Path(sys.argv[3]) if len(sys.argv) > 3 else Path('/data/liuyu/worktrees/l3_d5_vis_temporal/artifacts')
BASE = '/data/liuyu/outputs/l3_d5_vis_temporal_r1'

for d in [OUT_DIR, RPT_DIR, ART_DIR]: d.mkdir(parents=True, exist_ok=True)

EPISODES = {}
for seed in ['seed81', 'seed82']:
    sd = os.path.join(BASE, seed)
    if not os.path.isdir(sd): continue
    for cond in os.listdir(sd):
        cp = os.path.join(sd, cond)
        if os.path.isdir(cp):
            EPISODES[(seed, cond)] = cp

print('Auditing {} episodes'.format(len(EPISODES)))

# ── Per-episode audit ──
results = []
for (seed, cond), path in sorted(EPISODES.items()):
    tele = os.path.join(path, 'step_telemetry.csv')
    summ = os.path.join(path, 'episode_summary.json')
    if not os.path.isfile(tele): continue

    rows = list(csv.DictReader(open(tele)))
    summary = json.load(open(summ)) if os.path.isfile(summ) else {}

    n_steps = len(rows)
    d5_emit = int(summary.get('d5_emit_step', -1))

    # Attack frames
    atk_rows = [r for r in rows if r.get('attack_this') in ('True', 'true', '1')]
    n_atk = len(atk_rows)
    n_open_token = sum(1 for r in atk_rows if str(r.get('adv_token','')) not in ('','nan') and int(float(r['adv_token'])) == 31744)
    n_arm_ok = sum(1 for r in atk_rows if str(r.get('adv_arm','')) not in ('','nan') and int(float(r['adv_arm'])) >= 5)
    n_env_open = sum(1 for r in atk_rows if float(r.get('env_gripper', 1)) < 0)

    # Physical
    qpos_sum = [float(r['qpos_sum']) for r in rows if r['qpos_sum'] not in ('nan','')]
    q7_vals = [float(r['q7']) for r in rows if r.get('q7','') not in ('nan','')]
    q8_vals = [float(r['q8']) for r in rows if r.get('q8','') not in ('nan','')]

    if d5_emit >= 0:
        pre = [(int(r['step']), float(r['qpos_sum'])) for r in rows if d5_emit-5 <= int(r['step']) < d5_emit and r['qpos_sum'] not in ('nan','')]
        post = [(int(r['step']), float(r['qpos_sum'])) for r in rows if d5_emit <= int(r['step']) < d5_emit+30 and r['qpos_sum'] not in ('nan','')]
        pre_q = np.mean([q for _,q in pre]) if pre else float('nan')
        post_vals = [q for _,q in post]
        peak_q = min(post_vals) if post_vals else float('nan')
        mean_post = np.mean(post_vals) if post_vals else float('nan')
        peak_delta = pre_q - peak_q if not np.isnan(pre_q) and not np.isnan(peak_q) else float('nan')
        auc = sum(max(0, pre_q - q) for q in post_vals) if post_vals else 0

        # Latency: first step where qpos notably changes
        latency = None
        for s, q in post:
            if abs(q - pre_q) > 0.0005:
                latency = s - d5_emit; break
        # Sustain: consecutive steps with qpos < baseline
        sustain = 0
        for s, q in post:
            if q < pre_q: sustain += 1
            else: break
    else:
        pre_q = peak_q = mean_post = peak_delta = auc = float('nan')
        latency = sustain = None

    # Object/contact
    obj_z_vals = [(int(r['step']), float(r.get('obj_z', 0) or 0)) for r in rows if r.get('obj_z','') not in ('nan','')]
    eef_dist_vals = [(int(r['step']), float(r.get('eef_obj_dist', 0) or 0)) for r in rows if r.get('eef_obj_dist','') not in ('nan','')] if any('eef_obj_dist' in r for r in rows) else []

    # Contact quality proxies based on telemetry patterns
    success = bool(summary.get('task_success', False))

    r = {
        'seed': seed, 'condition': cond, 'n_steps': n_steps, 'd5_emit': d5_emit,
        'n_attack_frames': n_atk, 'token_open_count': n_open_token,
        'arm_ok_count': n_arm_ok, 'env_open_count': n_env_open,
        'token_duty': round(n_open_token/n_atk, 3) if n_atk>0 else 0,
        'env_duty': round(n_env_open/n_atk, 3) if n_atk>0 else 0,
        'arm_duty': round(n_arm_ok/n_atk, 3) if n_atk>0 else 0,
        'baseline_qpos': round(pre_q, 6) if not np.isnan(pre_q) else None,
        'peak_qpos': round(peak_q, 6) if not np.isnan(peak_q) else None,
        'mean_post_qpos': round(mean_post, 6) if not np.isnan(mean_post) else None,
        'peak_delta': round(peak_delta, 6) if not np.isnan(peak_delta) else None,
        'qpos_auc': round(auc, 6),
        'response_latency': latency, 'sustain_steps': sustain,
        'success': success,
    }
    results.append(r)
    print('  {} {}: atk={} tok={}/{} env={}/{} peak_d={} succ={}'.format(
        seed, cond, n_atk, n_open_token, n_atk, n_env_open, n_atk, r['peak_delta'], success))

if not results:
    print('No results yet — waiting for GPU jobs')
    sys.exit(0)

# ── Episode table ──
fields = list(results[0].keys())
with open(OUT_DIR / 'l3_temporal_episode_results.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(results)

# ── G1-G5 classification ──
def get_val(seed, cond, key):
    for r in results:
        if r['seed'] == seed and r['condition'] == cond:
            return r.get(key, 0) or 0
    return 0

gates = []
for seed in ['seed81', 'seed82']:
    # G1: D5 emit
    g1 = all(r['d5_emit'] == 60 for r in results if r['seed'] == seed)
    # G2: Semantic OPEN duty
    true_tok = get_val(seed, 'TRUE_TEMPORAL_K10', 'token_duty')
    rand_tok = get_val(seed, 'RAND_TEMPORAL_K10', 'token_duty')
    shuf_tok = get_val(seed, 'SHUFFLED_TEMPORAL_K10', 'token_duty')
    g2 = (true_tok >= 0.80) and (true_tok - max(rand_tok, shuf_tok) >= 0.50)
    # G3: Env OPEN duty
    true_env = get_val(seed, 'TRUE_TEMPORAL_K10', 'env_duty')
    rand_env = get_val(seed, 'RAND_TEMPORAL_K10', 'env_duty')
    shuf_env = get_val(seed, 'SHUFFLED_TEMPORAL_K10', 'env_duty')
    g3 = (true_env >= 0.80) and (true_env - max(rand_env, shuf_env) >= 0.50)
    # G4: Physical
    true_pd = get_val(seed, 'TRUE_TEMPORAL_K10', 'peak_delta') or 0
    clean_pd = get_val(seed, 'CLEAN_D5', 'peak_delta') or 0
    single_pd = get_val(seed, 'TRUE_SINGLE', 'peak_delta') or 0
    rand_pd = get_val(seed, 'RAND_TEMPORAL_K10', 'peak_delta') or 0
    shuf_pd = get_val(seed, 'SHUFFLED_TEMPORAL_K10', 'peak_delta') or 0
    max_ctrl_pd = max(clean_pd, single_pd, rand_pd, shuf_pd)
    g4 = (true_pd - max_ctrl_pd >= 0.002)  # 0.002 threshold for Butter qpos
    # G5A: Contact
    g5a = False  # placeholder — needs video
    # G5B: Task
    true_succ = get_val(seed, 'TRUE_TEMPORAL_K10', 'success')
    clean_succ = get_val(seed, 'CLEAN_D5', 'success')
    g5b = (clean_succ and not true_succ)

    gates.append({
        'seed': seed, 'G1_D5_emit': g1, 'G2_token_duty': g2, 'G3_env_duty': g3,
        'G4_physical': g4, 'G5A_contact': g5a, 'G5B_task': g5b,
        'true_token_duty': true_tok, 'true_env_duty': true_env,
        'true_peak_delta': true_pd, 'max_ctrl_peak_delta': max_ctrl_pd,
    })

# Overall
both_g2 = all(g['G2_token_duty'] for g in gates)
both_g3 = all(g['G3_env_duty'] for g in gates)
both_g4 = all(g['G4_physical'] for g in gates)
both_g5b = all(g['G5B_task'] for g in gates)

if both_g2 and both_g3 and both_g4 and both_g5b:
    classification = 'L3_D5_TRIGGERED_VIS_TEMPORAL_TASK_FAILURE_PASS'
elif both_g2 and both_g3 and both_g4:
    classification = 'L3_D5_TRIGGERED_VIS_TEMPORAL_PHYSICAL_BRIDGE_PASS_TASK_EFFECT_NOT_PROVEN'
elif both_g2 and both_g3:
    classification = 'VIS_TEMPORAL_SEMANTIC_COMMAND_DUTY_PASS_PHYSICAL_NOT_ESTABLISHED'
else:
    classification = 'VIS_TEMPORAL_INCOMPLETE'

print('\n=== GATES ===')
for g in gates:
    print('  {}: G1={} G2={} G3={} G4={} G5B={} tok={:.1f} env={:.1f} pd={:.4f}/{:.4f}'.format(
        g['seed'], g['G1_D5_emit'], g['G2_token_duty'], g['G3_env_duty'],
        g['G4_physical'], g['G5B_task'], g['true_token_duty'], g['true_env_duty'],
        g['true_peak_delta'], g['max_ctrl_peak_delta']))
print('Classification: {}'.format(classification))

with open(OUT_DIR / 'l3_temporal_final_gates.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(gates[0].keys())); w.writeheader(); w.writerows(gates)

gate_json = {'classification': classification, 'gates': gates, 'episodes': len(results)}
with open(ART_DIR / 'l3_temporal_final_gate.json', 'w') as f:
    json.dump(gate_json, f, indent=2)

print('\nDone. Tables: {}, Gate: {}'.format(OUT_DIR, ART_DIR))
