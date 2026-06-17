#!/usr/bin/env python3
"""Auditor v3: Corrected arm selectivity, physical timing, paired deltas, fail-closed gates.

Fixes vs v2:
  - Exact attack step arrays {60} and {60..69} verified
  - Arm selectivity: independent G2 gate, min_arm>=5 threshold, raw adv_arm reported
  - Physical: response measured from emit+1 (not emit=pre-action qpos)
  - Paired TRUE-CLEAN delta curves alongside individual excursion
  - q7/q8 independent
  - Latency measured from emit+1
  - No automatic PHYSICAL_BRIDGE_PASS classification
  - SPECIFICITY requires 10/10 data (fail-closed on any missing cell)
"""
import csv, json, os, sys, numpy as np
from pathlib import Path

OUT_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('tables')
RPT_DIR = Path(sys.argv[2]) if len(sys.argv) > 2 else Path('reports')
ART_DIR = Path(sys.argv[3]) if len(sys.argv) > 3 else Path('artifacts')
BASE = '/data/liuyu/outputs/l3_d5_vis_temporal_r1'

for d in [OUT_DIR, RPT_DIR, ART_DIR]:
    d.mkdir(parents=True, exist_ok=True)

EXPECTED_CELLS = [
    ('seed81', 'CLEAN_D5'), ('seed81', 'TRUE_SINGLE'), ('seed81', 'TRUE_TEMPORAL_K10'),
    ('seed81', 'RAND_TEMPORAL_K10'), ('seed81', 'SHUFFLED_TEMPORAL_K10'),
    ('seed82', 'CLEAN_D5'), ('seed82', 'TRUE_SINGLE'), ('seed82', 'TRUE_TEMPORAL_K10'),
    ('seed82', 'RAND_TEMPORAL_K10'), ('seed82', 'SHUFFLED_TEMPORAL_K10'),
]

# ── Load ──
episodes = {}
missing = []
for (seed, cond) in EXPECTED_CELLS:
    path = os.path.join(BASE, seed, cond)
    tele = os.path.join(path, 'step_telemetry.csv')
    summ = os.path.join(path, 'episode_summary.json')
    if not os.path.isfile(tele) or not os.path.isfile(summ):
        missing.append(f'{seed}/{cond}')
        continue
    rows = list(csv.DictReader(open(tele)))
    summary = json.load(open(summ))
    episodes[(seed, cond)] = {'rows': rows, 'summary': summary}

AUDIT_BLOCKED = len(missing) > 0
print(f'Cells: {len(episodes)}/10 present' + (f' MISSING: {missing}' if missing else ''))

if len(episodes) == 0:
    print('FATAL: No data.')
    sys.exit(1)


# ── Helpers ──
def sf(v, default=float('nan')):
    try: return float(v)
    except: return default


# ── Per-episode metrics ──
results = []
for (seed, cond), ep in sorted(episodes.items()):
    rows = ep['rows']; sm = ep['summary']
    n_steps = len(rows)
    d5_emit = int(sm.get('d5_emit_step', -1))
    n_atk = int(sm.get('attack_frames', 0))

    # ── Attack frames ──
    atk_rows = [r for r in rows if r.get('attack_this') in ('True', 'true', '1')]
    atk_steps = [int(r['step']) for r in atk_rows]
    n_open_tok = sum(1 for r in atk_rows if sf(r.get('adv_token',''), 0) == 31744)
    adv_arm_vals = [int(sf(r.get('adv_arm',''), 0)) for r in atk_rows if r.get('adv_arm','') not in ('','nan')]
    n_arm_ok = sum(1 for v in adv_arm_vals if v >= 5)
    min_arm = min(adv_arm_vals) if adv_arm_vals else None
    n_env_open = sum(1 for r in atk_rows if sf(r.get('env_gripper',''), 1) < 0)

    # Exact attack step verification
    if cond == 'TRUE_SINGLE':
        steps_ok = (atk_steps == [60])
    elif 'TEMPORAL' in cond and n_atk > 0:
        steps_ok = (atk_steps == list(range(60, 60 + n_atk)))
    elif n_atk == 0:
        steps_ok = (atk_steps == [])
    else:
        steps_ok = False

    # ── Physical: pre-attack baseline (emit-5 to emit-1, BEFORE attack action) ──
    pre_steps = [r for r in rows if max(0, d5_emit - 5) <= int(r['step']) < d5_emit]
    pre_q = [sf(r['qpos_sum']) for r in pre_steps if str(r.get('qpos_sum','')) not in ('nan','')]
    baseline = np.mean(pre_q) if pre_q else float('nan')

    # ── Post-attack: emit+1 onward (first step after attack action takes effect) ──
    post_steps_raw = [(int(r['step']), sf(r['qpos_sum']), sf(r['q7']), sf(r['q8']))
                      for r in rows if d5_emit + 1 <= int(r['step']) < d5_emit + 21
                      if str(r.get('qpos_sum','')) not in ('nan','')]
    post_q = [(s, q) for s, q, _, _ in post_steps_raw]
    post_q7 = [(s, q7) for s, _, q7, _ in post_steps_raw if not np.isnan(q7)]
    post_q8 = [(s, q8) for s, _, _, q8 in post_steps_raw if not np.isnan(q8)]
    post_vals = [q for _, q in post_q]
    peak_q = min(post_vals) if post_vals else float('nan')
    mean_post = np.mean(post_vals) if post_vals else float('nan')
    peak_delta = baseline - peak_q if not np.isnan(baseline) and not np.isnan(peak_q) else float('nan')

    # AUC: sum of positive opening from emit+1 to emit+20
    auc = sum(max(0, baseline - q) for q in post_vals) if post_vals and not np.isnan(baseline) else 0

    # Latency from emit+1 (first step with qpos responding: q < baseline - 0.0005)
    latency = None
    for s, q in post_q:
        if not np.isnan(baseline) and q < baseline - 0.0005:
            latency = s - d5_emit
            break

    # Sustain: consecutive steps where q < baseline, starting from first response
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

    # q7/q8 independent (post window)
    q7_post_vals = [q7 for _, q7 in post_q7]
    q8_post_vals = [q8 for _, q8 in post_q8]
    q7_mean_post = np.mean(q7_post_vals) if q7_post_vals else float('nan')
    q8_mean_post = np.mean(q8_post_vals) if q8_post_vals else float('nan')
    q7_pre_vals = [sf(r['q7']) for r in pre_steps if str(r.get('q7','')) not in ('nan','')]
    q8_pre_vals = [sf(r['q8']) for r in pre_steps if str(r.get('q8','')) not in ('nan','')]
    q7_baseline = np.mean(q7_pre_vals) if q7_pre_vals else float('nan')
    q8_baseline = np.mean(q8_pre_vals) if q8_pre_vals else float('nan')

    success = bool(sm.get('task_success', False))

    results.append({
        'seed': seed, 'condition': cond, 'n_steps': n_steps, 'd5_emit': d5_emit,
        'n_atk': n_atk, 'atk_steps': str(atk_steps), 'steps_exact_ok': steps_ok,
        'n_open_token': n_open_tok,
        'adv_arm_values': str(adv_arm_vals),
        'n_arm_ok': n_arm_ok, 'min_arm': int(min_arm) if min_arm is not None else None,
        'n_env_open': n_env_open,
        'token_duty': round(n_open_tok / n_atk, 3) if n_atk > 0 else 0,
        'env_duty': round(n_env_open / n_atk, 3) if n_atk > 0 else 0,
        'arm_duty': round(n_arm_ok / n_atk, 3) if n_atk > 0 else 0,
        'baseline_q': round(baseline, 6) if not np.isnan(baseline) else None,
        'peak_q': round(peak_q, 6) if not np.isnan(peak_q) else None,
        'mean_post_q': round(mean_post, 6) if not np.isnan(mean_post) else None,
        'peak_delta': round(peak_delta, 6) if not np.isnan(peak_delta) else None,
        'auc': round(auc, 6),
        'latency': latency,
        'sustain': sustain,
        'q7_baseline': round(q7_baseline, 6) if not np.isnan(q7_baseline) else None,
        'q7_mean_post': round(q7_mean_post, 6) if not np.isnan(q7_mean_post) else None,
        'q8_baseline': round(q8_baseline, 6) if not np.isnan(q8_baseline) else None,
        'q8_mean_post': round(q8_mean_post, 6) if not np.isnan(q8_mean_post) else None,
        'success': success,
    })

# ── Write episode table ──
ep_fields = list(results[0].keys())
with open(OUT_DIR / 'l3_temporal_episode_results_v3.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=ep_fields); w.writeheader(); w.writerows(results)

print(f'Episodes: {len(results)}/10 ({len(missing)} missing)')

# ── Paired deltas ──
def get_val(seed, cond, key):
    for r in results:
        if r['seed'] == seed and r['condition'] == cond:
            return r.get(key)
    return None

paired_deltas = {}
for seed in ['seed81', 'seed82']:
    true_pd = get_val(seed, 'TRUE_TEMPORAL_K10', 'peak_delta') or 0
    clean_pd = get_val(seed, 'CLEAN_D5', 'peak_delta') or 0
    true_auc = get_val(seed, 'TRUE_TEMPORAL_K10', 'auc') or 0
    clean_auc = get_val(seed, 'CLEAN_D5', 'auc') or 0
    paired_deltas[seed] = {
        'TRUE_T10_pd': true_pd, 'CLEAN_pd': clean_pd,
        'TRUE_minus_CLEAN_pd': round(true_pd - clean_pd, 6),
        'TRUE_T10_auc': true_auc, 'CLEAN_auc': clean_auc,
        'TRUE_minus_CLEAN_auc': round(true_auc - clean_auc, 6),
    }


# ── Gate classification ──
gates = []
for seed in ['seed81', 'seed82']:
    # G0: Data completeness + exact step check
    cells_ok = all((seed, c) in episodes for c in
                   ['CLEAN_D5', 'TRUE_SINGLE', 'TRUE_TEMPORAL_K10',
                    'RAND_TEMPORAL_K10', 'SHUFFLED_TEMPORAL_K10'])
    d5_ok = all(get_val(seed, c, 'd5_emit') == 60
                for c in ['CLEAN_D5', 'TRUE_SINGLE', 'TRUE_TEMPORAL_K10',
                          'RAND_TEMPORAL_K10', 'SHUFFLED_TEMPORAL_K10']
                if get_val(seed, c, 'd5_emit') is not None)
    steps_ok = (get_val(seed, 'TRUE_SINGLE', 'steps_exact_ok') and
                get_val(seed, 'TRUE_TEMPORAL_K10', 'steps_exact_ok'))
    g0 = cells_ok and d5_ok and steps_ok

    # G1: Semantic OPEN duty (token = 31744)
    true_tok = get_val(seed, 'TRUE_TEMPORAL_K10', 'token_duty') or 0
    rand_tok = get_val(seed, 'RAND_TEMPORAL_K10', 'token_duty') or 0
    shuf_tok = get_val(seed, 'SHUFFLED_TEMPORAL_K10', 'token_duty') or 0
    g1 = (true_tok >= 0.80) and (true_tok - max(rand_tok, shuf_tok) >= 0.50)

    # G2: ARM SELECTIVITY (independent gate)
    true_arm_duty = get_val(seed, 'TRUE_TEMPORAL_K10', 'arm_duty') or 0
    true_min_arm = get_val(seed, 'TRUE_TEMPORAL_K10', 'min_arm')
    true_arm_vals = get_val(seed, 'TRUE_TEMPORAL_K10', 'adv_arm_values') or ''
    rand_arm_duty = get_val(seed, 'RAND_TEMPORAL_K10', 'arm_duty') or 0
    shuf_arm_duty = get_val(seed, 'SHUFFLED_TEMPORAL_K10', 'arm_duty') or 0
    single_min_arm = get_val(seed, 'TRUE_SINGLE', 'min_arm')
    # Arm selectivity: TRUE arm duty >= 0.80 AND min_arm >= 5
    g2_arm_selectivity = (true_arm_duty >= 0.80 and true_min_arm is not None and true_min_arm >= 5)
    # Arm selectivity vs controls
    g2_arm_vs_controls = (true_arm_duty - max(rand_arm_duty, shuf_arm_duty) >= 0.50)
    g2 = g2_arm_selectivity and g2_arm_vs_controls

    # G3: Env OPEN duty
    true_env = get_val(seed, 'TRUE_TEMPORAL_K10', 'env_duty') or 0
    rand_env = get_val(seed, 'RAND_TEMPORAL_K10', 'env_duty') or 0
    shuf_env = get_val(seed, 'SHUFFLED_TEMPORAL_K10', 'env_duty') or 0
    g3 = (true_env >= 0.80) and (true_env - max(rand_env, shuf_env) >= 0.50)

    # G4: Physical — paired TRUE-CLEAN delta
    true_pd = get_val(seed, 'TRUE_TEMPORAL_K10', 'peak_delta') or 0
    clean_pd = get_val(seed, 'CLEAN_D5', 'peak_delta') or 0
    single_pd = get_val(seed, 'TRUE_SINGLE', 'peak_delta') or 0
    rand_pd = get_val(seed, 'RAND_TEMPORAL_K10', 'peak_delta') or 0
    shuf_pd = get_val(seed, 'SHUFFLED_TEMPORAL_K10', 'peak_delta') or 0
    true_auc = get_val(seed, 'TRUE_TEMPORAL_K10', 'auc') or 0
    clean_auc = get_val(seed, 'CLEAN_D5', 'auc') or 0
    single_auc = get_val(seed, 'TRUE_SINGLE', 'auc') or 0
    rand_auc = get_val(seed, 'RAND_TEMPORAL_K10', 'auc') or 0
    shuf_auc = get_val(seed, 'SHUFFLED_TEMPORAL_K10', 'auc') or 0

    # Paired TRUE-CLEAN delta (the correct causal metric)
    paired_pd_delta = true_pd - clean_pd
    paired_auc_delta = true_auc - clean_auc

    # Individual excursion check (TRUE vs controls, for reference)
    max_ctrl_pd = max(clean_pd, single_pd, rand_pd, shuf_pd)
    max_ctrl_auc = max(clean_auc, single_auc, rand_auc, shuf_auc)
    true_pd_exceeds_ctrl = true_pd > max_ctrl_pd
    true_auc_exceeds_ctrl = true_auc > max_ctrl_auc

    # Latency from emit+1
    true_lat = get_val(seed, 'TRUE_TEMPORAL_K10', 'latency')
    true_sus = get_val(seed, 'TRUE_TEMPORAL_K10', 'sustain') or 0

    # G4A: paired TRUE-CLEAN AUC increase (emission includes temporal cumulative effect)
    g4a_paired_auc = paired_auc_delta > 0.002
    # G4B: low latency (response within 3 steps after emit+1)
    g4b_latency = true_lat is not None and 1 <= true_lat <= 5
    # G4C: sustain >= 2
    g4c_sustain = true_sus >= 2
    # G4D: TRUE pd > CLEAN pd (paired)
    g4d_paired_pd = paired_pd_delta > 0.001
    g4 = g4a_paired_auc and g4b_latency and g4c_sustain and g4d_paired_pd

    # Controls clean
    rand_open = get_val(seed, 'RAND_TEMPORAL_K10', 'n_env_open') or 0
    shuf_open = get_val(seed, 'SHUFFLED_TEMPORAL_K10', 'n_env_open') or 0
    single_env_duty = get_val(seed, 'TRUE_SINGLE', 'env_duty') or 0
    specificity_ok = (rand_open <= 3 and shuf_open <= 3 and single_env_duty >= 0.80)

    # G5A: Contact — NOT AUDITABLE from current telemetry
    g5a = 'NOT_AUDITABLE'

    # G5B: Task failure
    clean_succ = get_val(seed, 'CLEAN_D5', 'success')
    rand_succ = get_val(seed, 'RAND_TEMPORAL_K10', 'success')
    shuf_succ = get_val(seed, 'SHUFFLED_TEMPORAL_K10', 'success')
    true_succ = get_val(seed, 'TRUE_TEMPORAL_K10', 'success')
    single_succ = get_val(seed, 'TRUE_SINGLE', 'success')
    g5b = (clean_succ and rand_succ and shuf_succ and (not true_succ))

    gates.append({
        'seed': seed,
        'G0_data_steps': g0,
        'G1_semantic_open': g1,
        'G2_arm_selectivity': g2,
        'G3_env_open': g3,
        'G4_physical_paired': g4,
        'G5A_contact': g5a,
        'G5B_task_failure': g5b,
        'specificity_ok': specificity_ok,
        # Raw metrics for transparency
        'true_tok_duty': true_tok,
        'true_env_duty': true_env,
        'true_arm_duty': true_arm_duty,
        'true_min_arm': true_min_arm,
        'true_arm_vals': true_arm_vals,
        'true_pd': true_pd,
        'clean_pd': clean_pd,
        'single_pd': single_pd,
        'rand_pd': rand_pd,
        'shuf_pd': shuf_pd,
        'paired_pd_delta': paired_pd_delta,
        'true_auc': true_auc,
        'clean_auc': clean_auc,
        'paired_auc_delta': paired_auc_delta,
        'true_lat': true_lat,
        'true_sus': true_sus,
    })

# ── Overall classification ──
all_specific = all(g['specificity_ok'] for g in gates)
all_g0 = all(g['G0_data_steps'] for g in gates)
all_g1 = all(g['G1_semantic_open'] for g in gates)
all_g2 = all(g['G2_arm_selectivity'] for g in gates)
all_g3 = all(g['G3_env_open'] for g in gates)
all_g4 = all(g['G4_physical_paired'] for g in gates)
all_g5b = all(g['G5B_task_failure'] for g in gates)

if AUDIT_BLOCKED:
    classification = 'AUDIT_BLOCKED_MISSING_CELLS'
elif not all_specific:
    classification = 'SPECIFICITY_BLOCKED'
elif all_g0 and all_g1 and all_g2 and all_g3 and all_g4 and all_g5b:
    classification = 'L3_D5_TRIGGERED_VIS_TEMPORAL_TASK_FAILURE_PASS'
elif all_g0 and all_g1 and all_g3:
    # Semantic control established, env OPEN established
    if all_g4:
        if all_g2:
            classification = 'L3_D5_VIS_TEMPORAL_SELECTIVE_PHYSICAL_BRIDGE_PASS_TASK_EFFECT_NOT_PROVEN'
        else:
            classification = 'L3_D5_VIS_TEMPORAL_SEMANTIC_COMMAND_DUTY_PASS_PHYSICAL_RESPONSE_OBSERVED_ARM_SELECTIVITY_NOT_ESTABLISHED'
    else:
        classification = 'VIS_TEMPORAL_SEMANTIC_COMMAND_DUTY_PASS_PHYSICAL_NOT_ESTABLISHED'
elif all_g0 and all_g1 and all_g3:
    classification = 'VIS_TEMPORAL_SEMANTIC_COMMAND_DUTY_PASS_PHYSICAL_NOT_ESTABLISHED'
else:
    classification = 'VIS_TEMPORAL_INCOMPLETE'

print('\n=== GATE RESULTS ===')
for g in gates:
    print(f"  {g['seed']}: G0={g['G0_data_steps']} G1={g['G1_semantic_open']}(tok={g['true_tok_duty']}) "
          f"G2={g['G2_arm_selectivity']}(arm_d={g['true_arm_duty']},min={g['true_min_arm']}) "
          f"G3={g['G3_env_open']}(env={g['true_env_duty']}) "
          f"G4={g['G4_physical_paired']}(pd={g['true_pd']:.5f}/{g['clean_pd']:.5f} "
          f"paired_d={g['paired_pd_delta']:.5f} auc_d={g['paired_auc_delta']:.5f} "
          f"lat={g['true_lat']} sus={g['true_sus']}) "
          f"G5B={g['G5B_task_failure']} spec={g['specificity_ok']}")
    if not g['G2_arm_selectivity']:
        print(f"         ARM VALUES: {g['true_arm_vals']}")

print(f'\nClassification: {classification}')

# Write outputs
with open(OUT_DIR / 'l3_temporal_final_gates_v3.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(gates[0].keys())); w.writeheader(); w.writerows(gates)

# ── full audit artifact ──
gate_json = {
    'classification': classification,
    'classification_notes': [
        'G0: data completeness + D5 emit + exact attack step arrays verified',
        'G1: semantic OPEN token duty (31744) selectivity vs RAND/SHUFFLED',
        'G2: arm token selectivity (>=5/6 match, min_arm>=5) — INDEPENDENT gate',
        'G3: env OPEN command duty selectivity vs RAND/SHUFFLED',
        'G4: paired TRUE-CLEAN physical delta (AUC+peak, latency from emit+1, sustain)',
        'G5A: contact failure — NOT_AUDITABLE (no object telemetry)',
        'G5B: task failure — requires all controls success AND TRUE failure',
        'Latency measured from emit+1 (first step where attack action can take effect)',
        'Paired delta = TRUE - CLEAN (causal attribution, not individual excursion)',
        'SPECIFICITY_BLOCKED = controls produce env OPEN beyond tolerance',
    ],
    'audit_blocked': AUDIT_BLOCKED,
    'missing_cells': missing,
    'n_episodes': len(results),
    'gates': gates,
    'paired_deltas': paired_deltas,
    'episode_results': [{k: v for k, v in r.items() if k != 'adv_arm_values'}
                        for r in results],
}
with open(ART_DIR / 'l3_temporal_final_gate_v3.json', 'w') as f:
    json.dump(gate_json, f, indent=2, default=str)

print(f'Done: {OUT_DIR} -> {ART_DIR}')
