#!/usr/bin/env python3
"""First-divergence analysis: compare repeatability runs against metric refresh."""
import json, os, hashlib, csv
from pathlib import Path
from collections import defaultdict

METRIC = Path('/mnt/sdc/dty_user/openvla_attack/evidence/phase7_object/metric_refresh_v2')
RPT = Path('/mnt/sdc/dty_user/openvla_attack/evidence/phase7_object/repeatability_v2')
OUT = Path('/mnt/sdc/dty_user/openvla_attack/reports/repeatability')
OUT.mkdir(parents=True, exist_ok=True)

# Map repeatability dir names to metric keys
# RPT dir name format: CELL_sSEED_TAG_rpt
# Hard-coded mapping from RPT dir name to (cell, seed, condition)
# From run_one_repeatability.sh JOBS array
RPT_MAP = {
    'alphabet_soup_s0_s456_armlock_rpt': ('alphabet_soup_s0', '456', 'tma_armlock'),
    'bbq_sauce_s0_s123_nolock_rpt': ('bbq_sauce_s0', '123', 'tma_nolock'),
    'bbq_sauce_s0_s42_nolock_rpt': ('bbq_sauce_s0', '42', 'prefix_nolock'),
    'bbq_sauce_s0_s456_nolock_rpt': ('bbq_sauce_s0', '456', 'tma_nolock'),
    'butter_s0_s123_nolock_rpt': ('butter_s0', '123', 'tma_nolock'),
    'butter_s0_s42_nolock_rpt': ('butter_s0', '42', 'tma_nolock'),
    'butter_s0_s456_nolock_rpt': ('butter_s0', '456', 'prefix_nolock'),
    'butter_s2_s42_armlock_rpt': ('butter_s2', '42', 'tma_armlock'),
    'butter_s2_s42_nolock_rpt': ('butter_s2', '42', 'prefix_nolock'),
    'butter_s2_s456_armlock_rpt': ('butter_s2', '456', 'tma_armlock'),
    'orange_juice_s0_s42_nolock_rpt': ('orange_juice_s0', '42', 'prefix_nolock'),
    'tomato_sauce_s0_s123_nolock_rpt': ('tomato_sauce_s0', '123', 'tma_nolock'),
    'tomato_sauce_s0_s42_nolock_rpt': ('tomato_sauce_s0', '42', 'prefix_nolock'),
    'tomato_sauce_s0_s456_armlock_rpt': ('tomato_sauce_s0', '456', 'tma_armlock'),
    'tomato_sauce_s0_s456_nolock_rpt': ('tomato_sauce_s0', '456', 'tma_nolock'),
}

def parse_rpt_dir(name):
    if name in RPT_MAP:
        cell, seed, cond = RPT_MAP[name]
        is_armlock = 'armlock' in cond
        return cell, seed, is_armlock, cond
    # Fallback
    return None, None, False, None

def load_telemetry(path):
    if not path.exists(): return []
    rows = []
    with open(path) as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows

def step_hash(row, fields):
    """Hash selected fields for comparison."""
    vals = [row.get(f, '') for f in fields]
    return hashlib.sha256('|'.join(vals).encode()).hexdigest()[:12]

# For each repeatability run, find matching metric run and compare
results = []
for d in sorted(RPT.iterdir()):
    if not d.is_dir(): continue
    cell, seed, is_armlock, cond = parse_rpt_dir(d.name)
    if not cell or not seed or not cond:
        print(f'SKIP {d.name}: cannot parse')
        continue

    # Find corresponding metric run
    metric_cell_dir = METRIC / cond
    metric_run = None
    for md in metric_cell_dir.iterdir():
        if not md.is_dir(): continue
        mname = md.name
        if cell in mname and ('_s' + seed) in mname:
            metric_run = md
            break

    if not metric_run:
        print(f'SKIP {d.name}: no metric match')
        continue

    # Load telemetry
    rpt_rows = load_telemetry(d / 'step_telemetry.csv')
    met_rows = load_telemetry(metric_run / 'step_telemetry.csv')

    # Load summaries
    rpt_sum = json.load(open(d / 'episode_summary.json'))
    met_sum = json.load(open(metric_run / 'episode_summary.json'))

    # Pre-trigger: find first step where clean_policy_action differs
    first_div_step = None
    first_div_field = None
    first_div_stage = None
    pre_trigger_match = True

    for i in range(min(len(rpt_rows), len(met_rows))):
        rr = rpt_rows[i]
        mr = met_rows[i]
        # Compare clean policy action
        rpt_clean = rr.get('clean_policy_action_7d', '')
        met_clean = mr.get('clean_policy_action_7d', '')
        if rpt_clean and met_clean:
            try:
                rpt_arr = json.loads(rpt_clean)
                met_arr = json.loads(met_clean)
                max_diff = max(abs(a-b) for a,b in zip(rpt_arr, met_arr))
                if max_diff > 1e-6:
                    first_div_step = i
                    first_div_field = f'clean_policy_action_7d (max_diff={max_diff:.2e})'
                    first_div_stage = 'pre_trigger'
                    pre_trigger_match = False
                    break
            except (json.JSONDecodeError, ValueError):
                if rpt_clean != met_clean:
                    first_div_step = i
                    first_div_field = 'clean_policy_action_7d (string mismatch)'
                    first_div_stage = 'pre_trigger'
                    pre_trigger_match = False
                    break

    # Attack window comparison
    emit_match = (rpt_sum.get('mlp_emit_step') == met_sum.get('mlp_emit_step'))
    atk_frames_match = (rpt_sum.get('attack_frames') == met_sum.get('attack_frames'))

    rpt_atk = [r for r in rpt_rows if r.get('attack_this') == 'True']
    met_atk = [r for r in met_rows if r.get('attack_this') == 'True']

    adv_token_match = True
    adv_action_max_diff = 0.0
    exec_action_max_diff = 0.0

    if len(rpt_atk) == len(met_atk) and len(rpt_atk) > 0:
        for i in range(len(rpt_atk)):
            # Compare adv token IDs
            rpt_tok = rpt_atk[i].get('adv_token_ids_7d', '')
            met_tok = met_atk[i].get('adv_token_ids_7d', '')
            if rpt_tok and met_tok:
                try:
                    rpt_tok_arr = json.loads(rpt_tok)
                    met_tok_arr = json.loads(met_tok)
                    if rpt_tok_arr != met_tok_arr:
                        adv_token_match = False
                except (json.JSONDecodeError, ValueError):
                    if rpt_tok != met_tok:
                        adv_token_match = False

            # Compare adv policy action
            rpt_adv = json.loads(rpt_atk[i].get('adv_policy_action_7d_before_lock', '[]'))
            met_adv = json.loads(met_atk[i].get('adv_policy_action_7d_before_lock', '[]'))
            if rpt_adv and met_adv:
                diff = max(abs(a-b) for a,b in zip(rpt_adv, met_adv))
                adv_action_max_diff = max(adv_action_max_diff, diff)

            # Compare executed action
            rpt_exec = json.loads(rpt_atk[i].get('executed_policy_action_7d_after_lock', '[]'))
            met_exec = json.loads(met_atk[i].get('executed_policy_action_7d_after_lock', '[]'))
            if rpt_exec and met_exec:
                diff = max(abs(a-b) for a,b in zip(rpt_exec, met_exec))
                exec_action_max_diff = max(exec_action_max_diff, diff)

    # Determine if divergence is physics-only
    physics_only = (pre_trigger_match and emit_match and adv_token_match
                    and adv_action_max_diff < 1e-7 and exec_action_max_diff < 1e-7)

    outcome_match = (rpt_sum.get('task_success') == met_sum.get('task_success'))

    result = {
        'cell': cell, 'seed': seed, 'condition': cond,
        'is_armlock': is_armlock,
        'metric_success': met_sum.get('task_success'),
        'rpt_success': rpt_sum.get('task_success'),
        'outcome_match': outcome_match,
        'pre_trigger_match': pre_trigger_match,
        'emit_match': emit_match,
        'atk_frames_match': atk_frames_match,
        'adv_token_match': adv_token_match,
        'adv_action_max_diff': adv_action_max_diff,
        'exec_action_max_diff': exec_action_max_diff,
        'physics_only_divergence': physics_only,
        'first_divergence_step': first_div_step if first_div_step is not None else -1,
        'first_divergence_stage': first_div_stage or 'none',
        'first_divergence_field': first_div_field or 'none',
    }
    results.append(result)

    status = 'PHYSICS_ONLY' if physics_only else 'PRE_TRIGGER' if not pre_trigger_match else 'ATTACK_DIFF' if not adv_token_match else 'OUTCOME_FLIP'
    print(f'{cell} s{seed} {cond}: {status} metric={met_sum.get("task_success")} rpt={rpt_sum.get("task_success")} pre={pre_trigger_match} emit={emit_match} tok={adv_token_match} adv_diff={adv_action_max_diff:.2e}')

# Summary
n_physics = sum(1 for r in results if r['physics_only_divergence'])
n_pre = sum(1 for r in results if not r['pre_trigger_match'])
n_attack = sum(1 for r in results if not r['adv_token_match'] and r['pre_trigger_match'])
n_outcome = sum(1 for r in results if not r['outcome_match'])

print(f'\n=== SUMMARY ({len(results)} runs) ===')
print(f'Physics-only divergence: {n_physics}/{len(results)}')
print(f'Pre-trigger divergence: {n_pre}/{len(results)}')
print(f'Attack-window divergence: {n_attack}/{len(results)}')
print(f'Outcome mismatch: {n_outcome}/{len(results)}')

# Write CSV
with open(OUT / 'REPEATABILITY_15_FINAL.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
    w.writeheader()
    w.writerows(results)
print(f'\nWritten: {OUT}/REPEATABILITY_15_FINAL.csv')

# Gate decision
if n_pre == 0 and n_attack == 0 and n_physics > 0:
    decision = 'PASS — physics-only divergence confirmed'
elif n_pre == 0 and n_attack <= 2:
    decision = 'CONDITIONAL PASS — minor attack-window variation, disclose'
else:
    decision = 'HOLD — pre-trigger or significant attack instability detected'

print(f'GATE: {decision}')
