#!/usr/bin/env python3
"""Metric refresh parity audit vs canonical 3-seed results."""
import json, os
from pathlib import Path
from collections import defaultdict

METRIC = Path('/mnt/sdc/dty_user/openvla_attack/evidence/phase7_object/metric_refresh_v2')
CANONICAL_BASE = Path('/mnt/sdc/dty_user/openvla_attack/evidence/phase7_object/attack_benchmark')
SUPP = Path('/mnt/sdc/dty_user/openvla_attack/evidence/phase7_object/supplement_7h')

def get_canonical_results():
    results = {}
    # TMA no-lock
    tma_vanilla = SUPP / 'tma_vanilla'
    if tma_vanilla.exists():
        for d in tma_vanilla.iterdir():
            if not d.is_dir(): continue
            sfile = d / 'episode_summary.json'
            if not sfile.exists(): continue
            s = json.load(open(sfile))
            name = d.name
            cell = name.rsplit('_tma_s', 1)[0] if '_tma_s' in name else name.rsplit('_s', 1)[0]
            seed = name.rsplit('_s', 1)[-1]
            try: seed_int = int(seed) if seed.isdigit() else 42
            except: seed_int = 42
            key = f'{cell}|tma_nolock|s{seed_int}'
            results[key] = {'success': s.get('task_success'), 'emit': s.get('mlp_emit_step')}

    # TMA ArmLock
    tma_al = SUPP / 'tma_armlock'
    if tma_al.exists():
        for d in tma_al.iterdir():
            if not d.is_dir(): continue
            sfile = d / 'episode_summary.json'
            if not sfile.exists(): continue
            s = json.load(open(sfile))
            name = d.name
            cell = name.rsplit('_armlock_s', 1)[0] if '_armlock_s' in name else name.rsplit('_s', 1)[0]
            seed = name.rsplit('_s', 1)[-1]
            try: seed_int = int(seed) if seed.isdigit() else 42
            except: seed_int = 42
            key = f'{cell}|tma_armlock|s{seed_int}'
            results[key] = {'success': s.get('task_success'), 'emit': s.get('mlp_emit_step')}

    # Prefix no-lock (VIS from attack_benchmark)
    for d in CANONICAL_BASE.iterdir():
        if not d.is_dir() or '_vis_' not in d.name: continue
        sfile = d / 'episode_summary.json'
        if not sfile.exists(): continue
        s = json.load(open(sfile))
        name = d.name
        cell = name.split('_vis_s')[0]
        seed = name.split('_vis_s')[1] if '_vis_s' in name else '42'
        try: seed_int = int(seed) if seed.isdigit() else 42
        except: seed_int = 42
        key = f'{cell}|prefix_nolock|s{seed_int}'
        results[key] = {'success': s.get('task_success'), 'emit': s.get('mlp_emit_step')}

    # Prefix ArmLock
    oa = SUPP / 'ours_armlock'
    if oa.exists():
        for d in oa.iterdir():
            if not d.is_dir(): continue
            sfile = d / 'episode_summary.json'
            if not sfile.exists(): continue
            s = json.load(open(sfile))
            name = d.name
            cell = name.rsplit('_ours_armlock_s', 1)[0] if '_ours_armlock_s' in name else name.rsplit('_s', 1)[0]
            seed = name.rsplit('_s', 1)[-1]
            try: seed_int = int(seed) if seed.isdigit() else 42
            except: seed_int = 42
            key = f'{cell}|prefix_armlock|s{seed_int}'
            results[key] = {'success': s.get('task_success'), 'emit': s.get('mlp_emit_step')}

    return results

def get_metric_results():
    results = {}
    for tag_dir in METRIC.iterdir():
        if not tag_dir.is_dir(): continue
        cond = tag_dir.name
        for d in tag_dir.iterdir():
            if not d.is_dir(): continue
            sfile = d / 'episode_summary.json'
            if not sfile.exists(): continue
            s = json.load(open(sfile))
            name = d.name
            # cell_sX_sSEED -> cell + seed
            parts = name.rsplit('_s', 1)
            cell = parts[0] if len(parts) >= 2 else name
            seed = parts[1] if len(parts) >= 2 else '42'
            try: seed_int = int(seed) if seed.isdigit() else 42
            except: seed_int = 42
            key = f'{cell}|{cond}|s{seed_int}'
            results[key] = {
                'success': s.get('task_success'),
                'emit': s.get('mlp_emit_step'),
                'token_duty': s.get('token_open_duty'),
            }
    return results

canon = get_canonical_results()
metric = get_metric_results()

print(f'Canonical keys: {len(canon)}')
print(f'Metric keys: {len(metric)}')

discordant = []
matched = 0

all_keys = set(canon.keys()) | set(metric.keys())
for key in sorted(all_keys):
    c = canon.get(key)
    m = metric.get(key)
    if c and m:
        matched += 1
        if c['success'] != m['success']:
            discordant.append({
                'key': key,
                'canon_success': c['success'],
                'metric_success': m['success'],
                'canon_emit': c['emit'],
                'metric_emit': m['emit'],
            })

print(f'Matched: {matched}')
print(f'Discordant: {len(discordant)}')

if discordant:
    print('\n=== DISCORDANT KEYS ===')
    for d in discordant:
        print(f"  {d['key']}: canon={d['canon_success']} metric={d['metric_success']} canon_emit={d['canon_emit']} metric_emit={d['metric_emit']}")

# Per-condition summary
print('\n=== PER-CONDITION CANONICAL vs METRIC ===')
for cond in ['tma_nolock', 'tma_armlock', 'prefix_nolock', 'prefix_armlock']:
    c_keys = [k for k in canon if cond in k]
    m_keys = [k for k in metric if cond in k]
    c_fail = sum(1 for k in c_keys if canon[k]['success'] is False)
    m_fail = sum(1 for k in m_keys if metric[k]['success'] is False)
    c_total = len(c_keys)
    m_total = len(m_keys)
    if c_total and m_total:
        print(f'{cond}: canon={c_fail}/{c_total}={c_fail/c_total:.3f} metric={m_fail}/{m_total}={m_fail/m_total:.3f}')

# Write report
report_path = Path('/mnt/sdc/dty_user/openvla_attack/reports/phase7_table1/METRIC_REFRESH_PARITY_AUDIT.md')
with open(report_path, 'w') as f:
    f.write('# Metric Refresh Parity Audit\n\n')
    f.write(f'- Canonical keys matched: {len(canon)}\n')
    f.write(f'- Metric refresh keys: {len(metric)}\n')
    f.write(f'- Matched pairs: {matched}\n')
    f.write(f'- Discordant outcomes: {len(discordant)}\n\n')
    if discordant:
        f.write('## Discordant Runs\n\n')
        f.write('| Key | Canonical | Metric | Canon Emit | Metric Emit |\n')
        f.write('|-----|:---------:|:------:|:----------:|:-----------:|\n')
        for d in discordant:
            f.write(f"| {d['key']} | {d['canon_success']} | {d['metric_success']} | {d['canon_emit']} | {d['metric_emit']} |\n")
        f.write('\n## Likely Causes\n\n')
        f.write('1. Bridge v1 vs v2 differences (additional telemetry fields, timing)\n')
        f.write('2. GPU numerical non-determinism in PGD optimization\n')
        f.write('3. Statistical noise in attack success (stochastic PGD)\n')
        f.write('\n## Recommendation\n\n')
        f.write('Re-run discordant runs with v2 bridge to confirm root cause.\n')
        f.write('Disclose in paper if confirmed as closed-loop numerical sensitivity.\n')

print(f'\nReport: {report_path}')
