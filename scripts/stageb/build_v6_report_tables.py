#!/usr/bin/env python3
"""Build V6 report tables from raw summaries with full audit logic.

Usage:
  python build_v6_report_tables.py \
    --p1-dir /path/to/phase1_v2 \
    --p2-dir /path/to/stageb_v6_rand_veto \
    --p3-dir /path/to/stageb_v6_vis_pilot \
    --output-dir /path/to/tables
"""

import argparse
import csv
import json
import os
import glob
import statistics
from collections import defaultdict


def load_summaries(phase_dir):
    """Load all summary JSONs from a directory, return list of dicts."""
    summaries = []
    for sf in sorted(glob.glob(os.path.join(phase_dir, 'summary_*.json'))):
        with open(sf) as f:
            s = json.load(f)
        s['_summary_path'] = sf
        s['_summary_sha256'] = _sha256_file(sf)
        summaries.append(s)
    return summaries


def _sha256_file(path):
    """Return SHA256 of file, or 'MISSING' if not found."""
    import hashlib
    if not os.path.exists(path):
        return 'MISSING'
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            h.update(chunk)
    return h.hexdigest()


def parent_id(task, state_id):
    """Canonical parent_id from task and state_id fields."""
    return f"{task}_s{state_id}"


def trace_path(phase_dir, summary):
    """Derive trace path from summary path (convention: summary_X.json -> trace_X.csv)."""
    base = os.path.basename(summary['_summary_path'])
    # summary_alphabet_soup_s10_v6_clean_observer_seed0_jobas10_r1.json
    # -> trace_alphabet_soup_s10_v6_clean_observer_seed0_jobas10_r1.csv
    trace_name = base.replace('summary_', 'trace_').replace('.json', '.csv')
    return os.path.join(phase_dir, trace_name)


def verify_expected_keys(summaries, phase_label, expected_parents, seeds_per_parent,
                          is_clean_phase=False):
    """Verify expected count and uniqueness. Returns list of issues."""
    issues = []

    # Build key map
    by_parent = defaultdict(list)
    for s in summaries:
        pid = parent_id(s['task'], s['state_id'])
        by_parent[pid].append(s)

    # Check total count
    expected_total = len(expected_parents) * seeds_per_parent
    if len(summaries) != expected_total:
        issues.append(
            f"{phase_label}: expected {expected_total} summaries, found {len(summaries)}"
        )

    # Check each parent
    for pid in expected_parents:
        entries = by_parent.get(pid, [])
        if len(entries) != seeds_per_parent:
            issues.append(
                f"{phase_label}: parent {pid} has {len(entries)}/{seeds_per_parent} entries"
            )
        # Check for duplicates within parent
        if is_clean_phase:
            # For clean phase, key by job_id (since seed is always 0)
            seen_jobs = set()
            for e in entries:
                jid = e.get('job_id', '')
                if jid in seen_jobs:
                    issues.append(
                        f"{phase_label}: duplicate job_id {jid} for parent {pid}"
                    )
                seen_jobs.add(jid)
        else:
            # For RAND/VIS phases, key by seed
            seen_seeds = set()
            for e in entries:
                seed = e.get('attack_seed')
                if seed in seen_seeds:
                    issues.append(
                        f"{phase_label}: duplicate seed {seed} for parent {pid}"
                    )
                seen_seeds.add(seed)

    # Check for unexpected parents
    for pid in by_parent:
        if pid not in expected_parents:
            issues.append(f"{phase_label}: unexpected parent {pid} ({len(by_parent[pid])} entries)")

    return issues, by_parent


def check_trace_exists(phase_dir, summary):
    """Verify trace file exists and has matching row count."""
    tp = trace_path(phase_dir, summary)
    if not os.path.exists(tp):
        return False, 'TRACE_MISSING', 0
    # Count rows (minus header)
    try:
        with open(tp) as f:
            reader = csv.DictReader(f)
            n = sum(1 for _ in reader)
    except Exception as e:
        return False, f'TRACE_READ_ERROR:{e}', 0
    return True, 'ok', n


def validate_attack_telemetry(summary):
    """Check for required attack telemetry fields. Returns list of missing fields."""
    required = [
        'eps_raw_pixels', 'pgd_steps', 'perturb_frame_count',
        'decode_path', 'preprocess_path', 'attention_mask',
    ]
    missing = [f for f in required if f not in summary]
    return missing


def compute_event_C2O_rate(c2o_count, attacked_close_count):
    """Compute per-episode event C2O rate. Returns (rate, flag)."""
    if attacked_close_count == 0:
        return None, 'NO_ATTACK_OPPORTUNITY'
    return round(c2o_count / attacked_close_count, 4), None


def main():
    parser = argparse.ArgumentParser(description='Build V6 report tables from raw summaries')
    parser.add_argument('--p1-dir', required=True)
    parser.add_argument('--p2-dir', required=True)
    parser.add_argument('--p3-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Expected parents ──
    ALL_PARENTS = [
        'cream_cheese_s2', 'butter_s2', 'alphabet_soup_s10',
        'bbq_sauce_s0', 'chocolate_pudding_s2', 'ketchup_s11',
    ]

    # ── Load all summaries ──
    p1_raw = load_summaries(args.p1_dir)
    p2_raw = load_summaries(args.p2_dir)
    p3_raw = load_summaries(args.p3_dir)

    all_issues = []

    # ── Verify expected keys ──
    p1_issues, p1_by_parent = verify_expected_keys(
        p1_raw, 'Phase1', ALL_PARENTS, 2, is_clean_phase=True
    )
    p2_issues, p2_by_parent = verify_expected_keys(
        p2_raw, 'Phase2', ALL_PARENTS, 3, is_clean_phase=False
    )
    # Phase 3 only has 3 VIS parents
    VIS_PARENTS = ['butter_s2', 'bbq_sauce_s0', 'chocolate_pudding_s2']
    p3_issues, p3_by_parent = verify_expected_keys(
        p3_raw, 'Phase3', VIS_PARENTS, 3, is_clean_phase=False
    )

    all_issues.extend(p1_issues)
    all_issues.extend(p2_issues)
    all_issues.extend(p3_issues)

    # ── Verify traces exist ──
    for phase_dir, summaries, label in [
        (args.p1_dir, p1_raw, 'P1'),
        (args.p2_dir, p2_raw, 'P2'),
        (args.p3_dir, p3_raw, 'P3'),
    ]:
        for s in summaries:
            ok, status, n_rows = check_trace_exists(phase_dir, s)
            if not ok:
                all_issues.append(f"{label}: {s.get('task')}_s{s.get('state_id')} seed={s.get('attack_seed')}: {status}")
            elif n_rows != s.get('n_steps', -1):
                all_issues.append(
                    f"{label}: {s.get('task')}_s{s.get('state_id')} seed={s.get('attack_seed')}: "
                    f"trace rows={n_rows} vs n_steps={s.get('n_steps')}"
                )

    # ── Check attack telemetry fields ──
    for label, summaries in [('P2', p2_raw), ('P3', p3_raw)]:
        for s in summaries:
            missing = validate_attack_telemetry(s)
            if missing:
                all_issues.append(
                    f"{label}: {s.get('task')}_s{s.get('state_id')} seed={s.get('attack_seed')}: "
                    f"EVIDENCE_FIELD_MISSING: {missing}"
                )

    # Hard fail on issues
    if all_issues:
        print("AUDIT ISSUES FOUND:")
        for issue in all_issues:
            print(f"  {issue}")
        print(f"\n{len(all_issues)} issues total. Fix before proceeding.")
        # Write issues file
        with open(os.path.join(args.output_dir, 's20d_v6_audit_issues.txt'), 'w') as f:
            f.write('\n'.join(all_issues))
        raise SystemExit(1 if any('EVIDENCE_FIELD_MISSING' in i or 'TRACE_MISSING' in i for i in all_issues) else 0)

    print(f"Audit PASS: {len(p1_raw)} P1 + {len(p2_raw)} P2 + {len(p3_raw)} P3 = {len(p1_raw)+len(p2_raw)+len(p3_raw)} summaries, all traces present, 0 evidence fields missing")

    # ═══════════════════════════════════════════════════════════════
    # TABLE 1: Phase 1 Clean Trigger
    # ═══════════════════════════════════════════════════════════════
    rows1 = []
    for s in sorted(p1_raw, key=lambda x: (x['task'], x['state_id'], x.get('job_id', ''))):
        rows1.append({
            'parent_id': parent_id(s['task'], s['state_id']),
            'task': s['task'],
            'state_id': s['state_id'],
            'job_id': s.get('job_id', ''),
            'repeat_index': s.get('job_id', '')[-2:],
            'trigger_found': s['trigger_found'],
            'trigger_step': s['trigger_step'],
            'trigger_clean_gripper_raw': s.get('trigger_clean_gripper_raw', ''),
            'n_steps': s['n_steps'],
            'success_primary': s['success_primary'],
            'timeout': s.get('timeout', False),
            'infra_status': s['infra_status'],
        })

    t1_path = os.path.join(args.output_dir, 's20d_v6_online_clean_trigger_complete.csv')
    with open(t1_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows1[0].keys()))
        w.writeheader()
        w.writerows(rows1)
    print(f'T1 clean trigger: {len(rows1)} rows -> {t1_path}')

    # ═══════════════════════════════════════════════════════════════
    # TABLE 2: Phase 2 RAND Veto
    # ═══════════════════════════════════════════════════════════════
    rows2 = []
    p2_data = defaultdict(lambda: {
        'trigger': 0, 'c2o_ep': 0, 'c2o_cnt': 0,
        'att': 0, 'succ': 0, 'n': 0,
        'episode_rates': [], 'no_attack_eps': 0,
    })

    for s in sorted(p2_raw, key=lambda x: (x['task'], x['state_id'], x['attack_seed'])):
        pid = parent_id(s['task'], s['state_id'])
        c2o_ep = 1 if s['C2O_count'] > 0 else 0
        attacked = s.get('attacked_close_count', 0)
        rate, rate_flag = compute_event_C2O_rate(s['C2O_count'], attacked)

        rows2.append({
            'parent_id': pid,
            'task': s['task'],
            'state_id': s['state_id'],
            'seed': s['attack_seed'],
            'trigger_found': s['trigger_found'],
            'trigger_step': s['trigger_step'],
            'attacked_close_count': attacked,
            'C2O_count': s['C2O_count'],
            'C2O_episode': c2o_ep,
            'event_C2O_rate': rate if rate is not None else 'NA',
            'event_C2O_rate_flag': rate_flag if rate_flag else '',
            'success_primary': s['success_primary'],
            'timeout': s.get('timeout', False),
            'infra_status': s['infra_status'],
            'perturb_frame_count': s.get('perturb_frame_count', ''),
            'eps_raw_pixels': s.get('eps_raw_pixels', ''),
            'pgd_steps': s.get('pgd_steps', ''),
        })

        d = p2_data[pid]
        d['trigger'] += int(s['trigger_found'])
        d['c2o_ep'] += c2o_ep
        d['c2o_cnt'] += s['C2O_count']
        d['att'] += attacked
        d['succ'] += int(s['success_primary'])
        d['n'] += 1
        if rate is not None:
            d['episode_rates'].append(rate)
        else:
            d['no_attack_eps'] += 1

    t2_path = os.path.join(args.output_dir, 's20d_v6_online_rand_veto_complete.csv')
    with open(t2_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows2[0].keys()))
        w.writeheader()
        w.writerows(rows2)
    print(f'T2 RAND veto: {len(rows2)} rows -> {t2_path}')

    # ═══════════════════════════════════════════════════════════════
    # TABLE 3: RAND Parent Classification
    # ═══════════════════════════════════════════════════════════════
    # Compute clean success per parent (Phase 1)
    clean_succ = defaultdict(int)
    clean_n = defaultdict(int)
    for s in p1_raw:
        pid = parent_id(s['task'], s['state_id'])
        if s['success_primary']:
            clean_succ[pid] += 1
        clean_n[pid] += 1

    rows3 = []
    for pid in sorted(p2_data.keys()):
        d = p2_data[pid]
        task = p2_by_parent[pid][0]['task']  # Read task from summary

        # Classification rules
        classification_rule_version = 'v6_audited_v1'
        if d['c2o_ep'] >= 2:
            cls = 'ONLINE_RANDOM_SENSITIVE_ABSTAIN'
            reason = f"RAND C2O episodes {d['c2o_ep']}/{d['n']} >= 2/3 threshold"
        elif d['trigger'] < 2:
            cls = 'ONLINE_TRIGGER_UNSTABLE'
            reason = f"trigger {d['trigger']}/{d['n']} < 2/3 threshold"
        elif d['succ'] >= 2:
            cls = 'ONLINE_RAND_STRICT'
            reason = (f"C2O {d['c2o_ep']}/{d['n']} <= 1/3, "
                      f"trigger {d['trigger']}/{d['n']}, "
                      f"success {d['succ']}/{d['n']} >= 2/3")
        else:
            cls = 'ONLINE_RAND_USABLE'
            reason = (f"C2O {d['c2o_ep']}/{d['n']} <= 1/3, "
                      f"trigger {d['trigger']}/{d['n']}, "
                      f"success {d['succ']}/{d['n']} < 2/3 (degraded)")

        # Compute rates
        pooled_rate = round(d['c2o_cnt'] / max(d['att'], 1), 4) if d['att'] > 0 else None
        median_rate = (
            round(statistics.median(d['episode_rates']), 4)
            if d['episode_rates'] else None
        )

        rows3.append({
            'parent_id': pid,
            'task': task,
            'trigger': f"{d['trigger']}/{d['n']}",
            'C2O_episodes': f"{d['c2o_ep']}/{d['n']}",
            'total_C2O_count': d['c2o_cnt'],
            'total_attacked_close': d['att'],
            'pooled_event_C2O_rate': pooled_rate if pooled_rate is not None else 'NA',
            'median_event_C2O_rate': median_rate if median_rate is not None else 'NA',
            'no_attack_opportunity_eps': d['no_attack_eps'],
            'success': f"{d['succ']}/{d['n']}",
            'clean_success': f"{clean_succ.get(pid, 0)}/{clean_n.get(pid, 0)}",
            'classification': cls,
            'classification_rule_version': classification_rule_version,
            'classification_reason': reason,
        })

    t3_path = os.path.join(args.output_dir, 's20d_v6_online_rand_parent_classification.csv')
    with open(t3_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows3[0].keys()))
        w.writeheader()
        w.writerows(rows3)

    # Summary stats
    strict_count = sum(1 for r in rows3 if r['classification'] == 'ONLINE_RAND_STRICT')
    usable_count = sum(1 for r in rows3 if r['classification'] == 'ONLINE_RAND_USABLE')
    print(f'T3 RAND classification: {len(rows3)} rows ({strict_count} STRICT, {usable_count} USABLE) -> {t3_path}')

    # ═══════════════════════════════════════════════════════════════
    # TABLE 4: Phase 3 VIS Pilot
    # ═══════════════════════════════════════════════════════════════
    rows4 = []
    p3_data = defaultdict(lambda: {
        'c2o_ep': 0, 'n': 0, 'episode_rates': [],
        'c2o_cnt': 0, 'att': 0, 'succ': 0, 'no_attack_eps': 0,
    })

    for s in sorted(p3_raw, key=lambda x: (x['task'], x['state_id'], x['attack_seed'])):
        pid = parent_id(s['task'], s['state_id'])
        c2o_ep = 1 if s['C2O_count'] > 0 else 0
        attacked = s.get('attacked_close_count', 0)
        rate, rate_flag = compute_event_C2O_rate(s['C2O_count'], attacked)

        rows4.append({
            'parent_id': pid,
            'task': s['task'],
            'state_id': s['state_id'],
            'seed': s['attack_seed'],
            'trigger_found': s['trigger_found'],
            'trigger_step': s['trigger_step'],
            'attacked_close_count': attacked,
            'C2O_count': s['C2O_count'],
            'C2O_episode': c2o_ep,
            'event_C2O_rate': rate if rate is not None else 'NA',
            'event_C2O_rate_flag': rate_flag if rate_flag else '',
            'success_primary': s['success_primary'],
            'timeout': s.get('timeout', False),
            'infra_status': s['infra_status'],
            'perturb_frame_count': s.get('perturb_frame_count', ''),
            'eps_raw_pixels': s.get('eps_raw_pixels', ''),
            'pgd_steps': s.get('pgd_steps', ''),
        })

        d = p3_data[pid]
        d['c2o_ep'] += c2o_ep
        d['n'] += 1
        d['c2o_cnt'] += s['C2O_count']
        d['att'] += attacked
        d['succ'] += int(s['success_primary'])
        if rate is not None:
            d['episode_rates'].append(rate)
        else:
            d['no_attack_eps'] += 1

    t4_path = os.path.join(args.output_dir, 's20d_v6_online_vis_pilot_complete.csv')
    with open(t4_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows4[0].keys()))
        w.writeheader()
        w.writerows(rows4)
    print(f'T4 VIS pilot: {len(rows4)} rows -> {t4_path}')

    # ═══════════════════════════════════════════════════════════════
    # TABLE 5: VIS Parent Comparison
    # ═══════════════════════════════════════════════════════════════
    rows5 = []
    for pid in VIS_PARENTS:
        vd = p3_data.get(pid, {'c2o_ep': 0, 'n': 0, 'episode_rates': [], 'c2o_cnt': 0, 'att': 0, 'succ': 0, 'no_attack_eps': 0})
        rd = p2_data.get(pid, {'c2o_ep': 0, 'n': 0, 'episode_rates': [], 'c2o_cnt': 0, 'att': 0, 'succ': 0, 'no_attack_eps': 0})

        # VIS classification
        if vd['c2o_ep'] >= 2 and rd['c2o_ep'] <= 1:
            cls = 'ONLINE_CMD_CANDIDATE'
        elif vd['c2o_ep'] == 1:
            cls = 'ONLINE_VIS_PARTIAL'
        else:
            cls = 'ONLINE_VIS_NO_EFFECT'

        vis_pooled = round(vd['c2o_cnt'] / max(vd['att'], 1), 4) if vd['att'] > 0 else None
        rand_pooled = round(rd['c2o_cnt'] / max(rd['att'], 1), 4) if rd['att'] > 0 else None

        rows5.append({
            'parent_id': pid,
            'task': pid.split('_s')[0].replace('_', ' '),  # Only used for display
            'VIS_C2O_episodes': f"{vd['c2o_ep']}/{vd['n']}",
            'VIS_total_C2O': vd['c2o_cnt'],
            'VIS_attacked_close': vd['att'],
            'VIS_pooled_event_C2O_rate': vis_pooled if vis_pooled is not None else 'NA',
            'VIS_no_attack_eps': vd['no_attack_eps'],
            'VIS_success': f"{vd['succ']}/{vd['n']}",
            'RAND_C2O_episodes': f"{rd['c2o_ep']}/{rd['n']}",
            'RAND_total_C2O': rd['c2o_cnt'],
            'RAND_attacked_close': rd['att'],
            'RAND_pooled_event_C2O_rate': rand_pooled if rand_pooled is not None else 'NA',
            'RAND_no_attack_eps': rd['no_attack_eps'],
            'physical_bridge': 'NOT_ESTABLISHED',
            'task_effect': 'NOT_ESTABLISHED',
            'classification': cls,
        })

    t5_path = os.path.join(args.output_dir, 's20d_v6_online_vis_parent_comparison.csv')
    with open(t5_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows5[0].keys()))
        w.writeheader()
        w.writerows(rows5)
    print(f'T5 VIS comparison: {len(rows5)} rows -> {t5_path}')

    # ═══════════════════════════════════════════════════════════════
    # TABLE 6: butter_s2 Command Candidate Evidence
    # ═══════════════════════════════════════════════════════════════
    rows6 = []
    for r in rows1:
        if r['parent_id'] == 'butter_s2':
            rows6.append({
                'condition': 'clean_observer',
                'seed': 'N/A',
                'job_id': r['job_id'],
                'trigger_step': r['trigger_step'],
                'C2O_count': 0,
                'C2O_episode': 0,
                'success': r['success_primary'],
                'infra': r['infra_status'],
            })
    for r in rows2:
        if r['parent_id'] == 'butter_s2':
            rows6.append({
                'condition': 'online_random_linf',
                'seed': r['seed'],
                'job_id': '',
                'trigger_step': r['trigger_step'],
                'C2O_count': r['C2O_count'],
                'C2O_episode': r['C2O_episode'],
                'success': r['success_primary'],
                'infra': r['infra_status'],
            })
    for r in rows4:
        if r['parent_id'] == 'butter_s2':
            rows6.append({
                'condition': 'online_vis_pgd',
                'seed': r['seed'],
                'job_id': '',
                'trigger_step': r['trigger_step'],
                'C2O_count': r['C2O_count'],
                'C2O_episode': r['C2O_episode'],
                'success': r['success_primary'],
                'infra': r['infra_status'],
            })

    t6_path = os.path.join(args.output_dir, 's20d_v6_butter_s2_command_candidate_evidence.csv')
    with open(t6_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows6[0].keys()))
        w.writeheader()
        w.writerows(rows6)
    print(f'T6 butter evidence: {len(rows6)} rows -> {t6_path}')

    # ═══════════════════════════════════════════════════════════════
    # TABLE 7: Registry Update
    # ═══════════════════════════════════════════════════════════════
    reg_path = os.path.join(args.output_dir, 'layer3_parent_registry.csv')
    existing = []
    if os.path.exists(reg_path):
        with open(reg_path) as f:
            existing = list(csv.DictReader(f))

    v6_new = [
        {
            'parent_id': 'butter_s2', 'stage': 'V6_online_L3',
            'task': 'butter',
            'status': 'ONLINE_CMD_CANDIDATE_PENDING_PROVENANCE_REPAIR',
            'rand_stability': 'ONLINE_RAND_STRICT',
            'vis_outcome': 'VIS_C2O_3/3',
            'rand_outcome': 'RAND_C2O_0/3',
            'layer3_confirmed': 'False',
            'layer3_class': 'ONLINE_CMD_CANDIDATE',
            'eligible_for_vis': 'True',
            'physical_bridge': 'NOT_ESTABLISHED',
            'task_effect': 'NOT_ESTABLISHED',
            'notes': 'V6 online trigger. VIS 3/3 C2O, RAND 0/3. PENDING_PROVENANCE_REPAIR.'
        },
        {
            'parent_id': 'bbq_sauce_s0', 'stage': 'V6_online_L3',
            'task': 'bbq_sauce',
            'status': 'ONLINE_VIS_PARTIAL',
            'rand_stability': 'ONLINE_RAND_STRICT',
            'vis_outcome': 'VIS_C2O_1/3',
            'rand_outcome': 'RAND_C2O_0/3',
            'layer3_confirmed': 'False',
            'layer3_class': 'ONLINE_VIS_PARTIAL',
            'eligible_for_vis': 'True',
            'physical_bridge': 'NOT_ESTABLISHED',
            'task_effect': 'NOT_ESTABLISHED',
            'notes': 'V6 online trigger. VIS 1/3 C2O.'
        },
        {
            'parent_id': 'chocolate_pudding_s2', 'stage': 'V6_online_L3',
            'task': 'chocolate_pudding',
            'status': 'ONLINE_VIS_NO_EFFECT',
            'rand_stability': 'ONLINE_RAND_STRICT',
            'vis_outcome': 'VIS_C2O_0/3',
            'rand_outcome': 'RAND_C2O_0/3',
            'layer3_confirmed': 'False',
            'layer3_class': 'ONLINE_VIS_NO_EFFECT',
            'eligible_for_vis': 'True',
            'physical_bridge': 'NOT_ESTABLISHED',
            'task_effect': 'NOT_ESTABLISHED',
            'notes': 'V6 online trigger. VIS 0/3 C2O.'
        },
    ]

    # Merge: update existing entries or append new ones
    existing_by_key = {(e['parent_id'], e.get('stage', '')): i for i, e in enumerate(existing)}
    for new_e in v6_new:
        key = (new_e['parent_id'], new_e['stage'])
        if key in existing_by_key:
            existing[existing_by_key[key]] = new_e
        else:
            existing.append(new_e)

    fnames = list(v6_new[0].keys())
    with open(reg_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fnames, extrasaction='ignore')
        w.writeheader()
        w.writerows(existing)
    print(f'T7 Registry: {len(existing)} entries -> {reg_path}')

    # ═══════════════════════════════════════════════════════════════
    # Audit Manifest Summary
    # ═══════════════════════════════════════════════════════════════
    manifest_path = os.path.join(args.output_dir, 's20d_v6_audit_manifest.csv')
    manifest_rows = []
    for label, summaries in [('P1_clean', p1_raw), ('P2_rand', p2_raw), ('P3_vis', p3_raw)]:
        for s in summaries:
            pid = parent_id(s['task'], s['state_id'])
            tp = trace_path(
                args.p1_dir if label == 'P1_clean' else (
                    args.p2_dir if label == 'P2_rand' else args.p3_dir
                ), s
            )
            manifest_rows.append({
                'phase': label,
                'parent_id': pid,
                'task': s['task'],
                'state_id': s['state_id'],
                'seed': s.get('attack_seed', ''),
                'job_id': s.get('job_id', ''),
                'condition': s.get('condition', ''),
                'trigger_found': s['trigger_found'],
                'C2O_count': s.get('C2O_count', ''),
                'success_primary': s['success_primary'],
                'infra_status': s['infra_status'],
                'summary_path': s['_summary_path'],
                'summary_sha256': s['_summary_sha256'],
                'trace_path': tp,
                'trace_sha256': _sha256_file(tp),
                'trace_exists': os.path.exists(tp),
                'included': True,
                'exclusion_reason': '',
            })

    m_path = os.path.join(args.output_dir, 's20d_v6_audit_manifest.csv')
    with open(m_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()))
        w.writeheader()
        w.writerows(manifest_rows)
    print(f'Manifest: {len(manifest_rows)} rows -> {m_path}')

    # Final summary
    print(f'\nAll tables in {args.output_dir}')
    print(f'Summary: {len(p1_raw)}+{len(p2_raw)}+{len(p3_raw)}={len(p1_raw)+len(p2_raw)+len(p3_raw)} episodes')
    print(f'RAND: {sum(1 for r in rows3 if r["classification"]=="ONLINE_RAND_STRICT")} STRICT, '
          f'{sum(1 for r in rows3 if r["classification"]=="ONLINE_RAND_USABLE")} USABLE')
    vis_cmd = sum(1 for r in rows5 if r['classification'] == 'ONLINE_CMD_CANDIDATE')
    print(f'VIS: {vis_cmd} CMD_CANDIDATE, '
          f'{sum(1 for r in rows5 if r["classification"]=="ONLINE_VIS_PARTIAL")} PARTIAL, '
          f'{sum(1 for r in rows5 if r["classification"]=="ONLINE_VIS_NO_EFFECT")} NO_EFFECT')


if __name__ == '__main__':
    main()
