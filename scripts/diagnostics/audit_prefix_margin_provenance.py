# -*- coding: utf-8 -*-
"""audit_prefix_margin_provenance.py — regenerate provenance from trace CSVs.

Usage::

    python scripts/diagnostics/audit_prefix_margin_provenance.py \
        --run-dirs /data/liuyu/outputs/milestone_7_vis_controlled_rollout_micro_20260601/runs \
        --output-csv tables/vis_prefix_margin_provenance.csv \
        --group-csv tables/vis_prefix_margin_group_summary.csv \
        --report reports/VIS_PREFIX_MARGIN_REPAIR_AUDIT.md
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

# Allow running from repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
from gripper_attack.gripper_semantics import (
    raw_gripper_is_open,
    CANONICAL_OPEN_SEMANTICS_VERSION,
)


def parse_args():
    ap = argparse.ArgumentParser(description='Regenerate VIS prefix-margin provenance from trace CSVs')
    ap.add_argument('--run-dirs', nargs='+', required=True,
                     help='One or more directories containing *_trace.csv files')
    ap.add_argument('--manual-audit-csv', default=None,
                     help='Optional manual audit CSV with failure_phase_manual annotations')
    ap.add_argument('--output-csv', default='tables/vis_prefix_margin_provenance.csv',
                     help='Output provenance CSV path')
    ap.add_argument('--group-csv', default='tables/vis_prefix_margin_group_summary.csv',
                     help='Output group summary CSV path')
    ap.add_argument('--report', default='reports/VIS_PREFIX_MARGIN_REPAIR_AUDIT.md',
                     help='Output audit report markdown path')
    return ap.parse_args()


# ── Provenance schema (canonical field names) ──
PROVENANCE_FIELDS = [
    'run_id',
    'task',
    'state_id',
    'condition',
    'objective',
    'eps_raw_pixels',
    'window_start',
    'window_end',
    'seed',
    'code_status',
    'attack_adapter_commit',
    'runner_commit',
    'semantics_version',
    'prefix_loss_version',
    'teacher_forced_fallback_allowed',
    'trace_generated_by_repaired_runner',
    'trace_path',
    'trace_schema_version',
    'generated_open_count_canonical',
    'generated_open_total',
    'generated_open_ratio',
    'generated_open_predicate_version',
    'qpos_pre_start',
    'qpos_pre_end',
    'qpos_post_start',
    'qpos_post_end',
    'qpos_delta_pre',
    'qpos_delta_post',
    'qpos_abs_after_min',
    'qpos_abs_after_max',
    'armL2_mean',
    'armL2_max',
    'token_flip_count',
    'official_done',
    'timeout',
    'success_semantics',
    'failure_phase_manual',
    'failure_phase_auto',
    'denominator_status',
    'matched_random_group_id',
    'validity',
    'invalid_reason',
]


def parse_bool(val):
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        return val.strip().lower() in ('true', '1', 'yes')
    return bool(val)


def _find_trace_files(run_dirs):
    """Recursively find all *_trace.csv files under run directories."""
    files = []
    for d in run_dirs:
        p = Path(d)
        if not p.exists():
            print(f"WARNING: run-dir does not exist: {d}")
            continue
        for f in p.rglob('*_trace.csv'):
            files.append(str(f))
    return sorted(files)


def _parse_run_id_from_filename(trace_path):
    """Extract run_id from trace filename, e.g. vis_ketchup_s0_vis_pgd_full_d18_w10_27_seed0_042831_trace.csv."""
    stem = Path(trace_path).stem
    if stem.endswith('_trace'):
        stem = stem[:-6]  # remove _trace suffix
    return stem


def _infer_params_from_run_id(run_id, trace_rows):
    """Infer experiment parameters from trace rows (primary) or run_id (fallback).

    Post-repair traces write objective/eps/window/state_id directly into each row.
    Pre-repair traces don't — fall back to filename heuristics.
    """
    info = {'run_id': run_id}

    # ── Primary: read from trace rows (post-repair or v2+ schema) ──
    if trace_rows:
        r0 = trace_rows[0]
        for key in ('task', 'condition', 'seed', 'state_id', 'objective',
                     'eps_raw_pixels', 'window_start', 'window_end'):
            if key in r0:
                try:
                    val = r0[key]
                    info[key] = int(val) if key in ('seed', 'state_id',
                        'eps_raw_pixels', 'window_start', 'window_end') else str(val)
                except (ValueError, TypeError):
                    info[key] = str(val)

    # ── Fallback: filename heuristics (pre-repair traces) ──
    parts = run_id.split('_')

    if 'task' not in info or not info.get('task'):
        if len(parts) >= 2 and parts[0] == 'vis':
            info['task'] = '_'.join(parts[1:3]) if parts[1] in ('cream', 'salad', 'ketchup', 'tomato') else parts[1]
            if info.get('task', '').startswith('cream_cheese'):
                info['task'] = 'cream_cheese'
            elif info.get('task', '').startswith('salad_dressing'):
                info['task'] = 'salad_dressing'
            elif info.get('task', '').startswith('tomato_sauce'):
                info['task'] = 'tomato_sauce'
            elif info.get('task', '').startswith('ketchup'):
                info['task'] = 'ketchup'

    if 'condition' not in info or not info.get('condition'):
        info['condition'] = 'unknown'
        if 'vis_pgd' in run_id:
            info['condition'] = 'vis_pgd'
        elif 'random_linf' in run_id or '_random_' in run_id:
            info['condition'] = 'random_linf'
        elif 'clean' in run_id:
            info['condition'] = 'clean'

    if 'state_id' not in info:
        info['state_id'] = 0
        for part in parts:
            if part.startswith('s') and len(part) >= 2 and part[1:].isdigit():
                info['state_id'] = int(part[1:])
                break

    if 'seed' not in info:
        info['seed'] = 0
        for part in parts:
            if part.startswith('seed'):
                try:
                    info['seed'] = int(part[4:])
                except ValueError:
                    pass
                break

    if 'window_start' not in info or 'window_end' not in info:
        # Regex fallback: handle pre-repair run_ids like
        #   vis_ketchup_s0_vis_pgd_full_d18_w10_27_seed0_042831
        # where split('_') breaks w10_27 into w10, 27.
        import re
        _wm = re.search(r'_w(\d+)_(\d+)', run_id)
        if _wm:
            info['window_start'] = int(_wm.group(1))
            info['window_end'] = int(_wm.group(2))

    if 'objective' not in info:
        info['objective'] = 'unknown'

    if 'eps_raw_pixels' not in info:
        info['eps_raw_pixels'] = ''

    return info


def _classify_failure_phase_auto(trace_rows, info):
    """Auto-classify failure phase from trace data.

    Returns one of:
    - early_grasp_disruption: OPEN in the first 1/3 of window
    - pre_release_drop: OPEN in the last 1/3 of window
    - mid_window_opening: OPEN in middle 1/3
    - no_open: no OPEN detected
    - task_success: official_done=True
    """
    done = any(parse_bool(r.get('done', 'False')) for r in trace_rows)
    if done:
        return 'task_success'

    window_rows = [r for r in trace_rows if parse_bool(r.get('in_window', 'False'))]
    if not window_rows:
        return 'no_open'

    n = len(window_rows)
    first_third = window_rows[:max(1, n // 3)]
    last_third = window_rows[-(max(1, n // 3)):]
    mid_third = window_rows[n // 3: 2 * n // 3]

    open_first = sum(1 for r in first_third if raw_gripper_is_open(float(r.get('adv_grip', 0.996))))
    open_last = sum(1 for r in last_third if raw_gripper_is_open(float(r.get('adv_grip', 0.996))))
    open_mid = sum(1 for r in mid_third if raw_gripper_is_open(float(r.get('adv_grip', 0.996))))

    if open_first + open_mid + open_last == 0:
        return 'no_open'

    if open_first >= max(open_mid, open_last) and open_first > 0:
        return 'early_grasp_disruption'
    if open_last >= max(open_first, open_mid) and open_last > 0:
        return 'pre_release_drop'
    return 'mid_window_opening'


def _compute_provenance(trace_rows, trace_path, info):
    """Compute a single provenance row from trace CSV rows."""
    row = dict.fromkeys(PROVENANCE_FIELDS)
    row.update(info)
    row['trace_path'] = trace_path

    # ── Provenance separation fields ──
    # Post-repair traces contain trace_generated_by_repaired_runner=True in every row.
    _is_repaired = any(
        parse_bool(r.get('trace_generated_by_repaired_runner', 'False'))
        for r in trace_rows
    )
    row['code_status'] = 'post_repair' if _is_repaired else 'pre_repair'
    row['trace_generated_by_repaired_runner'] = _is_repaired
    # Read attacker semantic version from trace if present
    _semver = None
    for r in trace_rows:
        if r.get('semantics_version'):
            _semver = str(r['semantics_version'])
            break
    row['semantics_version'] = _semver or CANONICAL_OPEN_SEMANTICS_VERSION
    row['prefix_loss_version'] = 'repaired_direct_row_v2' if _is_repaired else 'pre_repair_label_based'
    row['teacher_forced_fallback_allowed'] = not _is_repaired
    row['attack_adapter_commit'] = 'unknown'
    row['runner_commit'] = 'unknown'

    # Schema version detection
    has_qpos_post = any('qpos_post_step' in r for r in trace_rows)
    has_env_gripper = any('env_gripper' in r for r in trace_rows)
    if has_qpos_post and has_env_gripper:
        row['trace_schema_version'] = 'v3_post_fix'
    elif has_env_gripper:
        row['trace_schema_version'] = 'v2_env_gripper'
    else:
        row['trace_schema_version'] = 'v1_basic'
        row['validity'] = 'schema_incomplete'
        row['invalid_reason'] = 'missing qpos_post_step and env_gripper columns'

    # Window analysis
    window_rows = [r for r in trace_rows if parse_bool(r.get('in_window', 'False'))]
    # For PGD conditions, attacked_rows = pgd_applied=True within window.
    # For random_linf / clean, use all window_rows as effective rows for qpos tracking.
    is_pgd_condition = row.get('condition') in ('vis_pgd',)
    if is_pgd_condition:
        effective_rows = [r for r in window_rows if parse_bool(r.get('pgd_applied', 'False'))]
    else:
        effective_rows = list(window_rows)  # random/clean: all in-window steps

    # Canonical OPEN count (using repaired semantics: raw_gripper < 0.5 = OPEN)
    row['generated_open_count_canonical'] = sum(
        1 for r in window_rows if raw_gripper_is_open(float(r.get('adv_grip', 0.996)))
    )
    row['generated_open_total'] = len(window_rows)
    row['generated_open_ratio'] = round(
        row['generated_open_count_canonical'] / max(row['generated_open_total'], 1), 4)
    row['generated_open_predicate_version'] = CANONICAL_OPEN_SEMANTICS_VERSION

    # qpos metrics (from effective rows)
    if effective_rows:
        qpos_pre_vals = [float(r['qpos_pre_step']) for r in effective_rows if 'qpos_pre_step' in r]
        qpos_post_vals = [float(r['qpos_post_step']) for r in effective_rows if 'qpos_post_step' in r]

        if qpos_pre_vals:
            row['qpos_pre_start'] = round(qpos_pre_vals[0], 6)
            row['qpos_pre_end'] = round(qpos_pre_vals[-1], 6)
            row['qpos_delta_pre'] = round(
                max(abs(v - qpos_pre_vals[0]) for v in qpos_pre_vals) if len(qpos_pre_vals) > 1 else 0.0, 6)

        if qpos_post_vals:
            row['qpos_post_start'] = round(qpos_post_vals[0], 6)
            row['qpos_post_end'] = round(qpos_post_vals[-1], 6)
            row['qpos_delta_post'] = round(
                max(abs(v - qpos_post_vals[0]) for v in qpos_post_vals) if len(qpos_post_vals) > 1 else 0.0, 6)
            row['qpos_abs_after_min'] = round(min(qpos_post_vals), 6)
            row['qpos_abs_after_max'] = round(max(qpos_post_vals), 6)
        else:
            row['validity'] = 'schema_incomplete'
            row['invalid_reason'] = (row.get('invalid_reason', '') + '; missing qpos_post_step').strip('; ')
    else:
        row['validity'] = 'schema_incomplete'
        row['invalid_reason'] = (row.get('invalid_reason', '') + '; no effective rows').strip('; ')

    # Arm L2
    arm_l2_vals = [float(r['arm_l2']) for r in window_rows if 'arm_l2' in r]
    if arm_l2_vals:
        row['armL2_mean'] = round(float(np.mean(arm_l2_vals)), 6)
        row['armL2_max'] = round(float(np.max(arm_l2_vals)), 6)

    # Token flips
    row['token_flip_count'] = sum(1 for r in window_rows if parse_bool(r.get('token_flip', 'False')))

    # Done / timeout
    row['official_done'] = any(parse_bool(r.get('done', 'False')) for r in trace_rows)
    row['timeout'] = not row['official_done'] and len(trace_rows) >= 299

    # Success semantics
    if row['official_done'] and row['condition'] in ('random_linf', 'clean'):
        row['success_semantics'] = 'clean_success'
    elif row['official_done']:
        row['success_semantics'] = 'task_survived'
    elif row['timeout']:
        row['success_semantics'] = 'task_failed_timeout'
    else:
        row['success_semantics'] = 'task_failed_early'

    # Failure phase auto-classification
    row['failure_phase_auto'] = _classify_failure_phase_auto(trace_rows, info)

    return row


def _group_key(row):
    """Group by experiment config, NOT by condition.

    Prefix and random conditions must appear in the same group so
    denominator_status and claim_eligible can be computed correctly.
    """
    return (
        row.get('task', 'unknown'),
        row.get('state_id', 0),
        row.get('objective', 'unknown'),
        row.get('eps_raw_pixels', ''),
        row.get('window_start', ''),
        row.get('window_end', ''),
        row.get('code_status', 'pre_repair'),
    )


GROUP_SUMMARY_FIELDS = [
    'task', 'state_id', 'objective',
    'eps_raw_pixels', 'window_start', 'window_end',
    'code_status',
    'n_runs', 'n_valid', 'n_invalid',
    'valid_unique_seed_count',
    'prefix_fail', 'prefix_success',
    'random_fail', 'random_success',
    'canonical_open_min', 'canonical_open_max',
    'open_count_prefix_mean', 'open_count_random_mean',
    'qpos_delta_post_min', 'qpos_delta_post_max',
    'qpos_delta_post_prefix_mean',
    'armL2_max',
    'all_armL2_zero',
    'all_random_open_zero',
    'denominator_status',
    'denominator_detail',
    'claim_eligible',
    'claim_caveats',
    'failure_phase_mode',
    'claim_readiness',
]


def _compute_group_summary(rows):
    """Aggregate provenance rows into group summaries."""
    groups = defaultdict(list)
    for r in rows:
        groups[_group_key(r)].append(r)

    summaries = []
    for gk, group_rows in groups.items():
        s = dict(zip(GROUP_SUMMARY_FIELDS, gk + ('',) * (len(GROUP_SUMMARY_FIELDS) - len(gk))))
        s['n_runs'] = len(group_rows)
        s['n_valid'] = sum(1 for r in group_rows if r.get('validity') is None)
        s['n_invalid'] = s['n_runs'] - s['n_valid']

        valid = [r for r in group_rows if r.get('validity') is None]
        # Also exclude attack_infrastructure_failure runs
        valid = [r for r in valid if r.get('validity') != 'attack_infrastructure_failure']

        prefix = [r for r in valid if r.get('condition') == 'vis_pgd']
        randoms = [r for r in valid if r.get('condition') == 'random_linf']
        cleans = [r for r in valid if r.get('condition') == 'clean']

        # Unique seed count
        seeds = set()
        for r in valid:
            s_val = r.get('seed')
            if s_val is not None:
                seeds.add(s_val)
        s['valid_unique_seed_count'] = len(seeds)
        s['code_status'] = group_rows[0].get('code_status', 'unknown') if group_rows else 'unknown'

        s['prefix_fail'] = sum(1 for r in prefix if not r.get('official_done', True))
        s['prefix_success'] = sum(1 for r in prefix if r.get('official_done', False))
        s['random_fail'] = sum(1 for r in randoms if not r.get('official_done', True))
        s['random_success'] = sum(1 for r in randoms if r.get('official_done', False))

        # Canonical OPEN counts (per-condition)
        if prefix:
            s['canonical_open_min'] = min(r['generated_open_count_canonical'] for r in prefix)
            s['canonical_open_max'] = max(r['generated_open_count_canonical'] for r in prefix)
            s['open_count_prefix_mean'] = round(np.mean([r['generated_open_count_canonical'] for r in prefix]), 2)
        if randoms:
            s['open_count_random_mean'] = round(np.mean([r['generated_open_count_canonical'] for r in randoms]), 2)

        # qpos delta (per-condition)
        if prefix:
            qpos_pre_p = [r['qpos_delta_pre'] or 0 for r in prefix]
            qpos_post_p = [r['qpos_delta_post'] or 0 for r in prefix]
            s['qpos_delta_post_min'] = round(min(qpos_post_p), 6) if qpos_post_p else None
            s['qpos_delta_post_max'] = round(max(qpos_post_p), 6) if qpos_post_p else None
            s['qpos_delta_post_prefix_mean'] = round(np.mean(qpos_post_p), 6) if qpos_post_p else None

        # Arm L2
        if valid:
            s['armL2_max'] = round(np.max([r['armL2_max'] or 0 for r in valid]), 6)
            s['all_armL2_zero'] = all((r.get('armL2_max') or 0) < 1e-6 for r in valid)

        # Random OPEN check
        s['all_random_open_zero'] = all(
            r['generated_open_count_canonical'] == 0
            for r in randoms + cleans
        ) if (randoms or cleans) else None

        # ── Advisor-safe denominator ──
        rc = s['random_success'] + s['random_fail']
        if rc == 0:
            s['denominator_status'] = 'no_random_control'
            s['denominator_detail'] = 'missing same-budget random condition'
        elif s['all_random_open_zero'] == True and s['random_success'] > 0:
            s['denominator_status'] = 'clean'
            s['denominator_detail'] = (
                f'random {s["random_success"]}/{rc} success, '
                f'{s["valid_unique_seed_count"]} unique seeds, '
                f'ALL random OPEN=0')
        elif s['all_random_open_zero'] == True:
            s['denominator_status'] = 'minor_contamination'
            s['denominator_detail'] = (
                f'random {s["random_success"]}/{rc} success (some failure), '
                f'all random OPEN=0')
        elif rc > 0 and s['random_success'] / max(rc, 1) >= 0.8:
            s['denominator_status'] = 'polluted_mild'
            s['denominator_detail'] = 'random mostly clean but some OPEN in random'
        else:
            s['denominator_status'] = 'polluted'
            s['denominator_detail'] = 'random fails or produces OPEN — VIS-specific claim unsupported'

        # ── Claim eligibility ──
        claim_eligible = True
        caveats = []
        if s['prefix_fail'] == 0:
            claim_eligible = False
            caveats.append('no prefix failure')
        if s['denominator_status'] not in ('clean', 'minor_contamination'):
            claim_eligible = False
            caveats.append(f'denominator {s["denominator_status"]}')
        if s.get('code_status') == 'pre_repair':
            claim_eligible = False
            caveats.append('pre-repair code — OPEN semantics and prefix loss not canonical')
        s['claim_eligible'] = claim_eligible
        s['claim_caveats'] = '; '.join(caveats) if caveats else 'none'

        # Failure phase mode
        phases = [r.get('failure_phase_auto', 'unknown') for r in prefix if not r.get('official_done', True)]
        if phases:
            s['failure_phase_mode'] = max(set(phases), key=phases.count)

        # Claim readiness (derived from claim_eligible)
        if s['claim_eligible']:
            s['claim_readiness'] = 'admissible_for_claim'
        elif s['prefix_fail'] > 0:
            s['claim_readiness'] = 'denominator_polluted_or_missing_controls'
        else:
            s['claim_readiness'] = 'no_prefix_failure'

        summaries.append(s)

    return summaries


def generate_report(provenance_rows, group_summaries, args):
    """Generate the repair audit markdown report."""
    lines = []
    lines.append('# VIS prefix_margin — Provenance Repair Audit')
    lines.append(f'\n**Generated**: 2026-06-03')
    lines.append(f'**Canonical semantics**: `{CANONICAL_OPEN_SEMANTICS_VERSION}`')
    lines.append(f'**Source directories**: {", ".join(args.run_dirs)}')
    lines.append(f'**Total trace files processed**: {len(provenance_rows)}')
    lines.append('')

    # Summary statistics
    valid = [r for r in provenance_rows if r.get('validity') is None]
    invalid = [r for r in provenance_rows if r.get('validity') is not None]
    lines.append('## Summary')
    lines.append(f'- Valid traces: {len(valid)}')
    lines.append(f'- Invalid traces: {len(invalid)}')
    if invalid:
        reasons = defaultdict(int)
        for r in invalid:
            reasons[r.get('invalid_reason', 'unknown')] += 1
        lines.append('- Invalid reasons:')
        for reason, count in sorted(reasons.items(), key=lambda x: -x[1]):
            lines.append(f'  - {reason}: {count}')
    lines.append('')

    # Group summary table
    if group_summaries:
        lines.append('## Group Summary')
        lines.append('')
        lines.append('| Task | Window | eps | Code | Prefix F/S | Random F/S | OPEN pre/rand | qposΔ | armL2 | Denominator | Claim Ready? |')
        lines.append('|------|--------|-----|------|-----------|------------|---------------|-------|-------|-------------|-------------|')
        for gs in sorted(group_summaries, key=lambda g: (g['task'], g['window_start'])):
            _pref_open = gs.get('open_count_prefix_mean', '-')
            _rand_open = gs.get('open_count_random_mean', '-')
            _qpos = gs.get('qpos_delta_post_prefix_mean', '-')
            _arm = gs.get('armL2_max', '-')
            lines.append(
                f"| {gs['task']} | {gs['window_start']}-{gs['window_end']} "
                f"| {gs['eps_raw_pixels']} | {gs['code_status'][:8]} "
                f"| {gs['prefix_fail']}/{gs['prefix_success']} "
                f"| {gs['random_fail']}/{gs['random_success']} "
                f"| {_pref_open}/{_rand_open} "
                f"| {_qpos} "
                f"| {_arm} "
                f"| {gs['denominator_status']} "
                f"| {gs['claim_readiness']} |"
            )
        lines.append('')

    # Per-run detail
    lines.append('## Per-Run Details')
    lines.append('')
    lines.append('See [vis_prefix_margin_provenance.csv](../tables/vis_prefix_margin_provenance.csv) for full per-run data.')
    lines.append('')

    # Claim boundary after repair
    lines.append('## Repaired Claim Boundary')
    lines.append('')
    lines.append('### Semantics fix')
    lines.append(f'- OPEN is canonically defined as `raw_gripper < 0.5` (equivalent to `env_gripper > 0` after normalize→invert pipeline).')
    lines.append(f'- All OPEN counts recomputed using `gripper_semantics.raw_gripper_is_open()`.')
    lines.append(f'- Prior reports used inconsistent predicates; this audit is authoritative.')
    lines.append('')
    lines.append('### Prefix-loss fix')
    lines.append('- Gripper loss now computed directly from gripper logit row via `action_token_logit_row_index(action_dim-1, action_dim)` (= -2 for action_dim=7), independent of labels.')
    lines.append('- Arm CE computed from `_active_label_rows()` (arm dims only).')
    lines.append('- Loss = gripper_margin + arm_preserve_weight * mean(arm_CEs).')
    lines.append('')
    lines.append('### Teacher-forced fallback ban')
    lines.append('- Restart selection for gripper objectives MUST use actual autoregressive re-decode.')
    lines.append('- Missing adv_inputs or re-decode failure → RuntimeError (not silent fallback).')
    lines.append('')

    return '\n'.join(lines)


def main():
    args = parse_args()
    trace_files = _find_trace_files(args.run_dirs)
    print(f"Found {len(trace_files)} trace files")

    provenance_rows = []
    for tf in trace_files:
        try:
            with open(tf, 'r', newline='') as f:
                reader = csv.DictReader(f)
                trace_rows = list(reader)
            if not trace_rows:
                print(f"WARNING: empty trace file: {tf}")
                continue
        except Exception as e:
            print(f"WARNING: could not read {tf}: {e}")
            continue

        run_id = _parse_run_id_from_filename(tf)
        info = _infer_params_from_run_id(run_id, trace_rows)
        prov = _compute_provenance(trace_rows, tf, info)
        provenance_rows.append(prov)

    # Apply manual audit annotations if provided
    if args.manual_audit_csv and os.path.exists(args.manual_audit_csv):
        with open(args.manual_audit_csv, 'r', newline='') as f:
            manual = {row['run_id']: row for row in csv.DictReader(f)}
        for prov in provenance_rows:
            ma = manual.get(prov['run_id'])
            if ma:
                prov['failure_phase_manual'] = ma.get('failure_phase_manual', '')
                if ma.get('denominator_status'):
                    prov['denominator_status'] = ma['denominator_status']
                if ma.get('validity'):
                    prov['validity'] = ma['validity']
                if ma.get('invalid_reason'):
                    prov['invalid_reason'] = ma['invalid_reason']

    # Write provenance CSV
    os.makedirs(os.path.dirname(args.output_csv) or '.', exist_ok=True)
    with open(args.output_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=PROVENANCE_FIELDS, extrasaction='ignore')
        w.writeheader()
        w.writerows(provenance_rows)
    print(f"Wrote {len(provenance_rows)} rows to {args.output_csv}")

    # Compute group summaries
    group_summaries = _compute_group_summary(provenance_rows)

    # Write group summary CSV
    os.makedirs(os.path.dirname(args.group_csv) or '.', exist_ok=True)
    with open(args.group_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=GROUP_SUMMARY_FIELDS, extrasaction='ignore')
        w.writeheader()
        w.writerows(group_summaries)
    print(f"Wrote {len(group_summaries)} group summaries to {args.group_csv}")

    # Generate report
    report_text = generate_report(provenance_rows, group_summaries, args)
    os.makedirs(os.path.dirname(args.report) or '.', exist_ok=True)
    with open(args.report, 'w', encoding='utf-8') as f:
        f.write(report_text)
    print(f"Wrote report to {args.report}")


if __name__ == '__main__':
    main()
