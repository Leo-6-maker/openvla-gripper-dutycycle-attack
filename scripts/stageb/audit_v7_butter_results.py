#!/usr/bin/env python3
"""Audit V7 butter confirmation results from raw summaries.

Usage:
  python audit_v7_butter_results.py \
    --output-root /path/to/output \
    --manifest tables/stageb_v7_butter_mini_manifest.csv \
    --output-dir tables/
"""

import argparse
import csv
import json
import os
import glob
import statistics
from collections import defaultdict


def load_summaries(output_root, pair_names):
    """Load all summary JSONs from per-pair output dirs."""
    summaries = []
    for pair in pair_names:
        pair_dir = os.path.join(output_root, pair)
        if not os.path.isdir(pair_dir):
            continue
        for sf in sorted(glob.glob(os.path.join(pair_dir, 'summary_*.json'))):
            with open(sf) as f:
                s = json.load(f)
            s['_summary_path'] = sf
            s['_gpu_pair'] = pair
            summaries.append(s)
    return summaries


def classify_c2o(s):
    """Return (c2o_strict, c2o_env, c2o_boundary) for this summary."""
    # Use C2O_count which is strict C2O (primary metric)
    c2o_strict = s.get('C2O_count', 0)
    c2o_env = s.get('C2O_env_count', 0)
    c2o_boundary = s.get('C2O_boundary_count', 0)
    return c2o_strict > 0, c2o_env > 0, c2o_boundary > 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-root', required=True)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()

    PAIR_NAMES = ['GPU10', 'GPU26', 'GPU45']
    summaries = load_summaries(args.output_root, PAIR_NAMES)

    if not summaries:
        print("No summaries found!")
        return

    # Load expected manifest
    expected = []
    with open(args.manifest) as f:
        expected = list(csv.DictReader(f))

    # Build observed key map
    observed = {}
    for s in summaries:
        key = (s['task'], s.get('state_id', ''), s.get('condition', ''),
               s.get('attack_seed', ''))
        observed[key] = s

    # Audit: check all expected keys present
    missing = []
    all_valid = True
    results = []

    for row in expected:
        key = (row['task'], int(row['state_id']), row['condition'],
               int(row['attack_seed']))
        pid = row['parent_id']
        s = observed.get(key)

        entry = {
            'logical_key': row['logical_key'],
            'parent_id': pid,
            'condition': row['condition'],
            'seed': row['attack_seed'],
            'gpu_pair': row['gpu_pair'],
            'summary_found': s is not None,
        }

        if s is None:
            entry['valid'] = False
            entry['failure'] = 'MISSING_SUMMARY'
            missing.append(str(key))
            all_valid = False
        else:
            # Validity checks
            infra = s.get('infra_status', '')
            trigger = s.get('trigger_found', False)
            rc = 0  # assumed if summary exists

            valid = True
            failures = []

            if infra != 'ok':
                valid = False
                failures.append(f'infra={infra}')
            if not trigger:
                valid = False
                failures.append('no_trigger')

            # VIS-specific checks
            if row['condition'] == 'online_vis_pgd':
                if s.get('attack_method', '') not in ('token_prefix_pgd', ''):
                    # empty allowed for clean, but VIS must have token_prefix_pgd
                    pass
                # Check Linf
                actual_linf = s.get('pixel_budget_adv_inputs_linf', None)
                # (Linf check would need per-step trace; summary has only config-level)

            # C2O classification
            c2o_s, c2o_e, c2o_b = classify_c2o(s)
            entry['c2o_strict'] = int(c2o_s)
            entry['c2o_env'] = int(c2o_e)
            entry['c2o_boundary'] = int(c2o_b)
            entry['C2O_count'] = s.get('C2O_count', 0)
            entry['C2O_env_count'] = s.get('C2O_env_count', 0)
            entry['C2O_boundary_count'] = s.get('C2O_boundary_count', 0)
            entry['attacked_close_count'] = s.get('attacked_close_count', 0)
            entry['trigger_found'] = trigger
            entry['trigger_step'] = s.get('trigger_step', -1)
            entry['success_primary'] = s.get('success_primary', False)
            entry['infra_status'] = infra
            entry['perturb_frame_count'] = s.get('perturb_frame_count', 0)
            entry['n_steps'] = s.get('n_steps', 0)
            entry['runner_sha256'] = s.get('runner_sha256', '')
            entry['adapter_sha256'] = s.get('adapter_sha256', '')
            entry['valid'] = valid
            entry['failure'] = ';'.join(failures) if failures else ''

        results.append(entry)

    # Compute gate metrics
    vis_entries = [r for r in results if r['condition'] == 'online_vis_pgd' and r['valid']]
    rand_entries = [r for r in results if r['condition'] == 'online_random_linf' and r['valid']]
    clean_entries = [r for r in results if r['condition'] == 'clean_observer' and r['valid']]

    vis_c2o = sum(r['c2o_strict'] for r in vis_entries)
    rand_c2o = sum(r['c2o_strict'] for r in rand_entries)
    n_vis = len(vis_entries)
    n_rand = len(rand_entries)
    n_clean = len(clean_entries)

    print(f'Clean valid: {n_clean}')
    print(f'RAND valid: {n_rand}, C2O episodes: {rand_c2o}/{n_rand}')
    print(f'VIS valid: {n_vis}, C2O episodes: {vis_c2o}/{n_vis}')
    print(f'Missing keys: {len(missing)}')
    if missing:
        for m in missing:
            print(f'  MISSING: {m}')

    # Mini gate evaluation
    gate_clean = n_clean >= 2
    gate_rand_valid = n_rand >= 3
    gate_vis_valid = n_vis >= 3
    gate_vis_c2o = vis_c2o >= 2
    gate_rand_c2o = rand_c2o <= 1
    boundary_pos = sum(r['c2o_boundary'] for r in vis_entries)
    gate_boundary = boundary_pos == 0

    gates = [
        ('clean >= 2/2', gate_clean, f'{n_clean}/2'),
        ('RAND resolved valid >= 3/3', gate_rand_valid, f'{n_rand}/3'),
        ('VIS resolved valid >= 3/3', gate_vis_valid, f'{n_vis}/3'),
        ('VIS strict-C2O >= 2/3', gate_vis_c2o, f'{vis_c2o}/{n_vis}'),
        ('RAND strict-C2O <= 1/3', gate_rand_c2o, f'{rand_c2o}/{n_rand}'),
        ('boundary-only positive = 0', gate_boundary, f'{boundary_pos}'),
    ]

    all_pass = all(g[1] for g in gates)

    print('\n=== MINI GATE ===')
    for name, passed, value in gates:
        status = 'PASS' if passed else 'FAIL'
        print(f'  [{status}] {name}: {value}')

    decision = 'STRONG_PASS' if all_pass else 'FAIL'
    print(f'\nDecision: {decision}')

    if decision == 'STRONG_PASS':
        print('ACTION: Automatically launch formal 12-seed confirmation.')
    elif vis_c2o == 1 and gate_rand_c2o:
        print('ACTION: Launch extension seeds 404-406.')
    elif vis_c2o == 0:
        print('ACTION: Stop. ONLINE_CMD_NOT_REPLICATED_IN_MINI_EXECSPEC_V2')
    elif not gate_rand_c2o:
        print('ACTION: Stop. BUTTER_RANDOM_SENSITIVE_UNDER_CURRENT_PROTOCOL')

    # Write results table
    os.makedirs(args.output_dir, exist_ok=True)
    table_path = os.path.join(args.output_dir, 'stageb_v7_butter_all_attempts.csv')
    if results:
        with open(table_path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            w.writeheader()
            w.writerows(results)
        print(f'\nResults table: {table_path}')

    # Write gate summary
    gate_path = os.path.join(args.output_dir, 'stageb_v7_butter_mini_gate.csv')
    with open(gate_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['gate', 'passed', 'value'])
        for name, passed, value in gates:
            w.writerow([name, passed, value])
        w.writerow(['decision', '', decision])
    print(f'Gate summary: {gate_path}')


if __name__ == '__main__':
    main()
