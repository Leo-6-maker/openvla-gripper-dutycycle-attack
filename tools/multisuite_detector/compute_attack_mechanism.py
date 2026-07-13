#!/usr/bin/env python3
"""Compute attack mechanism metrics from step_records."""
import json, glob, sys, numpy as np
from collections import defaultdict

def compute_arm_metrics(cells_root):
    metas = glob.glob(cells_root + '/**/episode_metadata.json', recursive=True)
    results = defaultdict(lambda: defaultdict(list))

    for mp in sorted(metas):
        m = json.load(open(mp))
        suite = m['suite']
        cond = m['condition']
        pk = m['parent_key']
        trig = m.get('detector_trigger_step')
        atk_count = m.get('attack_delivery_count', 0)

        rec_path = mp.replace('episode_metadata.json', 'step_records.jsonl')
        if not glob.os.path.exists(rec_path):
            continue

        records = [json.loads(l) for l in open(rec_path) if l.strip()]
        attack_frames = [r for r in records if r.get('attack_delivered')]

        if not attack_frames:
            results[suite][cond].append({
                'parent_key': pk, 'triggered': False, 'atk_count': 0,
                'arm_l2_mean': 0.0, 'arm_l2_max': 0.0,
                'arm_changed_frames': 0, 'gripper_open_frames': 0,
                'total_attack_frames': 0,
                'clean_succ': m.get('success'),
            })
            continue

        arm_l2s = []
        arm_changed = 0
        raw_open = 0
        env_open = 0
        close_to_open_flip = 0
        gripper_deltas = []
        for r in attack_frames:
            clean_env = np.array(r.get('clean_env_action', [0]*7))
            exec_env = np.array(r.get('executed_env_action', [0]*7))
            arm_delta = exec_env[:6] - clean_env[:6]
            l2 = float(np.linalg.norm(arm_delta))
            arm_l2s.append(l2)
            if l2 > 1e-4:
                arm_changed += 1
            # LIBERO: env_gripper < -0.5 = OPEN, env_gripper > +0.5 = CLOSE
            if float(r.get('executed_gripper_raw', 0)) > 0.5:
                raw_open += 1
            if exec_env[-1] < -0.5:
                env_open += 1
            # Close-to-open flip: clean was CLOSE, executed is OPEN
            if clean_env[-1] > 0.5 and exec_env[-1] < -0.5:
                close_to_open_flip += 1
            gripper_deltas.append(float(exec_env[-1] - clean_env[-1]))

        results[suite][cond].append({
            'parent_key': pk, 'triggered': True, 'trig_step': trig,
            'atk_count': atk_count,
            'arm_l2_mean': float(np.mean(arm_l2s)),
            'arm_l2_max': float(np.max(arm_l2s)),
            'arm_changed_frames': arm_changed,
            'raw_open_frames': raw_open,
            'env_open_frames': env_open,
            'close_to_open_flip': close_to_open_flip,
            'gripper_delta_mean': float(np.mean(gripper_deltas)),
            'total_attack_frames': len(attack_frames),
            'clean_succ': m.get('success'),
            'attack_succ': m.get('success'),
        })

    return results

def print_mechanism_table(results):
    suites = ['libero_object', 'libero_spatial', 'libero_goal', 'libero_10']
    conditions = ['R9Q_DETECTOR_T10', 'RAND_T10', 'COMMAND_OPEN_ORACLE']

    print()
    print('=' * 120)
    print('Attack Mechanism Diagnostic Table')
    print('=' * 120)
    print(f"{'Condition':<22} {'Suite':<18} {'Trig n':>7} {'Arm dL2':>9} {'EnvOpen%':>9} {'RawOpen%':>9} {'Flip%':>7} {'GripD':>7} {'IndFail':>8}")
    print('-' * 120)

    for cond in conditions:
        for suite in suites:
            entries = results[suite].get(cond, [])
            trig_entries = [e for e in entries if e['triggered']]

            if not trig_entries:
                print(f"{cond:<22} {suite:<18} {'0':>7} {'-':>9} {'-':>9} {'-':>9} {'-':>7} {'-':>7} {'-':>8}")
                continue

            arm_means = [e['arm_l2_mean'] for e in trig_entries]
            total_frames = sum(e['total_attack_frames'] for e in trig_entries)
            changed_frames = sum(e['arm_changed_frames'] for e in trig_entries)
            env_open_frames = sum(e['env_open_frames'] for e in trig_entries)
            raw_open_frames = sum(e['raw_open_frames'] for e in trig_entries)
            flip_frames = sum(e['close_to_open_flip'] for e in trig_entries)
            grip_deltas = [e['gripper_delta_mean'] for e in trig_entries]
            induced = sum(1 for e in entries if e.get('clean_succ') and not e.get('attack_succ'))

            med_arm = np.median(arm_means)
            env_pct = env_open_frames / max(total_frames, 1) * 100
            raw_pct = raw_open_frames / max(total_frames, 1) * 100
            flip_pct = flip_frames / max(total_frames, 1) * 100
            med_grip_d = np.median(grip_deltas)

            print(f"{cond:<22} {suite:<18} {len(trig_entries):>7} {med_arm:>9.4f} {env_pct:>8.1f}% {raw_pct:>8.1f}% {flip_pct:>6.1f}% {med_grip_d:>7.2f} {str(induced)+'/'+str(len(entries)):>8}")
        print('-' * 120)

def print_r9q_specific(results):
    pk_conds = defaultdict(dict)
    for suite in ['libero_object', 'libero_spatial', 'libero_goal', 'libero_10']:
        for entry in results[suite].get('R9Q_DETECTOR_T10', []):
            pk_conds[entry['parent_key']]['R9Q'] = entry
        for entry in results[suite].get('COMMAND_OPEN_ORACLE', []):
            pk_conds[entry['parent_key']]['ORACLE'] = entry
        for entry in results[suite].get('CLEAN', []):
            pk_conds[entry['parent_key']]['CLEAN'] = entry

    r9q_specific = []
    for pk, conds in pk_conds.items():
        clean = conds.get('CLEAN', {})
        r9q = conds.get('R9Q', {})
        oracle = conds.get('ORACLE', {})
        c_s = clean.get('success') or clean.get('clean_succ')
        r_s = r9q.get('attack_succ') or r9q.get('success')
        o_s = oracle.get('attack_succ') or oracle.get('success') or oracle.get('clean_succ', True)
        if c_s and not r_s and o_s:
            r9q_specific.append(pk)

    print('R9Q-specific induced failure (Clean SUCC, R9Q FAIL, Oracle SUCC):')
    for pk in r9q_specific:
        print(f'  {pk}')
    print(f'  Total: {len(r9q_specific)}')

    improved = []
    for pk, conds in pk_conds.items():
        clean = conds.get('CLEAN', {})
        r9q = conds.get('R9Q', {})
        c_s = clean.get('success') or clean.get('clean_succ')
        r_s = r9q.get('attack_succ') or r9q.get('success')
        if not c_s and r_s:
            improved.append(pk)
    print(f'Attack-improved anomaly: {len(improved)}')

if __name__ == '__main__':
    cells = sys.argv[1]
    results = compute_arm_metrics(cells)
    print_mechanism_table(results)
    print_r9q_specific(results)
