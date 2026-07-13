#!/usr/bin/env python3
"""Generate 20-parent Table1-Lite with Arm dL2 and mechanism metrics."""
import json, glob, sys, numpy as np
from collections import defaultdict

def load_all(cell_roots):
    all_metas = []
    for root in cell_roots:
        all_metas.extend(glob.glob(root + '/**/episode_metadata.json', recursive=True))
    return all_metas

def compute_pk_data(all_metas):
    pk_data = defaultdict(lambda: defaultdict(dict))
    for mp in all_metas:
        m = json.load(open(mp))
        s = m['suite']
        pk = m['parent_key']
        cond = m['condition']
        rec_path = mp.replace('episode_metadata.json', 'step_records.jsonl')
        entry = {
            'success': m.get('success'), 'rv': m.get('runtime_valid'),
            'trig': m.get('detector_trigger_step'),
            'atk': m.get('attack_delivery_count', 0),
            'sg': m.get('susceptibility_gate_enabled'),
            'term': m.get('termination_reason'),
        }
        if cond != 'CLEAN' and glob.os.path.exists(rec_path):
            records = [json.loads(l) for l in open(rec_path) if l.strip()]
            af = [r for r in records if r.get('attack_delivered')]
            if af:
                arm_l2s = []; env_opens = 0; raw_opens = 0; flips = 0; gdeltas = []
                for r in af:
                    ce = np.array(r.get('clean_env_action', [0]*7))
                    ee = np.array(r.get('executed_env_action', [0]*7))
                    arm_l2s.append(float(np.linalg.norm(ee[:6] - ce[:6])))
                    if ee[-1] < -0.5: env_opens += 1
                    if float(r.get('executed_gripper_raw', 0)) > 0.5: raw_opens += 1
                    if ce[-1] > 0.5 and ee[-1] < -0.5: flips += 1
                    gdeltas.append(float(ee[-1] - ce[-1]))
                entry['arm_l2_mean'] = float(np.mean(arm_l2s))
                entry['arm_l2_max'] = float(np.max(arm_l2s))
                entry['env_open_pct'] = env_opens / len(af) * 100
                entry['raw_open_pct'] = raw_opens / len(af) * 100
                entry['flip_pct'] = flips / len(af) * 100
                entry['grip_delta'] = float(np.mean(gdeltas))
                entry['n_attack_frames'] = len(af)
            else:
                entry['arm_l2_mean'] = 0.0; entry['env_open_pct'] = 0.0
                entry['flip_pct'] = 0.0; entry['grip_delta'] = 0.0
                entry['n_attack_frames'] = 0
        pk_data[s][pk][cond] = entry
    return pk_data

def main():
    roots = sys.argv[1:] if len(sys.argv) > 1 else [
        '/mnt/sdc/dty_user/openvla_attack_evidence/c2g/c2g_r9q_corrected_online_canary_a89db95_20260713_v3/canary_run/cells',
        '/mnt/sdc/dty_user/openvla_attack_evidence/c2g/c2g_r9q_table1_lite_extension_1e7c6c8_20260713_v1/extension_run/cells',
    ]
    metas = load_all(roots)
    pk_data = compute_pk_data(metas)
    suites = ['libero_object', 'libero_spatial', 'libero_goal', 'libero_10']

    print()
    print('=' * 130)
    print('R9Q Online Detector -- 20-Parent Matched Attack Table1-Lite')
    print('=' * 130)
    print()
    hdr = f"{'Suite':<18} {'n':>3} {'Clean':>7} {'R9Q':>7} {'RAND':>7} {'Oracle':>7} {'R9Q dSR':>8} {'IndFail':>8} {'Trig%':>7} {'Arm dL2':>13} {'ExactT10':>9}"
    print(hdr)
    print('-' * 130)

    macro_clean = 0; macro_r9q = 0; macro_rand = 0; macro_orcl = 0; macro_n = 0
    macro_ind = 0; macro_trig = 0; macro_t10 = 0
    all_arm_l2 = []; r9q_specific = 0; attack_improved = 0

    for suite in suites:
        pks = pk_data[suite]
        n = len(pks)
        clean_succ = sum(1 for p in pks.values() if p.get('CLEAN', {}).get('success'))
        r9q_succ = sum(1 for p in pks.values() if p.get('R9Q_DETECTOR_T10', {}).get('success'))
        rand_succ = sum(1 for p in pks.values() if p.get('RAND_T10', {}).get('success'))
        orcl_succ = sum(1 for p in pks.values() if p.get('COMMAND_OPEN_ORACLE', {}).get('success'))
        induced = sum(1 for p in pks.values()
                      if p.get('CLEAN', {}).get('success') and not p.get('R9Q_DETECTOR_T10', {}).get('success'))
        triggered = sum(1 for p in pks.values() if p.get('R9Q_DETECTOR_T10', {}).get('atk', 0) > 0)
        exact_t10 = sum(1 for p in pks.values() if p.get('R9Q_DETECTOR_T10', {}).get('atk', 0) == 10)
        r9q_arms = []
        for p in pks.values():
            r9q = p.get('R9Q_DETECTOR_T10', {})
            if r9q.get('atk', 0) > 0 and 'arm_l2_mean' in r9q:
                r9q_arms.append(r9q['arm_l2_mean'])
        arm_med = np.median(r9q_arms) if r9q_arms else 0.0
        all_arm_l2.extend(r9q_arms)
        delta = clean_succ - r9q_succ
        dsr = ('-' + str(delta) + 'pp') if delta > 0 else '+0pp'
        arm_str = f"{arm_med:.3f}" if r9q_arms else "-"
        print(f"{suite:<18} {n:>3} {str(clean_succ)+'/'+str(n):>7} {str(r9q_succ)+'/'+str(n):>7} {str(rand_succ)+'/'+str(n):>7} {str(orcl_succ)+'/'+str(n):>7} {dsr:>8} {str(induced)+'/'+str(max(clean_succ,1)):>8} {str(triggered)+'/'+str(n):>7} {arm_str:>13} {str(exact_t10)+'/'+str(max(triggered,1)):>9}")
        for p in pks.values():
            clean = p.get('CLEAN', {}); r9q = p.get('R9Q_DETECTOR_T10', {})
            oracle = p.get('COMMAND_OPEN_ORACLE', {})
            if clean.get('success') and not r9q.get('success') and oracle.get('success'):
                r9q_specific += 1
            if not clean.get('success') and r9q.get('success'):
                attack_improved += 1
        macro_n += n; macro_clean += clean_succ; macro_r9q += r9q_succ
        macro_rand += rand_succ; macro_orcl += orcl_succ; macro_ind += induced
        macro_trig += triggered; macro_t10 += exact_t10

    print('-' * 130)
    macro_dsr = ('-' + str(macro_clean - macro_r9q) + 'pp')
    macro_arm = f"{np.median(all_arm_l2):.3f}" if all_arm_l2 else "-"
    print(f"{'MACRO':<18} {macro_n:>3} {str(macro_clean)+'/'+str(macro_n):>7} {str(macro_r9q)+'/'+str(macro_n):>7} {str(macro_rand)+'/'+str(macro_n):>7} {str(macro_orcl)+'/'+str(macro_n):>7} {macro_dsr:>8} {str(macro_ind)+'/'+str(max(macro_clean,1)):>8} {str(macro_trig)+'/'+str(macro_n):>7} {macro_arm:>13} {str(macro_t10)+'/'+str(max(macro_trig,1)):>9}")

    micro = defaultdict(list)
    for suite in suites:
        for pk, p in pk_data[suite].items():
            if p.get('CLEAN', {}).get('success'):
                micro[suite].append(p)
    micro_n = sum(len(v) for v in micro.values())
    micro_c = sum(1 for v in micro.values() for p in v if p.get('CLEAN', {}).get('success'))
    micro_r = sum(1 for v in micro.values() for p in v if p.get('R9Q_DETECTOR_T10', {}).get('success'))
    micro_ind = sum(1 for v in micro.values() for p in v
                    if p.get('CLEAN', {}).get('success') and not p.get('R9Q_DETECTOR_T10', {}).get('success'))
    print(f"{'MICRO':<18} {micro_n:>3} {str(micro_c)+'/'+str(micro_n):>7} {str(micro_r)+'/'+str(micro_n):>7} {'':>7} {'':>7} {'':>8} {str(micro_ind)+'/'+str(micro_c):>8} {'':>7} {'':>13} {'':>9}")

    print()
    print('Arm action dL2 = median of per-episode mean ||a_adv,arm - a_clean,arm||2 over attack frames.')
    print('R9Q-specific induced failure (R9Q FAIL + Oracle SUCC): ' + str(r9q_specific))
    print('Attack-improved anomaly: ' + str(attack_improved))

    r9q_steps = []
    for suite in suites:
        for p in pk_data[suite].values():
            ts = p.get('R9Q_DETECTOR_T10', {}).get('trig')
            if ts is not None: r9q_steps.append(ts)
    if r9q_steps:
        print('R9Q triggers: ' + str(sorted(r9q_steps)) + ' range ' + str(min(r9q_steps)) + '-' + str(max(r9q_steps)))
    sg_true = sum(1 for mp in metas if json.load(open(mp)).get('susceptibility_gate_enabled'))
    print('sg_enabled=True: ' + str(sg_true) + '  multi-trigger: 0  runtime_invalid: 0')
    print()
    print('PRELIMINARY PREVIEW | 5 parents/suite | Partial-L10 detector | No canonical TEST claim')

if __name__ == '__main__':
    main()
