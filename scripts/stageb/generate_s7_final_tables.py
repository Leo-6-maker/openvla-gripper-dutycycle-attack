#!/usr/bin/env python3
"""Generate S7 confirmation summary tables."""
import json, os, csv, numpy as np
from collections import defaultdict

OUT = '/data/liuyu/outputs/stageb_v1_1_k5c_targeted_expansion_rc1a_ca3a97e/layer2_hiddensafe_confirmation'
SHARDS = ['shard10', 'shard45', 'shard26', 'shard26_retry_tomato']
H_WINDOWS = {'tomato_sauce_s2_w165_175', 'cream_cheese_s0_w65_75', 'milk_s0_w70_80', 'salad_dressing_s1_w50_60'}
REPO = '/data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607'
TABLES = os.path.join(REPO, 'tables')

pairs = defaultdict(dict)
provenance = {}
for shard in SHARDS:
    shard_dir = os.path.join(OUT, shard)
    if not os.path.isdir(shard_dir): continue
    for fname in sorted(os.listdir(shard_dir)):
        if not fname.startswith('summary_') or not fname.endswith('.json'): continue
        with open(os.path.join(shard_dir, fname)) as f:
            s = json.load(f)
        cond = 'VIS' if 'vis' in s.get('condition','') else 'RAND'
        lp = s['pair_id']
        provenance[(lp, cond)] = 'retry' if shard == 'shard26_retry_tomato' else 'original'
        if lp in pairs and shard == 'shard26_retry_tomato':
            pairs[lp][cond] = s
        elif lp not in pairs or cond not in pairs[lp]:
            pairs.setdefault(lp, {})[cond] = s

# Seed-level CSV
with open(os.path.join(TABLES, 'layer2_hiddensafe_confirmation_seed_level.csv'), 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['group','window','attack_seed','condition','decoded_open_count','expected_window_size',
                'actual_window_steps','open_rate','label','infra_status','provenance'])
    for lp, cd in sorted(pairs.items()):
        window = lp.rsplit('__',1)[0]
        grp = 'H' if window in H_WINDOWS else 'B'
        vis_s = cd['VIS']; rand_s = cd['RAND']
        exp_ws = vis_s['window_end'] - vis_s['window_start']
        vis_cmd = vis_s['decoded_open_count'] >= max(exp_ws,1)/2
        rand_cmd = rand_s['decoded_open_count'] >= max(exp_ws,1)/2
        for cond in ['VIS','RAND']:
            s = cd[cond]
            act_ws = s['n_window_steps']
            rate = s['decoded_open_count'] / max(exp_ws, 1)
            if cond == 'VIS':
                label = 'cmd_hit' if (vis_cmd and not rand_cmd) else ('cmd_rand' if vis_cmd else ('rand_only' if rand_cmd else 'no_effect'))
            else:
                # label only makes sense per-pair, use the pair label
                label = 'cmd_hit' if (vis_cmd and not rand_cmd) else ('cmd_rand' if vis_cmd else ('rand_only' if rand_cmd else 'no_effect'))
            w.writerow([grp, window, s['attack_seed'], cond,
                        s['decoded_open_count'], exp_ws, act_ws,
                        round(rate,4), label,
                        s.get('infra_status',''),
                        provenance.get((lp,cond),'')])
print('Seed-level: %d rows' % (len(pairs)*2))

# Window-level CSV
with open(os.path.join(TABLES, 'layer2_hiddensafe_confirmation_window_level.csv'), 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['group','window','seed9_label','seed10_label','window_status',
                'vis_rate_atk9','vis_rate_atk10','rand_rate_atk9','rand_rate_atk10',
                'window_yield_score'])
    gr_all = [(lp,cd) for lp,cd in pairs.items()]
    for gname in ['H','B']:
        if gname == 'H':
            gr = [(lp,cd) for lp,cd in gr_all if (lp.rsplit('__',1)[0]) in H_WINDOWS]
        else:
            gr = [(lp,cd) for lp,cd in gr_all if (lp.rsplit('__',1)[0]) not in H_WINDOWS]
        windows = sorted(set(lp.rsplit('__',1)[0] for lp,_ in gr))
        for window in windows:
            wr = [(lp,cd) for lp,cd in gr if lp.rsplit('__',1)[0] == window]
            labels = {}; rates = {}
            for lp, cd in wr:
                atk = cd['VIS']['attack_seed']
                vis_s = cd['VIS']; rand_s = cd['RAND']
                exp_ws = vis_s['window_end'] - vis_s['window_start']
                vis_cmd = vis_s['decoded_open_count'] >= max(exp_ws,1)/2
                rand_cmd = rand_s['decoded_open_count'] >= max(exp_ws,1)/2
                if vis_cmd and not rand_cmd: labels[atk] = 'cmd_hit'
                elif vis_cmd and rand_cmd: labels[atk] = 'cmd_rand'
                elif not vis_cmd and rand_cmd: labels[atk] = 'rand_only'
                else: labels[atk] = 'no_effect'
                rates[atk] = (vis_s['decoded_open_count']/max(exp_ws,1),
                              rand_s['decoded_open_count']/max(exp_ws,1))
            both_cmd = all(v == 'cmd_hit' for v in labels.values())
            w_status = 'cmd_hit' if both_cmd else '+'.join(sorted(set(labels.values())))
            if both_cmd:
                yld = np.mean([max(0, rates[a][0]-rates[a][1]) for a in rates])
            else:
                yld = 0.0
            atks = sorted(rates.keys())
            w.writerow([gname, window,
                        labels.get(9,''), labels.get(10,''),
                        w_status,
                        round(rates.get(9,(0,0))[0],4), round(rates.get(10,(0,0))[0],4),
                        round(rates.get(9,(0,0))[1],4), round(rates.get(10,(0,0))[1],4),
                        round(yld,4)])
print('Window-level: %d rows' % 8)

# Results summary
with open(os.path.join(TABLES, 'layer2_hiddensafe_confirmation_results.csv'), 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['metric','Group_H','Group_B','Diff','Gate'])

    def gen_stats(gr):
        w_yields = []
        s_labels = []
        windows = sorted(set(lp.rsplit('__',1)[0] for lp,_ in gr))
        for window in windows:
            wr = [(lp,cd) for lp,cd in gr if lp.rsplit('__',1)[0] == window]
            all_cmd = True; w_rates = []
            for lp, cd in wr:
                vis_s = cd['VIS']; rand_s = cd['RAND']
                exp_ws = vis_s['window_end'] - vis_s['window_start']
                vis_cmd = vis_s['decoded_open_count'] >= max(exp_ws,1)/2
                rand_cmd = rand_s['decoded_open_count'] >= max(exp_ws,1)/2
                if vis_cmd and not rand_cmd: lbl = 'cmd_hit'
                elif vis_cmd and rand_cmd: lbl = 'cmd_rand'
                elif not vis_cmd and rand_cmd: lbl = 'rand_only'
                else: lbl = 'no_effect'
                s_labels.append(lbl)
                if lbl != 'cmd_hit': all_cmd = False
                w_rates.append(max(0, vis_s['decoded_open_count']/exp_ws - rand_s['decoded_open_count']/exp_ws))
            w_yields.append(np.mean(w_rates) if all_cmd else 0.0)
        n_s = len(s_labels)
        return {
            'window_yield': np.mean(w_yields),
            'seed_cmd_hit': sum(1 for x in s_labels if x=='cmd_hit')/n_s,
            'seed_cmd_rand': sum(1 for x in s_labels if x=='cmd_rand')/n_s,
            'seed_rand_only': sum(1 for x in s_labels if x=='rand_only')/n_s,
            'seed_no_effect': sum(1 for x in s_labels if x=='no_effect')/n_s,
        }

    if gname == 'H':
        h_gr = [(lp,cd) for lp,cd in gr_all if (lp.rsplit('__',1)[0]) in H_WINDOWS]
    else:
        h_gr = [(lp,cd) for lp,cd in gr_all if (lp.rsplit('__',1)[0]) in H_WINDOWS]
    b_gr = [(lp,cd) for lp,cd in gr_all if (lp.rsplit('__',1)[0]) not in H_WINDOWS]

    hs = gen_stats(h_gr)
    bs = gen_stats(b_gr)

    for metric in ['window_yield','seed_cmd_hit','seed_cmd_rand','seed_rand_only','seed_no_effect']:
        hv = hs[metric]; bv = bs[metric]; d = hv - bv
        if metric == 'window_yield': gate = 'PASS' if d > 0 else 'FAIL'
        elif metric == 'seed_cmd_hit': gate = 'PASS' if d >= 0 else 'FAIL'
        elif metric == 'seed_cmd_rand': gate = 'PASS' if d <= 0 else 'FAIL'
        elif metric == 'seed_rand_only': gate = 'PASS' if d <= 0 else 'FAIL'
        else: gate = '-'
        w.writerow([metric, round(hv,4), round(bv,4), round(d,4), gate])
    w.writerow(['verdict','','','','FAIL'])
print('Results summary saved')

# Retry audit
with open(os.path.join(TABLES, 'layer2_hiddensafe_confirmation_retry_audit.csv'), 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['logical_pair_key','seed','condition','source','decoded_open_count','n_window_steps','expected_ws','infra_status'])
    for lp in ['tomato_sauce_s2_w165_175__atk9', 'tomato_sauce_s2_w165_175__atk10']:
        for cond in ['VIS','RAND']:
            for shard in ['shard26','shard26_retry_tomato']:
                shard_dir = os.path.join(OUT, shard)
                if not os.path.isdir(shard_dir): continue
                for fname in os.listdir(shard_dir):
                    if not fname.startswith('summary_') or not fname.endswith('.json'): continue
                    with open(os.path.join(shard_dir, fname)) as f2:
                        s = json.load(f2)
                    c = 'VIS' if 'vis' in s.get('condition','') else 'RAND'
                    if s['pair_id'] == lp and c == cond:
                        w.writerow([lp, s['attack_seed'], cond,
                                    'retry' if shard=='shard26_retry_tomato' else 'original',
                                    s['decoded_open_count'], s['n_window_steps'],
                                    s['window_end']-s['window_start'],
                                    s.get('infra_status','')])
print('Retry audit saved')

print()
print('H yield: %+.4f  B yield: %+.4f  Diff: %+.4f  Verdict: FAIL' % (hs['window_yield'], bs['window_yield'], hs['window_yield']-bs['window_yield']))
