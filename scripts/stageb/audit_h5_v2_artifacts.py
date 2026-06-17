#!/usr/bin/env python3
"""Independent H5-V2 artifact auditor — GPU(2,6)."""
import csv, hashlib, io, json, os, sys, torch
EPOCH = '/data/liuyu/outputs/l3_h5_v2_candidate_epoch_20260617_r1'

def tsha(t):
    b = io.BytesIO(); torch.save(t.detach().cpu(), b); return hashlib.sha256(b.getvalue()).hexdigest()

results = []
for seed in [81, 82]:
    sd = f'{EPOCH}/seed{seed}'
    meta = json.load(open(f'{sd}/source_run_metadata.json'))
    clean = torch.load(f'{sd}/clean_pixel_values.pt', map_location='cpu', weights_only=True)
    clean_sha = tsha(clean)
    print('Seed {}: clean_pv={}'.format(seed, clean_sha[:16]))

    sel_rows = list(csv.DictReader(open(f'{sd}/selected_candidates.csv')))
    for sr in sel_rows:
        cond = sr['condition']
        cid = int(sr['candidate_id'])
        grip = int(sr['official_gripper_token'])
        arm = int(sr['arm_match'])
        cond_short = cond.split('_')[0].lower()
        pv = torch.load(f'{sd}/{cond_short}_cand{cid}_adv_pv.pt', map_location='cpu', weights_only=True)
        delta = torch.load(f'{sd}/{cond_short}_cand{cid}_delta.pt', map_location='cpu', weights_only=True)
        linf = delta.float().abs().max().item()
        ok = linf <= 0.02353
        print('  {} id={}: grip={} arm={}/6 linf={:.6f} {}'.format(cond_short, cid, grip, arm, linf, 'OK' if ok else 'FAIL'))
        results.append({'seed': seed, 'condition': cond_short, 'cid': cid, 'grip': grip, 'arm': arm, 'linf': linf, 'linf_ok': ok})

all_ok = all(r['linf_ok'] for r in results)
print('\nAUDIT: {}/{} artifacts OK'.format(sum(1 for r in results if r['linf_ok']), len(results)))
