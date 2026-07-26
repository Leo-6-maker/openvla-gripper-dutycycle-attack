"""G2: Run 32-episode canary label production."""
import sys, json, os, hashlib, time

DIR = os.path.dirname(__file__)
sys.path.insert(0, DIR)

# Patch module-level paths BEFORE importing process_episode
import run_pilot_12_v3 as pilot
pilot.MANIFEST_PATH = os.path.join(DIR, 'canary_32_manifest_full.json')
pilot.PILOT_OUT = os.path.join(DIR, 'canary_32_output')

from run_pilot_12_v3 import process_episode

MANIFEST_PATH = pilot.MANIFEST_PATH
OUT = pilot.PILOT_OUT

with open(MANIFEST_PATH) as f:
    manifest = json.load(f)
n_eps = len(manifest['episodes'])
print('G2 Canary: ' + str(n_eps) + ' episodes')
print('Output: ' + OUT)

gates = {'ok': 0, 'missing': 0, 'unk2neg': 0}
results = []
for ep in manifest['episodes']:
    identity = ep['suite'] + '/' + ep['task'] + '/' + ep['state']
    r = process_episode(ep, manifest, dry_run=False)
    if 'error' in r:
        err = str(r.get('error', ''))[:80]
        print('  ' + identity + ': ERR ' + err)
        gates['missing'] += 1
    else:
        gates['ok'] += 1
        results.append(r)
        n = r['n_steps']
        s = r['stats']
        cp = s['n_critical'] / max(1, n) * 100
        npct = s.get('n_critical_negative', 0) / max(1, n) * 100
        print('  ' + identity + ': ' + str(n) + 's crit=' + str(s['n_critical']) + '(' + str(int(cp)) + '%) neg=' + str(s.get('n_critical_negative',0)) + '(' + str(int(npct)) + '%) unk=' + str(s['n_critical_unknown']) + ' k10=' + str(s['n_k10_feasible']) + ' safe=' + str(s['n_safe_release']))
        for l in r['steps']:
            if not l['physical_criticality']['valid_mask'] and l['physical_criticality']['value'] == 0:
                gates['unk2neg'] += 1

os.makedirs(OUT, exist_ok=True)
receipt = {
    'canary': 'G2_CANARY_32',
    'timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    'manifest_sha': hashlib.sha256(open(MANIFEST_PATH, 'rb').read()).hexdigest(),
    'gates': gates,
    'results': results,
    'summary': {
        'n_processed': gates['ok'],
        'total_steps': sum(r['n_steps'] for r in results),
        'total_critical': sum(r['stats']['n_critical'] for r in results),
        'total_critical_neg': sum(r['stats'].get('n_critical_negative', 0) for r in results),
        'total_critical_unk': sum(r['stats']['n_critical_unknown'] for r in results),
        'total_attack_opp': sum(r['stats']['n_attack_opportunity'] for r in results),
        'total_gripper_close': sum(r['stats']['n_gripper_closing'] for r in results),
        'total_safe_release': sum(r['stats']['n_safe_release'] for r in results),
        'total_instability': sum(r['stats']['n_instability'] for r in results),
    },
}
rp = os.path.join(OUT, 'CANARY_RECEIPT.json')
with open(rp, 'w') as f:
    json.dump(receipt, f, indent=2, default=str)
ok = gates['ok'] == n_eps and gates['unk2neg'] == 0
print('G2 CANARY: ' + ('PASS' if ok else 'FAIL') + ' (' + str(gates['ok']) + '/' + str(n_eps) + ') unk2neg=' + str(gates['unk2neg']))
