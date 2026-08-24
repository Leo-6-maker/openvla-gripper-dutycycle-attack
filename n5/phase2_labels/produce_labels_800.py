"""G4: 800-episode Label V2 production on 16 CPU shards."""
import sys, json, os, hashlib, time, multiprocessing as mp
from collections import defaultdict

DIR = os.path.dirname(__file__)
sys.path.insert(0, DIR)

# Patch paths for production
import run_pilot_12_v3 as pilot
from label_contract_v2 import N5_ALLOWED_ROOT
pilot.PILOT_OUT = os.path.join(N5_ALLOWED_ROOT, 'phase2_labels', 'g4_label_production')
pilot.MANIFEST_PATH = None  # Not used in process_episode
from run_pilot_12_v3 import process_episode

# Load training identities
IDENTITY_MANIFEST = '/mnt/sdc/dty_user/openvla_attack_evidence/c2g/c2g_cs200_official_v3_20260716/ops/FACTORIZED_PHASE_B2_DETERMINISTIC_ALLOCATION_V3_804113EE_20260723/checkpoint_training_identity_manifest.json'
OUT_ROOT = pilot.PILOT_OUT
N_SHARDS = 16

with open(IDENTITY_MANIFEST) as f:
    splits_data = json.load(f)

# Collect all training identities across all folds
all_train = set()
for fold_key, fold_data in splits_data['splits'].items():
    all_train.update(fold_data.get('checkpoint_training', []))
identities = sorted(all_train)

print(f'G4 Label Production: {len(identities)} episodes on {N_SHARDS} CPU shards')
print(f'Output: {OUT_ROOT}')

# Build full episode specs from identities
episodes = []
for identity in identities:
    suite, task, state = identity.split('/')
    episodes.append({
        'suite': suite, 'task': task, 'state': state,
        'input_files': {},  # verification via CS200 direct paths
    })

# Shard episodes
shard_size = (len(episodes) + N_SHARDS - 1) // N_SHARDS
shards = [episodes[i:i+shard_size] for i in range(0, len(episodes), shard_size)]
print(f'Shards: {len(shards)} (size ~{shard_size})')

def label_shard(shard_idx, shard_eps):
    """Label one shard of episodes."""
    shard_gates = {'ok': 0, 'missing': 0, 'unk2neg': 0, 'steps': 0}
    for ep in shard_eps:
        identity = f"{ep['suite']}/{ep['task']}/{ep['state']}"
        r = process_episode(ep, {'episodes': shard_eps}, dry_run=False)
        if 'error' in r:
            shard_gates['missing'] += 1
        else:
            shard_gates['ok'] += 1
            shard_gates['steps'] += r['n_steps']
            for l in r['steps']:
                if not l['physical_criticality']['valid_mask'] and l['physical_criticality']['value'] == 0:
                    shard_gates['unk2neg'] += 1
        if shard_gates['ok'] % 10 == 0:
            print(f'  Shard {shard_idx}: {shard_gates["ok"]}/{len(shard_eps)} done')
    return shard_gates

print('Launching shards...')
with mp.Pool(N_SHARDS) as pool:
    results = pool.starmap(label_shard, enumerate(shards))

# Aggregate
total_gates = {'ok': 0, 'missing': 0, 'unk2neg': 0, 'steps': 0}
for r in results:
    for k in total_gates:
        total_gates[k] += r[k]

# Receipt
receipt = {
    'production': 'G4_LABEL_PRODUCTION_800',
    'timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    'n_shards': N_SHARDS,
    'n_episodes': len(episodes),
    'identity_source': IDENTITY_MANIFEST,
    'gates': total_gates,
}
os.makedirs(OUT_ROOT, exist_ok=True)
rp = os.path.join(OUT_ROOT, 'PRODUCTION_RECEIPT.json')
with open(rp, 'w') as f:
    json.dump(receipt, f, indent=2, default=str)

ok = total_gates['ok'] == len(episodes) and total_gates['unk2neg'] == 0
print(f'\nG4 PRODUCTION: {"PASS" if ok else "FAIL"} ({total_gates["ok"]}/{len(episodes)})')
print(f'Steps: {total_gates["steps"]}, unk2neg: {total_gates["unk2neg"]}, missing: {total_gates["missing"]}')
print(f'Receipt: {rp}')
