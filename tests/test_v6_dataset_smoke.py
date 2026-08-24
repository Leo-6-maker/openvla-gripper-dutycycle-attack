"""Smoke test for V2 dataset adapter."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from gripper_attack.v6_critical_dataset import (
    load_v2_episodes, CriticalEpisodeDataset, collate_v2_batch)
import json

# Mock manifest for H1 data
manifest = {
    'splits': {
        'o0_i0': {'heldout_l3': [
            'libero_10/task_00/state_30',
            'libero_10/task_01/state_31',
        ]},
    }
}

FEAT_ROOT = '/mnt/sdc/dty_user/openvla_attack_evidence/c2g/c2g_cs200_official_v3_20260716/clean'
TEACHER_ROOT = '/tmp/ft_H/labels'

eps = load_v2_episodes(FEAT_ROOT, TEACHER_ROOT, manifest, exclude_parser_contradictions=True)
print(f'Loaded {len(eps)} episodes')
assert len(eps) > 0, 'No episodes loaded'

ep = eps[0]
print(f'First episode: {ep.eid} T={ep.T} has_opp={ep.has_opportunity} reason={ep.absence_reason}')
print(f'  f25d shape: {ep.features_25d.shape}')
print(f'  p9d shape: {ep.policy_9d.shape}')
print(f'  g9d shape: {ep.gripper_9d.shape}')
print(f'  k10_pos: {ep.k10_startable.sum()} / {ep.k10_known.sum()} known')
assert ep.features_25d.shape == (ep.T, 25)
assert ep.policy_9d.shape == (ep.T, 9)
assert ep.gripper_9d.shape == (ep.T, 9)

# Test dataset + collation
ds = CriticalEpisodeDataset(eps)
print(f'Dataset: {len(ds)} episodes, opp_rate={ds.opportunity_rate:.2%}')
print(f'Absence summary: {ds.absence_summary}')

batch = collate_v2_batch([ds[i] for i in range(min(2, len(ds)))])
print(f'Batch keys: {list(batch.keys())}')
print(f'x_25d shape: {batch["x_25d"].shape}')
print(f'x_policy shape: {batch["x_policy"].shape}')
print(f'k10_startable shape: {batch["k10_startable"].shape}')

print()
print('PASS: V2 dataset adapter')
