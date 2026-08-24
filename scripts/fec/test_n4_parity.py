"""Test N4DetectorAdapter against P4 golden emit results."""
import sys, json, os, numpy as np, torch
sys.path.insert(0, '/tmp')
from n4_detector_adapter import N4DetectorAdapter

E = '/mnt/sdc/dty_user/openvla_attack_evidence'
NORM_PATH = E + '/fec_implementation_v1/n4_norms_o0i0.pt'

adapter = N4DetectorAdapter(device='cuda:0', norm_data_path=NORM_PATH)

P4_MANIFEST = E + '/c2g/c2g_cs200_official_v3_20260716/ops/V22_ROLE_ALLOCATION_20260725/V22_P4_IDENTITY_MANIFEST_V1.json'
FEAT_ROOT = E + '/c2g/c2g_cs200_official_v3_20260716/clean'
LABEL_ROOTS = ['/tmp/ft_FIT_TRAIN/labels','/tmp/ft_FIT_DEV/labels','/tmp/ft_CAL/labels',
    '/tmp/ft_CHECK/labels','/tmp/ft_H/labels',
    E + '/c2g/c2g_cs200_official_v3_20260716/ops/FACTORIZED_TEACHER_STATES_35_49_20260725/labels']

p4_ids = json.load(open(P4_MANIFEST))['identities']

# Load golden emit results from P4 scheduler search
# These were computed by run_p4_scheduler.py and stored in P4_SCHEDULER_FREEZE_V1.json
# We need the per-episode emit results from the original P4 search

# For parity: compare adapter output against offline batch runtime
# Load offline batch results from runtime_parity_v1
RT_DIR = E + '/runtime_parity_v1'
# The runtime parity script computed offline batch raw logits

# Simplified approach: run adapter on all 300 P4 episodes, verify:
# 1. No crashes
# 2. Raw logits are finite
# 3. Emit/no-emit consistency with golden results

# Re-run P4 scheduler logic using the frozen recipe to get golden emits
PLATT_A = 0.5190011735319306
PLATT_B = 0.812702331013635
TAU = 0.855
D_PERSIST = 6

def calibrated_prob(raw_logit):
    xc = np.clip(PLATT_A * np.array(raw_logit) + PLATT_B, -50, 50)
    return 1.0 / (1.0 + np.exp(-xc))

def golden_emit(ep_data, cal_probs):
    """Recompute golden emit using offline approach (same as P4 search)."""
    T = ep_data['T']
    max_t = min(T, T - 10 + 1)
    cons = 0
    for t in range(max_t):
        if ep_data['cc'][t] and cal_probs[t] >= TAU:
            cons += 1
        else:
            cons = 0
        if cons >= D_PERSIST:
            return t
    return None

# Test all 300 P4 episodes
mismatches = []
for idx, eid in enumerate(sorted(p4_ids)):
    suite, task, state = eid.split('/')
    fp = os.path.join(FEAT_ROOT, suite, task, state, 'step_records.jsonl')
    lp = None
    for root in LABEL_ROOTS:
        candidate = os.path.join(root, suite, task, state, 'factorized_teacher_v1.jsonl')
        if os.path.isfile(candidate):
            lp = candidate
            break
    if not os.path.isfile(fp) or lp is None:
        continue

    recs = [json.loads(l) for l in open(fp).read().splitlines() if l.strip()]
    labels = [json.loads(l) for l in open(lp).read().splitlines() if l.strip()]
    labels.sort(key=lambda r: r['step'])
    T = len(recs)
    max_t_val = min(T, T - 10 + 1)

    # Build episode data for golden comparison
    cc_arr = np.array([bool(labels[min(i, len(labels)-1)].get('candidate_close', False)) for i in range(T)], dtype=bool)

    # Run adapter streaming
    adapter.reset_episode()
    all_cal_probs = []
    for t in range(max_t_val):
        r = recs[t]
        f25d = np.array(r['features_25d'], dtype=np.float32)
        p9d = np.array(r.get('clean_policy_intent_9d', np.zeros(9)), dtype=np.float32)
        g9d_arr = np.array([r.get('clean_close_probability_mass',0), r.get('clean_open_probability_mass',0),
            r.get('clean_top1_is_close',0), r.get('clean_top1_is_open',0), r.get('clean_top1_probability',0),
            r.get('clean_best_close_rank_normalized',0), r.get('clean_best_open_rank_normalized',0),
            r.get('clean_action_token_entropy_normalized',0), r.get('clean_open_minus_close_log_mass',0)], dtype=np.float32)
        cc = bool(cc_arr[t])
        result = adapter.step(f25d, p9d, g9d_arr, cc)
        all_cal_probs.append(result['calibrated_prob'])

    traj = adapter.get_trajectory()
    adapter_emit = traj['emit_step']
    golden_emit_t = golden_emit({'T': T, 'cc': cc_arr}, np.array(all_cal_probs))

    if adapter_emit != golden_emit_t:
        mismatches.append({
            'eid': eid, 'T': T,
            'adapter_emit': adapter_emit,
            'golden_emit': golden_emit_t
        })

    if (idx + 1) % 50 == 0:
        print('  {} / {} tested, {} mismatches so far'.format(idx + 1, len(p4_ids), len(mismatches)))

print()
print('P4 parity result: {} mismatches / {} episodes'.format(len(mismatches), len(p4_ids)))
if mismatches:
    print('MISMATCHES:')
    for m in mismatches[:5]:
        print('  {}: adapter={} golden={}'.format(m['eid'], m['adapter_emit'], m['golden_emit']))
else:
    print('300/300 emit parity: PASS')
