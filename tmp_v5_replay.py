"""V5 scheduler replay + long-streak audit + improved L1/L2 Pareto."""
import json, csv, hashlib, time, sys
from collections import defaultdict
from pathlib import Path
import numpy as np

# Add scheduler to path
sys.path.insert(0, "/mnt/sdc/dty_user/openvla_attack/src")
from gripper_attack.v5_scheduler import V5OneShotScheduler, V5SchedulerConfig

BASE = Path("/mnt/sdc/dty_user/openvla_attack_evidence/c2g/c2g_cs200_official_v3_20260716/ops/OFFICIAL_V3_FACTORIZED_STUDENT_V2_RECOMMENDED_EXACT_W32_V1_20260721")
OUT = BASE / "analysis/student_trigger_calibration"
OUT.mkdir(parents=True, exist_ok=True)
THIS_SHA = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

# Scheduler contract hash
SCHEDULER_PATH = Path("/mnt/sdc/dty_user/openvla_attack/src/gripper_attack/v5_scheduler.py")
SCHEDULER_SHA = hashlib.sha256(SCHEDULER_PATH.read_bytes()).hexdigest()

def discover():
    splits = []
    for d in sorted(BASE.iterdir()):
        if not d.is_dir() or not d.name.startswith("V2B_EXACT_W32_H64_D0.1_WD1e-4_"):
            continue
        if not (BASE / f"audit_{d.name}.json").is_file(): continue
        if not (d / "SHA256SUMS").is_file(): continue
        splits.append(d.name)
    return sorted(splits)

def load_steps(sl):
    path = BASE / f"predict_{sl}" / "heldout_step_predictions.jsonl"
    steps = []
    with open(path) as f:
        for line in f: steps.append(json.loads(line))
    return steps

def load_platt(sl):
    """Load pre-computed Platt params for this split."""
    pf = BASE / "analysis/student_trigger_calibration/platt_calibration_results.json"
    if pf.is_file():
        data = json.loads(pf.read_text())
        for r in data:
            if r.get('split') == sl.replace('V2B_EXACT_W32_H64_D0.1_WD1e-4_', ''):
                return r
    return None

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -50, 50)))

# ── L1/L2 metrics ──

def compute_l1_l2(steps, tau_g=0.5, tau_r=0.5, platt_params=None):
    """Compute L1 (per-step) and L2 (per-episode) bg emit with known_mask."""
    bg_steps = 0; bg_emit = 0
    eps_bg = defaultdict(lambda: {'total': 0, 'emit': False})

    a_g, b_g = 1.0, 0.0
    a_r, b_r = 1.0, 0.0
    if platt_params and platt_params.get('head') == 'grasp':
        a_g = platt_params.get('a', 1.0)
        b_g = platt_params.get('b', 0.0)
    # Get release platt from same split
    if platt_params:
        pf = BASE / "analysis/student_trigger_calibration/platt_calibration_results.json"
        if pf.is_file():
            data = json.loads(pf.read_text())
            split_short = platt_params.get('split', '')
            for r in data:
                if r.get('split') == split_short and r.get('head') == 'release':
                    a_r = r.get('a', 1.0)
                    b_r = r.get('b', 0.0)

    for s in steps:
        if s['event_id'] < 0 and s['route_supported']:
            cid = s['canonical_parent_key']
            gk = s.get('grasp_known_mask', False)
            rk = s.get('release_known_mask', False)

            gp = s.get('grasp_prob', 0)
            rp = s.get('release_prob', 0)
            # Apply Platt
            if 0 < gp < 1:
                z = np.log(gp/(1-gp))
                gp = sigmoid(a_g * z + b_g)
            if 0 < rp < 1:
                z = np.log(rp/(1-rp))
                rp = sigmoid(a_r * z + b_r)

            any_known = gk or rk
            if any_known:
                bg_steps += 1
                eps_bg[cid]['total'] += 1
            emit = (gk and gp >= tau_g) or (rk and rp >= tau_r)
            if emit:
                bg_emit += 1
                eps_bg[cid]['emit'] = True

    L1 = bg_emit / max(1, bg_steps)
    L2 = sum(1 for v in eps_bg.values() if v['emit']) / max(1, len(eps_bg))
    return L1, L2, bg_steps, len(eps_bg)

# ── V5 Scheduler Replay ──

def v5_scheduler_replay(steps, platt_params=None):
    """Replay actual V5 scheduler per episode.

    candidate_close = route_supported AND event_id >= 0 (approximate)
    utility_probability = grasp_prob
    release_probability = release_prob
    regrasp_probability = manipulation_prob
    """
    config = V5SchedulerConfig()
    a_g, b_g = 1.0, 0.0
    a_r, b_r = 1.0, 0.0
    a_m, b_m = 1.0, 0.0

    if platt_params:
        pf = BASE / "analysis/student_trigger_calibration/platt_calibration_results.json"
        if pf.is_file():
            data = json.loads(pf.read_text())
            split_short = platt_params.get('split', '')
            for r in data:
                if r.get('split') == split_short:
                    if r.get('head') == 'grasp': a_g, b_g = r.get('a', 1.0), r.get('b', 0.0)
                    if r.get('head') == 'release': a_r, b_r = r.get('a', 1.0), r.get('b', 0.0)

    episodes = defaultdict(list)
    for s in steps:
        episodes[s['canonical_parent_key']].append(s)

    results = []
    for ep_key, ep_steps in sorted(episodes.items()):
        ep_sorted = sorted(ep_steps, key=lambda x: x['step_index'])
        scheduler = V5OneShotScheduler(config)

        for s in ep_sorted:
            # candidate_close approximation
            candidate_close = s['route_supported'] and s['event_id'] >= 0

            gp = s.get('grasp_prob', 0); rp = s.get('release_prob', 0); mp = s.get('manipulation_prob', 0)
            if 0 < gp < 1: gp = sigmoid(a_g * np.log(gp/(1-gp)) + b_g)
            if 0 < rp < 1: rp = sigmoid(a_r * np.log(rp/(1-rp)) + b_r)

            result = scheduler.update(
                step=s['step_index'],
                candidate_close=candidate_close,
                valid=s['route_supported'],
                utility_probability=gp,
                release_probability=rp,
                regrasp_probability=mp,
                uncertainty_probability=0.0,
            )

        results.append({
            'identity': ep_key,
            'route': ep_sorted[0].get('mechanism_route', '?'),
            'emitted': result['one_shot_emitted'],
            'emit_step': result['emit_step'],
            'final_state': result['state'],
            'candidate_dwell_max': result['candidate_dwell'],
        })

    total = len(results)
    emitted = sum(1 for r in results if r['emitted'])
    return {
        'total_episodes': total,
        'emitted_episodes': emitted,
        'emit_rate': emitted / max(1, total),
        'per_episode': results,
    }

# ── Long streak semantic audit ──

def semantic_streak_audit(steps, split_label, top_n=30):
    """Classify long background streaks."""
    bg_eps = defaultdict(list)
    event_eps = defaultdict(list)
    for s in steps:
        cid = s['canonical_parent_key']
        if s['event_id'] < 0 and s['route_supported']:
            bg_eps[cid].append(s)
        elif s['event_id'] >= 0 and s['route_supported']:
            event_eps[cid].append(s)

    streaks = []
    for ep_key, ep_bg in bg_eps.items():
        ep_sorted = sorted(ep_bg, key=lambda x: x['step_index'])
        events_sorted = sorted(event_eps.get(ep_key, []), key=lambda x: x['step_index'])

        seq = 0; start = None; prev_idx = None
        for bs in ep_sorted:
            gk = bs.get('grasp_known_mask', False); rk = bs.get('release_known_mask', False)
            emits = []
            if gk: emits.append(bs['grasp_prob'])
            if rk: emits.append(bs['release_prob'])
            any_s = max(emits) if emits else 0
            curr_idx = bs['step_index']

            if prev_idx is not None and curr_idx != prev_idx + 1:
                if seq > 0 and start is not None:
                    streaks.append({'ep': ep_key, 'start': start, 'end': prev_idx, 'length': seq,
                                    'route': bs.get('mechanism_route', '?')})
                seq = 0; start = None
            if any_s >= 0.5:
                if seq == 0: start = curr_idx
                seq += 1
            else:
                if seq > 0 and start is not None:
                    streaks.append({'ep': ep_key, 'start': start, 'end': prev_idx, 'length': seq,
                                    'route': bs.get('mechanism_route', '?')})
                seq = 0; start = None
            prev_idx = curr_idx

    streaks.sort(key=lambda x: -x['length'])

    # Classify top streaks
    classified = []
    for st in streaks[:top_n]:
        ep_key = st['ep']
        events = sorted(event_eps.get(ep_key, []), key=lambda x: x['step_index'])

        # Find nearest event
        min_dist = float('inf')
        nearest_event_id = None
        for ev in events:
            dist = min(abs(st['start'] - ev['step_index']), abs(st['end'] - ev['step_index']))
            if dist < min_dist:
                min_dist = dist
                nearest_event_id = ev['event_id']

        # Classification
        if min_dist <= 10:
            if st['start'] < min(ev['step_index'] for ev in events if ev['event_id'] == nearest_event_id):
                classification = 'PRE_EVENT_ANTICIPATION'
            else:
                classification = 'POST_EVENT_TAIL'
        elif min_dist <= 30:
            classification = 'LABEL_BOUNDARY_AMBIGUITY'
        elif min_dist > 100:
            classification = 'TRUE_BACKGROUND_FALSE_POSITIVE'
        else:
            classification = 'CANNOT_DETERMINE'

        classified.append({**st, 'nearest_event_dist': min_dist,
                          'classification': classification, 'split': split_label})

    return classified

# ── MAIN ──
splits = discover()
print(f"Found {len(splits)} splits")
print(f"Scheduler SHA256: {SCHEDULER_SHA[:16]}")
print(f"Analysis script SHA256: {THIS_SHA[:16]}")

# ── 1. V5 Scheduler Replay (L3) ──
print("\n=== V5 SCHEDULER REPLAY (L3) ===")
print("candidate_close = route_supported AND event_id >= 0 (APPROXIMATE)")
print(f"{'split':<8} {'eps':>5} {'emitted':>7} {'rate':>7}")
print("-" * 35)

l3_results = {}
for sl in splits:
    steps = load_steps(sl)
    short = sl.replace('V2B_EXACT_W32_H64_D0.1_WD1e-4_', '')
    # Try with raw and Platt
    platt = load_platt(sl)
    l3_raw = v5_scheduler_replay(steps, None)
    l3_platt = v5_scheduler_replay(steps, platt)
    l3_results[short] = {'raw': l3_raw, 'platt': l3_platt}
    print(f"{short:<8} raw={l3_raw['emitted_episodes']:>3}/{l3_raw['total_episodes']:<3} ({l3_raw['emit_rate']:.3f}) platt={l3_platt['emitted_episodes']:>3}/{l3_platt['total_episodes']:<3} ({l3_platt['emit_rate']:.3f})")

# ── 2. L1/L2 with Platt ──
print("\n=== L1/L2 WITH PLATT ===")
print(f"{'split':<8} {'L1_raw':>8} {'L1_platt':>9} {'L2_raw':>8} {'L2_platt':>9}")
print("-" * 50)

l1l2_results = {}
for sl in splits:
    steps = load_steps(sl)
    short = sl.replace('V2B_EXACT_W32_H64_D0.1_WD1e-4_', '')
    platt = load_platt(sl)
    L1r, L2r, _, _ = compute_l1_l2(steps, 0.5, 0.5, None)
    L1p, L2p, _, _ = compute_l1_l2(steps, 0.5, 0.5, platt)
    l1l2_results[short] = {'L1_raw': L1r, 'L2_raw': L2r, 'L1_platt': L1p, 'L2_platt': L2p}
    l1p_ok = 'OK' if L1p <= 0.10 else '--'
    print(f"{short:<8} {L1r:>8.4f} {L1p:>9.4f} {L2r:>8.4f} {L2p:>9.4f}  {l1p_ok}")

# ── 3. Long Streak Audit ──
print("\n=== LONG STREAK SEMANTIC AUDIT ===")
all_classified = []
for sl in splits[:6]:  # First 6 splits
    steps = load_steps(sl)
    short = sl.replace('V2B_EXACT_W32_H64_D0.1_WD1e-4_', '')
    classified = semantic_streak_audit(steps, short, top_n=15)
    all_classified.extend(classified)

# Summary counts
counts = defaultdict(int)
for c in all_classified:
    counts[c['classification']] += 1
total = len(all_classified)
print(f"Total streaks classified: {total}")
for cat in ['PRE_EVENT_ANTICIPATION', 'POST_EVENT_TAIL', 'LABEL_BOUNDARY_AMBIGUITY',
            'TRUE_BACKGROUND_FALSE_POSITIVE', 'CANNOT_DETERMINE']:
    n = counts[cat]
    print(f"  {cat}: {n} ({100*n/max(1,total):.1f}%)")

print(f"\nTop-10 longest:")
for s in all_classified[:10]:
    print(f"  {s['split']} {s['ep'][:30]} len={s['length']} dist={s['nearest_event_dist']} → {s['classification']}")

# ── 4. Write outputs ──

# L3 replay
with open(OUT / "v5_scheduler_l3_replay.json", "w") as f:
    json.dump({
        'scheduler_sha256': SCHEDULER_SHA,
        'candidate_close_approximation': 'route_supported AND event_id >= 0',
        'candidate_close_status': 'APPROXIMATE_NOT_AUTHORITATIVE',
        'required_missing_field': 'candidate_close from action_states',
        'results': {k: {'raw_emit_rate': v['raw']['emit_rate'],
                        'platt_emit_rate': v['platt']['emit_rate'],
                        'raw_emitted': v['raw']['emitted_episodes'],
                        'platt_emitted': v['platt']['emitted_episodes'],
                        'total_episodes': v['raw']['total_episodes']}
                    for k, v in l3_results.items()},
        'analysis_script_sha256': THIS_SHA,
    }, f, indent=2)

# L1/L2 summary
with open(OUT / "l1_l2_platt_comparison.json", "w") as f:
    json.dump({
        'per_split': l1l2_results,
        'n_l1_pass_raw': sum(1 for v in l1l2_results.values() if v['L1_raw'] <= 0.10),
        'n_l1_pass_platt': sum(1 for v in l1l2_results.values() if v['L1_platt'] <= 0.10),
        'n_l1_near_platt': sum(1 for v in l1l2_results.values() if 0.10 < v['L1_platt'] <= 0.12),
        'analysis_script_sha256': THIS_SHA,
    }, f, indent=2)

# Streak audit
with open(OUT / "long_streak_semantic_audit.csv", "w", newline="") as f:
    if all_classified:
        w = csv.DictWriter(f, fieldnames=['split','ep','start','end','length','route',
                                          'nearest_event_dist','classification'])
        w.writeheader()
        for s in all_classified:
            w.writerow({k: s.get(k, '') for k in w.fieldnames})

# Scheduler contract record
with open(OUT / "v5_scheduler_contract.json", "w") as f:
    json.dump({
        'contract_file': 'src/gripper_attack/v5_scheduler.py',
        'scheduler_sha256': SCHEDULER_SHA,
        'config': {
            'utility_threshold': 0.5,
            'release_veto_threshold': 0.5,
            'regrasp_veto_threshold': 0.5,
            'release_veto_enabled': True,
            'regrasp_veto_enabled': True,
            'minimum_candidate_dwell': 10,
            'persistence_rule': '3-of-5 local-peak one-shot',
        },
        'input_field_mapping': {
            'utility_probability': 'grasp_prob (factorized Student)',
            'release_probability': 'release_prob (factorized Student)',
            'regrasp_probability': 'manipulation_prob (factorized Student, proxy)',
            'candidate_close': 'APPROXIMATE: route_supported AND event_id >= 0 (MISSING from prediction)',
            'valid': 'route_supported',
            'uncertainty_probability': '0.0 (uncertainty_veto_enabled=False)',
        },
        'missing_for_authoritative_replay': ['candidate_close from action_states'],
    }, f, indent=2)

# ── Summary ──
print(f"\n{'='*60}")
print("SUMMARY")
n_l1p = sum(1 for v in l1l2_results.values() if v['L1_platt'] <= 0.10)
print(f"L1 Platt pass: {n_l1p}/12")
print(f"L3 raw emit rate range: {min(v['raw']['emit_rate'] for v in l3_results.values()):.3f} - {max(v['raw']['emit_rate'] for v in l3_results.values()):.3f}")
print(f"L3 platt emit rate range: {min(v['platt']['emit_rate'] for v in l3_results.values()):.3f} - {max(v['platt']['emit_rate'] for v in l3_results.values()):.3f}")
print(f"Long streak classification: PRE_EVENT={counts['PRE_EVENT_ANTICIPATION']} TRUE_BG={counts['TRUE_BACKGROUND_FALSE_POSITIVE']} OTHER={total-counts['PRE_EVENT_ANTICIPATION']-counts['TRUE_BACKGROUND_FALSE_POSITIVE']}")
print(f"L3 STATUS: APPROXIMATE — candidate_close NOT authoritative")
print(f"{'='*60}")
