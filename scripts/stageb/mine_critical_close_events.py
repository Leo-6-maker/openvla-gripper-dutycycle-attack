#!/usr/bin/env python3
"""Phase 1: Mine critical-CLOSE events from V4-aligned clean traces.
Output: critical_close_event_candidates.csv

Classification:
  P0: clean_close_steps >= 2 in 5-step event window, clean_open_count <= 2,
      phase in {grasp_transition, early_transport}, not dummy-wait zone
  P1: clean_close_steps >= 1, phase in {grasp_transition, early_transport, transport}
  P2: diagnostic only (place_or_done, or very sparse close)
"""

import csv, json, glob, os, sys
from collections import defaultdict
from pathlib import Path

SCAN_DIR = '/data/liuyu/outputs/stageb_s20m4_clean_scan_20260613'
REGISTRY = '/data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607/tables/layer3_registry.csv'
OUT = os.environ.get('OUT', '/data/liuyu/outputs/stageb_v5_critical_close_overnight_20260613_0100')

os.makedirs(os.path.join(OUT, 'tables'), exist_ok=True)

# ── Phase estimation by normalized step progress ──
# LIBERO Object: grasp → transport → place
# Heuristic phases based on normalized position in episode:
#   0.00-0.10: dummy_wait
#   0.10-0.35: grasp_transition
#   0.35-0.65: transport (split into early_transport 0.35-0.50, late_transport 0.50-0.65)
#   0.65-0.85: preplace
#   0.85-1.00: place_or_done

def estimate_phase(step, n_steps):
    frac = step / max(n_steps, 1)
    if frac < 0.10:
        return 'dummy_wait_or_init'
    elif frac < 0.35:
        return 'grasp_transition'
    elif frac < 0.50:
        return 'early_transport'
    elif frac < 0.65:
        return 'transport'
    elif frac < 0.85:
        return 'preplace'
    else:
        return 'place_or_done'

# ── Load RAND status from registry ──
print('Loading registry...')
rand_status = {}
if os.path.exists(REGISTRY):
    with open(REGISTRY) as f:
        for r in csv.DictReader(f):
            cid = r.get('parent_id', '')
            rand_status[cid] = {
                'rand_stability': r.get('rand_stability', ''),
                'status': r.get('status', ''),
            }
print(f'  {len(rand_status)} registry entries')

# ── Scan all traces ──
traces = sorted(glob.glob(os.path.join(SCAN_DIR, 'trace_*.csv')))
summaries = sorted(glob.glob(os.path.join(SCAN_DIR, 'summary_*.json')))
print(f'Traces: {len(traces)}, Summaries: {len(summaries)}')

# Build task→summary lookup
summary_by_id = {}
for sf in summaries:
    try:
        s = json.load(open(sf))
        cid = f"{s['task']}_s{s['state_id']}_w{s['window_start']}_{s['window_end']}"
        summary_by_id[cid] = s
    except Exception:
        pass

# ── Mine events ──
candidates = []

for tf in traces:
    try:
        rows = list(csv.DictReader(open(tf)))
    except Exception:
        continue
    if not rows:
        continue

    # Parse task/state_id from first row
    task = rows[0].get('task', '')
    state_id = rows[0].get('state_id', '')
    if not task:
        continue

    n_steps = len(rows)
    gripper_vals = [float(r['clean_gripper_env']) for r in rows]
    qpos_before = [float(r.get('gripper_qpos_before', 0) or 0) for r in rows]
    qpos_after = [float(r.get('gripper_qpos_after', 0) or 0) for r in rows]

    # Find close onsets and streaks
    close_events = []

    # Close onsets: OPEN -> CLOSE
    for i in range(1, n_steps):
        if gripper_vals[i] > 0.5 and gripper_vals[i-1] < -0.5:
            phase = estimate_phase(i, n_steps)
            if phase != 'dummy_wait_or_init':
                close_events.append({
                    'type': 'close_onset',
                    'center_step': i,
                    'close_steps_in_event': [],
                    'phase': phase,
                })

    # Close streaks >= 2
    in_streak = False
    streak_start = 0
    for i, v in enumerate(gripper_vals):
        is_close = v > 0.5
        if is_close and not in_streak:
            streak_start = i
            in_streak = True
        elif not is_close and in_streak:
            slen = i - streak_start
            if slen >= 2:
                phase = estimate_phase(streak_start + slen // 2, n_steps)
                if phase != 'dummy_wait_or_init':
                    cs_list = list(range(streak_start, i))
                    close_events.append({
                        'type': 'close_streak',
                        'center_step': streak_start + slen // 2,
                        'close_steps_in_event': cs_list,
                        'phase': phase,
                        'streak_len': slen,
                    })
            in_streak = False
    if in_streak:
        slen = n_steps - streak_start
        if slen >= 2:
            phase = estimate_phase(streak_start + slen // 2, n_steps)
            if phase != 'dummy_wait_or_init':
                cs_list = list(range(streak_start, n_steps))
                close_events.append({
                    'type': 'close_streak',
                    'center_step': streak_start + slen // 2,
                    'close_steps_in_event': cs_list,
                    'phase': phase,
                    'streak_len': slen,
                })

    # For each event, create windows of lengths 3, 5, 7
    for evt in close_events:
        center = evt['center_step']
        for wlen in [5, 3, 7]:
            half = wlen // 2
            ws = max(10, center - half)  # skip dummy wait
            we = min(n_steps - 1, center + half)
            if we - ws + 1 < wlen:
                continue  # window too short

            # Count CLOSE/OPEN in window
            window_grips = gripper_vals[ws:we+1]
            clean_close_steps = sum(1 for v in window_grips if v > 0.5)
            clean_open_steps = sum(1 for v in window_grips if v < -0.5)
            clean_close_streak = 0
            cur = 0
            for v in window_grips:
                if v > 0.5:
                    cur += 1
                    clean_close_streak = max(clean_close_streak, cur)
                else:
                    cur = 0

            # Qpos dynamics
            qpos_b_mean = sum(qpos_before[ws:we+1]) / len(qpos_before[ws:we+1])
            qpos_a_vals = qpos_after[ws:we+1]
            qpos_delta = qpos_a_vals[-1] - qpos_a_vals[0] if qpos_a_vals else 0

            # Phase
            phase = estimate_phase(center, n_steps)

            # Candidate ID
            candidate_id = f"{task}_s{state_id}_w{ws}_{we}_c{center}_{evt['type']}"

            # Query RAND registry (approximate match)
            # The clean traces use fixed w0_10 windows, so match by task+state
            base_cid = f"{task}_s{state_id}"
            rand_info = {}
            for key, val in rand_status.items():
                if key.startswith(base_cid):
                    rand_info = val
                    break

            # Scoring
            score = 0
            if phase == 'grasp_transition':
                score += 3
            elif phase == 'early_transport':
                score += 2
            if clean_close_streak >= 2:
                score += 2
            rand_stab = rand_info.get('rand_stability', '')
            if 'STRICT' in rand_stab:
                score += 2
            elif 'USABLE' in rand_stab:
                score += 1
            if qpos_delta > 0.01:
                score += 1  # actual physical closing
            if clean_open_steps >= 3:
                score -= 3  # naturally open window
            if phase in ('preplace', 'place_or_done'):
                score -= 3

            # P0/P1/P2 classification
            # Discovery: UNKNOWN rand is accepted, marked rand_verified=False
            is_rand_verified = ('STRICT' in rand_stab.upper() or 'USABLE' in rand_stab.upper())

            if (clean_close_steps >= 2 and clean_open_steps <= 2
                    and phase in ('grasp_transition', 'early_transport')):
                tier = 'P0'
            elif (clean_close_steps >= 1
                    and phase in ('grasp_transition', 'early_transport', 'transport')):
                tier = 'P1'
            else:
                tier = 'P2'

            candidates.append({
                'candidate_id': candidate_id,
                'task': task,
                'state_id': state_id,
                'event_type': evt['type'],
                'event_center_step': center,
                'window_start': ws,
                'window_end': we,
                'window_len': wlen,
                'phase': phase,
                'clean_close_steps': clean_close_steps,
                'clean_open_steps': clean_open_steps,
                'clean_close_streak': clean_close_streak,
                'close_onset_present': int(evt['type'] == 'close_onset'),
                'qpos_before_mean': round(qpos_b_mean, 6),
                'qpos_after_delta': round(qpos_delta, 6),
                'rand_status': rand_info.get('rand_stability', 'UNKNOWN'),
                'registry_status': rand_info.get('status', ''),
                'priority_score': score,
                'rand_verified': is_rand_verified,
                'tier': tier,
                'n_trace_steps': n_steps,
            })

print(f'\nTotal raw candidates: {len(candidates)}')

# Deduplicate: keep best (highest score, then longest window) per approximate window
# Group by task+state, dedup overlapping windows
deduped = []
seen_windows = set()
for c in sorted(candidates, key=lambda x: (-x['priority_score'], -x['window_len'])):
    key = (c['task'], c['state_id'], c['window_start'], c['window_end'])
    # Allow overlapping windows from different events if center_step differs significantly
    overlap = False
    for (t, s, ws2, we2) in seen_windows:
        if t == c['task'] and s == c['state_id']:
            if not (c['window_end'] < ws2 or c['window_start'] > we2):
                overlap = True
                break
    if not overlap:
        seen_windows.add(key)
        deduped.append(c)

print(f'After dedup: {len(deduped)}')

# Sort by score and break ties by close_streak
deduped.sort(key=lambda x: (-x['priority_score'], -x['clean_close_streak'], -x['clean_close_steps']))

# ── Output ──
fieldnames = [
    'rank', 'candidate_id', 'task', 'state_id', 'event_type', 'event_center_step',
    'window_start', 'window_end', 'window_len', 'phase', 'tier',
    'clean_close_steps', 'clean_open_steps', 'clean_close_streak',
    'close_onset_present', 'qpos_before_mean', 'qpos_after_delta',
    'rand_status', 'rand_verified', 'registry_status', 'priority_score', 'n_trace_steps',
]

out_path = os.path.join(OUT, 'tables', 'critical_close_event_candidates.csv')
with open(out_path, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
    w.writeheader()
    for i, c in enumerate(deduped):
        c['rank'] = i + 1
        w.writerow(c)

# Also copy to repo tables/
repo_tables = '/data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607/tables'
os.makedirs(repo_tables, exist_ok=True)
repo_path = os.path.join(repo_tables, 's20d_v5_critical_close_event_candidates.csv')
with open(repo_path, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
    w.writeheader()
    for i, c in enumerate(deduped):
        c['rank'] = i + 1
        w.writerow(c)

# ── Summary ──
p0 = sum(1 for c in deduped if c['tier'] == 'P0')
p1 = sum(1 for c in deduped if c['tier'] == 'P1')
p2 = sum(1 for c in deduped if c['tier'] == 'P2')
print(f'\n=== Phase 1 Results ===')
print(f'P0: {p0}, P1: {p1}, P2: {p2}')
print(f'Top 10:')
for c in deduped[:10]:
    print(f'  #{c["rank"]:2d} {c["candidate_id"]:60s} phase={c["phase"]:20s} '
          f'tier={c["tier"]} score={c["priority_score"]} '
          f'close={c["clean_close_steps"]}/{c["clean_close_streak"]} '
          f'open={c["clean_open_steps"]} rand={c["rand_status"]}')

print(f'\nOutputs:')
print(f'  {out_path}')
print(f'  {repo_path}')

# Phase gate
if p0 + p1 < 5:
    print('\n*** GATE FAIL: < 5 P0/P1 candidates found ***')
    print('ACTION: Stop GPU smoke, report NO_CRITICAL_CLOSE_EVENTS_FOUND')
else:
    print(f'\n*** GATE PASS: {p0+p1} P0/P1 candidates ***')
    print(f'Proceed to Phase 2: eps6 event-window smoke on top 30 candidates')
