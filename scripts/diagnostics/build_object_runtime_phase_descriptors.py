#!/usr/bin/env python3
"""build_object_runtime_phase_descriptors.py — CPU-only offline phase localization audit.

Builds runtime phase proxy events from Object clean rollouts, then classifies
candidate attack windows into phase bins.

Inputs: NPZ (X_raw), meta CSV, phase_event_summary CSV, candidate CSVs.
Outputs: runtime phase events table, window phase descriptors, candidate ranking.

Do NOT use GPU. Do NOT train student. Do NOT run VIS.
"""

from __future__ import annotations
import argparse, csv, os, sys
from collections import defaultdict
import numpy as np

GRIPPER_IDX = 0; QPOS_IDX = 1; EEF_VX = 7; EEF_VY = 8; EEF_VZ = 9; EEF_X = 4; EEF_Y = 5; EEF_Z = 6

TASK_KEY_MAP = {
    'pick_up_the_alphabet_soup_and_place_it_in_the_basket': 'alphabet_soup',
    'pick_up_the_cream_cheese_and_place_it_in_the_basket': 'cream_cheese',
    'pick_up_the_salad_dressing_and_place_it_in_the_basket': 'salad_dressing',
    'pick_up_the_bbq_sauce_and_place_it_in_the_basket': 'bbq_sauce',
    'pick_up_the_ketchup_and_place_it_in_the_basket': 'ketchup',
    'pick_up_the_tomato_sauce_and_place_it_in_the_basket': 'tomato_sauce',
    'pick_up_the_butter_and_place_it_in_the_basket': 'butter',
    'pick_up_the_milk_and_place_it_in_the_basket': 'milk',
    'pick_up_the_chocolate_pudding_and_place_it_in_the_basket': 'chocolate_pudding',
    'pick_up_the_orange_juice_and_place_it_in_the_basket': 'orange_juice',
}


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz-path", default="data/detector/object_clean_sequences_v3.npz")
    ap.add_argument("--meta-csv", default="data/detector/object_clean_sequences_v3_meta.csv")
    ap.add_argument("--phase-csv", default="tables/object_phase_event_summary.csv")
    ap.add_argument("--candidate-csvs", nargs="+", default=[
        "tables/object_teacher_delay50_vis_smoke_candidates.csv",
        "tables/object_teacher_delay50_vis_smoke_batch1.csv",
    ])
    ap.add_argument("--sweep-csv", default="tables/object_detector_negative_delay_sweep.csv")
    ap.add_argument("--output-events", default="tables/object_runtime_phase_events.csv")
    ap.add_argument("--output-descriptors", default="tables/object_teacher_window_phase_descriptors.csv")
    ap.add_argument("--output-ranking", default="tables/object_phase_response_candidate_ranking.csv")
    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args()


def find_first_sustained(raw_gc, K=2, threshold=0.5):
    streak = 0
    for t in range(len(raw_gc)):
        if raw_gc[t] < threshold:
            streak += 1
            if streak >= K: return t - K + 1
        else: streak = 0
    return None


def find_qpos_motion(qpos, delta_min=0.002):
    """First step where qpos changes by at least delta_min from initial."""
    if len(qpos) < 3: return None
    baseline = qpos[0]
    for t in range(1, len(qpos)):
        if abs(qpos[t] - baseline) >= delta_min:
            return t
    return None


def find_qpos_min(qpos, start=0):
    if start >= len(qpos): return None
    return int(np.argmin(qpos[start:])) + start


def find_qpos_stable(qpos, after_step, K=3, eps=0.0005):
    if after_step is None or after_step >= len(qpos): return None
    for t in range(after_step + 1, len(qpos) - K):
        window = qpos[t:t+K]
        if max(window) - min(window) <= eps:
            return t
    return None


def find_eef_slowdown(eef_speed, percentile=20):
    if len(eef_speed) < 5: return None
    threshold = np.percentile(eef_speed, percentile)
    for t in range(1, len(eef_speed)):
        if eef_speed[t] < threshold and eef_speed[t-1] >= threshold:
            return t
    return None


def classify_window(ws, we, T, raw_gc, qpos, eef_speed, events):
    """Classify window into a phase bin proxy."""
    open_ratio = np.mean(raw_gc[ws:we+1] < 0.5)
    tg = events.get("T_gform"); tm = events.get("T_qpos_motion")
    ts = events.get("T_qpos_stable"); te = events.get("T_eef_slowdown")

    # Natural-open confounded
    if open_ratio > 0.3:
        return "natural_open_or_release_proxy"

    # Stable grasp: window after qpos has stabilized or well after T_gform
    if ts is not None and ws >= ts:
        return "stable_grasp_or_lift_proxy"
    if tg is not None and ws > tg + 10:
        return "stable_grasp_or_lift_proxy"

    # Grasp formation zone: overlaps qpos motion -> stability
    if tm is not None and ts is not None and we >= tm and ws <= ts:
        return "grasp_formation_pre_lock_proxy"
    if tg is not None and tg - 20 <= ws <= tg + 10:
        return "grasp_formation_pre_lock_proxy"

    # Pre-grasp CLOSED: gripper closed, before qpos motion
    if open_ratio <= 0.1:
        if tm is not None and we < tm:
            return "pre_grasp_closed_proxy"
        if tg is not None and we < tg:
            return "pre_grasp_closed_proxy"
        return "pre_grasp_closed_proxy"

    # Approach: far from grasp, EEF moving
    if tg is not None and ws < tg - 20:
        if eef_speed[ws:we+1].mean() > 0.003:
            return "approach_far_proxy"
        return "approach_near_proxy"

    return "unknown_proxy"


def main():
    args = parse_args()

    # ── Load NPZ ──
    data = np.load(args.npz_path, allow_pickle=True)
    Xr = data["X_raw"]; mask_all = data["mask"]
    ep_ids = list(data.get("episode_ids", []))
    print(f"Loaded {len(ep_ids)} episodes from NPZ")

    # ── Load meta ──
    meta = {}
    if os.path.exists(args.meta_csv):
        with open(args.meta_csv, newline="") as f:
            for r in csv.DictReader(f): meta[r["episode_id"]] = r

    # ── Load phase events ──
    phases = {}
    if os.path.exists(args.phase_csv):
        with open(args.phase_csv, newline="") as f:
            for r in csv.DictReader(f):
                if r.get("run_id"): phases[r["run_id"]] = r

    # ── 1. Build runtime phase events for each episode ──
    print("\nBuilding runtime phase events...")
    runtime_events = []
    for orig_idx, eid in enumerate(ep_ids):
        eid_str = str(eid)
        ep_meta = meta.get(eid_str, {})
        tg_str = ep_meta.get("T_gform", "")
        tg = int(tg_str) if tg_str else None
        task = ep_meta.get("task_name", "?")
        task_key = TASK_KEY_MAP.get(task, task[:20])
        state_id = ep_meta.get("state_id", "?")

        m = mask_all[orig_idx]; T = int(m.sum())
        raw_gc = Xr[orig_idx, :T, GRIPPER_IDX]
        qpos = Xr[orig_idx, :T, QPOS_IDX]
        eef_vx = Xr[orig_idx, :T, EEF_VX]; eef_vy = Xr[orig_idx, :T, EEF_VY]
        eef_vz = Xr[orig_idx, :T, EEF_VZ]
        eef_speed = np.sqrt(eef_vx**2 + eef_vy**2 + eef_vz**2)

        # Detect proxy events
        t_open = find_first_sustained(raw_gc)
        t_qmotion = find_qpos_motion(qpos)
        t_qmin = find_qpos_min(qpos, t_open if t_open is not None else 0)
        t_qstable = find_qpos_stable(qpos, t_qmin)
        t_eefflow = find_eef_slowdown(eef_speed)

        validity = "full"
        if tg is None: validity = "missing_T_gform"
        if t_open is None: validity = "missing_open_onset"

        runtime_events.append(dict(
            episode_id=eid_str, task_key=task_key, state_id=state_id,
            T=tg if tg is not None else "",
            T_open_onset=t_open if t_open is not None else "",
            T_qpos_motion=t_qmotion if t_qmotion is not None else "",
            T_qpos_min=t_qmin if t_qmin is not None else "",
            T_qpos_stable=t_qstable if t_qstable is not None else "",
            T_eef_slowdown=t_eefflow if t_eefflow is not None else "",
            T_eef_motion_peak=int(np.argmax(eef_speed)) if len(eef_speed) > 0 else "",
            T_done=T,
            phase_event_validity=validity,
        ))

    # Write events
    ev_fields = ["episode_id","task_key","state_id","T","T_open_onset","T_qpos_motion",
                 "T_qpos_min","T_qpos_stable","T_eef_slowdown","T_eef_motion_peak",
                 "T_done","phase_event_validity"]
    os.makedirs(os.path.dirname(args.output_events) or ".", exist_ok=True)
    with open(args.output_events, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=ev_fields, extrasaction="ignore")
        w.writeheader(); w.writerows(runtime_events)
    print(f"Wrote {len(runtime_events)} episodes to {args.output_events}")

    # ── 2. Build event lookup ──
    ev_lookup = {r["episode_id"]: r for r in runtime_events}

    # ── 3. Load candidate windows and classify ──
    all_candidates = []
    for csv_path in args.candidate_csvs:
        if os.path.exists(csv_path):
            with open(csv_path, newline="") as f:
                candidates = list(csv.DictReader(f))
            source = "teacher_oracle" if "teacher" in csv_path else "detector_pred"
            for c in candidates:
                c["candidate_source"] = source
            all_candidates.extend(candidates)
            print(f"  Loaded {len(candidates)} from {csv_path}")

    # Add sweep if available
    if os.path.exists(args.sweep_csv):
        with open(args.sweep_csv, newline="") as f:
            sweep = list(csv.DictReader(f))
        for s in sweep:
            s["candidate_source"] = "negative_delay_sweep"
        all_candidates.extend(sweep)
        print(f"  Loaded {len(sweep)} from sweep CSV")

    # Deduplicate
    seen = set(); deduped = []
    for c in all_candidates:
        key = (c.get("episode_id",""), c.get("window_start",""), c.get("delay",""))
        if key not in seen:
            seen.add(key); deduped.append(c)
    print(f"  Total unique candidates: {len(deduped)}")

    # ── 4. Classify each candidate window ──
    descriptors = []
    for c in deduped:
        eid = c.get("episode_id", "")
        ws_str = c.get("window_start", ""); we_str = c.get("window_end", "")
        if not ws_str or not we_str: continue
        ws = int(ws_str); we = int(we_str)

        ev = ev_lookup.get(eid, {})
        tg_str = c.get("T_gform", ev.get("T", ""))
        tg = int(tg_str) if tg_str else None

        # Find orig_idx for this episode
        orig_idx = None
        for i, e in enumerate(ep_ids):
            if str(e) == eid:
                orig_idx = i; break
        if orig_idx is None: continue

        m = mask_all[orig_idx]; T = int(m.sum())
        if ws < 0 or we >= T: continue

        raw_gc = Xr[orig_idx, :T, GRIPPER_IDX]
        qpos = Xr[orig_idx, :T, QPOS_IDX]
        eef_vx = Xr[orig_idx, :T, EEF_VX]; eef_vy = Xr[orig_idx, :T, EEF_VY]
        eef_vz = Xr[orig_idx, :T, EEF_VZ]
        eef_speed = np.sqrt(eef_vx**2 + eef_vy**2 + eef_vz**2)
        eef_x = Xr[orig_idx, :T, EEF_X]; eef_y = Xr[orig_idx, :T, EEF_Y]
        eef_z = Xr[orig_idx, :T, EEF_Z]

        w_gc = raw_gc[ws:we+1]; w_qpos = qpos[ws:we+1]; w_speed = eef_speed[ws:we+1]
        open_cnt = int((w_gc < 0.5).sum()); open_ratio = round(open_cnt / len(w_gc), 4)

        # Build events dict for classification
        events = dict(
            T_gform=tg,
            T_qpos_motion=int(ev["T_qpos_motion"]) if ev.get("T_qpos_motion") else None,
            T_qpos_stable=int(ev["T_qpos_stable"]) if ev.get("T_qpos_stable") else None,
            T_eef_slowdown=int(ev["T_eef_slowdown"]) if ev.get("T_eef_slowdown") else None,
        )

        phase_bin = classify_window(ws, we, T, raw_gc, qpos, eef_speed, events)

        task = c.get("task_name", c.get("task_key", "?"))
        task_key = TASK_KEY_MAP.get(task, task) if len(task) > 20 else task

        # Distances to events
        def dist(t_val):
            if t_val is not None and ws is not None: return ws - t_val
            return ""

        descriptors.append(dict(
            episode_id=eid, task_key=task_key, state_id=c.get("state_id","?"),
            split=c.get("split",""), window_start=ws, window_end=we,
            actual_window_len=we-ws+1,
            T_gform=tg if tg is not None else "",
            relative_lead=dist(tg),
            candidate_source=c.get("candidate_source",""),
            delay=c.get("delay",""),
            clean_open_count=open_cnt, clean_open_ratio=open_ratio,
            raw_gripper_mean=round(float(w_gc.mean()), 4),
            qpos_start=round(float(w_qpos[0]), 6), qpos_end=round(float(w_qpos[-1]), 6),
            qpos_min=round(float(w_qpos.min()), 6), qpos_max=round(float(w_qpos.max()), 6),
            qpos_delta_abs=round(float(w_qpos.max()-w_qpos.min()), 6),
            qpos_opening_proxy=round(float(w_qpos[0]-w_qpos.min()), 6),
            qpos_velocity_mean=round(float(np.abs(np.diff(w_qpos)).mean()), 6) if len(w_qpos)>1 else 0,
            eef_speed_mean=round(float(w_speed.mean()), 6),
            eef_speed_max=round(float(w_speed.max()), 6),
            eef_displacement=round(float(np.sqrt((eef_x[we]-eef_x[ws])**2+(eef_y[we]-eef_y[ws])**2+(eef_z[we]-eef_z[ws])**2)), 6),
            eef_z_delta=round(float(eef_z[we]-eef_z[ws]), 6),
            dist_to_T_open_onset=dist(int(ev["T_open_onset"]) if ev.get("T_open_onset") else None),
            dist_to_T_qpos_motion=dist(int(ev["T_qpos_motion"]) if ev.get("T_qpos_motion") else None),
            dist_to_T_qpos_min=dist(int(ev["T_qpos_min"]) if ev.get("T_qpos_min") else None),
            dist_to_T_qpos_stable=dist(int(ev["T_qpos_stable"]) if ev.get("T_qpos_stable") else None),
            dist_to_T_eef_slowdown=dist(int(ev["T_eef_slowdown"]) if ev.get("T_eef_slowdown") else None),
            dist_to_T_done=dist(T),
            phase_bin_proxy=phase_bin,
            notes="",
        ))

    # Write descriptors
    d_fields = ["episode_id","task_key","state_id","split","window_start","window_end",
                "actual_window_len","T_gform","relative_lead","candidate_source","delay",
                "clean_open_count","clean_open_ratio","raw_gripper_mean",
                "qpos_start","qpos_end","qpos_min","qpos_max","qpos_delta_abs",
                "qpos_opening_proxy","qpos_velocity_mean",
                "eef_speed_mean","eef_speed_max","eef_displacement","eef_z_delta",
                "dist_to_T_open_onset","dist_to_T_qpos_motion","dist_to_T_qpos_min",
                "dist_to_T_qpos_stable","dist_to_T_eef_slowdown","dist_to_T_done",
                "phase_bin_proxy","notes"]
    with open(args.output_descriptors, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=d_fields, extrasaction="ignore")
        w.writeheader(); w.writerows(descriptors)
    print(f"\nWrote {len(descriptors)} descriptors to {args.output_descriptors}")

    # ── 5. Phase bin distribution ──
    bin_counts = defaultdict(int)
    for d in descriptors: bin_counts[d["phase_bin_proxy"]] += 1
    print("\nPhase bin distribution:")
    for b, n in sorted(bin_counts.items(), key=lambda x: -x[1]):
        print(f"  {b:40s}: {n:4d}")

    # ── 6. Candidate ranking ──
    BIN_PRIORITY = {
        "grasp_formation_pre_lock_proxy": 1,
        "pre_grasp_closed_proxy": 2,
        "approach_near_proxy": 3,
        "approach_far_proxy": 4,
        "stable_grasp_or_lift_proxy": 5,
        "natural_open_or_release_proxy": 6,
        "unknown_proxy": 7,
    }

    # Filter to strict-eligible only, then rank
    eligible = [d for d in descriptors if d["clean_open_ratio"] <= 0.1]
    eligible.sort(key=lambda d: (BIN_PRIORITY.get(d["phase_bin_proxy"], 99), d["clean_open_ratio"]))

    # Per-task top picks
    per_task = defaultdict(list)
    for d in eligible: per_task[d["task_key"]].append(d)

    ranking = []
    for task_key in sorted(per_task):
        for i, d in enumerate(per_task[task_key][:3]):  # top 3 per task
            d["rank"] = i + 1; d["reason"] = f"phase_bin={d['phase_bin_proxy']}_open={d['clean_open_ratio']}"
            ranking.append(d)

    # Take top 20 overall
    ranking.sort(key=lambda d: (BIN_PRIORITY.get(d["phase_bin_proxy"], 99), d["clean_open_ratio"]))
    ranking = ranking[:20]

    r_fields = ["episode_id","task_key","state_id","window_start","window_end","T_gform",
                "relative_lead","phase_bin_proxy","clean_open_ratio",
                "candidate_source","rank","reason"]
    with open(args.output_ranking, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=r_fields, extrasaction="ignore")
        w.writeheader(); w.writerows(ranking)

    print(f"\nTop 20 phase-based candidates:")
    for i, r in enumerate(ranking[:10]):
        print(f"  {i+1:2d}. {r['task_key']:20s} [{r['window_start']},{r['window_end']}] "
              f"{r['phase_bin_proxy']:35s} lead={r['relative_lead']} open={r['clean_open_ratio']}")

    # ── 7. Report on 3 running VIS windows ──
    running_vis = [
        ("alphabet_soup", 3, 20),
        ("butter", 29, 46),
        ("ketchup", 16, 33),
    ]
    print("\nRunning VIS window phase analysis:")
    for task_key, ws, we in running_vis:
        matches = [d for d in descriptors if d["task_key"] == task_key
                   and d["window_start"] == ws and d["window_end"] == we]
        if matches:
            d = matches[0]
            print(f"  {task_key:20s} [{ws},{we}]: {d['phase_bin_proxy']:35s} "
                  f"lead={d['relative_lead']} open={d['clean_open_ratio']}")
        else:
            print(f"  {task_key:20s} [{ws},{we}]: NOT FOUND in descriptors")

    print(f"\nDone. Outputs: {args.output_events}, {args.output_descriptors}, {args.output_ranking}")


if __name__ == "__main__":
    main()
