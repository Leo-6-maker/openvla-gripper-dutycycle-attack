#!/usr/bin/env python3
"""build_clean_phase_dataset.py — clean-only phase/event-labeled dataset builder.

Physical semantics (verified by server smoke):
  OPEN:  qpos decreases (e.g. 0.020 -> 0.001), env_gripper=+1, raw_gripper<0.5
  CLOSE: qpos increases (e.g. 0.021 -> 0.039), env_gripper=-1, raw_gripper>=0.5
"""

from __future__ import annotations
import argparse, csv, os, sys, json
from pathlib import Path
try: import numpy as np
except ImportError: np = None

REPO = Path(os.environ.get("ATTACK_REPO", "/data/liuyu/repos/openvla-gripper-dutycycle-attack-clean-main-20260524"))
_src = str(REPO / "src")
if os.path.isdir(_src): sys.path.insert(0, _src)
try: from gripper_attack.gripper_semantics import raw_gripper_is_open, CANONICAL_OPEN_SEMANTICS_VERSION
except ImportError:
    def raw_gripper_is_open(v): return float(v) < 0.5
    CANONICAL_OPEN_SEMANTICS_VERSION = "dry_run_fallback"

# Shared gripper field fallback order (must match audit_phase_conditioned_vis.py)
GRIP_FIELDS = ("adv_grip", "raw_gripper", "clean_grip", "clean_gripper_action", "adv_gripper_action", "action_gripper")


def get_raw_gripper(row):
    for k in GRIP_FIELDS:
        if k in row and row[k] not in ("", None):
            try: return float(row[k])
            except (ValueError, TypeError): continue
    return None

PHASE_6CLASS = {0:"approach",1:"pregrasp",2:"grasp_formation",3:"stable_grasp_or_lift",4:"carry_or_place",5:"release_or_done"}
PHASE_3CLASS = {0:"pre_grasp",1:"grasp_formation",2:"post_grasp"}

# ── Physical constants (confirmed by server smoke) ──
QPOS_OPEN_MAX = 0.005   # qpos <= this is definitely open
QPOS_CLOSED_MIN = 0.03  # qpos >= this is definitely closed

# Runtime feature names (must match train_phase_selector.py DEFAULT_FEATURES)
RUNTIME_FEATURES = [
    "gripper_command", "gripper_qpos", "gripper_width",
    "eef_x", "eef_y", "eef_z",
    "eef_vx", "eef_vy", "eef_vz",
    "action_dx", "action_dy", "action_dz", "action_gripper",
]

PRIVILEGED_FEATURES = [
    "object_x", "object_y", "object_z",
    "target_x", "target_y", "target_z",
    "eef_object_dist", "object_target_dist",
    "object_lifted", "contact_flag",
]


def parse_bool(v):
    if isinstance(v, bool): return v
    return str(v).strip().lower() in ("true","1","yes")


def _safe_float(v, default=0.0):
    try: return float(v)
    except (ValueError, TypeError): return default


def _finite_diff(arr):
    """Compute finite differences, padding first element with 0."""
    if len(arr) < 2: return np.zeros_like(arr)
    d = np.diff(arr); return np.concatenate([[0.0], d])


def load_traces(run_dirs, task_filter=None, seed_filter=None):
    traces = []
    for d in run_dirs:
        for f in Path(d).rglob("*_trace.csv"):
            traces.append(str(f))
    loaded = []
    for tp in sorted(traces):
        try:
            with open(tp, newline="") as fh:
                rows = list(csv.DictReader(fh))
            if not rows: continue
            r0 = rows[0]
            cond = r0.get("condition","")
            if cond != "clean": continue
            task = r0.get("task","")
            seed = r0.get("seed","0")
            if task_filter and task not in task_filter: continue
            if seed_filter is not None and int(seed) not in seed_filter: continue
            loaded.append((tp, rows))
        except Exception as e:
            print(f"  SKIP: {tp} ({e})")
    return loaded


def detect_events(steps):
    """Heuristic phase event detection using CORRECTED qpos direction.

    OPEN: qpos LOW (near 0), CLOSE: qpos HIGH (near 0.039).
    """
    n = len(steps)
    if n < 5: return {}, np.full(n,-1,int), np.full(n,-1,int)

    # Extract time series
    # env_gripper fallback: if missing, use raw_gripper semantics. If raw also missing, mark incomplete.
    has_env_grip = any(s.get("env_gripper","") not in ("", None) for s in steps)
    has_any_grip = any(get_raw_gripper(s) is not None for s in steps)
    if not has_any_grip:
        return {}, np.full(n,-1,int), np.full(n,-1,int)  # cannot label without gripper

    if has_env_grip:
        env_grip = np.array([_safe_float(s.get("env_gripper", 0.0)) for s in steps])
        is_close_env = env_grip < 0
    else:
        raw_grip = np.array([get_raw_gripper(s) or 0.996 for s in steps])
        is_close_env = np.array([not raw_gripper_is_open(v) for v in raw_grip])

    raw_grip = np.array([get_raw_gripper(s) or 0.996 for s in steps])
    qpos = np.array([_safe_float(s.get("qpos_post_step", s.get("gripper_qpos", 0.03))) for s in steps])
    eef_z = np.array([_safe_float(s.get("eef_z", 0)) for s in steps])
    done = np.array([parse_bool(s.get("done","False")) for s in steps])
    is_open_canon = np.array([raw_gripper_is_open(_safe_float(s.get("raw_gripper", s.get("adv_grip",0.996)))) for s in steps])
    is_open_phys = qpos <= QPOS_OPEN_MAX   # PHYSICAL open
    is_closed_phys = qpos >= QPOS_CLOSED_MIN  # PHYSICAL closed

    # T_gripper_close_onset: first sustained env CLOSE + qpos starts INCREASING
    T_close = None
    streak = 0
    for i in range(n):
        streak = streak+1 if is_close_env[i] else 0
        if streak >= 3:
            T_close = i - 2; break

    # T_grasp_formation_start: close onset + qpos increasing (gripper physically closing)
    T_gform = T_close
    if T_close is not None:
        s0, s1 = max(0,T_close-5), min(n,T_close+10)
        if s1 > s0+1:
            dq = np.diff(qpos[s0:s1])
            for j in range(len(dq)):
                if dq[j] > 0.001:  # qpos INCREASING = closing (corrected)
                    T_gform = s0+j; break

    # T_grasp_lock: stable HIGH qpos after closing
    T_lock = None
    if T_close is not None:
        for i in range(T_close+3, min(n, T_close+30) - 5):
            w = qpos[i:i+5]
            if np.std(w) < 0.0005 and np.mean(w) >= QPOS_CLOSED_MIN:  # HIGH qpos = closed
                T_lock = i; break

    # T_lift_start: EEF z increasing
    T_lift = None
    if T_lock is not None:
        s0 = T_lock; s1 = min(n, s0+40)
        if s1 > s0+3:
            dz = np.diff(eef_z[s0:s1])
            st = 0
            for j,dval in enumerate(dz):
                st = st+1 if dval > 0.001 else 0
                if st >= 3: T_lift = s0+j-2; break

    # T_release_start: natural canonical OPEN + qpos DECREASING after post-grasp
    T_rel = None
    s0 = (T_lift or 0)+10; s1 = n
    if s1 > s0+3:
        st = 0
        for i in range(s0, s1):
            st = st+1 if is_open_canon[i] else 0
            if st >= 3: T_rel = i-2; break
        # If not found by canonical, try physical: qpos dropping below open threshold
        if T_rel is None:
            for i in range(s0, s1):
                if qpos[i] <= QPOS_OPEN_MAX:
                    T_rel = i; break

    T_dn = next((i for i in range(n) if done[i]), None)

    events = {"T_gripper_close_onset":T_close,"T_grasp_formation_start":T_gform,
              "T_grasp_lock":T_lock,"T_lift_start":T_lift,
              "T_release_start":T_rel,"T_done":T_dn,"n_steps":n}

    # Per-step labels
    ph6 = np.full(n, -1, int)
    for i in range(n):
        if T_close is not None and i < T_close: ph6[i]=1
        if T_gform is not None and i>=(T_gform or 0) and (T_lock is None or i<(T_lock or n)): ph6[i]=max(ph6[i],2)
        if T_lock is not None and i>=T_lock and (T_lift is None or i<(T_lift or n)): ph6[i]=3
        if T_lift is not None and i>=T_lift and (T_rel is None or i<(T_rel or n)): ph6[i]=4
        if T_rel is not None and i>=T_rel: ph6[i]=5

    ph3 = np.full(n, -1, int)
    for i in range(n):
        if ph6[i] in (0,1): ph3[i]=0
        elif ph6[i]==2: ph3[i]=1
        elif ph6[i] in (3,4,5): ph3[i]=2

    return events, ph6, ph3


def _build_feature_columns(steps):
    """Extract runtime features from trace rows, computing velocities via finite diff."""
    n = len(steps)
    raw_grip = np.array([_safe_float(s.get("raw_gripper", s.get("adv_grip",""))) for s in steps])
    qpos_vals = np.array([_safe_float(s.get("qpos_post_step", s.get("gripper_qpos",""))) for s in steps])
    eef_x = np.array([_safe_float(s.get("eef_x","")) for s in steps])
    eef_y = np.array([_safe_float(s.get("eef_y","")) for s in steps])
    eef_z = np.array([_safe_float(s.get("eef_z","")) for s in steps])

    eef_vx = _finite_diff(eef_x); eef_vy = _finite_diff(eef_y); eef_vz = _finite_diff(eef_z)

    action_dx = np.zeros(n); action_dy = np.zeros(n); action_dz = np.zeros(n)
    for i,s in enumerate(steps):
        raw = s.get("raw_gripper", s.get("adv_grip",""))
        if raw: raw_grip[i] = _safe_float(raw)
        # Try to extract arm deltas from action columns if present
        for dim, col in [(0,"action_dx"),(1,"action_dy"),(2,"action_dz")]:
            val = s.get(col, s.get(f"action_{dim}", ""))
            if val:
                [action_dx, action_dy, action_dz][dim][i] = _safe_float(val)

    width = np.array([_safe_float(s.get("gripper_width","")) for s in steps])

    features = {}
    for i in range(n):
        feats = {
            "gripper_command": raw_grip[i],
            "gripper_qpos": qpos_vals[i],
            "gripper_width": width[i] if width[i] != 0 else "",
            "eef_x": eef_x[i], "eef_y": eef_y[i], "eef_z": eef_z[i],
            "eef_vx": eef_vx[i], "eef_vy": eef_vy[i], "eef_vz": eef_vz[i],
            "action_dx": action_dx[i], "action_dy": action_dy[i], "action_dz": action_dz[i],
            "action_gripper": raw_grip[i],
        }
        missing = 0
        for k, v in feats.items():
            if v == "" or (isinstance(v, float) and v == 0.0 and k == "gripper_width"):
                missing += 1
        feats["feature_validity"] = "ok" if missing <= 3 else "partial"
        feats["missing_feature_count"] = missing
        features[i] = feats

    return features


def main():
    ap = argparse.ArgumentParser(description="Build clean phase-labeled dataset")
    ap.add_argument("--run-dirs", nargs="+", required=True)
    ap.add_argument("--tasks", nargs="+")
    ap.add_argument("--seeds", type=int, nargs="+")
    ap.add_argument("--output-csv", default="tables/phase_alignment_clean_rollouts.csv")
    ap.add_argument("--summary-csv", default="tables/phase_event_summary.csv")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.dry_run:
        print("DRY RUN: build_clean_phase_dataset")
        print(f"  Run dirs: {args.run_dirs}")
        print(f"  Runtime features: {RUNTIME_FEATURES}")
        print(f"  Privileged excluded: {PRIVILEGED_FEATURES}")
        print(f"  QPOS_OPEN_MAX={QPOS_OPEN_MAX} QPOS_CLOSED_MIN={QPOS_CLOSED_MIN}")
        return

    loaded = load_traces(args.run_dirs, args.tasks, args.seeds)
    print(f"Loaded {len(loaded)} clean traces")

    all_rows = []; summaries = []
    for tp, steps in loaded:
        r0 = steps[0]; task = r0.get("task","?"); seed = r0.get("seed","?")
        events, ph6, ph3 = detect_events(steps)
        features = _build_feature_columns(steps)

        _has_grip = any(get_raw_gripper(s) is not None for s in steps)
        _validity = "incomplete_missing_gripper" if not _has_grip else \
            ("heuristic" if events.get("T_grasp_formation_start") is not None else "incomplete")
        summaries.append({"task":task,"seed":seed,"rollout_id":Path(tp).stem,
            "trace_path":tp,**{k: v if v is not None else "" for k,v in events.items()},
            "label_validity": _validity})

        for i,s in enumerate(steps):
            feats = features.get(i, {})
            row = {"task":task,"seed":seed,"rollout_id":Path(tp).stem,"trace_path":tp,
                "policy_step": int(_safe_float(s.get("policy_step", str(i)), i)),
                "raw_gripper": get_raw_gripper(s) if get_raw_gripper(s) is not None else "",
                "env_gripper":s.get("env_gripper",""), "done":s.get("done","False"),
                "phase_label_6class":PHASE_6CLASS.get(int(ph6[i]),"unknown"), "phase_label_6class_id":int(ph6[i]),
                "phase_label_3class":PHASE_3CLASS.get(int(ph3[i]),"unknown"), "phase_label_3class_id":int(ph3[i]),
                "T_gripper_close_onset":events.get("T_gripper_close_onset",""),
                "T_grasp_formation_start":events.get("T_grasp_formation_start",""),
                "T_grasp_lock":events.get("T_grasp_lock",""),"T_lift_start":events.get("T_lift_start",""),
                "T_release_start":events.get("T_release_start",""),"T_done":events.get("T_done",""),
                "label_confidence":"medium","label_source":"heuristic",
                "label_validity":"heuristic" if events.get("T_grasp_formation_start") is not None else "incomplete",
            }
            for k in RUNTIME_FEATURES:
                row[f"feat_{k}"] = feats.get(k, "")
            row["feature_validity"] = feats.get("feature_validity", "unknown")
            row["missing_feature_count"] = feats.get("missing_feature_count", 0)
            all_rows.append(row)
        print(f"  {task} seed{seed}: T_close={events.get('T_gripper_close_onset')} T_gform={events.get('T_grasp_formation_start')} T_lock={events.get('T_grasp_lock')} T_lift={events.get('T_lift_start')} T_rel={events.get('T_release_start')}")

    if all_rows:
        os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
        with open(args.output_csv,"w",newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys())); w.writeheader(); w.writerows(all_rows)
        print(f"Wrote {len(all_rows)} rows to {args.output_csv}")
    if summaries:
        os.makedirs(os.path.dirname(args.summary_csv) or ".", exist_ok=True)
        with open(args.summary_csv,"w",newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(summaries[0].keys())); w.writeheader(); w.writerows(summaries)
        print(f"Wrote {len(summaries)} summaries to {args.summary_csv}")


if __name__ == "__main__":
    main()
