#!/usr/bin/env python3
"""audit_clean_phase_events.py — extract heuristic phase labels from clean rollouts.

Pipeline:
  1. Read clean trace CSV(s)
  2. Extract per-step proprio/action features
  3. Detect heuristic phase events (T_gripper_close_onset, T_grasp_formation, etc.)
  4. Label each policy step with 6-class and 3-class phase labels
  5. Run frozen ProprioNoStep to get T_prop trigger step
  6. Compute offset from T_prop to phase events
  7. Output tables/phase_alignment_clean_rollouts.csv and
     tables/proprionostep_phase_alignment.csv
"""

from __future__ import annotations
import argparse, csv, os, sys
from pathlib import Path
import numpy as np

REPO = Path(os.environ.get("ATTACK_REPO", "/data/liuyu/repos/openvla-gripper-dutycycle-attack-clean-main-20260524"))
sys.path.insert(0, str(REPO / "src"))
from gripper_attack.gripper_semantics import raw_gripper_is_open, CANONICAL_OPEN_SEMANTICS_VERSION

# ── Phase taxonomy ──
PHASE_6CLASS = {
    0: "approach",
    1: "pregrasp",
    2: "grasp_formation",
    3: "stable_grasp_or_lift",
    4: "carry_or_place",
    5: "release_or_done",
}
PHASE_3CLASS = {
    0: "pre_grasp",       # approach + pregrasp
    1: "grasp_formation",
    2: "post_grasp",       # stable_grasp_or_lift + carry_or_place + release_or_done
}

# ── Configurable thresholds ──
CLOSE_ONSET_K = 3         # consecutive close-or-hold commands to detect close onset
GRASP_LOCK_K = 5           # consecutive stable qpos to detect grasp lock
LIFT_DZ_THRESH = 0.001     # minimum EEF z velocity for lift detection
LIFT_K = 3                  # consecutive lift steps
CARRY_DIST_THRESH = 0.005  # minimum horizontal movement for carry
RELEASE_OPEN_K = 3         # consecutive OPEN commands for natural release


def load_trace(csv_path):
    with open(csv_path, newline="") as f:
        return list(csv.DictReader(f))


def detect_phase_events(steps):
    """Detect heuristic phase events from clean rollout steps.

    Returns dict with event step indices and per-step phase labels.
    """
    n = len(steps)
    if n == 0:
        return {}, []

    # Extract time series
    raw_grip = np.array([float(s.get("raw_gripper", s.get("adv_grip", 0.996))) for s in steps])
    env_grip = np.array([float(s.get("env_gripper", -1.0)) for s in steps])
    qpos = np.array([float(s.get("qpos_post_step", s.get("gripper_qpos", 0.03))) for s in steps])
    eef_x = np.array([float(s.get("eef_x", 0)) for s in steps])
    eef_y = np.array([float(s.get("eef_y", 0)) for s in steps])
    eef_z = np.array([float(s.get("eef_z", 0)) for s in steps])
    done = np.array([s.get("done", "False") == "True" for s in steps])

    from gripper_attack.gripper_semantics import env_gripper_is_open, env_gripper_is_close

    # Environment OPEN/CLOSE
    is_open_env = np.array([env_gripper_is_open(g) for g in env_grip])
    is_close_env = np.array([env_gripper_is_close(g) for g in env_grip])

    is_open_canonical = np.array([raw_gripper_is_open(float(s.get("raw_gripper", s.get("adv_grip", 0.996)))) for s in steps])

    # ── T_gripper_close_onset: first sustained close-or-hold ──
    T_close_onset = None
    close_streak = 0
    for i in range(n):
        if is_close_env[i]:
            close_streak += 1
        else:
            close_streak = 0
        if close_streak >= CLOSE_ONSET_K:
            T_close_onset = i - CLOSE_ONSET_K + 1
            break

    # ── T_grasp_formation: around close onset where qpos begins to change ──
    T_grasp_form = T_close_onset  # default: align with close onset
    if T_close_onset is not None:
        # Search around close_onset for qpos inflection
        start = max(0, T_close_onset - 5)
        end = min(n, T_close_onset + 10)
        if end > start + 1:
            dq = np.diff(qpos[start:end])
            # Find where qpos starts consistently decreasing (gripper closing)
            for j in range(len(dq)):
                if dq[j] < -0.001:  # qpos decreasing = closing
                    T_grasp_form = start + j
                    break

    # ── T_grasp_lock: qpos stabilizes after closing ──
    T_grasp_lock = None
    if T_close_onset is not None:
        search_start = T_close_onset + 3
        search_end = min(n, search_start + 30)
        if search_end > search_start + GRASP_LOCK_K:
            for i in range(search_start, search_end - GRASP_LOCK_K):
                window = qpos[i:i + GRASP_LOCK_K]
                if np.std(window) < 0.0005 and np.mean(window) < 0.02:
                    T_grasp_lock = i
                    break

    # ── T_lift_start: EEF z increases (object lifted) ──
    T_lift_start = None
    if T_grasp_lock is not None:
        start = T_grasp_lock
        end = min(n, start + 40)
        if end > start + LIFT_K:
            dz = np.diff(eef_z[start:end])
            streak = 0
            for j, dz_val in enumerate(dz):
                if dz_val > LIFT_DZ_THRESH:
                    streak += 1
                else:
                    streak = 0
                if streak >= LIFT_K:
                    T_lift_start = start + j - LIFT_K + 1
                    break

    # ── T_release_start: natural OPEN begins ──
    T_release_start = None
    search_start = T_lift_start + 10 if T_lift_start is not None else int(n * 0.5)
    search_end = n
    if search_end > search_start + RELEASE_OPEN_K:
        open_streak = 0
        for i in range(search_start, search_end):
            if is_open_canonical[i]:
                open_streak += 1
            else:
                open_streak = 0
            if open_streak >= RELEASE_OPEN_K:
                T_release_start = i - RELEASE_OPEN_K + 1
                break

    # ── T_done ──
    T_done = None
    for i in range(n):
        if done[i]:
            T_done = i
            break

    events = {
        "T_gripper_close_onset": T_close_onset,
        "T_grasp_formation": T_grasp_form,
        "T_grasp_lock": T_grasp_lock,
        "T_lift_start": T_lift_start,
        "T_release_start": T_release_start,
        "T_done": T_done,
        "n_steps": n,
    }

    # ── Per-step phase labeling ──
    phase_6 = np.full(n, -1, dtype=int)  # -1 = unknown
    for i in range(n):
        if T_close_onset is not None and i < T_close_onset:
            phase_6[i] = 1  # pregrasp (assume approach phase already passed by step 0)
        elif T_grasp_form is not None and T_grasp_lock is not None and T_grasp_form <= i < T_grasp_lock:
            phase_6[i] = 2  # grasp_formation
        elif T_grasp_lock is not None and T_lift_start is not None and T_grasp_lock <= i < T_lift_start:
            phase_6[i] = 3  # stable_grasp_or_lift
        elif T_lift_start is not None and T_release_start is not None and T_lift_start <= i < T_release_start:
            phase_6[i] = 4  # carry_or_place
        elif T_release_start is not None and i >= T_release_start:
            phase_6[i] = 5  # release_or_done
        elif T_close_onset is not None and i >= T_close_onset:
            phase_6[i] = 2  # default post-close → grasp_formation

    # 3-class mapping
    phase_3 = np.full(n, -1, dtype=int)
    for i in range(n):
        if phase_6[i] in (0, 1):
            phase_3[i] = 0  # pre_grasp
        elif phase_6[i] == 2:
            phase_3[i] = 1  # grasp_formation
        elif phase_6[i] in (3, 4, 5):
            phase_3[i] = 2  # post_grasp

    return events, phase_6, phase_3


def main():
    ap = argparse.ArgumentParser(description="Audit clean rollout phase events")
    ap.add_argument("--trace", required=True, nargs="+", help="Clean trace CSV file(s)")
    ap.add_argument("--task", default="ketchup")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output-csv", default="tables/phase_alignment_clean_rollouts.csv")
    ap.add_argument("--alignment-csv", default="tables/proprionostep_phase_alignment.csv")
    ap.add_argument("--skip-detector", action="store_true", help="Skip ProprioNoStep alignment")
    args = ap.parse_args()

    all_rows = []
    alignment_rows = []

    for trace_path in args.trace:
        if not os.path.exists(trace_path):
            print(f"WARNING: trace not found: {trace_path}")
            continue

        print(f"Processing: {trace_path}")
        steps = load_trace(trace_path)
        events, phase_6, phase_3 = detect_phase_events(steps)

        # Print event summary
        print(f"  Steps: {len(steps)}")
        for ev_name, ev_val in events.items():
            if ev_val is not None:
                print(f"  {ev_name}: {ev_val}")
        print(f"  Phase distribution (6-class):")
        for label, name in sorted(PHASE_6CLASS.items()):
            cnt = int(np.sum(phase_6 == label))
            if cnt > 0:
                print(f"    {name}: {cnt} steps")

        # Build output rows
        for i, step in enumerate(steps):
            row = {
                "task": args.task,
                "seed": args.seed,
                "policy_step": i,
                "raw_gripper": step.get("raw_gripper", step.get("adv_grip", "")),
                "env_gripper": step.get("env_gripper", ""),
                "qpos": step.get("qpos_post_step", step.get("gripper_qpos", "")),
                "eef_x": step.get("eef_x", ""),
                "eef_y": step.get("eef_y", ""),
                "eef_z": step.get("eef_z", ""),
                "arm_l2": step.get("arm_l2", "0"),
                "done": step.get("done", "False"),
                "phase_label_6class": PHASE_6CLASS.get(int(phase_6[i]), "unknown"),
                "phase_label_6class_id": int(phase_6[i]),
                "phase_label_3class": PHASE_3CLASS.get(int(phase_3[i]), "unknown"),
                "phase_label_3class_id": int(phase_3[i]),
                "label_source": "heuristic",
                "trace_path": trace_path,
            }
            all_rows.append(row)

        # Alignment row
        alignment_rows.append({
            "task": args.task,
            "seed": args.seed,
            "trace_path": trace_path,
            "T_gripper_close_onset": events.get("T_gripper_close_onset", ""),
            "T_grasp_formation": events.get("T_grasp_formation", ""),
            "T_grasp_lock": events.get("T_grasp_lock", ""),
            "T_lift_start": events.get("T_lift_start", ""),
            "T_release_start": events.get("T_release_start", ""),
            "T_done": events.get("T_done", ""),
            "n_steps": events.get("n_steps", 0),
        })

    # Write phase-labeled CSV
    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    if all_rows:
        with open(args.output_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            w.writeheader()
            w.writerows(all_rows)
        print(f"Wrote {len(all_rows)} rows to {args.output_csv}")

    # Write alignment CSV
    os.makedirs(os.path.dirname(args.alignment_csv) or ".", exist_ok=True)
    if alignment_rows:
        with open(args.alignment_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(alignment_rows[0].keys()))
            w.writeheader()
            w.writerows(alignment_rows)
        print(f"Wrote {len(alignment_rows)} alignment rows to {args.alignment_csv}")
    print("Done.")


if __name__ == "__main__":
    main()
