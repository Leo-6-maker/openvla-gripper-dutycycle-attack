#!/usr/bin/env python3
"""Gate 2: R10.4E E-R3a offline Teacher-Student matched audit.

Reads sealed privileged sidecar + step_records + detector_records.
Reconstructs Teacher grasp_established intervals from privileged evidence,
then compares against Student probability/FSM behavior.

NO OpenVLA. NO LIBERO. Read-only. Does not modify sealed roots.

Output for each episode:
  - Teacher grasp intervals (start, end, duration)
  - Student max probability in each interval
  - Student >= 0.5 intervals + persistence
  - FSM vertical-lift analysis
  - Final no-emit classification (A/B/C/D)
"""

import json, sys
from pathlib import Path
from typing import Any

import numpy as np

# Frozen FSM constants
GRASP_THRESHOLD = 0.5
GRASP_PERSISTENCE = 3
VERTICAL_LIFT_M = 0.02

# Gripper closed detection: sum of absolute finger positions < this threshold
GRIPPER_CLOSED_THRESHOLD = 0.1

# Proximity: object z must be within this distance of gripper z to be "grasped"
OBJECT_ELEVATED_MIN_DZ = 0.03  # object must be at least this much above initial z


def load_jsonl(path: Path):
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def extract_teacher_grasp_intervals(sidecar):
    """Reconstruct Teacher grasp_established intervals from privileged data.

    Uses a simple heuristic based on:
    - Gripper close state (sum of finger positions < threshold)
    - Object proximity to gripper (object z near eef_z)
    - Contact between gripper and objects

    Returns list of {start_step, end_step, duration, max_obj_z, n_contacts}
    """
    intervals = []
    in_grasp = False
    grasp_start = -1
    initial_obj_z = None

    for row in sidecar:
        step = row["step"]
        gripper = np.abs(np.array(row["robot0_gripper_qpos"], dtype=np.float64)).sum()
        eef_z = np.array(row["robot0_eef_pos"], dtype=np.float64)[2]
        obj_state = np.array(row["object_state"], dtype=np.float64)
        contacts = row.get("mujoco_contact_pairs", [])

        # Object positions: every 7 values = [x, y, z, qw, qx, qy, qz]
        # We care about z-height of first 2 objects
        obj_zs = []
        for i in range(min(2, len(obj_state) // 7)):
            obj_z = float(obj_state[i * 7 + 2])
            obj_zs.append(obj_z)

        # Track initial object z for lift detection
        if step == 0 and obj_zs:
            initial_obj_z = obj_zs[0]

        gripper_closed = gripper < GRIPPER_CLOSED_THRESHOLD
        obj_near = any(abs(z - eef_z) < 0.15 for z in obj_zs) if obj_zs else False
        has_gripper_contact = any(
            "gripper" in c[0].lower() or "gripper" in c[1].lower()
            for c in contacts
        )

        # Teacher grasp: gripper closed AND (object near OR contact)
        teacher_grasp = gripper_closed and (obj_near or has_gripper_contact or gripper < 0.03)

        if teacher_grasp and not in_grasp:
            in_grasp = True
            grasp_start = step
        elif not teacher_grasp and in_grasp:
            in_grasp = False
            intervals.append({
                "start_step": grasp_start,
                "end_step": step - 1,
                "duration": step - grasp_start,
                "max_obj_z": max(obj_zs) if obj_zs else 0.0,
                "n_contacts": len(contacts),
            })

    if in_grasp:
        intervals.append({
            "start_step": grasp_start,
            "end_step": len(sidecar) - 1,
            "duration": len(sidecar) - grasp_start,
            "max_obj_z": 0.0,
            "n_contacts": 0,
        })

    return intervals


def analyze_student_in_interval(
    interval: dict,
    detector_records,
    sidecar,
):
    """Analyze Student behavior within a Teacher grasp interval."""
    start, end = interval["start_step"], interval["end_step"]
    probs = []
    max_prob = 0.0
    max_step = -1
    persistence_runs = []
    current_run = 0

    for step in range(start, min(end + 1, len(detector_records))):
        det = detector_records[step]
        prob = det.get("grasp_probability", 0.0)
        probs.append((step, prob))
        if prob > max_prob:
            max_prob = prob
            max_step = step

        if prob >= GRASP_THRESHOLD:
            current_run += 1
        else:
            if current_run > 0:
                persistence_runs.append({
                    "start": step - current_run,
                    "end": step - 1,
                    "length": current_run,
                })
            current_run = 0

    if current_run > 0:
        persistence_runs.append({
            "start": end + 1 - current_run,
            "end": end,
            "length": current_run,
        })

    passed_persistence = any(r["length"] >= GRASP_PERSISTENCE for r in persistence_runs)

    # FSM vertical lift from first persistence run >= 3
    fsm_would_arm = False
    anchor_step = -1
    anchor_eef_z = 0.0
    lift_pass_step = -1
    for run in persistence_runs:
        if run["length"] >= GRASP_PERSISTENCE:
            anchor_step = run["start"]
            if anchor_step < len(sidecar):
                anchor_eef_z = float(np.array(sidecar[anchor_step]["robot0_eef_pos"], dtype=np.float64)[2])
            # Check if EEF lifted >= 0.02m after anchor
            for s in range(anchor_step, min(end + 1, len(sidecar))):
                eef_z = float(np.array(sidecar[s]["robot0_eef_pos"], dtype=np.float64)[2])
                if eef_z - anchor_eef_z >= VERTICAL_LIFT_M:
                    fsm_would_arm = True
                    lift_pass_step = s
                    break
            break

    return {
        "max_prob": max_prob,
        "max_prob_step": max_step,
        "n_steps_above_threshold": sum(1 for _, p in probs if p >= GRASP_THRESHOLD),
        "persistence_runs": persistence_runs,
        "passed_persistence": passed_persistence,
        "fsm_would_arm": fsm_would_arm,
        "anchor_step": anchor_step if passed_persistence else -1,
        "lift_pass_step": lift_pass_step,
    }


def classify_no_emit(
    teacher_intervals,
    student_analyses,
    detector_records,
):
    """Classify why no emit occurred: A/B/C/D."""
    if not teacher_intervals:
        return "A_NO_TEACHER_EVENT"

    # Check B: Teacher has event, Student never above threshold
    any_persistence = any(a["student_analysis"]["passed_persistence"] for a in student_analyses)
    any_above = any(a["student_analysis"]["max_prob"] >= GRASP_THRESHOLD for a in student_analyses)

    if not any_above:
        return "B_TEACHER_EVENT_STUDENT_BELOW_THRESHOLD"

    # Check C: Student has persistence but no lift
    if any_persistence:
        any_lift = any(a["student_analysis"]["fsm_would_arm"] for a in student_analyses)
        if not any_lift:
            return "C_STUDENT_POSITIVE_FSM_NO_LIFT"

    # Check D: Something else blocked
    return "D_OTHER_BLOCK"


def audit_episode(ep_dir: Path, identity: str):
    """Run full Teacher-Student matched audit on one episode."""
    sidecar = load_jsonl(ep_dir / "privileged_teacher_sidecar.jsonl")
    step_recs = load_jsonl(ep_dir / "step_records.jsonl")
    det_recs = load_jsonl(ep_dir / "detector_records.jsonl")

    teacher_intervals = extract_teacher_grasp_intervals(sidecar)

    # Analyze each Teacher interval
    analyses = []
    for interval in teacher_intervals:
        analysis = analyze_student_in_interval(interval, det_recs, sidecar)
        analyses.append({
            "teacher_interval": interval,
            "student_analysis": analysis,
        })

    # Overall Student stats
    all_probs = [d.get("grasp_probability", 0.0) for d in det_recs]
    max_overall = max(all_probs) if all_probs else 0.0
    mean_overall = float(np.mean(all_probs)) if all_probs else 0.0

    no_emit_class = classify_no_emit(teacher_intervals, analyses, det_recs)

    return {
        "identity": identity,
        "n_steps": len(step_recs),
        "n_teacher_intervals": len(teacher_intervals),
        "teacher_intervals": [
            {"start": i["teacher_interval"]["start_step"],
             "end": i["teacher_interval"]["end_step"],
             "duration": i["teacher_interval"]["duration"]}
            for i in analyses
        ],
        "student_max_overall": max_overall,
        "student_mean_overall": round(mean_overall, 6),
        "student_by_interval": [
            {
                "teacher_start": a["teacher_interval"]["start_step"],
                "teacher_end": a["teacher_interval"]["end_step"],
                "max_prob": round(a["student_analysis"]["max_prob"], 6),
                "max_prob_step": a["student_analysis"]["max_prob_step"],
                "persistence_runs": [{
                    "start": r["start"], "end": r["end"], "length": r["length"]
                } for r in a["student_analysis"]["persistence_runs"]],
                "passed_persistence": a["student_analysis"]["passed_persistence"],
                "fsm_would_arm": a["student_analysis"]["fsm_would_arm"],
                "lift_pass_step": a["student_analysis"]["lift_pass_step"],
            }
            for a in analyses
        ],
        "no_emit_classification": no_emit_class,
    }


def main():
    # Paths on server
    task00_root = Path("/mnt/sdc/dty_user/openvla_attack_evidence/r10_4d_passive_smoke_output_20260720")
    task01_root = Path("/mnt/sdc/dty_user/openvla_attack_evidence/r10_4e_e_r3a_output_20260720/libero_10_task_01_state_20")

    print("=" * 60)
    print("Gate 2: Offline Teacher-Student Matched Audit")
    print("=" * 60)

    for label, root, identity in [
        ("task_00", task00_root, "libero_10/task_00/state_20"),
        ("task_01", task01_root, "libero_10/task_01/state_20"),
    ]:
        print(f"\n--- {label} ---")
        result = audit_episode(root, identity)

        print(f"  Steps: {result['n_steps']}")
        print(f"  Teacher grasp intervals: {result['n_teacher_intervals']}")
        if result["teacher_intervals"]:
            for i, t in enumerate(result["teacher_intervals"]):
                print(f"    Interval {i}: steps [{t['start']}-{t['end']}] duration={t['duration']}")
        else:
            print(f"    (none)")

        print(f"  Student max probability: {result['student_max_overall']:.6f}")
        print(f"  Student mean probability: {result['student_mean_overall']:.6f}")

        for i, s in enumerate(result["student_by_interval"]):
            print(f"  Interval {i} Student analysis:")
            print(f"    max_prob: {s['max_prob']:.6f} @ step {s['max_prob_step']}")
            print(f"    persistence runs: {s['persistence_runs']}")
            print(f"    passed_persistence: {s['passed_persistence']}")
            print(f"    fsm_would_arm: {s['fsm_would_arm']}")
            print(f"    lift_pass_step: {s['lift_pass_step']}")

        print(f"  Classification: {result['no_emit_classification']}")

        print()

    # Summary table
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    for label, root, identity in [
        ("task_00", task00_root, "libero_10/task_00/state_20"),
        ("task_01", task01_root, "libero_10/task_01/state_20"),
    ]:
        result = audit_episode(root, identity)
        print(f"  {label}: {result['n_teacher_intervals']} Teacher intervals, "
              f"max_prob={result['student_max_overall']:.4f}, "
              f"class={result['no_emit_classification']}")


if __name__ == "__main__":
    main()
