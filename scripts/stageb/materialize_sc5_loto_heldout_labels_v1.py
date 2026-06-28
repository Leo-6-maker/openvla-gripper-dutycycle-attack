#!/usr/bin/env python3
"""
Phase B: Materialize held-out teacher labels from frozen fold-specific teacher config.

For each fold, loads the FROZEN teacher_config JSON and held-out privileged records,
generates test teacher labels using V2PrivilegedTeacher(frozen_cfg).

Usage (per fold):
  python materialize_sc5_loto_heldout_labels_v1.py \
    --fold 01 --fold_dir <FOLD01_OUTPUT_DIR> \
    --output_dir <FOLD01_OUTPUT_DIR>
"""
import argparse, json, os, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from gripper_attack.v2_privileged_teacher import (
    V2PrivilegedTeacher, TeacherConfig,
    find_sc5_anchor_v2, compute_sc5_valid_start_corridor
)

TASKS = ["alphabet_soup","cream_cheese","salad_dressing","bbq_sauce","ketchup",
         "tomato_sauce","butter","milk","chocolate_pudding","orange_juice"]
WAVE1_ROOT = "/mnt/sdc/dty_user/openvla_attack/evidence/sc5_object_privileged_loto_v1/wave1_50_0280c85_20260627T175204Z"
WAVE2_ROOT = "/mnt/sdc/dty_user/openvla_attack/evidence/sc5_object_privileged_loto_v1/wave2_remaining_states_0280c85_20260627T183812Z"
K_SC5 = 10; GUARD_SC5 = 5

# Single source of truth
FOLD_MATRIX = {
    "00": {"test": 8, "val": 6, "train": [0,1,2,3,4,5,7,9]},
    "01": {"test": 9, "val": 7, "train": [0,1,2,3,4,5,6,8]},
    "02": {"test": 0, "val": 8, "train": [1,2,3,4,5,6,7,9]},
    "03": {"test": 1, "val": 9, "train": [0,2,3,4,5,6,7,8]},
    "04": {"test": 2, "val": 0, "train": [1,3,4,5,6,7,8,9]},
    "05": {"test": 3, "val": 1, "train": [0,2,4,5,6,7,8,9]},
    "06": {"test": 4, "val": 2, "train": [0,1,3,5,6,7,8,9]},
    "07": {"test": 5, "val": 3, "train": [0,1,2,4,6,7,8,9]},
    "08": {"test": 6, "val": 4, "train": [0,1,2,3,5,7,8,9]},
    "09": {"test": 7, "val": 5, "train": [0,1,2,3,4,6,8,9]},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", required=True)
    ap.add_argument("--fold_dir", required=True, help="Directory with FOLDxx_teacher_config.json")
    ap.add_argument("--output_dir", required=True)
    args = ap.parse_args()

    fold_id = args.fold; fc = FOLD_MATRIX[fold_id]
    test_task = fc["test"]
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # Load frozen teacher config
    tc_path = os.path.join(args.fold_dir, "FOLD%s_teacher_config.json" % fold_id)
    with open(tc_path) as f: tc = json.load(f)
    th = tc["thresholds"]

    cfg = TeacherConfig()
    for k in ["grasp_close_sustain","eef_obj_dist_max","eef_obj_dist_stable_var",
              "lift_z_threshold","lift_sustain_steps","carry_obj_z_var_max","carry_window",
              "preplace_target_dist_min","preplace_target_dist_max",
              "release_target_dist_max","regrasp_eef_obj_dist_max","stability_window"]:
        setattr(cfg, k, th.get(k, getattr(cfg, k, 0)))
    cfg.calibrated_from = tc["calibrated_from"]
    cfg.version = tc.get("version", "loto_fold%s" % fold_id)

    teacher = V2PrivilegedTeacher(cfg)

    # Load test privileged records
    all_labels = []
    episode_anchors = []

    for s in range(50):
        wave_root = WAVE1_ROOT if s <= 4 else WAVE2_ROOT
        priv_path = os.path.join(wave_root, "jobs", "task_%d_%s" % (test_task, TASKS[test_task]),
                                 "state_%d" % s, "attempt_1", "privileged_step_records.jsonl")
        with open(priv_path) as f:
            records = [json.loads(line) for line in f if line.strip()]

        labels = teacher.label_trajectory(records)
        ep_labels = []
        for step_idx, lab in enumerate(labels):
            if lab is not None:
                lab["task_idx"] = test_task; lab["task_name"] = TASKS[test_task]
                lab["state_id"] = s; lab["step_idx"] = step_idx
                lab["split"] = "held_out"
                all_labels.append(lab)
                ep_labels.append(lab)

        # Compute anchor and valid-start corridor for this episode
        sc5 = find_sc5_anchor_v2(ep_labels, K=K_SC5, guard=GUARD_SC5)
        anchor = sc5["anchor"]
        if sc5["valid"]:
            corridor_info = compute_sc5_valid_start_corridor(ep_labels, anchor, K=K_SC5)
            corridor_active = sorted(corridor_info["corridor_active_at_t"])
        else:
            corridor_active = []

        episode_anchors.append({
            "task_idx": test_task, "state_id": s,
            "sc5_anchor": anchor, "sc5_valid": sc5["valid"],
            "sc5_reason": sc5.get("reason", ""),
            "stable_carry_start": sc5.get("stable_carry_start", -1),
            "corridor_active_at_t": corridor_active,
            "corridor_first": corridor_active[0] if corridor_active else -1,
            "corridor_last": corridor_active[-1] if corridor_active else -1,
        })

    # Write labels
    labels_path = out_dir / ("FOLD%s_teacher_labels_heldout.jsonl" % fold_id)
    with open(labels_path, "w") as f:
        for lab in all_labels:
            f.write(json.dumps(lab) + "\n")

    # Write anchors
    anchors_path = out_dir / ("FOLD%s_heldout_episode_anchors.json" % fold_id)
    with open(anchors_path, "w") as f:
        json.dump({"fold": fold_id, "test_task": test_task,
                   "test_task_name": TASKS[test_task],
                   "episodes": episode_anchors,
                   "teacher_config_sha256": None}, f, indent=2)

    pos_eps = sum(1 for a in episode_anchors if a["sc5_valid"])
    print("Fold %s heldout labels: %d rows, %d episodes, %d teacher-positive" % (
        fold_id, len(all_labels), len(episode_anchors), pos_eps))
    print("  Labels: %s" % labels_path)
    print("  Anchors: %s" % anchors_path)


if __name__ == "__main__":
    main()
