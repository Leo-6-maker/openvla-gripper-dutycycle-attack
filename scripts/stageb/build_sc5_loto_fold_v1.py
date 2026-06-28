#!/usr/bin/env python3
"""
Generic LOTO Fold Builder V1.

Reads the combined 500-episode corpus and produces per-fold:
  - Train+Val feature CSV (FOLDxx_TRAIN_VAL_FEATURE_DATASET.csv)
  - Heldout feature CSV (FOLDxx_HELDOUT_FEATURE_DATASET.csv)
  - Train+Val teacher labels JSONL (FOLDxx_teacher_labels_train_val.jsonl)
  - Train-only normalization JSON (FOLDxx_TRAIN_NORMALIZATION.json)
  - Teacher config JSON (FOLDxx_teacher_config.json)
  - Fold manifest JSON (FOLDxx_MANIFEST.json)

Usage:
  python build_sc5_loto_fold_v1.py --fold 01 --output_dir <dir>
"""
import argparse, csv, hashlib, json, os, sys, numpy as np
from collections import defaultdict, Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from gripper_attack.v2_privileged_teacher import (
    calibrate_thresholds, V2PrivilegedTeacher, TeacherConfig
)

# ═══════════════════════════════════════════════
# Frozen constants
# ═══════════════════════════════════════════════
TASKS = ["alphabet_soup","cream_cheese","salad_dressing","bbq_sauce","ketchup",
         "tomato_sauce","butter","milk","chocolate_pudding","orange_juice"]
SOURCE_COMMIT = "0280c8564773a5e6ca0482c740891d8f9eddad84"

# LOTO fold matrix: test_task -> (val_task, train_tasks)
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

# Paths to combined corpus
WAVE1_ROOT = "/mnt/sdc/dty_user/openvla_attack/evidence/sc5_object_privileged_loto_v1/wave1_50_0280c85_20260627T175204Z"
WAVE2_ROOT = "/mnt/sdc/dty_user/openvla_attack/evidence/sc5_object_privileged_loto_v1/wave2_remaining_states_0280c85_20260627T183812Z"

FEATURE_NAMES = [
    "gripper_command","gripper_qpos","gripper_opening_proxy",
    "eef_x","eef_y","eef_z","eef_vx","eef_vy","eef_vz",
    "action_dx","action_dy","action_dz","action_gripper",
    "recent_close_streak","recent_open_streak","recent_gripper_flip_count",
    "close_onset","time_since_close","eef_speed",
    "eef_z_delta_since_close","qpos_delta_1","qpos_delta_3",
    "opening_proxy_delta_3","opening_proxy_variance_5","eef_speed_variance_5",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", required=True, help="Fold ID: 00-09")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--teacher_labels", required=True,
                    help="Combined 500-episode teacher_labels JSONL (full)")
    ap.add_argument("--feature_dataset", required=True,
                    help="Combined 500-episode FOLD00_FEATURE_DATASET.csv (full)")
    args = ap.parse_args()

    fold_id = args.fold
    if fold_id not in FOLD_MATRIX:
        raise ValueError("Unknown fold: %s" % fold_id)

    fold_cfg = FOLD_MATRIX[fold_id]
    test_task = fold_cfg["test"]
    val_task = fold_cfg["val"]
    train_tasks = set(fold_cfg["train"])

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Building Fold %s: test=%d(%s) val=%d(%s) train=%s" % (
        fold_id, test_task, TASKS[test_task], val_task, TASKS[val_task],
        sorted(train_tasks)))

    # ── 1. Load teacher labels, split by task ──
    print("Loading teacher labels...")
    all_labels = {"train": [], "val": [], "test": []}
    with open(args.teacher_labels) as f:
        for line in f:
            if not line.strip(): continue
            lab = json.loads(line)
            t = lab["task_idx"]
            if t == test_task:
                all_labels["test"].append(lab)
            elif t == val_task:
                all_labels["val"].append(lab)
            elif t in train_tasks:
                all_labels["train"].append(lab)

    print("  Train labels: %d rows" % len(all_labels["train"]))
    print("  Val labels:   %d rows" % len(all_labels["val"]))
    print("  Test labels:  %d rows" % len(all_labels["test"]))

    # ── 2. Load feature dataset, split by task ──
    print("Loading feature dataset...")
    all_features = {"train": [], "val": [], "test": []}
    with open(args.feature_dataset) as f:
        reader = csv.DictReader(f)
        for r in reader:
            t = int(r["task_idx"])
            if t == test_task:
                all_features["test"].append(r)
            elif t == val_task:
                all_features["val"].append(r)
            elif t in train_tasks:
                all_features["train"].append(r)

    print("  Train rows: %d (%d episodes)" % (
        len(all_features["train"]),
        len(set((r["task_idx"], r["state_id"]) for r in all_features["train"]))))
    print("  Val rows:   %d (%d episodes)" % (
        len(all_features["val"]),
        len(set((r["task_idx"], r["state_id"]) for r in all_features["val"]))))
    print("  Test rows:  %d (%d episodes)" % (
        len(all_features["test"]),
        len(set((r["task_idx"], r["state_id"]) for r in all_features["test"]))))

    # ── 3. Train-only teacher calibration ──
    print("\nTeacher calibration on %d train label rows..." % len(all_labels["train"]))
    # Write temp JSONL for calibration input
    tmp_calib = out_dir / "_temp_calib_train.jsonl"
    with open(tmp_calib, "w") as f:
        for lab in all_labels["train"]:
            f.write(json.dumps(lab) + "\n")

    # Calibrate using privileged records, not teacher labels
    # We need privileged_step_records.jsonl paths for train episodes
    train_priv_paths = []
    train_eps = set((lab["task_idx"], lab["state_id"]) for lab in all_labels["train"])
    for t, s in sorted(train_eps):
        wave_root = WAVE1_ROOT if s <= 4 else WAVE2_ROOT
        priv_path = os.path.join(wave_root, "jobs", "task_%d_%s" % (t, TASKS[t]),
                                 "state_%d" % s, "attempt_1", "privileged_step_records.jsonl")
        if os.path.exists(priv_path):
            train_priv_paths.append(priv_path)

    print("  Found %d privileged record paths" % len(train_priv_paths))
    cfg = calibrate_thresholds(train_priv_paths)
    cfg.calibrated_from = "Fold_%s_train_%d_episodes_tasks_%s" % (
        fold_id, len(train_priv_paths), str(sorted(train_tasks)))
    cfg.version = "v2_teacher_loto_fold%s" % fold_id

    teacher_config = {
        "gate": "LOTO_FOLD%s_TEACHER_CONFIG" % fold_id,
        "fold": fold_id, "test_task": test_task, "val_task": val_task,
        "train_tasks": sorted(train_tasks),
        "calibrated_from": cfg.calibrated_from,
        "thresholds": {
            "grasp_close_sustain": cfg.grasp_close_sustain,
            "grasp_open_proxy_max": cfg.grasp_open_proxy_max,
            "eef_obj_dist_max": cfg.eef_obj_dist_max,
            "eef_obj_dist_stable_var": cfg.eef_obj_dist_stable_var,
            "lift_z_threshold": cfg.lift_z_threshold,
            "lift_sustain_steps": cfg.lift_sustain_steps,
            "carry_obj_z_var_max": cfg.carry_obj_z_var_max,
            "carry_window": cfg.carry_window,
            "preplace_target_dist_min": cfg.preplace_target_dist_min,
            "preplace_target_dist_max": cfg.preplace_target_dist_max,
            "release_target_dist_max": cfg.release_target_dist_max,
            "regrasp_eef_obj_dist_max": cfg.regrasp_eef_obj_dist_max,
            "stability_window": cfg.stability_window,
        },
    }
    tmp_calib.unlink()

    tc_path = out_dir / ("FOLD%s_teacher_config.json" % fold_id)
    with open(tc_path, "w") as f:
        json.dump(teacher_config, f, indent=2)
    print("  Saved: %s" % tc_path)
    print("  Thresholds: grasp_open_proxy=%.4f eef_obj_dist=%.4f lift_z=%.4f" % (
        cfg.grasp_open_proxy_max, cfg.eef_obj_dist_max, cfg.lift_z_threshold))

    # ── 4. Train-only normalization ──
    print("\nComputing train-only normalization...")
    Xtr = np.array([[float(r["f_" + n]) for n in FEATURE_NAMES]
                    for r in all_features["train"]], dtype=np.float64)
    train_mean = Xtr.mean(axis=0).tolist()
    train_std = np.maximum(Xtr.std(axis=0), 1e-8).tolist()

    norm = {
        "mean": {"f_" + FEATURE_NAMES[i]: train_mean[i] for i in range(25)},
        "std": {"f_" + FEATURE_NAMES[i]: train_std[i] for i in range(25)},
        "computed_from": "Fold %s train-only (%d rows)" % (fold_id, len(all_features["train"])),
    }
    norm_path = out_dir / ("FOLD%s_TRAIN_NORMALIZATION.json" % fold_id)
    with open(norm_path, "w") as f:
        json.dump(norm, f, indent=2)
    print("  Saved: %s" % norm_path)

    # ── 5. Write train+val feature CSV ──
    print("\nWriting train+val feature CSV...")
    tv_path = out_dir / ("FOLD%s_TRAIN_VAL_FEATURE_DATASET.csv" % fold_id)
    with open(tv_path, "w", newline="") as f:
        fieldnames = ["task_idx","state_id","split","step"] + ["f_" + n for n in FEATURE_NAMES]
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        w.writeheader()
        for r in all_features["train"]:
            r["split"] = "train"
            w.writerow(r)
        for r in all_features["val"]:
            r["split"] = "val"
            w.writerow(r)
    print("  %s (%d train + %d val rows)" % (
        tv_path, len(all_features["train"]), len(all_features["val"])))

    # ── 6. Write heldout feature CSV ──
    print("Writing heldout feature CSV...")
    ho_path = out_dir / ("FOLD%s_HELDOUT_FEATURE_DATASET.csv" % fold_id)
    with open(ho_path, "w", newline="") as f:
        fieldnames = ["task_idx","state_id","split","step"] + ["f_" + n for n in FEATURE_NAMES]
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        w.writeheader()
        for r in all_features["test"]:
            r["split"] = "held_out"
            w.writerow(r)
    print("  %s (%d rows)" % (ho_path, len(all_features["test"])))

    # ── 7. Write train+val teacher labels JSONL ──
    print("Writing train+val teacher labels...")
    tl_path = out_dir / ("FOLD%s_teacher_labels_train_val.jsonl" % fold_id)
    with open(tl_path, "w") as f:
        for lab in all_labels["train"] + all_labels["val"]:
            f.write(json.dumps(lab) + "\n")
    print("  %s (%d rows)" % (tl_path, len(all_labels["train"]) + len(all_labels["val"])))

    # ── 8. Write fold manifest ──
    print("Writing fold manifest...")
    with open(tv_path, "rb") as f: tv_sha = hashlib.sha256(f.read()).hexdigest()
    with open(ho_path, "rb") as f: ho_sha = hashlib.sha256(f.read()).hexdigest()
    with open(tl_path, "rb") as f: tl_sha = hashlib.sha256(f.read()).hexdigest()
    with open(norm_path, "rb") as f: nm_sha = hashlib.sha256(f.read()).hexdigest()
    with open(tc_path, "rb") as f: tc_sha = hashlib.sha256(f.read()).hexdigest()

    manifest = {
        "gate": "LOTO_FOLD%s_MANIFEST" % fold_id,
        "fold": fold_id,
        "test_task": test_task, "test_task_name": TASKS[test_task],
        "val_task": val_task, "val_task_name": TASKS[val_task],
        "train_tasks": sorted(train_tasks),
        "train_episodes": len(set((r["task_idx"], r["state_id"]) for r in all_features["train"])),
        "val_episodes": len(set((r["task_idx"], r["state_id"]) for r in all_features["val"])),
        "test_episodes": len(set((r["task_idx"], r["state_id"]) for r in all_features["test"])),
        "train_rows": len(all_features["train"]),
        "val_rows": len(all_features["val"]),
        "test_rows": len(all_features["test"]),
        "train_val_csv_sha256": tv_sha,
        "heldout_csv_sha256": ho_sha,
        "teacher_labels_sha256": tl_sha,
        "normalization_sha256": nm_sha,
        "teacher_config_sha256": tc_sha,
        "source_commit": SOURCE_COMMIT,
        "test_rows_loaded_during_build": len(all_features["test"]),
    }
    manifest_path = out_dir / ("FOLD%s_MANIFEST.json" % fold_id)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print("  %s" % manifest_path)

    print("\n=== FOLD %s BUILD COMPLETE ===" % fold_id)
    print("Test: %d (%s) — DO NOT OPEN" % (test_task, TASKS[test_task]))
    print("Val:  %d (%s)" % (val_task, TASKS[val_task]))
    print("Train: %d episodes, %d rows" % (manifest["train_episodes"], manifest["train_rows"]))


if __name__ == "__main__":
    main()
