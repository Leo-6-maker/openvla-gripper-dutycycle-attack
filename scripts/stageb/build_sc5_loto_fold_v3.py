#!/usr/bin/env python3
"""
LOTO Fold Builder V3 — reads single-source protocol JSON. No hardcoded matrix.
Preserves original teacher step_idx (does NOT overwrite). Fail-closed throughout.
"""
import argparse, csv, hashlib, json, os, sys, numpy as np
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from gripper_attack.v2_privileged_teacher import (
    calibrate_thresholds, V2PrivilegedTeacher, TeacherConfig
)

TASKS = ["alphabet_soup","cream_cheese","salad_dressing","bbq_sauce","ketchup",
         "tomato_sauce","butter","milk","chocolate_pudding","orange_juice"]
SOURCE_COMMIT = "0280c8564773a5e6ca0482c740891d8f9eddad84"
FEATURE_NAMES = [
    "gripper_command","gripper_qpos","gripper_opening_proxy",
    "eef_x","eef_y","eef_z","eef_vx","eef_vy","eef_vz",
    "action_dx","action_dy","action_dz","action_gripper",
    "recent_close_streak","recent_open_streak","recent_gripper_flip_count",
    "close_onset","time_since_close","eef_speed",
    "eef_z_delta_since_close","qpos_delta_1","qpos_delta_3",
    "opening_proxy_delta_3","opening_proxy_variance_5","eef_speed_variance_5",
]
WAVE1_ROOT = "/mnt/sdc/dty_user/openvla_attack/evidence/sc5_object_privileged_loto_v1/wave1_50_0280c85_20260627T175204Z"
WAVE2_ROOT = "/mnt/sdc/dty_user/openvla_attack/evidence/sc5_object_privileged_loto_v1/wave2_remaining_states_0280c85_20260627T183812Z"


def load_privileged_paths(task_filter):
    paths = []
    for t in range(10):
        for s in range(50):
            if not task_filter(t, s): continue
            wave_root = WAVE1_ROOT if s <= 4 else WAVE2_ROOT
            path = os.path.join(wave_root, "jobs", "task_%d_%s" % (t, TASKS[t]),
                                "state_%d" % s, "attempt_1", "privileged_step_records.jsonl")
            if not os.path.exists(path):
                raise FileNotFoundError("Missing: %s" % path)
            paths.append(path)
    return paths


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--feature_dataset", required=True)
    ap.add_argument("--protocol", required=True,
                    help="LOTO_10FOLD_PROTOCOL_FREEZE_V2.json (single truth source)")
    args = ap.parse_args(argv)

    fold_id = args.fold
    with open(args.protocol) as f:
        protocol = json.load(f)
    matrix = protocol["fold_matrix"]
    if fold_id not in matrix:
        raise ValueError("Fold %s not in protocol" % fold_id)

    fc = matrix[fold_id]
    test_task = fc["test"]; val_task = fc["val"]
    train_tasks = set(fc["train"])

    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    print("Fold %s: test=%d(%s) val=%d(%s) train=%s" % (
        fold_id, test_task, TASKS[test_task], val_task, TASKS[val_task], sorted(train_tasks)))

    # ── 1. Collect privileged paths (fail-closed: exactly 400 train + 50 val) ──
    print("\nCollecting privileged record paths...")
    train_paths = load_privileged_paths(lambda t, s: t in train_tasks)
    val_paths = load_privileged_paths(lambda t, s: t == val_task)
    assert len(train_paths) == 400, "Expected 400, got %d" % len(train_paths)
    assert len(val_paths) == 50, "Expected 50, got %d" % len(val_paths)
    assert len(set(train_paths)) == 400, "Duplicate train paths"
    train_task_counts = defaultdict(int)
    for p in train_paths:
        for seg in p.split("/"):
            if seg.startswith("task_") and not seg.endswith(".jsonl"):
                train_task_counts[int(seg.split("_")[1])] += 1; break
    for t in train_tasks:
        assert train_task_counts[t] == 50, "Task %d: %d paths" % (t, train_task_counts[t])

    # ── 2. Train-only teacher calibration ──
    print("\nTeacher calibration on %d train records..." % len(train_paths))
    cfg = calibrate_thresholds(train_paths)
    cfg.calibrated_from = "Fold_%s_train_400_tasks_%s" % (fold_id, str(sorted(train_tasks)))
    cfg.version = "v2_loto_fold%s" % fold_id

    teacher_config = {
        "gate": "LOTO_FOLD%s_TEACHER_CONFIG" % fold_id,
        "fold": fold_id, "test_task": test_task, "val_task": val_task,
        "train_tasks": sorted(train_tasks),
        "calibrated_from": cfg.calibrated_from,
        "thresholds": {f.name: getattr(cfg, f.name) for f in [
            type(cfg).__dataclass_fields__[k] for k in type(cfg).__dataclass_fields__
        ] if f.name not in ("version", "calibrated_from")},
    }
    # Explicit threshold dict (safer than dataclass reflection across versions)
    teacher_config["thresholds"] = {
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
    }
    tc_path = out_dir / ("FOLD%s_teacher_config.json" % fold_id)
    with open(tc_path, "w") as f: json.dump(teacher_config, f, indent=2)
    print("  Saved: %s" % tc_path)

    # ── 3. Regenerate labels (PRESERVE original step_idx) ──
    print("\nRegenerating labels with fold-specific teacher...")
    teacher = V2PrivilegedTeacher(cfg)

    def label_episodes(paths):
        all_labels = []
        for path in sorted(paths):
            parts = path.split("/")
            task_dir = [p for p in parts if p.startswith("task_")][0]
            state_dir = [p for p in parts if p.startswith("state_")][0]
            t = int(task_dir.split("_")[1]); s = int(state_dir.split("_")[1])
            with open(path) as f:
                records = [json.loads(line) for line in f if line.strip()]
            labels = teacher.label_trajectory(records)
            for idx, lab in enumerate(labels):
                if lab is None: continue
                # Preserve original step_idx — do NOT overwrite with enumerate index
                orig_step = int(lab["step_idx"])
                raw_step = int(records[idx].get("step_idx", idx))
                assert orig_step == raw_step, \
                    "Step mismatch: teacher=%d raw=%d in t%d s%d" % (orig_step, raw_step, t, s)
                lab["task_idx"] = t; lab["task_name"] = TASKS[t]
                lab["state_id"] = s; lab["split"] = "train" if t in train_tasks else "val"
                all_labels.append(lab)
        return all_labels

    train_labels = label_episodes(train_paths)
    val_labels = label_episodes(val_paths)

    tr_corr = sum(1 for lab in train_labels if lab.get("phase") == "stable_carry")
    vl_corr = sum(1 for lab in val_labels if lab.get("phase") == "stable_carry")
    print("  Train: %d rows, stable_carry=%d" % (len(train_labels), tr_corr))
    print("  Val:   %d rows, stable_carry=%d" % (len(val_labels), vl_corr))

    # ── 4. Write teacher labels (convert numpy scalars) ──
    def to_native(obj):
        if isinstance(obj, dict):
            return {k: to_native(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [to_native(v) for v in obj]
        if hasattr(obj, "item"):  # numpy scalar
            return obj.item()
        return obj

    tl_path = out_dir / ("FOLD%s_teacher_labels_train_val.jsonl" % fold_id)
    with open(tl_path, "w") as f:
        for lab in train_labels + val_labels:
            f.write(json.dumps(to_native(lab)) + "\n")

    # ── 5. Train-only normalization ──
    print("\nTrain-only normalization...")
    Xtr_rows = []
    with open(args.feature_dataset) as f:
        for r in csv.DictReader(f):
            if int(r["task_idx"]) in train_tasks:
                Xtr_rows.append([float(r["f_" + n]) for n in FEATURE_NAMES])
    Xtr = np.array(Xtr_rows, dtype=np.float64)
    norm = {"mean": {"f_" + FEATURE_NAMES[i]: float(Xtr.mean(0)[i]) for i in range(25)},
            "std": {"f_" + FEATURE_NAMES[i]: float(max(Xtr.std(0)[i], 1e-8)) for i in range(25)},
            "computed_from": "Fold %s train-only (%d rows)" % (fold_id, len(Xtr_rows))}
    norm_path = out_dir / ("FOLD%s_TRAIN_NORMALIZATION.json" % fold_id)
    with open(norm_path, "w") as f: json.dump(norm, f, indent=2)

    # ── 6. Split feature CSV ──
    print("\nSplitting feature CSV...")
    tv_rows = []; ho_rows = []
    tr_eps = set(); vl_eps = set(); ho_eps = set()
    with open(args.feature_dataset) as f:
        for r in csv.DictReader(f):
            t = int(r["task_idx"]); s = int(r["state_id"])
            if t in train_tasks:
                r["split"] = "train"; tv_rows.append(r); tr_eps.add((t, s))
            elif t == val_task:
                r["split"] = "val"; tv_rows.append(r); vl_eps.add((t, s))
            elif t == test_task:
                r["split"] = "held_out"; ho_rows.append(r); ho_eps.add((t, s))

    tv_path = out_dir / ("FOLD%s_TRAIN_VAL_FEATURE_DATASET.csv" % fold_id)
    with open(tv_path, "w", newline="") as f:
        fieldnames = ["task_idx","state_id","split","step"] + ["f_" + n for n in FEATURE_NAMES]
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        w.writeheader(); w.writerows(tv_rows)

    ho_path = out_dir / ("FOLD%s_HELDOUT_FEATURE_DATASET.csv" % fold_id)
    with open(ho_path, "w", newline="") as f:
        fieldnames = ["task_idx","state_id","split","step"] + ["f_" + n for n in FEATURE_NAMES]
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        w.writeheader(); w.writerows(ho_rows)

    # ── 7. Manifest ──
    with open(tv_path, "rb") as f: tv_sha = hashlib.sha256(f.read()).hexdigest()
    with open(ho_path, "rb") as f: ho_sha = hashlib.sha256(f.read()).hexdigest()
    with open(tl_path, "rb") as f: tl_sha = hashlib.sha256(f.read()).hexdigest()
    with open(norm_path, "rb") as f: nm_sha = hashlib.sha256(f.read()).hexdigest()
    with open(tc_path, "rb") as f: tc_sha = hashlib.sha256(f.read()).hexdigest()
    with open(args.protocol, "rb") as f: proto_sha = hashlib.sha256(f.read()).hexdigest()

    manifest = {
        "gate": "LOTO_FOLD%s_MANIFEST_V3" % fold_id, "fold": fold_id,
        "test_task": test_task, "test_task_name": TASKS[test_task],
        "val_task": val_task, "val_task_name": TASKS[val_task],
        "train_tasks": sorted(train_tasks),
        "train_episodes": len(tr_eps), "val_episodes": len(vl_eps), "test_episodes": len(ho_eps),
        "train_rows": sum(1 for r in tv_rows if r["split"]=="train"),
        "val_rows": sum(1 for r in tv_rows if r["split"]=="val"),
        "test_rows": len(ho_rows),
        "train_labels_rows": len(train_labels), "val_labels_rows": len(val_labels),
        "train_stable_carry_rows": tr_corr, "val_stable_carry_rows": vl_corr,
        "train_val_csv_sha256": tv_sha, "heldout_csv_sha256": ho_sha,
        "teacher_labels_sha256": tl_sha, "normalization_sha256": nm_sha,
        "teacher_config_sha256": tc_sha, "protocol_sha256": proto_sha,
        "source_commit": SOURCE_COMMIT,
        "heldout_labels_phase": "B only",
    }
    manifest_path = out_dir / ("FOLD%s_MANIFEST.json" % fold_id)
    with open(manifest_path, "w") as f: json.dump(manifest, f, indent=2)

    print("\n=== FOLD %s BUILD COMPLETE ===" % fold_id)
    print("Heldout labels: NOT generated (Phase B)")


if __name__ == "__main__":
    main()
