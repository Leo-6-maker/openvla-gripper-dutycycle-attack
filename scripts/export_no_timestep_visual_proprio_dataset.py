#!/usr/bin/env python3
"""Export no-timestep visual/proprio student dataset from artifact-rich step_records.

Excludes normalized_step and absolute timestep as model features.
Teacher labels and privileged fields are labels/eval only, not features.
"""
import argparse, csv, json, sys
from pathlib import Path
from collections import defaultdict

# Deployment-allowed feature groups (matching deployment_detector_feature_schema.yaml)
IDENTITY_COLS = ["run_id", "episode_key", "suite", "task_id", "task_name",
                 "task_instruction", "state_id", "seed", "step_idx"]

TASK_LANG_COLS = ["parsed_target_object", "parsed_target_receptacle",
                  "mechanism_type", "parse_confidence"]

VISUAL_COLS = ["image_path", "image_path_available", "visual_feature_path",
               "visual_feature_available", "visual_encoder_name"]

GRIPPER_COLS = ["gripper_command", "gripper_qpos", "gripper_width",
                "recent_close_streak", "recent_open_streak", "recent_gripper_flip_count"]

EEF_COLS = ["eef_x", "eef_y", "eef_z", "eef_vx", "eef_vy", "eef_vz"]

ACTION_COLS = ["action_dx", "action_dy", "action_dz", "action_gripper"]

LABEL_COLS = ["teacher_phase", "teacher_hazard", "teacher_release_safe", "teacher_confidence"]

EVAL_COLS = ["teacher_window_start", "teacher_window_end", "teacher_anchor_step"]

PROVENANCE_COLS = ["deployed_features_use_privileged_state", "uses_attack_outcome",
                   "uses_manual_outcome", "normalized_step_in_deployment_features"]

FORBIDDEN_DEPLOYMENT = ["normalized_step", "object_pose", "target_pose",
                        "object_to_target_distance", "future_done", "success",
                        "attack_outcome", "oracle_outcome", "random_outcome",
                        "manual_outcome", "teacher_window_start", "teacher_window_end",
                        "teacher_anchor_step", "absolute_timestep", "max_steps"]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact_root", required=True)
    ap.add_argument("--teacher_label_root", default="")
    ap.add_argument("--visual_feature_manifest", default="")
    ap.add_argument("--output_root", required=True)
    return ap.parse_args()


def main():
    args = parse_args()
    root = Path(args.artifact_root)
    out = Path(args.output_root)
    out.mkdir(parents=True, exist_ok=True)
    tables_dir = out / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    runs_dir = root / "runs"
    dataset_rows = []
    n_total = 0
    n_with_labels = 0
    n_with_images = 0

    # Load visual feature manifest if available
    visual_map = {}
    if args.visual_feature_manifest:
        vf_path = Path(args.visual_feature_manifest)
        if vf_path.exists():
            with open(vf_path) as f:
                for row in csv.DictReader(f):
                    key = (row.get("run_id", ""), int(row.get("step_idx", 0) or 0))
                    visual_map[key] = row

    # Load teacher labels if available
    teacher_map = {}
    if args.teacher_label_root:
        tl_root = Path(args.teacher_label_root)
        tl_file = tl_root / "tables" / "teacher_phase_labels.csv"
        if tl_file.exists():
            with open(tl_file) as f:
                for row in csv.DictReader(f):
                    key = (row.get("run_id", ""), int(row.get("step_idx", 0) or 0))
                    teacher_map[key] = row

    # Walk step_records
    for suite_dir in sorted(runs_dir.glob("*")):
        if not suite_dir.is_dir():
            continue
        suite = suite_dir.name

        for run_dir in sorted(suite_dir.glob("*_state*")):
            if not run_dir.is_dir():
                continue
            step_file = run_dir / "step_records.jsonl"
            if not step_file.exists():
                continue

            with open(step_file) as f:
                for line in f:
                    if not line.strip():
                        continue
                    rec = json.loads(line)
                    n_total += 1

                    run_id = rec.get("run_id", run_dir.name)
                    step_idx = int(rec.get("step_idx", 0))
                    policy_step = int(rec.get("policy_step_idx", 0))

                    # Skip wait steps
                    if rec.get("phase") == "wait":
                        continue

                    # Episode key
                    ep_key = f"{rec.get('suite',suite)}::{rec.get('task_id','')}::{rec.get('task_name','')}::{rec.get('state_id','')}::{rec.get('seed','')}::{run_id}"

                    row = {}

                    # Identity (evaluation only, not model features)
                    row["run_id"] = run_id
                    row["episode_key"] = ep_key
                    row["suite"] = rec.get("suite", suite)
                    row["task_id"] = rec.get("task_id", "")
                    row["task_name"] = rec.get("task_name", "")
                    row["task_instruction"] = rec.get("task_instruction", "")
                    row["state_id"] = rec.get("state_id", "")
                    row["seed"] = rec.get("seed", "")
                    row["step_idx"] = step_idx

                    # Task language (placeholder — will be parsed by task_language_parser)
                    row["parsed_target_object"] = ""
                    row["parsed_target_receptacle"] = ""
                    row["mechanism_type"] = ""
                    row["parse_confidence"] = ""

                    # Visual
                    img_path = rec.get("image_path", "")
                    img_avail = rec.get("image_path_available", False)
                    if img_avail:
                        n_with_images += 1
                    vf_key = (run_id, step_idx)
                    vf_row = visual_map.get(vf_key, {})
                    row["image_path"] = img_path
                    row["image_path_available"] = img_avail
                    row["visual_feature_path"] = vf_row.get("visual_feature_path", "")
                    row["visual_feature_available"] = vf_row.get("visual_feature_available", False)
                    row["visual_encoder_name"] = vf_row.get("visual_encoder_name", "")

                    # Gripper
                    row["gripper_command"] = rec.get("gripper_command", 0)
                    row["gripper_qpos"] = rec.get("gripper_qpos", 0)
                    row["gripper_width"] = rec.get("gripper_width", 0)
                    row["recent_close_streak"] = 0
                    row["recent_open_streak"] = 0
                    row["recent_gripper_flip_count"] = 0

                    # EEF
                    row["eef_x"] = rec.get("eef_x", 0)
                    row["eef_y"] = rec.get("eef_y", 0)
                    row["eef_z"] = rec.get("eef_z", 0)
                    row["eef_vx"] = rec.get("eef_vx", 0)
                    row["eef_vy"] = rec.get("eef_vy", 0)
                    row["eef_vz"] = rec.get("eef_vz", 0)

                    # Action history
                    env_action = rec.get("env_action", [0]*7)
                    if isinstance(env_action, list) and len(env_action) >= 7:
                        row["action_dx"] = env_action[0]
                        row["action_dy"] = env_action[1]
                        row["action_dz"] = env_action[2]
                        row["action_gripper"] = env_action[-1]
                    else:
                        row["action_dx"] = rec.get("action_dx", 0)
                        row["action_dy"] = rec.get("action_dy", 0)
                        row["action_dz"] = rec.get("action_dz", 0)
                        row["action_gripper"] = rec.get("action_gripper", 0)

                    # Labels (if available)
                    tl_key = (run_id, step_idx)
                    tl_row = teacher_map.get(tl_key, {})
                    row["teacher_phase"] = tl_row.get("teacher_phase", "")
                    row["teacher_hazard"] = tl_row.get("teacher_hazard", "")
                    row["teacher_release_safe"] = tl_row.get("teacher_release_safe", "")
                    row["teacher_confidence"] = tl_row.get("teacher_confidence", "")
                    if row["teacher_phase"]:
                        n_with_labels += 1

                    # Evaluation-only
                    row["teacher_window_start"] = tl_row.get("teacher_window_start", "")
                    row["teacher_window_end"] = tl_row.get("teacher_window_end", "")
                    row["teacher_anchor_step"] = tl_row.get("teacher_anchor_step", "")

                    # Provenance
                    row["deployed_features_use_privileged_state"] = "false"
                    row["uses_attack_outcome"] = "false"
                    row["uses_manual_outcome"] = "false"
                    row["normalized_step_in_deployment_features"] = "false"

                    dataset_rows.append(row)

    if not dataset_rows:
        print("No policy steps found. Dataset empty.")
        return 1

    # Verify no forbidden features
    deployment_feature_names = (TASK_LANG_COLS + VISUAL_COLS + GRIPPER_COLS + EEF_COLS + ACTION_COLS)
    for forbidden in FORBIDDEN_DEPLOYMENT:
        if forbidden in deployment_feature_names:
            print(f"ERROR: Forbidden feature {forbidden} in deployment features!")
            return 1
    print("Deployment feature audit: PASSED (no forbidden features)")

    # Write dataset
    output_fields = (IDENTITY_COLS + TASK_LANG_COLS + VISUAL_COLS + GRIPPER_COLS +
                     EEF_COLS + ACTION_COLS + LABEL_COLS + EVAL_COLS + PROVENANCE_COLS)
    output_path = tables_dir / "no_timestep_visual_proprio_student_dataset.csv"
    with open(output_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=output_fields, extrasaction='ignore')
        w.writeheader()
        w.writerows(dataset_rows)

    print(f"Dataset exported: {len(dataset_rows)} policy steps")
    print(f"  Steps with images: {n_with_images}")
    print(f"  Steps with teacher labels: {n_with_labels}")
    print(f"  normalized_step in deployment features: false")
    print(f"Output: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
