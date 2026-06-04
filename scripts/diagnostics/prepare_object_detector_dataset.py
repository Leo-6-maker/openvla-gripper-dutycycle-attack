#!/usr/bin/env python3
"""prepare_object_detector_dataset.py — Clean, filter, and convert Object-100 flat dataset
into temporal sequence data for causal TCN early-grasp detector training.

Inputs:
  --dataset-csv: flat per-step dataset (no_timestep_visual_proprio_student_dataset.csv)
  --phase-csv: phase event summary (object_phase_event_summary.csv)
  --manifest-csv: official clean artifact manifest

Outputs:
  tables/object_detector_episode_manifest_clean.csv  — per-episode filtering manifest
  data/detector/object_clean_sequences_v1.npz         — [N_ep, T_max, D] array
  data/detector/object_clean_feature_schema_v1.json   — feature names and stats
  data/detector/object_clean_label_schema_v1.json     — label encoding and rules
  tables/object_detector_split_plan_clean.csv          — train/val/test split
"""

from __future__ import annotations
import argparse, csv, json, os, sys
from pathlib import Path
from collections import defaultdict
import hashlib

import numpy as np

# ── Runtime features for detector input ──
DETECTOR_FEATURES = [
    "gripper_command",     # raw gripper action (0=OPEN, 0.996=CLOSE)
    "gripper_qpos",        # physical gripper position
    "gripper_width",       # gripper opening width
    "eef_x", "eef_y", "eef_z",           # end-effector position
    "eef_vx", "eef_vy", "eef_vz",        # end-effector velocity
    "action_dx", "action_dy", "action_dz",  # action delta
    "action_gripper",      # gripper action
]

# ── Label encoding ──
PHASE_CLASSES = ["pre_grasp", "grasp_formation", "post_grasp"]
PHASE_TO_ID = {p: i for i, p in enumerate(PHASE_CLASSES)}
WINDOW_HALF = 9  # grasp_formation window = T_gform +/- WINDOW_HALF


def parse_args():
    ap = argparse.ArgumentParser(description="Prepare Object detector dataset")
    ap.add_argument("--dataset-csv",
                    default="/data/liuyu/outputs/milestone_2e2_object100_privileged_artifact_rich_20260527/tables/no_timestep_visual_proprio_student_dataset.csv")
    ap.add_argument("--phase-csv",
                    default="tables/object_phase_event_summary.csv")
    ap.add_argument("--manifest-csv",
                    default="/data/liuyu/outputs/milestone_2e2_object100_privileged_artifact_rich_20260527/tables/official_clean_artifact_rich_manifest.csv")
    ap.add_argument("--output-manifest", default="tables/object_detector_episode_manifest_clean.csv")
    ap.add_argument("--output-npz", default="data/detector/object_clean_sequences_v1.npz")
    ap.add_argument("--output-feature-schema", default="data/detector/object_clean_feature_schema_v1.json")
    ap.add_argument("--output-label-schema", default="data/detector/object_clean_label_schema_v1.json")
    ap.add_argument("--output-split", default="tables/object_detector_split_plan_clean.csv")
    ap.add_argument("--allow-manual-review", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args()


def load_phase_labels(phase_csv):
    """Load per-episode phase events."""
    phases = {}
    if not os.path.exists(phase_csv):
        print(f"WARNING: Phase CSV not found: {phase_csv}")
        return phases
    with open(phase_csv, newline="") as f:
        for r in csv.DictReader(f):
            rid = r.get("run_id", "")
            if rid:
                phases[rid] = r
    return phases


def load_manifest(manifest_csv):
    """Load official manifest with success status."""
    manifest = {}
    if not os.path.exists(manifest_csv):
        return manifest
    with open(manifest_csv, newline="") as f:
        for r in csv.DictReader(f):
            rid = r.get("run_id", "")
            manifest[rid] = {
                "success": r.get("success", "").lower() == "true",
                "num_steps": int(r.get("num_steps", 0)),
            }
    return manifest


def scene_hash(scene, modulus=100):
    return int(hashlib.md5(scene.encode()).hexdigest(), 16) % modulus


def main():
    args = parse_args()

    if not os.path.exists(args.dataset_csv):
        print(f"ERROR: Dataset CSV not found: {args.dataset_csv}")
        sys.exit(1)

    # ── Load inputs ──
    print("Loading flat dataset...")
    with open(args.dataset_csv, newline="") as f:
        flat_rows = list(csv.DictReader(f))
    print(f"  {len(flat_rows)} rows")

    phases = load_phase_labels(args.phase_csv)
    print(f"  {len(phases)} episodes with phase labels")

    manifest = load_manifest(args.manifest_csv)
    print(f"  {len(manifest)} episodes in manifest")

    # ── Group by episode_key, sort by step_idx ──
    episodes = defaultdict(list)
    for r in flat_rows:
        ek = r.get("episode_key", r.get("run_id", "unknown"))
        episodes[ek].append(r)

    for ek in episodes:
        episodes[ek].sort(key=lambda r: int(r.get("step_idx", 0)))

    print(f"  {len(episodes)} unique episodes")

    # ── 1. Build episode manifest with filtering ──
    manifest_rows = []
    duplicate_groups = defaultdict(list)

    for ek, steps in sorted(episodes.items()):
        r0 = steps[0]
        run_id = r0.get("run_id", ek)
        task = r0.get("task_name", "?")
        state_id = r0.get("state_id", "?")
        seed = r0.get("seed", "?")
        scene = f"{task}_s{state_id}"
        n_steps = len(steps)

        # Group by (task, state_id, seed) for duplicate detection
        dup_key = (task, state_id, seed)
        duplicate_groups[dup_key].append(run_id)

        # Get phase info
        ph = phases.get(run_id, {})
        success_info = manifest.get(run_id, {})
        is_success = success_info.get("success", False)

        tg = ph.get("T_grasp_formation", "")
        tl = ph.get("T_qpos_stable", "")
        tr = ph.get("T_release", "")
        lv = ph.get("label_validity", "unknown")

        # Check runtime feature completeness
        feat_missing = 0
        for feat in DETECTOR_FEATURES:
            for s in steps[:10]:  # check first 10 steps
                v = s.get(feat, "")
                if v is None or v == "":
                    feat_missing += 1
                    break
        runtime_complete = feat_missing == 0

        manifest_rows.append({
            "episode_id": run_id,
            "episode_key": ek,
            "task_name": task,
            "scene_id": scene,
            "state_id": state_id,
            "seed": seed,
            "num_steps": n_steps,
            "clean_success": is_success,
            "label_validity": lv,
            "T_gform": tg,
            "T_lock": tl,
            "T_release": tr,
            "runtime_feature_complete": runtime_complete,
            "duplicate_excluded": False,
            "needs_manual_review": False,
            "train_usable": False,
            "exclusion_reason": "",
        })

    # ── Detect and mark duplicates ──
    dup_count = 0
    for dup_key, rids in duplicate_groups.items():
        if len(rids) <= 1:
            continue
        # Find entries for these rids
        entries = [r for r in manifest_rows if r["episode_id"] in rids]
        entries.sort(key=lambda r: (not r["clean_success"], -r["num_steps"]))
        # Keep first (prefer success, then longer trace), mark rest as duplicate
        for e in entries[1:]:
            e["duplicate_excluded"] = True
            e["exclusion_reason"] = "duplicate_rerun"
            dup_count += 1
    print(f"  Marked {dup_count} duplicate episodes")

    # ── Apply filtering ──
    train_usable_count = 0
    oscillation_episodes = {"obj100_ketchup_ketchup_s8", "obj100_milk_milk_s2"}

    for r in manifest_rows:
        reasons = []
        if r["duplicate_excluded"]:
            reasons.append("duplicate")
        if not r["clean_success"]:
            reasons.append("clean_failure")
        if r["label_validity"] not in ("heuristic", "partial_missing_qpos"):
            reasons.append(f"invalid_label_{r['label_validity']}")
        if not r["runtime_feature_complete"]:
            reasons.append("feature_incomplete")
        if r["episode_id"] in oscillation_episodes:
            r["needs_manual_review"] = True
            if not args.allow_manual_review:
                reasons.append("needs_manual_review_oscillation")

        if reasons:
            r["train_usable"] = False
            r["exclusion_reason"] = "+".join(reasons)
        else:
            r["train_usable"] = True
            train_usable_count += 1

    print(f"  Train-usable episodes: {train_usable_count}")

    # ── Write manifest ──
    manifest_fields = [
        "episode_id", "episode_key", "task_name", "scene_id", "state_id",
        "seed", "num_steps", "clean_success", "label_validity",
        "T_gform", "T_lock", "T_release", "runtime_feature_complete",
        "duplicate_excluded", "needs_manual_review", "train_usable",
        "exclusion_reason",
    ]
    os.makedirs(os.path.dirname(args.output_manifest) or ".", exist_ok=True)
    with open(args.output_manifest, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=manifest_fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(manifest_rows)
    print(f"Wrote {len(manifest_rows)} rows to {args.output_manifest}")

    if args.dry_run:
        return

    # ── 2. Build temporal sequence dataset ──
    train_episodes = {r["episode_id"] for r in manifest_rows if r["train_usable"]}
    print(f"Building sequence dataset from {len(train_episodes)} train-usable episodes...")

    sequences = {}
    max_len = 0

    for ek, steps in episodes.items():
        r0 = steps[0]
        run_id = r0.get("run_id", ek)
        if run_id not in train_episodes:
            continue

        T = len(steps)
        max_len = max(max_len, T)

        # Build feature matrix
        X = np.zeros((T, len(DETECTOR_FEATURES)), dtype=np.float32)
        for t, s in enumerate(steps):
            for d, feat in enumerate(DETECTOR_FEATURES):
                v = s.get(feat, "")
                if v is not None and v != "":
                    try:
                        X[t, d] = float(v)
                    except (ValueError, TypeError):
                        X[t, d] = 0.0

        # Build phase labels
        ph = phases.get(run_id, {})
        tg_str = ph.get("T_grasp_formation", "")
        tl_str = ph.get("T_qpos_stable", "")
        tg = int(tg_str) if tg_str else None
        tl = int(tl_str) if tl_str else None

        y = np.full(T, -100, dtype=np.int32)  # -100 = ignore index
        if tg is not None:
            lock = tl if tl is not None else tg + 18
            lock = min(lock, T)
            for t in range(T):
                if t < tg:
                    y[t] = PHASE_TO_ID["pre_grasp"]
                elif tg <= t < lock:
                    y[t] = PHASE_TO_ID["grasp_formation"]
                else:
                    y[t] = PHASE_TO_ID["post_grasp"]

        mask = np.ones(T, dtype=np.bool_)

        task = r0.get("task_name", "?")
        state_id = r0.get("state_id", "?")

        sequences[run_id] = {
            "X": X, "y": y, "mask": mask,
            "task_name": task, "state_id": state_id,
            "seed": r0.get("seed", "?"),
            "T_gform": tg,
            "num_steps": T,
        }

    # ── Pad to max_len ──
    N = len(sequences)
    D = len(DETECTOR_FEATURES)
    print(f"  {N} sequences, max_len={max_len}, feature_dim={D}")

    X_padded = np.zeros((N, max_len, D), dtype=np.float32)
    y_padded = np.full((N, max_len), -100, dtype=np.int32)
    mask_padded = np.zeros((N, max_len), dtype=np.bool_)
    episode_meta = []

    for i, (rid, seq) in enumerate(sorted(sequences.items())):
        T = seq["num_steps"]
        X_padded[i, :T] = seq["X"]
        y_padded[i, :T] = seq["y"]
        mask_padded[i, :T] = seq["mask"]
        episode_meta.append({
            "episode_id": rid,
            "task_name": seq["task_name"],
            "state_id": seq["state_id"],
            "seed": seq["seed"],
            "T_gform": seq["T_gform"],
            "num_steps": T,
        })

    # ── Save NPZ ──
    os.makedirs(os.path.dirname(args.output_npz) or ".", exist_ok=True)
    np.savez_compressed(
        args.output_npz,
        X=X_padded, y=y_padded, mask=mask_padded,
        feature_names=np.array(DETECTOR_FEATURES),
        max_len=max_len, feature_dim=D,
        episode_ids=np.array([m["episode_id"] for m in episode_meta]),
    )

    # Also save metadata CSV alongside
    meta_path = args.output_npz.replace(".npz", "_meta.csv")
    with open(meta_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["episode_id", "task_name", "state_id", "seed", "T_gform", "num_steps"])
        w.writeheader()
        w.writerows(episode_meta)
    print(f"Saved {N} sequences to {args.output_npz}")
    print(f"Saved metadata to {meta_path}")

    # ── Save feature schema ──
    feature_schema = {
        "version": "v1",
        "feature_names": DETECTOR_FEATURES,
        "feature_dim": D,
        "feature_source": "runtime_only",
        "forbidden_features": ["object_pose", "target_pose", "normalized_step", "attack_outcome"],
        "normalization": "none_yet",
        "description": "13-D runtime proprioceptive features for causal early-grasp detector",
    }
    with open(args.output_feature_schema, "w") as f:
        json.dump(feature_schema, f, indent=2)
    print(f"Saved feature schema to {args.output_feature_schema}")

    # ── Save label schema ──
    label_schema = {
        "version": "v1",
        "num_classes": 3,
        "classes": PHASE_CLASSES,
        "class_to_id": PHASE_TO_ID,
        "label_rule": {
            "pre_grasp": "t < T_grasp_formation",
            "grasp_formation": "T_grasp_formation <= t < T_lock (or T_gform+18 if no T_lock)",
            "post_grasp": "t >= T_lock (or T_gform+18)",
        },
        "ignore_index": -100,
        "label_source": "heuristic from gripper_command and gripper_qpos transitions",
    }
    with open(args.output_label_schema, "w") as f:
        json.dump(label_schema, f, indent=2)
    print(f"Saved label schema to {args.output_label_schema}")

    # ── 3. Split plan ──
    train_usable_eps = [r for r in manifest_rows if r["train_usable"]]
    all_tasks = sorted(set(r["task_name"] for r in train_usable_eps))

    # Task holdout split
    task_hash = {t: scene_hash(t, 100) for t in all_tasks}
    split_rows = []

    for r in train_usable_eps:
        task = r["task_name"]
        scene = r["scene_id"]

        # Task split
        th = task_hash[task]
        if th < 70:
            split_task = "train"
        elif th < 85:
            split_task = "val"
        else:
            split_task = "test"

        # State split (within each task, hold out 2 states)
        sh = scene_hash(scene, 10)
        if sh < 7:
            split_state = "train"
        elif sh < 8:
            split_state = "val"
        else:
            split_state = "test"

        split_rows.append({
            "episode_id": r["episode_id"],
            "task_name": task,
            "scene_id": scene,
            "state_id": r["state_id"],
            "seed": r["seed"],
            "split_task_holdout": split_task,
            "split_state_holdout": split_state,
        })

    with open(args.output_split, "w", newline="") as f:
        fieldnames = ["episode_id", "task_name", "scene_id", "state_id", "seed",
                      "split_task_holdout", "split_state_holdout"]
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(split_rows)

    task_counts = defaultdict(lambda: defaultdict(int))
    state_counts = defaultdict(lambda: defaultdict(int))
    for r in split_rows:
        task_counts[r["split_task_holdout"]][r["task_name"]] += 1
        state_counts[r["split_state_holdout"]][r["task_name"]] += 1

    print(f"Split plan saved to {args.output_split}")
    print(f"  Task holdout: train={sum(task_counts['train'].values())}, "
          f"val={sum(task_counts['val'].values())}, "
          f"test={sum(task_counts['test'].values())}")
    print(f"  State holdout: train={sum(state_counts['train'].values())}, "
          f"val={sum(state_counts['val'].values())}, "
          f"test={sum(state_counts['test'].values())}")

    # ── Summary ──
    print(f"\nDone. {train_usable_count} train-usable episodes, {N} in NPZ sequence dataset.")


if __name__ == "__main__":
    main()
