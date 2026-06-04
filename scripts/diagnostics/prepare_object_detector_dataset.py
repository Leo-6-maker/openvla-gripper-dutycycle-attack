#!/usr/bin/env python3
"""prepare_object_detector_dataset.py — Clean, filter, and convert Object-100 flat dataset
into temporal sequence data for causal TCN early-grasp detector training.

v2 changes (fixing TCN F1=0 root causes):
  - --grasp-label-mode fixed_window (default): grasp_formation = [T_gform, T_gform+18)
    No longer uses T_qpos_stable which was often T_gform+1, making the class 1-step tiny.
  - Feature normalization: train mean/std saved and applied.
  - Class distribution printed per split.
  - Naming: T_open_onset (first sustained OPEN), T_gform (qpos min after opening).
    raw_gripper < 0.5 = OPEN, NOT CLOSE.
"""

from __future__ import annotations
import argparse, csv, json, os, sys
from pathlib import Path
from collections import defaultdict
import hashlib

import numpy as np

# ── Runtime features (13-D) — no privileged fields ──
# raw_gripper < 0.5 = OPEN (canonical semantics)
DETECTOR_FEATURES = [
    "gripper_command",     # raw gripper action (<0.5=OPEN, >0.5=CLOSE)
    "gripper_qpos",        # physical gripper position (decreases = opening)
    "gripper_width",       # gripper opening width
    "eef_x", "eef_y", "eef_z",
    "eef_vx", "eef_vy", "eef_vz",
    "action_dx", "action_dy", "action_dz",
    "action_gripper",
]

PHASE_CLASSES = ["pre_grasp", "grasp_formation", "post_grasp"]
PHASE_TO_ID = {p: i for i, p in enumerate(PHASE_CLASSES)}


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-csv",
                    default="/data/liuyu/outputs/milestone_2e2_object100_privileged_artifact_rich_20260527/tables/no_timestep_visual_proprio_student_dataset.csv")
    ap.add_argument("--phase-csv", default="tables/object_phase_event_summary.csv")
    ap.add_argument("--manifest-csv",
                    default="/data/liuyu/outputs/milestone_2e2_object100_privileged_artifact_rich_20260527/tables/official_clean_artifact_rich_manifest.csv")
    ap.add_argument("--output-manifest", default="tables/object_detector_episode_manifest_clean.csv")
    ap.add_argument("--output-npz", default="data/detector/object_clean_sequences_v2.npz")
    ap.add_argument("--output-feature-schema", default="data/detector/object_clean_feature_schema_v2.json")
    ap.add_argument("--output-label-schema", default="data/detector/object_clean_label_schema_v2.json")
    ap.add_argument("--output-split", default="tables/object_detector_split_plan_clean.csv")
    ap.add_argument("--output-norm-stats", default="data/detector/object_clean_feature_norm_stats_v2.json")
    ap.add_argument("--grasp-label-mode", choices=["fixed_window"], default="fixed_window")
    ap.add_argument("--grasp-window-len", type=int, default=18)
    ap.add_argument("--allow-manual-review", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args()


def load_phase_labels(phase_csv):
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


def build_labels_fixed_window(T, T_gform, window_len):
    """grasp_formation = [T_gform, T_gform + window_len).

    This is the DEFAULT mode. No longer uses T_qpos_stable which was
    often T_gform+1 and made the grasp class 1-step tiny.
    """
    y = np.full(T, -100, dtype=np.int32)
    if T_gform is None:
        return y
    gform_end = min(T_gform + window_len, T)
    for t in range(T):
        if t < T_gform:
            y[t] = PHASE_TO_ID["pre_grasp"]
        elif T_gform <= t < gform_end:
            y[t] = PHASE_TO_ID["grasp_formation"]
        else:
            y[t] = PHASE_TO_ID["post_grasp"]
    return y


def main():
    args = parse_args()

    if not os.path.exists(args.dataset_csv):
        print(f"ERROR: Dataset CSV not found: {args.dataset_csv}")
        sys.exit(1)

    print("Loading flat dataset...")
    with open(args.dataset_csv, newline="") as f:
        flat_rows = list(csv.DictReader(f))
    print(f"  {len(flat_rows)} rows")

    phases = load_phase_labels(args.phase_csv)
    print(f"  {len(phases)} episodes with phase labels")

    manifest = load_manifest(args.manifest_csv)
    print(f"  {len(manifest)} episodes in manifest")

    episodes = defaultdict(list)
    for r in flat_rows:
        ek = r.get("episode_key", r.get("run_id", "unknown"))
        episodes[ek].append(r)
    for ek in episodes:
        episodes[ek].sort(key=lambda r: int(r.get("step_idx", 0)))
    print(f"  {len(episodes)} unique episodes")

    # ── 1. Build manifest with filtering ──
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

        dup_key = (task, state_id, seed)
        duplicate_groups[dup_key].append(run_id)

        ph = phases.get(run_id, {})
        success_info = manifest.get(run_id, {})
        is_success = success_info.get("success", False)

        tg = ph.get("T_grasp_formation", "")
        tl = ph.get("T_qpos_stable", "")
        tr = ph.get("T_release", "")
        lv = ph.get("label_validity", "unknown")

        feat_missing = 0
        for feat in DETECTOR_FEATURES:
            for s in steps[:10]:
                v = s.get(feat, "")
                if v is None or v == "":
                    feat_missing += 1
                    break
        runtime_complete = feat_missing == 0

        manifest_rows.append({
            "episode_id": run_id, "episode_key": ek, "task_name": task,
            "scene_id": scene, "state_id": state_id, "seed": seed,
            "num_steps": n_steps, "clean_success": is_success,
            "label_validity": lv,
            "T_open_onset": ph.get("T_close_onset", ""),  # renamed: raw<0.5 = OPEN
            "T_gform": tg, "T_qpos_open_min": tg,         # T_gform = qpos min after opening
            "T_qpos_stable": tl, "T_release": tr,
            "runtime_feature_complete": runtime_complete,
            "duplicate_excluded": False, "needs_manual_review": False,
            "train_usable": False, "exclusion_reason": "",
        })

    # Duplicate removal
    dup_count = 0
    for dup_key, rids in duplicate_groups.items():
        if len(rids) <= 1:
            continue
        entries = [r for r in manifest_rows if r["episode_id"] in rids]
        entries.sort(key=lambda r: (not r["clean_success"], -r["num_steps"]))
        for e in entries[1:]:
            e["duplicate_excluded"] = True
            e["exclusion_reason"] = "duplicate_rerun"
            dup_count += 1
    print(f"  Marked {dup_count} duplicate episodes")

    # Filtering
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

    os.makedirs(os.path.dirname(args.output_manifest) or ".", exist_ok=True)
    with open(args.output_manifest, "w", newline="") as f:
        fields = ["episode_id", "episode_key", "task_name", "scene_id", "state_id",
                  "seed", "num_steps", "clean_success", "label_validity",
                  "T_open_onset", "T_gform", "T_qpos_open_min", "T_qpos_stable",
                  "T_release", "runtime_feature_complete", "duplicate_excluded",
                  "needs_manual_review", "train_usable", "exclusion_reason"]
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader(); w.writerows(manifest_rows)
    print(f"Wrote {len(manifest_rows)} rows to {args.output_manifest}")

    if args.dry_run:
        return

    # ── 2. Build temporal sequences ──
    train_episodes = {r["episode_id"] for r in manifest_rows if r["train_usable"]}
    print(f"Building sequence dataset from {len(train_episodes)} train-usable episodes...")

    sequences = {}
    class_counts = defaultdict(int)  # per-class step count
    grasp_lengths = []
    max_len = 0

    for ek, steps in episodes.items():
        r0 = steps[0]
        run_id = r0.get("run_id", ek)
        if run_id not in train_episodes:
            continue

        T = len(steps)
        max_len = max(max_len, T)

        X = np.zeros((T, len(DETECTOR_FEATURES)), dtype=np.float32)
        for t, s in enumerate(steps):
            for d, feat in enumerate(DETECTOR_FEATURES):
                v = s.get(feat, "")
                if v is not None and v != "":
                    try: X[t, d] = float(v)
                    except (ValueError, TypeError): X[t, d] = 0.0

        ph = phases.get(run_id, {})
        tg_str = ph.get("T_grasp_formation", "")
        tg = int(tg_str) if tg_str else None

        # FIXED: use fixed_window, NOT T_qpos_stable
        y = build_labels_fixed_window(T, tg, args.grasp_window_len)

        for c in range(len(PHASE_CLASSES)):
            cnt = int((y == c).sum())
            class_counts[c] += cnt
        if tg is not None:
            gform_len = min(T - tg, args.grasp_window_len)
            grasp_lengths.append(gform_len)

        mask = np.ones(T, dtype=np.bool_)
        sequences[run_id] = {
            "X": X, "y": y, "mask": mask,
            "task_name": r0.get("task_name", "?"),
            "state_id": r0.get("state_id", "?"),
            "seed": r0.get("seed", "?"),
            "T_gform": tg, "num_steps": T,
        }

    # ── Print class distribution ──
    total_labeled = sum(class_counts.values())
    print(f"\nClass distribution ({args.grasp_label_mode}, window={args.grasp_window_len}):")
    for cls_name, cls_id in sorted(PHASE_TO_ID.items(), key=lambda x: x[1]):
        cnt = class_counts.get(cls_id, 0)
        pct = 100 * cnt / max(total_labeled, 1)
        bar = "#" * int(pct / 2)
        print(f"  {cls_name:20s}: {cnt:6d} ({pct:5.1f}%) {bar}")
    print(f"  grasp_formation len: min={min(grasp_lengths) if grasp_lengths else 'N/A'}, "
          f"mean={np.mean(grasp_lengths):.1f}, max={max(grasp_lengths) if grasp_lengths else 'N/A'}")

    # ── Pad and save ──
    N = len(sequences)
    D = len(DETECTOR_FEATURES)
    print(f"\n  {N} sequences, max_len={max_len}, feature_dim={D}")

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
            "episode_id": rid, "task_name": seq["task_name"],
            "state_id": seq["state_id"], "seed": seq["seed"],
            "T_gform": seq["T_gform"], "num_steps": T,
        })

    # ── Feature normalization: train mean/std ──
    split_csv_path = args.output_split
    # Compute split first so we know which episodes are train
    train_usable_eps = [r for r in manifest_rows if r["train_usable"]]
    all_tasks = sorted(set(r["task_name"] for r in train_usable_eps))
    task_hash = {t: scene_hash(t, 100) for t in all_tasks}
    split_rows = []
    for r in train_usable_eps:
        task = r["task_name"]; scene = r["scene_id"]
        th = task_hash[task]; sh = scene_hash(scene, 10)
        split_rows.append({
            "episode_id": r["episode_id"], "task_name": task, "scene_id": scene,
            "state_id": r["state_id"], "seed": r["seed"],
            "split_task_holdout": "train" if th < 70 else ("val" if th < 85 else "test"),
            "split_state_holdout": "train" if sh < 7 else ("val" if sh < 8 else "test"),
        })

    # Identify train split episodes
    train_ids = {r["episode_id"] for r in split_rows if r["split_state_holdout"] == "train"}
    train_X = np.concatenate([seq["X"] for rid, seq in sequences.items() if rid in train_ids], axis=0)
    feat_mean = train_X.mean(axis=0).tolist()
    feat_std = train_X.std(axis=0).tolist()
    # Clip zero std
    feat_std = [max(s, 1e-8) for s in feat_std]

    print(f"  Feature normalization: computed from {len(train_ids)} train episodes")
    for i, fn in enumerate(DETECTOR_FEATURES):
        print(f"    {fn:20s}: mean={feat_mean[i]:8.4f}  std={feat_std[i]:8.4f}")

    # Apply normalization to all sequences
    for rid, seq in sequences.items():
        seq["X"] = (seq["X"] - np.array(feat_mean)) / np.array(feat_std)

    # Re-pad with normalized features
    for i, (rid, seq) in enumerate(sorted(sequences.items())):
        T = seq["num_steps"]
        X_padded[i, :T] = seq["X"]

    # Save norm stats
    norm_stats = {
        "version": "v2",
        "feature_names": DETECTOR_FEATURES,
        "mean": feat_mean,
        "std": feat_std,
        "computed_on": "train_split_state_holdout",
        "n_train_episodes": len(train_ids),
    }
    os.makedirs(os.path.dirname(args.output_norm_stats) or ".", exist_ok=True)
    with open(args.output_norm_stats, "w") as f:
        json.dump(norm_stats, f, indent=2)
    print(f"Saved norm stats to {args.output_norm_stats}")

    # Save NPZ
    os.makedirs(os.path.dirname(args.output_npz) or ".", exist_ok=True)
    np.savez_compressed(
        args.output_npz,
        X=X_padded, y=y_padded, mask=mask_padded,
        feature_names=np.array(DETECTOR_FEATURES),
        max_len=max_len, feature_dim=D,
        episode_ids=np.array([m["episode_id"] for m in episode_meta]),
    )
    meta_path = args.output_npz.replace(".npz", "_meta.csv")
    with open(meta_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["episode_id", "task_name", "state_id", "seed", "T_gform", "num_steps"])
        w.writeheader(); w.writerows(episode_meta)
    print(f"Saved {N} sequences to {args.output_npz}")

    # Save feature schema
    feature_schema = {
        "version": "v2",
        "feature_names": DETECTOR_FEATURES,
        "feature_dim": D,
        "feature_source": "runtime_only",
        "forbidden_features": ["object_pose", "target_pose", "normalized_step", "attack_outcome"],
        "normalization": "train_mean_std",
        "norm_stats_path": args.output_norm_stats,
        "description": "13-D normalized runtime proprioceptive features for causal early-grasp detector",
    }
    with open(args.output_feature_schema, "w") as f:
        json.dump(feature_schema, f, indent=2)

    # Save label schema
    label_schema = {
        "version": "v2",
        "num_classes": 3,
        "classes": PHASE_CLASSES,
        "class_to_id": PHASE_TO_ID,
        "label_mode": args.grasp_label_mode,
        "grasp_window_len": args.grasp_window_len,
        "label_rule": f"pre_grasp: t < T_gform; grasp_formation: T_gform <= t < T_gform+{args.grasp_window_len}; post_grasp: after",
        "T_gform_definition": "Heuristic: step where gripper_qpos reaches minimum after first sustained OPEN command (gripper_command < 0.5 for K=2 consecutive steps). This is a runtime-only early-grasp/coupling trigger candidate, NOT a privileged contact/sensor label.",
        "ignore_index": -100,
    }
    with open(args.output_label_schema, "w") as f:
        json.dump(label_schema, f, indent=2)

    # Save split
    with open(args.output_split, "w", newline="") as f:
        fieldnames = ["episode_id", "task_name", "scene_id", "state_id", "seed",
                      "split_task_holdout", "split_state_holdout"]
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader(); w.writerows(split_rows)

    sc = defaultdict(lambda: defaultdict(int))
    for r in split_rows:
        sc[r["split_state_holdout"]][r["task_name"]] += 1
    print(f"Split: train={sum(sc['train'].values())}, val={sum(sc['val'].values())}, test={sum(sc['test'].values())}")

    print(f"\nDone. {train_usable_count} train-usable, {N} in NPZ.")
    print(f"Class distribution: pre={class_counts[0]}, grasp={class_counts[1]}, post={class_counts[2]}")


if __name__ == "__main__":
    main()
