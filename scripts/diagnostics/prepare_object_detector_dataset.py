#!/usr/bin/env python3
"""prepare_object_detector_dataset.py v3 — Save X_raw and X_norm in NPZ.

v3: NPZ contains both X_raw and X_norm so that:
  - TCN model uses X_norm (standardized)
  - Rule baseline uses X_raw (raw gripper_command < 0.5 canonical semantics)
  - Proposal scripts use X_raw for clean_natural_open_ratio

Default split: split_state_holdout (NOT task-holdout).
Current v3 results are state-holdout smoke only, not task-holdout generalization.
"""

from __future__ import annotations
import argparse, csv, json, os, sys
from pathlib import Path
from collections import defaultdict
import hashlib
import numpy as np

DETECTOR_FEATURES = [
    "gripper_command",
    "gripper_qpos",
    "gripper_width",
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
    ap.add_argument("--output-npz", default="data/detector/object_clean_sequences_v3.npz")
    ap.add_argument("--output-feature-schema", default="data/detector/object_clean_feature_schema_v3.json")
    ap.add_argument("--output-label-schema", default="data/detector/object_clean_label_schema_v3.json")
    ap.add_argument("--output-split", default="tables/object_detector_split_plan_clean.csv")
    ap.add_argument("--output-norm-stats", default="data/detector/object_clean_feature_norm_stats_v3.json")
    ap.add_argument("--grasp-label-mode", choices=["fixed_window"], default="fixed_window")
    ap.add_argument("--grasp-window-len", type=int, default=18)
    ap.add_argument("--allow-manual-review", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args()


def load_phase_labels(phase_csv):
    phases = {}
    if not os.path.exists(phase_csv): return phases
    with open(phase_csv, newline="") as f:
        for r in csv.DictReader(f):
            if r.get("run_id"): phases[r["run_id"]] = r
    return phases


def load_manifest(manifest_csv):
    m = {}
    if not os.path.exists(manifest_csv): return m
    with open(manifest_csv, newline="") as f:
        for r in csv.DictReader(f):
            m[r.get("run_id","")] = {"success": r.get("success","").lower()=="true",
                                     "num_steps": int(r.get("num_steps",0))}
    return m


def scene_hash(scene, modulus=100):
    return int(hashlib.md5(scene.encode()).hexdigest(), 16) % modulus


def build_labels_fixed_window(T, T_gform, window_len):
    y = np.full(T, -100, dtype=np.int32)
    if T_gform is None: return y
    gform_end = min(T_gform + window_len, T)
    for t in range(T):
        if t < T_gform: y[t] = 0
        elif t < gform_end: y[t] = 1
        else: y[t] = 2
    return y


def main():
    args = parse_args()
    if not os.path.exists(args.dataset_csv):
        print(f"ERROR: Dataset CSV not found: {args.dataset_csv}"); sys.exit(1)

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

    # ── 1. Manifest ──
    manifest_rows = []; duplicate_groups = defaultdict(list)
    for ek, steps in sorted(episodes.items()):
        r0 = steps[0]; run_id = r0.get("run_id", ek)
        task = r0.get("task_name","?"); state_id = r0.get("state_id","?")
        seed = r0.get("seed","?"); scene = f"{task}_s{state_id}"
        n_steps = len(steps)
        duplicate_groups[(task, state_id, seed)].append(run_id)
        ph = phases.get(run_id, {})
        sinfo = manifest.get(run_id, {})
        is_success = sinfo.get("success", False)
        tg = ph.get("T_grasp_formation","")
        lv = ph.get("label_validity","unknown")
        feat_missing = sum(1 for feat in DETECTOR_FEATURES
                          if any(s.get(feat,"") in (None,"") for s in steps[:10]))
        manifest_rows.append(dict(episode_id=run_id, episode_key=ek, task_name=task,
            scene_id=scene, state_id=state_id, seed=seed, num_steps=n_steps,
            clean_success=is_success, label_validity=lv,
            T_open_onset=ph.get("T_close_onset",""),
            T_gform=tg, T_qpos_open_min=tg,
            T_qpos_stable=ph.get("T_qpos_stable",""), T_release=ph.get("T_release",""),
            runtime_feature_complete=feat_missing==0,
            duplicate_excluded=False, needs_manual_review=False,
            train_usable=False, exclusion_reason=""))

    # Duplicates
    dup_count = 0
    for dup_key, rids in duplicate_groups.items():
        if len(rids) <= 1: continue
        entries = [r for r in manifest_rows if r["episode_id"] in rids]
        entries.sort(key=lambda r: (not r["clean_success"], -r["num_steps"]))
        for e in entries[1:]:
            e["duplicate_excluded"] = True; e["exclusion_reason"] = "duplicate_rerun"
            dup_count += 1
    print(f"  Marked {dup_count} duplicate episodes")

    # Filter
    train_usable_count = 0
    oscillation = {"obj100_ketchup_ketchup_s8", "obj100_milk_milk_s2"}
    for r in manifest_rows:
        reasons = []
        if r["duplicate_excluded"]: reasons.append("duplicate")
        if not r["clean_success"]: reasons.append("clean_failure")
        if r["label_validity"] not in ("heuristic","partial_missing_qpos"):
            reasons.append(f"invalid_label_{r['label_validity']}")
        if not r["runtime_feature_complete"]: reasons.append("feature_incomplete")
        if r["episode_id"] in oscillation:
            r["needs_manual_review"] = True
            if not args.allow_manual_review: reasons.append("needs_manual_review_oscillation")
        if reasons:
            r["train_usable"] = False; r["exclusion_reason"] = "+".join(reasons)
        else:
            r["train_usable"] = True; train_usable_count += 1
    print(f"  Train-usable: {train_usable_count}")

    os.makedirs(os.path.dirname(args.output_manifest) or ".", exist_ok=True)
    fields = ["episode_id","episode_key","task_name","scene_id","state_id","seed",
              "num_steps","clean_success","label_validity","T_open_onset","T_gform",
              "T_qpos_open_min","T_qpos_stable","T_release","runtime_feature_complete",
              "duplicate_excluded","needs_manual_review","train_usable","exclusion_reason"]
    with open(args.output_manifest, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader(); w.writerows(manifest_rows)
    print(f"Wrote manifest: {args.output_manifest}")
    if args.dry_run: return

    # ── 2. Build sequences with X_raw ──
    train_eps = {r["episode_id"] for r in manifest_rows if r["train_usable"]}
    print(f"Building sequences from {len(train_eps)} episodes...")
    sequences = {}; class_counts = defaultdict(int); grasp_lengths = []; max_len = 0

    for ek, steps in episodes.items():
        r0 = steps[0]; run_id = r0.get("run_id", ek)
        if run_id not in train_eps: continue
        T = len(steps); max_len = max(max_len, T)
        X = np.zeros((T, len(DETECTOR_FEATURES)), dtype=np.float32)
        for t, s in enumerate(steps):
            for d, feat in enumerate(DETECTOR_FEATURES):
                v = s.get(feat,"")
                if v not in (None,""):
                    try: X[t,d] = float(v)
                    except (ValueError,TypeError): pass
        ph = phases.get(run_id,{})
        tg_str = ph.get("T_grasp_formation",""); tg = int(tg_str) if tg_str else None
        y = build_labels_fixed_window(T, tg, args.grasp_window_len)
        for c in range(3): class_counts[c] += int((y==c).sum())
        if tg is not None: grasp_lengths.append(min(T-tg, args.grasp_window_len))
        sequences[run_id] = dict(X_raw=X, y=y, mask=np.ones(T,dtype=np.bool_),
            task_name=r0.get("task_name","?"), state_id=r0.get("state_id","?"),
            seed=r0.get("seed","?"), T_gform=tg, num_steps=T)

    # Class stats
    total_labeled = sum(class_counts.values())
    print(f"\nClass distribution ({args.grasp_label_mode}, window={args.grasp_window_len}):")
    for name, cid in sorted(PHASE_TO_ID.items(), key=lambda x:x[1]):
        cnt = class_counts.get(cid,0)
        print(f"  {name:20s}: {cnt:6d} ({100*cnt/max(total_labeled,1):5.1f}%) {'#'*int(100*cnt/max(total_labeled,1)/2)}")
    print(f"  grasp_formation len: min={min(grasp_lengths) if grasp_lengths else 'N/A'}, "
          f"mean={np.mean(grasp_lengths):.1f}, max={max(grasp_lengths) if grasp_lengths else 'N/A'}")

    # ── Split ──
    train_usable_eps = [r for r in manifest_rows if r["train_usable"]]
    all_tasks = sorted(set(r["task_name"] for r in train_usable_eps))
    task_hash = {t: scene_hash(t,100) for t in all_tasks}
    split_rows = []
    for r in train_usable_eps:
        th = task_hash[r["task_name"]]; sh = scene_hash(r["scene_id"],10)
        split_rows.append(dict(episode_id=r["episode_id"], task_name=r["task_name"],
            scene_id=r["scene_id"], state_id=r["state_id"], seed=r["seed"],
            split_task_holdout="train" if th<70 else ("val" if th<85 else "test"),
            split_state_holdout="train" if sh<7 else ("val" if sh<8 else "test")))

    train_ids = {r["episode_id"] for r in split_rows if r["split_state_holdout"]=="train"}
    train_X_raw = np.concatenate([s["X_raw"] for rid,s in sequences.items() if rid in train_ids], axis=0)
    feat_mean = train_X_raw.mean(axis=0).tolist()
    feat_std = [max(s,1e-8) for s in train_X_raw.std(axis=0).tolist()]
    print(f"\nNormalization from {len(train_ids)} train episodes:")
    for i,fn in enumerate(DETECTOR_FEATURES):
        print(f"  {fn:20s}: mean={feat_mean[i]:8.4f}  std={feat_std[i]:8.4f}")

    # Build X_norm
    for rid, seq in sequences.items():
        seq["X_norm"] = (seq["X_raw"].copy() - np.array(feat_mean)) / np.array(feat_std)

    # ── Pad and save NPZ ──
    N = len(sequences); D = len(DETECTOR_FEATURES)
    Xr = np.zeros((N,max_len,D), dtype=np.float32)
    Xn = np.zeros((N,max_len,D), dtype=np.float32)
    yp = np.full((N,max_len), -100, dtype=np.int32)
    mp = np.zeros((N,max_len), dtype=np.bool_)
    episode_meta = []
    for i,(rid,seq) in enumerate(sorted(sequences.items())):
        T = seq["num_steps"]
        Xr[i,:T]=seq["X_raw"]; Xn[i,:T]=seq["X_norm"]
        yp[i,:T]=seq["y"]; mp[i,:T]=seq["mask"]
        episode_meta.append(dict(episode_id=rid, task_name=seq["task_name"],
            state_id=seq["state_id"], seed=seq["seed"], T_gform=seq["T_gform"], num_steps=T))

    os.makedirs(os.path.dirname(args.output_npz) or ".", exist_ok=True)
    np.savez_compressed(args.output_npz,
        X_raw=Xr, X_norm=Xn, y=yp, mask=mp,
        feature_names=np.array(DETECTOR_FEATURES), max_len=max_len, feature_dim=D,
        episode_ids=np.array([m["episode_id"] for m in episode_meta]))
    print(f"Saved NPZ: {args.output_npz}  (X_raw + X_norm, {N} eps)")

    # Meta CSV
    mp_csv = args.output_npz.replace(".npz","_meta.csv")
    with open(mp_csv,"w",newline="") as f:
        w = csv.DictWriter(f, fieldnames=["episode_id","task_name","state_id","seed","T_gform","num_steps"])
        w.writeheader(); w.writerows(episode_meta)

    # Norm stats
    norm_stats = dict(version="v3", feature_names=DETECTOR_FEATURES, mean=feat_mean, std=feat_std,
        computed_on="train_split_state_holdout", n_train_episodes=len(train_ids),
        note="X_raw=raw features for rule baseline, X_norm=standardized for TCN training")
    with open(args.output_norm_stats,"w") as f: json.dump(norm_stats,f,indent=2)

    # Schemas
    fs = dict(version="v3", feature_names=DETECTOR_FEATURES, feature_dim=D,
        feature_source="runtime_only", normalization="train_mean_std_on_X_norm",
        norm_stats_path=args.output_norm_stats,
        npz_keys=["X_raw","X_norm","y","mask"],
        X_raw_for="rule_baseline_and_clean_open_ratio",
        X_norm_for="TCN_model_training_and_inference")
    with open(args.output_feature_schema,"w") as f: json.dump(fs,f,indent=2)

    ls = dict(version="v3", num_classes=3, classes=PHASE_CLASSES, class_to_id=PHASE_TO_ID,
        label_mode=args.grasp_label_mode, grasp_window_len=args.grasp_window_len,
        label_rule=f"pre_grasp: t<T_gform; grasp: T_gform<=t<T_gform+{args.grasp_window_len}; post: after",
        T_gform_definition="Heuristic qpos-min after first sustained OPEN (gripper_command<0.5 for K=2). Runtime-only trigger candidate, NOT privileged contact label.",
        ignore_index=-100)
    with open(args.output_label_schema,"w") as f: json.dump(ls,f,indent=2)

    # Split
    with open(args.output_split,"w",newline="") as f:
        fn2 = ["episode_id","task_name","scene_id","state_id","seed","split_task_holdout","split_state_holdout"]
        w = csv.DictWriter(f, fieldnames=fn2, extrasaction="ignore")
        w.writeheader(); w.writerows(split_rows)

    sc = defaultdict(lambda: defaultdict(int))
    for r in split_rows: sc[r["split_state_holdout"]][r["task_name"]] += 1
    print(f"Split: train={sum(sc['train'].values())}, val={sum(sc['val'].values())}, test={sum(sc['test'].values())}")
    print(f"\nDone. {train_usable_count} usable, {N} in NPZ.")
    print(f"Class dist: pre={class_counts[0]}, grasp={class_counts[1]}, post={class_counts[2]}")


if __name__ == "__main__":
    main()
