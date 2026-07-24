#!/usr/bin/env python3
"""Phase 6D: Prepare Primary-only and Primary-oversampled step datasets for ablation.
Creates:
  1. SC5_V2_STEP_DATASET_PRIMARY_ONLY.csv  — train=Primary 250 only, val unchanged
  2. SC5_V2_STEP_DATASET_PRIMARY_OVERSAMPLED.csv — Primary 250 with NC episodes
     oversampled to match the total negative-step count of Primary+Reserve (M2).

Run on GPU server with access to the full step dataset.
"""
import argparse, csv, hashlib, json, os, sys
from collections import Counter, defaultdict
from pathlib import Path
from copy import deepcopy

REPO = Path(__file__).resolve().parents[2]


def load_dataset(csv_path):
    """Load step dataset, return (all_rows, train_episodes, val_episodes)."""
    rows = list(csv.DictReader(open(csv_path)))
    tr_eps = defaultdict(list)
    vl_eps = defaultdict(list)
    for r in rows:
        eid = r["episode_id"]
        if r["split"] == "train":
            tr_eps[eid].append(r)
        elif r["split"] == "val":
            vl_eps[eid].append(r)
    return rows, tr_eps, vl_eps


def make_label_key(rows):
    """Build label lookup key from episode rows. Labels keyed by (task, state, source)."""
    r0 = rows[0]
    task = int(r0["task_idx"])
    state = int(r0["parent_state_id"])
    source = r0.get("source_pool", "primary")
    return (task, state, source)


def count_nc_steps(ep_rows, labels):
    """Count total steps in NC (no-corridor) episodes."""
    nc = 0
    tv = 0
    for eid, rows in ep_rows.items():
        key = make_label_key(rows)
        lbl = labels.get(key, {})
        is_tv = lbl.get("teacher_valid") == "True"
        if is_tv:
            tv += len(rows)
        else:
            nc += len(rows)
    return tv, nc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_csv", required=True, help="Path to full SC5_V2_STEP_DATASET.csv")
    ap.add_argument("--labels_csv", required=True, help="Path to train+dev combined labels CSV")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--allow_overwrite", action="store_true", help="Allow overwriting existing output directory")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        if not args.allow_overwrite:
            print(f"ERROR: Output directory {out_dir} exists and is non-empty. Use --allow_overwrite to proceed.")
            sys.exit(1)
        print(f"WARNING: Overwriting existing output directory {out_dir}")
    os.makedirs(args.output_dir, exist_ok=True)

    # Record commit SHA
    import subprocess
    try:
        commit_sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(REPO), text=True).strip()
    except Exception:
        commit_sha = "unknown"

    # Input SHAs
    dataset_full_sha = hashlib.sha256(open(args.dataset_csv, "rb").read()).hexdigest()
    labels_sha = hashlib.sha256(open(args.labels_csv, "rb").read()).hexdigest()
    script_sha = hashlib.sha256(open(__file__, "rb").read()).hexdigest()
    print(f"Source dataset SHA: {dataset_full_sha[:16]}")
    print(f"Labels SHA: {labels_sha[:16]}")

    # Load labels
    labels = {}
    for r in csv.DictReader(open(args.labels_csv)):
        key = (int(r["task"]), int(r["state"]), r["source"])
        labels[key] = r
    print(f"Loaded {len(labels)} episode labels")

    # Load dataset
    all_rows, tr_eps, vl_eps = load_dataset(args.dataset_csv)
    print(f"Full dataset: {len(all_rows)} steps, {len(tr_eps)} train eps, {len(vl_eps)} val eps")

    # Classify train episodes by source pool
    primary_eps = {}
    reserve_eps = {}
    for eid, rows in tr_eps.items():
        source = rows[0].get("source_pool", "primary")
        if source == "primary":
            primary_eps[eid] = rows
        else:
            reserve_eps[eid] = rows
    print(f"Train: {len(primary_eps)} primary, {len(reserve_eps)} reserve")

    # Count steps
    tv_steps_primary, nc_steps_primary = count_nc_steps(primary_eps, labels)
    tv_steps_reserve, nc_steps_reserve = count_nc_steps(reserve_eps, labels)
    print(f"Primary: TV={tv_steps_primary} NC={nc_steps_primary}")
    print(f"Reserve: TV={tv_steps_reserve} NC={nc_steps_reserve}")
    print(f"Combined (M2): TV={tv_steps_primary+tv_steps_reserve} NC={nc_steps_primary+nc_steps_reserve}")

    # ── 1. Primary-only dataset ──
    primary_only_rows = []
    for r in all_rows:
        if r["split"] == "val":
            primary_only_rows.append(r)
        elif r["episode_id"] in primary_eps:
            primary_only_rows.append(r)
    # else: skip reserve episodes

    m1_path = os.path.join(args.output_dir, "SC5_V2_STEP_DATASET_PRIMARY_ONLY.csv")
    with open(m1_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=all_rows[0].keys())
        w.writeheader()
        w.writerows(primary_only_rows)
    m1_sha = hashlib.sha256(open(m1_path, "rb").read()).hexdigest()
    print(f"\nM1 (Primary-only): {len(primary_only_rows)} steps, SHA={m1_sha[:16]}")

    # Verify splits
    m1_tr = sum(1 for r in primary_only_rows if r["split"] == "train")
    m1_vl = sum(1 for r in primary_only_rows if r["split"] == "val")
    print(f"  train={m1_tr} val={m1_vl}")

    # ── 2. Primary oversampled dataset ──
    # M2 total NC steps = nc_steps_primary + nc_steps_reserve
    m2_nc_total = nc_steps_primary + nc_steps_reserve
    # M1 NC steps = nc_steps_primary
    # Need to add (m2_nc_total - nc_steps_primary) NC steps via oversampling
    oversample_needed = m2_nc_total - nc_steps_primary
    print(f"\nM2 NC total: {m2_nc_total}, M1 NC: {nc_steps_primary}, Need: {oversample_needed}")

    # Identify NC episodes in Primary only
    primary_nc_eps = {}
    primary_tv_eps = {}
    for eid, rows in primary_eps.items():
        key = make_label_key(rows)
        lbl = labels.get(key, {})
        if lbl.get("teacher_valid") == "True":
            primary_tv_eps[eid] = rows
        else:
            primary_nc_eps[eid] = rows

    print(f"Primary NC episodes: {len(primary_nc_eps)}, TV episodes: {len(primary_tv_eps)}")

    # Duplicate NC episodes deterministically (sorted episode IDs, round-robin)
    oversampled_rows = list(primary_only_rows)  # start with M1
    nc_ep_list = sorted(primary_nc_eps.items(), key=lambda x: x[0])  # sort by episode_id
    added = 0
    idx = 0
    while added < oversample_needed and nc_ep_list:
        eid, rows = nc_ep_list[idx % len(nc_ep_list)]
        for r in rows:
            if added >= oversample_needed:
                break
            # Deep copy and add duplicate label
            rr = deepcopy(r)
            rr["oversample_duplicate"] = "1"
            oversampled_rows.append(rr)
            added += 1
        idx += 1

    m1os_path = os.path.join(args.output_dir, "SC5_V2_STEP_DATASET_PRIMARY_OVERSAMPLED.csv")
    # Add oversample_duplicate column if not present
    fields = list(all_rows[0].keys())
    if "oversample_duplicate" not in fields:
        fields.append("oversample_duplicate")
    # Ensure all rows have the column
    for r in oversampled_rows:
        if "oversample_duplicate" not in r:
            r["oversample_duplicate"] = "0"

    with open(m1os_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(oversampled_rows)
    m1os_sha = hashlib.sha256(open(m1os_path, "rb").read()).hexdigest()
    print(f"M1-OS (Primary oversampled): {len(oversampled_rows)} steps, SHA={m1os_sha[:16]}")

    # Verify
    m1os_tr = sum(1 for r in oversampled_rows if r["split"] == "train")
    m1os_vl = sum(1 for r in oversampled_rows if r["split"] == "val")
    m1os_nc = sum(1 for r in oversampled_rows if r["split"] == "train" and r.get("teacher_sc5_corridor_active") != "1.0")
    print(f"  train={m1os_tr} val={m1os_vl}")

    # ── Save config ──
    config = {
        "gate": "SC5_V2_DATA_ABLATION_CONFIG",
        "run_commit_sha": commit_sha,
        "script_sha256": script_sha,
        "source_dataset_sha256": dataset_full_sha,
        "source_labels_sha256": labels_sha,
        "oversampling": {
            "method": "deterministic_round_robin",
            "seed": "none (sorted episode_id order)",
            "nc_episode_order": [eid for eid, _ in nc_ep_list],
            "nc_episodes_available": len(nc_ep_list),
            "episodes_copied": idx if idx > 0 else 0,
            "rows_added": added,
            "rows_target": oversample_needed,
        },
        "models": {
            "M0": {"name": "SC5-V1", "training_data": "N/A (pre-existing)", "steps": "N/A"},
            "M1": {"name": "Primary-only", "dataset": m1_path, "sha256": m1_sha,
                   "train_steps": m1_tr, "val_steps": m1_vl,
                   "description": "Primary 250 episodes only, no Reserve"},
            "M1_OS": {"name": "Primary-oversampled", "dataset": m1os_path, "sha256": m1os_sha,
                      "train_steps": m1os_tr, "val_steps": m1os_vl,
                      "description": f"Primary 250 with NC oversampling to match M2 negative count ({m2_nc_total} NC steps)"},
            "M2": {"name": "Primary+Reserve (current)", "dataset": args.dataset_csv,
                   "description": "Primary 250 + Reserve 120 = 370 episodes, 84015 steps"},
        },
        "training_config": {
            "architecture": "SC5_MLP_v2_Hybrid (6604 params)",
            "seeds": [42, 123, 456, 789, 1024],
            "loss": "phase CE(inverse_freq) + 0.5 * corridor BCE(pos_weight=5)",
            "release_head": "V1 frozen injection",
            "optimizer": "Adam(lr=0.001, weight_decay=1e-4)",
            "batch_size": 64,
            "max_epochs": 80,
            "early_stopping": "best val_phase_CE",
            "evaluation": "Same dev 90, same thresholds (tau_c=0.3, tau_r=0.3, guard=5)",
        },
        "comparisons": {
            "M2_vs_M1": "Reserve data overall value",
            "M1_OS_vs_M1": "Pure NC oversampling value (same episodes, more weight)",
            "M2_vs_M1_OS": "Reserve hard-negative diversity value (different episodes)",
        },
    }
    config_path = os.path.join(args.output_dir, "SC5_V2_DATA_ABLATION_CONFIG.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"\nConfig saved: {config_path}")


if __name__ == "__main__":
    main()
