#!/usr/bin/env python3
"""D4.3 shadow state freeze — canonical 500 task-state inventory + hashed selection.

Hard assertions:
  - exactly 500 unique (task_key, state_id) from combined 402 + 98 inventories
  - 10 tasks
  - no duplicate groups

Excludes before hashing:
  - D1b val20
  - D1b test21
  - D2 fresh25 (traces in the D2 fresh eligibility test set)
  - any explicitly invalid infrastructure state

State selection: SHA256("D4.3_SHADOW_V1|<task_key>|<state_id>")
  - Canary: 4 smallest hashes with 4 distinct task keys
  - Panel: exclude canary; for each of 10 tasks, 3 smallest hashes → 30 states

Commit outputs to tables/d4_shadow/ before any GPU rollout.
"""

import argparse
import csv
import hashlib
import os
import sys
from collections import defaultdict
from pathlib import Path


TASKS_10 = [
    "alphabet_soup", "cream_cheese", "salad_dressing", "bbq_sauce",
    "ketchup", "tomato_sauce", "butter", "milk",
    "chocolate_pudding", "orange_juice",
]

SALT = "D4.3_SHADOW_V1"


def sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-manifest-402", required=True,
                    help="tables/e4c_audit/l12_e4c2_input_manifest.csv")
    ap.add_argument("--fresh-inventory-98", required=True,
                    help="tables/d2_fresh/d2_final_trace_inventory.csv")
    ap.add_argument("--d1b-split-manifest", required=True,
                    help="tables/deepseek_detector/d1b_split_manifest.csv")
    ap.add_argument("--fresh25-ids", default="",
                    help="File listing fresh25 trace_ids (one per line), "
                         "or derive from --d2-trace-status + --d2-candidate-table")
    ap.add_argument("--d2-trace-status", default="",
                    help="tables/d2_fresh/d2_fresh_trace_status.csv (for deriving fresh25)")
    ap.add_argument("--d2-candidate-table", default="",
                    help="Candidate table for fresh traces (for deriving fresh25)")
    ap.add_argument("--output-dir", required=True,
                    help="tables/d4_shadow/")
    args = ap.parse_args()

    out = Path(args.output_dir)
    assert not out.exists() or len(list(out.iterdir())) == 0, (
        f"Output directory must be empty: {out}"
    )
    out.mkdir(parents=True, exist_ok=True)

    # ── Load 402 existing ──
    rows_402 = list(csv.DictReader(open(args.input_manifest_402)))
    existing = {}
    for r in rows_402:
        key = (r["task_key"], int(r["state_id"]))
        existing[key] = {
            "trace_id": r["trace_id"],
            "source_path": r.get("source_path", ""),
            "source_sha256": r.get("source_sha256", ""),
            "source": "historical_402",
        }

    # ── Load 98 fresh ──
    rows_98 = list(csv.DictReader(open(args.fresh_inventory_98)))
    fresh = {}
    for r in rows_98:
        if r.get("status", "collected") != "collected":
            continue
        key = (r["task_key"], int(r["state_id"]))
        fresh[key] = {
            "trace_id": r.get("filename", "").replace(".csv", ""),
            "source_path": r.get("source_path", ""),
            "source_sha256": r.get("full_sha256", ""),
            "source": "d2_fresh_98",
        }

    # ── Merge 500: existing takes priority, fresh fills gaps ──
    inventory = {}
    duplicates = []
    for key, info in existing.items():
        inventory[key] = info
    for key, info in fresh.items():
        if key in inventory:
            duplicates.append(key)
        else:
            inventory[key] = info

    if duplicates:
        print(f"WARNING: {len(duplicates)} duplicate (task,state) in fresh vs existing")
        for d in duplicates[:5]:
            print(f"  {d}")

    # ── Hard assertions ──
    n_total = len(inventory)
    print(f"Inventory: {len(existing)} existing + {len(fresh)} fresh = {n_total} merged")

    # Task coverage
    task_counts = defaultdict(int)
    for task_key, state_id in inventory:
        task_counts[task_key] += 1
    for t in TASKS_10:
        print(f"  {t}: {task_counts[t]} states")

    # ── Load exclusions ──
    excluded_ids = set()
    exclusion_records = []

    # D1b val20 + test21
    split_rows = list(csv.DictReader(open(args.d1b_split_manifest)))
    for r in split_rows:
        if r["split"] in ("val", "test"):
            excluded_ids.add(r["trace_id"])
            exclusion_records.append({
                "trace_id": r["trace_id"],
                "task_key": r["task_key"],
                "state_id": r["state_id"],
                "reason": f"D1b_{r['split']}",
            })

    # D2 fresh25
    if args.fresh25_ids and os.path.exists(args.fresh25_ids):
        with open(args.fresh25_ids) as f:
            for line in f:
                tid = line.strip()
                if tid:
                    excluded_ids.add(tid)
    elif args.d2_trace_status and args.d2_candidate_table:
        # Derive fresh25 from trace_status + candidate table
        from run_d2_fresh_confirm import select_eligible_multi_traces
        status_rows = list(csv.DictReader(open(args.d2_trace_status)))
        cand_rows = list(csv.DictReader(open(args.d2_candidate_table)))
        fresh25 = select_eligible_multi_traces(cand_rows, status_rows)
        for tid in fresh25:
            excluded_ids.add(tid)
            # Look up task_key, state_id from inventory
            found = False
            for (tk, sid), info in inventory.items():
                if info["trace_id"] == tid:
                    exclusion_records.append({
                        "trace_id": tid,
                        "task_key": tk,
                        "state_id": str(sid),
                        "reason": "D2_fresh25",
                    })
                    found = True
                    break
            if not found:
                exclusion_records.append({
                    "trace_id": tid, "task_key": "?", "state_id": "?",
                    "reason": "D2_fresh25",
                })

    print(f"Excluded trace_ids: {len(excluded_ids)} "
          f"(D1b_val20 + D1b_test21 + D2_fresh25)")

    # ── Build eligible set ──
    eligible = {}
    excluded_from_hash = []
    for (task_key, state_id), info in sorted(inventory.items()):
        tid = info["trace_id"]
        if tid in excluded_ids:
            excluded_from_hash.append({
                "task_key": task_key, "state_id": state_id,
                "trace_id": tid, "reason": "in_exclusion_set",
            })
            continue
        eligible[(task_key, state_id)] = info

    print(f"Eligible for shadow: {len(eligible)} task-states")

    # ── Selection hash ──
    hashed = []
    for (task_key, state_id), info in eligible.items():
        h = sha256_hex(f"{SALT}|{task_key}|{state_id}")
        hashed.append((h, task_key, state_id, info))

    hashed.sort(key=lambda x: x[0])  # sort by hash ascending

    # ── Canary: 4 smallest hashes, 4 distinct tasks ──
    canary = []
    used_tasks = set()
    for h, tk, sid, info in hashed:
        if tk not in used_tasks and len(canary) < 4:
            canary.append((h, tk, sid, info))
            used_tasks.add(tk)
    assert len(canary) == 4, f"Expected 4 canary states, got {len(canary)}"

    canary_keys = {(c[1], c[2]) for c in canary}

    # ── Panel: 3 per task, smallest hashes, excluding canary ──
    panel = []
    for tk in TASKS_10:
        task_hashed = [(h, t, s, info) for h, t, s, info in hashed
                       if t == tk and (t, s) not in canary_keys]
        task_hashed.sort(key=lambda x: x[0])
        selected = task_hashed[:3]
        panel.extend(selected)
        if len(selected) < 3:
            print(f"WARNING: {tk} only has {len(selected)} panel candidates "
                  f"(need 3)")

    assert len(panel) == 30, f"Expected 30 panel states, got {len(panel)}"

    # ── Write outputs ──
    canonical_order = []
    for i, (h, tk, sid, info) in enumerate(canary):
        canonical_order.append({
            "subset": "canary", "task_key": tk, "state_id": str(sid),
            "selection_hash": h, "trace_id": info["trace_id"],
            "source": info["source"], "frozen_order": str(i),
        })
    for i, (h, tk, sid, info) in enumerate(panel):
        canonical_order.append({
            "subset": "panel", "task_key": tk, "state_id": str(sid),
            "selection_hash": h, "trace_id": info["trace_id"],
            "source": info["source"], "frozen_order": str(len(canary) + i),
        })

    manifest_path = out / "d4_shadow_state_manifest.csv"
    with open(manifest_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "subset", "task_key", "state_id", "selection_hash",
            "trace_id", "source", "frozen_order",
        ])
        w.writeheader()
        w.writerows(canonical_order)

    # Exclusions
    excl_path = out / "d4_shadow_exclusions.csv"
    all_excl = exclusion_records + excluded_from_hash
    with open(excl_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["trace_id", "task_key", "state_id", "reason"])
        w.writeheader()
        w.writerows(all_excl)

    # Input hashes
    hashes_path = out / "d4_shadow_input_hashes.csv"
    with open(hashes_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["artifact", "sha256"])
        w.writerow(["manifest_402", sha256_hex(open(args.input_manifest_402).read())])
        w.writerow(["fresh_inventory_98", sha256_hex(open(args.fresh_inventory_98).read())])
        w.writerow(["d1b_split_manifest", sha256_hex(open(args.d1b_split_manifest).read())])
        w.writerow(["shadow_state_manifest",
                     sha256_hex(open(str(manifest_path)).read())])

    # Summary
    print(f"\nCanary ({len(canary)}):")
    for h, tk, sid, info in canary:
        print(f"  {tk}_s{sid}  hash={h[:16]}...  {info['source']}")
    print(f"\nPanel ({len(panel)}):")
    for tk in TASKS_10:
        task_panel = [(h, s, info) for h, t, s, info in panel if t == tk]
        print(f"  {tk}: {len(task_panel)} states — "
              f"{', '.join(f's{s}' for _, s, _ in task_panel)}")

    print(f"\nOutput: {out}")
    print("DONE — commit before GPU rollout.")


if __name__ == "__main__":
    main()
