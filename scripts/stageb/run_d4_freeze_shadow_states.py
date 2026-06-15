#!/usr/bin/env python3
"""D4.3 shadow state freeze — canonical 500 task-state inventory + hashed selection.

Hard assertions (all enforced before output):
  1. Input manifest 402: exactly 402 rows, 402 unique (task_key, state_id)
  2. Fresh inventory 98: exactly 98 collected rows, 98 unique keys
  3. Overlap 402 ∩ 98 = 0
  4. Merged inventory = exactly 500 unique (task_key, state_id)
  5. Exactly 10 tasks, each exactly 50 states
  6. D1b val=20, test=21 (exact counts, all mapped)
  7. D2 fresh25=25 (exact count, all mapped to task-state keys)
  8. Exclusions by (task_key, state_id) key, not trace_id

Excludes before hashing:
  - D1b val20 (by task-state key)
  - D1b test21 (by task-state key)
  - D2 fresh25 (by task-state key, derived from d2_fresh_close_candidates.csv)
  - Any explicitly invalid infrastructure state

State selection: SHA256("D4.3_SHADOW_V1|<task_key>|<state_id>")
  - Canary: 4 smallest hashes with 4 distinct task keys
  - Panel: exclude canary; for each of 10 tasks, 3 smallest hashes -> 30 states

Commit outputs to tables/d4_shadow/ before any GPU rollout.
"""

import argparse
import csv
import hashlib
import os
import sys
from collections import defaultdict
from pathlib import Path

TASKS_10 = frozenset([
    "alphabet_soup", "cream_cheese", "salad_dressing", "bbq_sauce",
    "ketchup", "tomato_sauce", "butter", "milk",
    "chocolate_pudding", "orange_juice",
])

TASKS_LIST = sorted(TASKS_10)

SALT = "D4.3_SHADOW_V1"


def sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def sha256_file(path: str) -> str:
    if not os.path.isfile(path):
        return ""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-manifest-402", required=True)
    ap.add_argument("--fresh-inventory-98", required=True)
    ap.add_argument("--d1b-split-manifest", required=True)
    ap.add_argument("--d2-fresh-candidate-table", required=True,
                    help="tables/d2_fresh/d2_fresh_close_candidates.csv")
    ap.add_argument("--d2-trace-status", required=True,
                    help="tables/d2_fresh/d2_fresh_trace_status.csv")
    ap.add_argument("--d2-fresh-inventory-for-mapping", required=True,
                    help="tables/d2_fresh/d2_final_trace_inventory.csv (for tid->key mapping)")
    ap.add_argument("--invalid-state-manifest", default="",
                    help="Manually excluded invalid (task,state) CSV (can be empty)")
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    out = Path(args.output_dir)
    assert not out.exists() or len(list(out.iterdir())) == 0, (
        f"FATAL: Output directory must be empty: {out}"
    )
    out.mkdir(parents=True, exist_ok=True)

    # ═══════════════════════════════════════════════════════════
    # 1. Load 402 existing
    # ═══════════════════════════════════════════════════════════
    rows_402 = list(csv.DictReader(open(args.input_manifest_402)))
    n_402 = len(rows_402)
    keys_402 = set()
    dup_402 = 0
    existing = {}
    for r in rows_402:
        tk = r["task_key"]
        sid = int(r["state_id"])
        key = (tk, sid)
        if key in keys_402:
            dup_402 += 1
            continue
        keys_402.add(key)
        existing[key] = {
            "trace_id": r["trace_id"],
            "source_path": r.get("source_path", ""),
            "source_sha256": r.get("source_sha256", ""),
            "source": "historical_402",
        }

    print(f"Input-402: {n_402} rows, {len(keys_402)} unique keys (internal dups={dup_402})")
    assert n_402 == 402, f"FATAL: expected 402 rows in input manifest, got {n_402}"
    assert len(keys_402) == 402, f"FATAL: expected 402 unique keys, got {len(keys_402)}"
    assert dup_402 == 0, f"FATAL: {dup_402} internal duplicates in 402 manifest"

    # ═══════════════════════════════════════════════════════════
    # 2. Load 98 fresh
    # ═══════════════════════════════════════════════════════════
    rows_98_all = list(csv.DictReader(open(args.fresh_inventory_98)))
    rows_98 = [r for r in rows_98_all if r.get("status", "collected") == "collected"]
    n_98 = len(rows_98)
    keys_98 = set()
    dup_98 = 0
    fresh = {}
    for r in rows_98:
        tk = r["task_key"]
        sid = int(r["state_id"])
        key = (tk, sid)
        if key in keys_98:
            dup_98 += 1
            continue
        keys_98.add(key)
        fresh[key] = {
            "trace_id": r.get("filename", "").replace(".csv", ""),
            "source_path": r.get("source_path", ""),
            "source_sha256": r.get("full_sha256", ""),
            "source": "d2_fresh_98",
        }

    print(f"Fresh-98: {n_98} collected rows, {len(keys_98)} unique keys (internal dups={dup_98})")
    assert n_98 == 98, f"FATAL: expected 98 collected rows, got {n_98}"
    assert len(keys_98) == 98, f"FATAL: expected 98 unique keys, got {len(keys_98)}"
    assert dup_98 == 0, f"FATAL: {dup_98} internal duplicates in fresh inventory"

    # ═══════════════════════════════════════════════════════════
    # 3. Merge: 402 + 98 = 500, zero overlap
    # ═══════════════════════════════════════════════════════════
    overlap = keys_402 & keys_98
    assert len(overlap) == 0, (
        f"FATAL: {len(overlap)} overlapping (task,state) between 402 and 98: "
        f"{sorted(overlap)[:10]}"
    )

    inventory = {}
    for key, info in existing.items():
        inventory[key] = info
    for key, info in fresh.items():
        inventory[key] = info

    n_inv = len(inventory)
    print(f"Merged inventory: {len(existing)} + {len(fresh)} = {n_inv}")
    assert n_inv == 500, f"FATAL: expected 500 merged keys, got {n_inv}"

    # ═══════════════════════════════════════════════════════════
    # 4. Task coverage: exactly 10 tasks x 50 states
    # ═══════════════════════════════════════════════════════════
    task_counts = defaultdict(int)
    task_state_detail = defaultdict(set)
    for tk, sid in inventory:
        task_counts[tk] += 1
        task_state_detail[tk].add(sid)
    inventory_tasks = set(task_counts.keys())

    assert inventory_tasks == TASKS_10, (
        f"FATAL: expected exactly 10 tasks {sorted(TASKS_10)}, "
        f"got {sorted(inventory_tasks)}"
    )
    for t in TASKS_LIST:
        assert task_counts[t] == 50, (
            f"FATAL: task {t} has {task_counts[t]} states, expected 50"
        )
        expected_states = set(range(50))
        actual_states = task_state_detail[t]
        missing = expected_states - actual_states
        extra = actual_states - expected_states
        assert not missing, f"FATAL: {t} missing states: {sorted(missing)}"
        assert not extra, f"FATAL: {t} extra states (>=50): {sorted(extra)}"
        print(f"  {t}: {task_counts[t]} states [0-49] OK")

    # ═══════════════════════════════════════════════════════════
    # 5. Build exclusion set by (task_key, state_id)
    # ═══════════════════════════════════════════════════════════
    excluded_keys = set()
    exclusion_records = []
    unmapped_exclusions = 0

    # 5a. D1b val + test
    split_rows = list(csv.DictReader(open(args.d1b_split_manifest)))
    n_val = 0
    n_test = 0
    for r in split_rows:
        tk = r["task_key"]
        sid = int(r["state_id"])
        key = (tk, sid)
        if r["split"] == "val":
            excluded_keys.add(key)
            exclusion_records.append({
                "task_key": tk, "state_id": str(sid), "trace_id": r["trace_id"],
                "reason": "D1b_val",
            })
            n_val += 1
        elif r["split"] == "test":
            excluded_keys.add(key)
            exclusion_records.append({
                "task_key": tk, "state_id": str(sid), "trace_id": r["trace_id"],
                "reason": "D1b_test",
            })
            n_test += 1

    assert n_val == 20, f"FATAL: D1b val count = {n_val}, expected 20"
    assert n_test == 21, f"FATAL: D1b test count = {n_test}, expected 21"
    print(f"D1b exclusions: val={n_val}, test={n_test}")

    # 5b. D2 fresh25 — from correct table: d2_fresh_close_candidates.csv
    from run_d2_fresh_confirm import select_eligible_multi_traces
    status_rows = list(csv.DictReader(open(args.d2_trace_status)))
    cand_rows = list(csv.DictReader(open(args.d2_fresh_candidate_table)))
    fresh25_map = select_eligible_multi_traces(cand_rows, status_rows)

    n_fresh25 = len(fresh25_map)
    assert n_fresh25 == 25, f"FATAL: fresh25 count = {n_fresh25}, expected 25"
    print(f"D2 fresh25 exclusions: {n_fresh25}")

    # Map fresh25 trace_ids to (task_key, state_id) via the D2 fresh inventory
    d2_map_rows = list(csv.DictReader(open(args.d2_fresh_inventory_for_mapping)))
    d2_tid_to_key = {}
    for r in d2_map_rows:
        if r.get("status", "collected") != "collected":
            continue
        tid = r.get("filename", "").replace(".csv", "")
        key = (r["task_key"], int(r["state_id"]))
        d2_tid_to_key[tid] = key

    for tid in fresh25_map:
        key = d2_tid_to_key.get(tid)
        if key is None:
            unmapped_exclusions += 1
            print(f"  WARNING: fresh25 trace {tid[:50]}... not found in fresh inventory")
            continue
        excluded_keys.add(key)
        exclusion_records.append({
            "task_key": key[0], "state_id": str(key[1]),
            "trace_id": tid, "reason": "D2_fresh25",
        })

    assert unmapped_exclusions == 0, (
        f"FATAL: {unmapped_exclusions} fresh25 traces could not be mapped to task-state keys"
    )

    # 5c. Invalid infrastructure states
    if args.invalid_state_manifest and os.path.exists(args.invalid_state_manifest):
        invalid_rows = list(csv.DictReader(open(args.invalid_state_manifest)))
        n_invalid = 0
        for r in invalid_rows:
            key = (r["task_key"], int(r["state_id"]))
            excluded_keys.add(key)
            exclusion_records.append({
                "task_key": key[0], "state_id": str(key[1]),
                "trace_id": r.get("trace_id", ""),
                "reason": r.get("reason", "invalid_infrastructure"),
            })
            n_invalid += 1
        print(f"Invalid infrastructure exclusions: {n_invalid}")
    else:
        print("Invalid infrastructure exclusions: 0 (no manifest provided)")

    # ═══════════════════════════════════════════════════════════
    # 6. Build eligible set (exclude by key)
    # ═══════════════════════════════════════════════════════════
    eligible = {}
    excluded_from_hash = []
    for key, info in sorted(inventory.items()):
        if key in excluded_keys:
            excluded_from_hash.append({
                "task_key": key[0], "state_id": key[1],
                "trace_id": info["trace_id"],
                "reason": "in_exclusion_set",
            })
            continue
        eligible[key] = info

    n_excluded_total = len(excluded_keys)
    n_eligible = len(eligible)
    # All exclusions must be in the inventory
    excluded_not_in_inv = excluded_keys - set(inventory.keys())
    assert len(excluded_not_in_inv) == 0, (
        f"FATAL: {len(excluded_not_in_inv)} excluded keys not in inventory"
    )
    print(f"Excluded (unique keys): {n_excluded_total}")
    print(f"Eligible for shadow selection: {n_eligible} task-states")

    # ═══════════════════════════════════════════════════════════
    # 7. Selection hash and canary/panel selection
    # ═══════════════════════════════════════════════════════════
    hashed = []
    for (tk, sid), info in eligible.items():
        h = sha256_hex(f"{SALT}|{tk}|{sid}")
        hashed.append((h, tk, sid, info))
    hashed.sort(key=lambda x: x[0])

    # Canary: 4 smallest hashes, 4 distinct tasks
    canary = []
    used_tasks = set()
    for h, tk, sid, info in hashed:
        if tk not in used_tasks and len(canary) < 4:
            canary.append((h, tk, sid, info))
            used_tasks.add(tk)
    assert len(canary) == 4, f"FATAL: expected 4 canary states, got {len(canary)}"
    canary_tasks = {c[1] for c in canary}
    assert len(canary_tasks) == 4, f"FATAL: canary tasks should be 4 distinct, got {canary_tasks}"

    canary_keys = {(c[1], c[2]) for c in canary}

    # Panel: 3 per task, smallest hashes, excluding canary
    panel = []
    for tk in TASKS_LIST:
        task_hashed = [(h, t, s, info) for h, t, s, info in hashed
                       if t == tk and (t, s) not in canary_keys]
        task_hashed.sort(key=lambda x: x[0])
        selected = task_hashed[:3]
        panel.extend(selected)
        assert len(selected) == 3, (
            f"FATAL: {tk} has only {len(selected)} eligible panel states (need 3)"
        )

    assert len(panel) == 30, f"FATAL: expected 30 panel states, got {len(panel)}"

    # Verify panel: 10 tasks x 3 states
    panel_task_counts = defaultdict(int)
    for _, tk, sid, _ in panel:
        panel_task_counts[tk] += 1
    for tk in TASKS_LIST:
        assert panel_task_counts[tk] == 3, (
            f"FATAL: panel task {tk} has {panel_task_counts[tk]} states"
        )

    # Canary and panel must be disjoint
    panel_keys = {(p[1], p[2]) for p in panel}
    assert len(canary_keys & panel_keys) == 0, "FATAL: canary-panel overlap"

    # ═══════════════════════════════════════════════════════════
    # 8. Write outputs
    # ═══════════════════════════════════════════════════════════
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
        w = csv.DictWriter(f, fieldnames=["task_key", "state_id", "trace_id", "reason"])
        w.writeheader()
        w.writerows(all_excl)

    # Input hashes — all selection inputs
    input_hash_files = [
        ("manifest_402", args.input_manifest_402),
        ("fresh_inventory_98", args.fresh_inventory_98),
        ("d1b_split_manifest", args.d1b_split_manifest),
        ("d2_fresh_candidate_table", args.d2_fresh_candidate_table),
        ("d2_trace_status", args.d2_trace_status),
        ("d2_fresh_inventory_for_mapping", args.d2_fresh_inventory_for_mapping),
        ("invalid_state_manifest", args.invalid_state_manifest),
    ]
    hashes_path = out / "d4_shadow_input_hashes.csv"
    with open(hashes_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["artifact", "sha256"])
        for label, path in input_hash_files:
            h = sha256_file(path) if path else "N/A_no_file"
            w.writerow([label, h])
        w.writerow(["shadow_state_manifest", sha256_file(str(manifest_path))])
        w.writerow(["exclusion_manifest", sha256_file(str(excl_path))])
        w.writerow(["freeze_runner_sha", sha256_file(__file__)])
        # Also hash the exclusion counts themselves
        w.writerow(["exclusion_counts_val_test_fresh",
                     sha256_hex(f"val={n_val}|test={n_test}|fresh25={n_fresh25}")])

    # Summary
    print(f"\nCanary ({len(canary)}):")
    for h, tk, sid, info in canary:
        print(f"  {tk}_s{sid}  hash={h[:16]}...  source={info['source']}")
    print(f"\nPanel ({len(panel)}):")
    for tk in TASKS_LIST:
        task_panel = [(h, s, info) for h, t, s, info in panel if t == tk]
        print(f"  {tk}: {len(task_panel)} states — "
              f"{', '.join(f's{s}' for _, s, _ in task_panel)}")

    print(f"\nOutput: {out}")
    print("DONE — commit before GPU rollout.")


if __name__ == "__main__":
    main()
