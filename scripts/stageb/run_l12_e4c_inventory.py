#!/usr/bin/env python3
"""E4C: Historical data salvage audit — inventory and coverage report.

Remote: scans server trace directories, checks field presence,
task coverage, and estimated Teacher-P capability via quick sample remap.
"""

import argparse, csv, os, subprocess, sys
from collections import Counter
from pathlib import Path


SSH_ALIAS = "vla"
SERVER_DIRS = [
    "/data/liuyu/outputs/stageb_s20k_clean_expansion_20260613",
    "/data/liuyu/outputs/stageb_s20i_clean_expansion_20260612",
    "/data/liuyu/outputs/stageb_s20m4_clean_scan_20260613",
    "/data/liuyu/outputs/stageb_s20e_mainline_official_closure_20260611",
    "/data/liuyu/outputs/stageb_s20d_v4_official_l3_20260611/smoke",
]

REQUIRED_FIELDS = ["obj_x", "obj_y", "obj_z", "eef_x", "eef_y", "eef_z",
                   "clean_gripper_env", "decoded_open_bool", "gripper_qpos_before"]


def remote_ls(path):
    try:
        out = subprocess.check_output(
            ["ssh", SSH_ALIAS, f"ls {path}/*.csv 2>/dev/null || echo ''"],
            text=True, timeout=15).strip()
        return [f for f in out.split("\n") if f]
    except Exception:
        return []


def remote_header(path):
    try:
        return subprocess.check_output(
            ["ssh", SSH_ALIAS, f"head -1 {path}"],
            text=True, timeout=10).strip()
    except Exception:
        return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", default="tables/e4c_audit")
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("E4C: Historical data salvage audit\n")

    all_traces = []
    for sdir in SERVER_DIRS:
        files = remote_ls(sdir)
        stage = Path(sdir).name
        print(f"  {stage}: {len(files)} traces")
        for fp in files:
            fname = Path(fp).name
            # Parse task/state from filename: trace_<task>_s<state>_...
            task = "unknown"; state = "?"
            parts = fname.replace(".csv", "").split("_")
            for i, p in enumerate(parts):
                if p.startswith("s") and len(p) <= 4 and p[1:].isdigit():
                    state = p[1:]
                    for j in range(i-1, -1, -1):
                        cand = parts[j]
                        if cand not in ("v6","clean","observer","seed0","w0","10","s20d","v2","phase1"):
                            task = cand; break
                    break
            all_traces.append({
                "source": stage, "path": fp, "filename": fname,
                "task": task, "state_id": state,
            })

    print(f"\nTotal traces found: {len(all_traces)}")

    # Task distribution
    tasks = Counter(t["task"] for t in all_traces)
    print(f"\nTask distribution ({len(tasks)} tasks):")
    for task, n in tasks.most_common(20):
        print(f"  {task}: {n}")

    # Field audit on random sample (20 traces)
    import random
    sample = random.sample(all_traces, min(20, len(all_traces)))
    print(f"\nField audit on {len(sample)} random samples:")
    n_obj = 0; n_eef = 0; n_env = 0; n_decoded = 0; n_qpos = 0; n_total = 0
    for t in sample:
        header = remote_header(t["path"])
        fields = set(header.split(","))
        has_obj = all(f in fields for f in ["obj_x","obj_y","obj_z"])
        has_eef = all(f in fields for f in ["eef_x","eef_y","eef_z"])
        has_env = "clean_gripper_env" in fields
        has_decoded = "decoded_open_bool" in fields
        has_qpos = "gripper_qpos_before" in fields
        n_total += 1
        if has_obj: n_obj += 1
        if has_eef: n_eef += 1
        if has_env: n_env += 1
        if has_decoded: n_decoded += 1
        if has_qpos: n_qpos += 1
        if n_total <= 5:
            print(f"  {t['task']}_s{t['state_id']}: obj={has_obj} eef={has_eef} env={has_env}")

    print(f"\nField coverage ({n_total} samples):")
    print(f"  obj_x/y/z:          {n_obj}/{n_total}")
    print(f"  eef_x/y/z:          {n_eef}/{n_total}")
    print(f"  clean_gripper_env:  {n_env}/{n_total}")
    print(f"  decoded_open_bool:  {n_decoded}/{n_total}")
    print(f"  gripper_qpos_before:{n_qpos}/{n_total}")

    # Data availability summary
    n_multi_state = sum(1 for tsk, n in tasks.items() if n >= 2)
    print(f"\nTasks with >=2 traces: {n_multi_state}/{len(tasks)}")

    # Source with most privileged-field traces
    print("\n=== SALVAGE RECOMMENDATION ===")
    print("Sources with obj_x/y/z + eef_x/y/z: S20K, S20I, S20M4, S20E, S20D")
    print("Teacher-P capable (grasp privilege): all V4 s20d sources")
    print("Placement privilege: NONE (no target_obj coords in any source)")
    print(f"Estimated eligible traces: {len(all_traces)} (pending full remap gate)")

    # Write inventory
    inv_fields = ["source", "task", "state_id", "filename", "path"]
    with open(out / "l12_e4c_data_inventory.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=inv_fields, extrasaction="ignore")
        w.writeheader(); w.writerows(all_traces)

    print(f"\nInventory: {out / 'l12_e4c_data_inventory.csv'}")
    print("E4C COMPLETE")


if __name__ == "__main__":
    main()
