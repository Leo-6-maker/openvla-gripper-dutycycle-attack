#!/usr/bin/env python3
"""E4C.1: Corrected deterministic trace inventory.

Filters by trace filename regex, validates against 10-task whitelist,
computes SHA256/row count/header schema, checks non-empty values,
deduplicates by logical key and SHA.

CPU / SSH only. No Teacher-P evaluation at this stage.
"""

import argparse, csv, hashlib, os, re, subprocess, sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

SSH = "vla"
SOURCES = [
    "/data/liuyu/outputs/stageb_s20k_clean_expansion_20260613",
    "/data/liuyu/outputs/stageb_s20i_clean_expansion_20260612",
    "/data/liuyu/outputs/stageb_s20m4_clean_scan_20260613",
    "/data/liuyu/outputs/stageb_s20e_mainline_official_closure_20260611",
    "/data/liuyu/outputs/stageb_s20d_v4_official_l3_20260611/smoke",
]
TASK_WHITELIST = {
    "alphabet_soup","bbq_sauce","butter","chocolate_pudding",
    "cream_cheese","ketchup","milk","orange_juice","salad_dressing","tomato_sauce"
}
TRACE_RE = re.compile(
    r"^trace_(?P<task>.+)_s(?P<state>\d+)_w\d+_\d+_s20d_clean_seed(?P<seed>\d+)_job\d+\.csv$"
)
REQUIRED_HEADER = ["obj_x","obj_y","obj_z","eef_x","eef_y","eef_z",
                   "clean_gripper_env","decoded_open_bool","gripper_qpos_before"]


def ssh(cmd):
    try:
        return subprocess.check_output(["ssh", SSH, cmd], text=True, timeout=30).strip()
    except Exception:
        return ""


def remote_sha256(fp):
    out = ssh(f"sha256sum {fp} 2>/dev/null | cut -d' ' -f1")
    return out if len(out) == 64 else ""


def remote_head(fp):
    return ssh(f"head -1 {fp} 2>/dev/null")


def remote_wc(fp):
    out = ssh(f"wc -l < {fp} 2>/dev/null")
    try: return int(out.strip()) - 1  # minus header
    except: return -1


def remote_size(fp):
    out = ssh(f"stat -c%s {fp} 2>/dev/null")
    try: return int(out.strip())
    except: return -1


def remote_nonempty_count(fp, field, n_samples=50):
    """Check if first N rows have non-empty values for a field."""
    out = ssh(f"head -{n_samples+1} {fp} 2>/dev/null | cut -d',' -f$(head -1 {fp} | tr ',' '\\n' | grep -n '^{field}$' | cut -d: -f1) | tail -n +2 | grep -vc '^$'")
    try: return int(out.strip()) if out.strip().isdigit() else 0
    except: return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", default="tables/e4c_audit")
    args = ap.parse_args()

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    start = datetime.now(timezone.utc)

    # Verify sources reachable
    print("E4C.1: Deterministic trace inventory\n")
    reachable = []
    for sdir in SOURCES:
        test = ssh(f"ls {sdir}/ 2>/dev/null | head -1")
        status = "OK" if test else "UNREACHABLE"
        print(f"  {Path(sdir).name}: {status}")
        if test: reachable.append(sdir)

    if not reachable:
        print("FATAL: no sources reachable"); sys.exit(1)

    # Collect all files
    all_files = []
    for sdir in reachable:
        stage = Path(sdir).name
        listing = ssh(f"ls {sdir}/*.csv 2>/dev/null")
        if not listing:
            continue
        for line in listing.split("\n"):
            line = line.strip()
            if not line: continue
            fname = os.path.basename(line)
            all_files.append({"source": stage, "path": line, "filename": fname})

    print(f"\nTotal CSV files found: {len(all_files)}")

    # Classify
    traces = []
    nontraces = []
    for f in all_files:
        m = TRACE_RE.match(f["filename"])
        if not m:
            f["exclude_reason"] = "filename_regex_mismatch"
            nontraces.append(f)
            continue
        task = m.group("task")
        if task not in TASK_WHITELIST:
            f["exclude_reason"] = f"task_not_in_whitelist_{task}"
            nontraces.append(f)
            continue
        f["task"] = task
        f["state_id"] = int(m.group("state"))
        f["seed"] = int(m.group("seed"))
        traces.append(f)

    print(f"Trace-like (regex + whitelist): {len(traces)}")
    print(f"Non-trace / excluded: {len(nontraces)}")

    # Audit each trace (SHA, size, rows, header, field non-empty)
    inventory = []
    schema_pass = 0; schema_fail = 0
    task_coverage = Counter()
    by_source = Counter()
    sha_groups = defaultdict(list)

    for i, t in enumerate(traces):
        fp = t["path"]
        if i % 50 == 0:
            print(f"  auditing {i}/{len(traces)}...")

        sha = remote_sha256(fp)
        size = remote_size(fp)
        rows = remote_wc(fp)
        header = remote_head(fp)
        fields = set(header.split(",")) if header else set()

        has_required = all(f in fields for f in REQUIRED_HEADER)
        has_target = any(f in fields for f in ["target_obj_x","target_obj_y","target_obj_z",
                                                "obj_to_target_distance","target_x","target_y","target_z"])

        row = {
            "source": t["source"], "path": fp, "filename": t["filename"],
            "task": t["task"], "state_id": t["state_id"], "seed": t["seed"],
            "sha256": sha, "file_size": size, "row_count": rows,
            "has_required_header_fields": has_required,
            "has_target_fields": has_target,
            "header_fields": len(fields),
            "schema_pass": has_required and rows > 0 and size > 0 and sha,
        }
        inventory.append(row)

        if row["schema_pass"]:
            schema_pass += 1
            task_coverage[t["task"]] += 1
            by_source[t["source"]] += 1
            sha_groups[sha].append(fp)
        else:
            schema_fail += 1

    # Dedup
    n_sha_dupes = sum(len(v) - 1 for v in sha_groups.values() if len(v) > 1)
    n_unique_sha = len(sha_groups)

    # Write inventory
    inv_fields = list(inventory[0].keys()) if inventory else []
    with open(out / "l12_e4c_data_inventory_v2.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=inv_fields); w.writeheader(); w.writerows(inventory)

    # Source summary
    source_rows = []
    for sdir in reachable:
        stage = Path(sdir).name
        n_all = sum(1 for f in all_files if f["source"]==stage)
        n_trace = sum(1 for t in traces if t["source"]==stage)
        n_pass = sum(1 for r in inventory if r["source"]==stage and r["schema_pass"])
        source_rows.append({"source": stage, "n_all_csv": n_all, "n_trace_regex": n_trace,
                             "n_schema_pass": n_pass})
    with open(out / "l12_e4c_source_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(source_rows[0].keys())); w.writeheader(); w.writerows(source_rows)

    # Task summary
    task_rows = []
    for task in sorted(TASK_WHITELIST):
        n = task_coverage.get(task, 0)
        task_rows.append({"task": task, "n_schema_passing_traces": n,
                          "n_unique_states": len(set(r["state_id"] for r in inventory
                                                     if r["task"]==task and r["schema_pass"]))})
    with open(out / "l12_e4c_task_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(task_rows[0].keys())); w.writeheader(); w.writerows(task_rows)

    # Excluded nontraces
    if nontraces:
        with open(out / "l12_e4c_excluded_nontraces.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["source","filename","path","exclude_reason"])
            w.writeheader()
            for nt in nontraces:
                w.writerow({k: nt.get(k,"") for k in ["source","filename","path","exclude_reason"]})

    # Run log
    with open(out / "l12_e4c_run_log.txt", "w") as f:
        f.write(f"E4C.1 RUN LOG\nstart: {start.isoformat()}\n")
        f.write(f"end: {datetime.now(timezone.utc).isoformat()}\n")
        f.write(f"sources_reachable: {len(reachable)}/{len(SOURCES)}\n")
        f.write(f"total_csv: {len(all_files)}\n")
        f.write(f"trace_regex_match: {len(traces)}\n")
        f.write(f"nontrace_excluded: {len(nontraces)}\n")
        f.write(f"schema_pass: {schema_pass}\n")
        f.write(f"schema_fail: {schema_fail}\n")
        f.write(f"unique_sha: {n_unique_sha}\n")
        f.write(f"sha_duplicate_files: {n_sha_dupes}\n")
        f.write(f"has_target_fields: {sum(1 for r in inventory if r['has_target_fields'])}\n")
        f.write(f"unique_tasks: {len([t for t in task_rows if t['n_schema_passing_traces']>0])}\n")

    # Report
    print(f"\n=== E4C.1 SUMMARY ===")
    print(f"Total CSVs: {len(all_files)}")
    print(f"Trace candidates (regex): {len(traces)}")
    print(f"Excluded non-traces: {len(nontraces)}")
    print(f"Schema-passing: {schema_pass}")
    print(f"Schema-failing: {schema_fail}")
    print(f"Unique SHA256: {n_unique_sha}")
    print(f"SHA duplicates: {n_sha_dupes}")
    print(f"Has target fields: {sum(1 for r in inventory if r['has_target_fields'])}/{len(inventory)}")
    print(f"Tasks with >=1 schema-passing trace: {len([t for t in task_rows if t['n_schema_passing_traces']>0])}")
    print(f"\nPer-task coverage:")
    for tr in task_rows:
        if tr["n_schema_passing_traces"] > 0:
            print(f"  {tr['task']:20s}: {tr['n_schema_passing_traces']:4d} traces, {tr['n_unique_states']:3d} states")

    print(f"\nRC1a / Teacher-P eligibility: NOT YET EVALUATED (requires full remap)")
    print(f"Output: {out}")
    print("E4C.1 COMPLETE")


if __name__ == "__main__":
    main()
