#!/usr/bin/env python3
"""Recover Table1 results from worker logs, ignoring stale partial CSV artifacts."""
import csv, re, sys
from pathlib import Path
from collections import defaultdict

OUT = Path("/data/liuyu/outputs/table1_official_eval_recovery_20260527")
for d in ["tables", "reports"]:
    OUT.joinpath(d).mkdir(parents=True, exist_ok=True)

# Worker log paths and metadata
WORKERS = {
    "spatial_w13": {
        "log": "/tmp/r4_s.log",
        "suite": "libero_spatial",
        "gpu": "1,3",
        "status": "running",
    },
    "goal_w26": {
        "log": "/tmp/r4_g.log",
        "suite": "libero_goal",
        "gpu": "2,6",
        "status": "running",
    },
    "libero10_w45": {
        "log": "/tmp/r4_l.log",
        "suite": "libero_10",
        "gpu": "4,5",
        "status": "running",
    },
}

STALE_CSV = Path("/data/liuyu/outputs/table1_clean_patched_v4_remaining3_20260527/tables/object_official_script_100_manifest.csv")
STALE_SUMMARY = Path("/data/liuyu/outputs/table1_clean_patched_v4_remaining3_20260527/tables/object_official_script_100_summary_by_task.csv")

# ── Parse worker logs ──
worker_results = {}
for wid, w in WORKERS.items():
    log_path = Path(w["log"])
    tasks = {}
    if log_path.exists():
        content = log_path.read_text(errors="replace")
        # Extract task SR lines: "  Task SR: X/Y = Z"
        for match in re.finditer(r'--- Task (\d+)/(\d+): (.+?) ---.*?Task SR: (\d+)/(\d+) = ([\d.]+)', content, re.DOTALL):
            task_num = int(match.group(1))
            task_name = match.group(3).strip()
            success = int(match.group(4))
            total = int(match.group(5))
            sr = float(match.group(6))
            tasks[task_name] = {"n": total, "success": success, "sr": sr}
        # Check for FINAL line
        if "=== FINAL ===" in content:
            w["status"] = "complete"
        elif tasks:
            w["status"] = "partial"

    worker_results[wid] = {
        "suite": w["suite"],
        "gpu": w["gpu"],
        "status": w["status"],
        "tasks": tasks,
        "total_success": sum(t["success"] for t in tasks.values()),
        "total_episodes": sum(t["n"] for t in tasks.values()),
    }

# ── Worker status table ──
status_rows = []
for wid, w in WORKERS.items():
    wr = worker_results[wid]
    status_rows.append({
        "worker_id": wid,
        "suite": w["suite"],
        "gpu": w["gpu"],
        "log_status": w["status"],
        "tasks_found": len(wr["tasks"]),
        "total_success": wr["total_success"],
        "total_episodes": wr["total_episodes"],
        "sr": f"{wr['total_success']/max(1,wr['total_episodes']):.3f}" if wr["total_episodes"] else "pending",
    })

with open(OUT / "tables" / "table1_worker_status.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(status_rows[0].keys()))
    w.writeheader()
    w.writerows(status_rows)

# ── Stale artifact audit ──
stale_rows = []
if STALE_CSV.exists():
    stale_rows.append({
        "path": str(STALE_CSV),
        "size_bytes": STALE_CSV.stat().st_size,
        "rows": sum(1 for _ in open(STALE_CSV)),
        "status": "stale_crashed_partial",
        "exclude_from_final": True,
        "reason": "Written by crashed Goal worker (old GPU indices, 0/100). Overwritten filename shared by all 3 workers.",
    })
if STALE_SUMMARY.exists():
    stale_rows.append({
        "path": str(STALE_SUMMARY),
        "size_bytes": STALE_SUMMARY.stat().st_size,
        "rows": sum(1 for _ in open(STALE_SUMMARY)),
        "status": "stale_crashed_partial",
        "exclude_from_final": True,
        "reason": "Same source as manifest — crashed Goal worker partial output.",
    })

with open(OUT / "tables" / "table1_artifact_validity_audit.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(stale_rows[0].keys()) if stale_rows else ["path","status"])
    w.writeheader()
    w.writerows(stale_rows)

# ── Recovery report ──
report = [
    "# Table1 Recovery Audit",
    "",
    "## Worker Status",
    "",
]
for row in status_rows:
    report.append(f"- **{row['worker_id']}** ({row['suite']}, GPU {row['gpu']}): {row['log_status']}, {row['tasks_found']} tasks, SR={row['sr']}")

report.extend([
    "",
    "## Stale Partial CSV Exclusion",
    "",
])
for row in stale_rows:
    report.append(f"- `{row['path']}`: {row['status']} ({row['reason']})")

report.extend([
    "",
    "## CSV Overwrite Bug",
    "",
    "The original official eval script writes all results to a single hardcoded filename",
    "(`object_official_script_100_manifest.csv`), regardless of suite or worker count.",
    "When multiple workers run in parallel, the last worker to finish overwrites all prior data.",
    "",
    "**Fix:** Patched script now writes worker+specific shards to `tables/shards/`.",
    "Aggregation is done by a separate script after all workers complete.",
    "",
    "## Recovery Strategy",
    "",
    "Final Table1 SR is reconstructed from worker stdout logs, not from CSV files.",
    "Each worker's task-level SR is parsed from the log output.",
    "Stale partial CSVs are excluded from final aggregation.",
])

with open(OUT / "reports" / "TABLE1_RECOVERY_AUDIT.md", "w") as f:
    f.write("\n".join(report))

# ── CSV bug audit ──
bug_report = [
    "# CSV Overwrite Bug Audit",
    "",
    "## Root Cause",
    "",
    "`tmp_official_eval.py` line 339 hardcodes:",
    "```python",
    'open(out / "tables" / "object_official_script_100_manifest.csv", "w")',
    "```",
    "Three parallel workers all write to the same file. Last writer wins.",
    "",
    "## Impact",
    "",
    "- Goal crash partial (295 rows, 0/100) overwrote prior valid data",
    "- Object R2 per-state data lost (earlier R3 run)",
    "- Current Spatial/Goal/L10 runs at risk of mutual overwrite",
    "",
    "## Fix Applied",
    "",
    "Patched to write worker+specific shards:",
    "```",
    "tables/shards/manifest_{suite}_{worker_id}.csv",
    "tables/shards/summary_{suite}_{worker_id}.csv",
    "```",
    "Separate aggregator merges after all workers complete.",
    "",
    "## Migration",
    "",
    "The artifact-rich runner (`run_official_eval_artifact_rich.py`) already implements",
    "parallel-safe sharding from Milestone 2D Phase A.",
]
with open(OUT / "reports" / "CSV_OVERWRITE_BUG_AUDIT.md", "w") as f:
    f.write("\n".join(bug_report))

# ── Summary ──
print("Table1 Recovery Audit")
print("=" * 60)
for row in status_rows:
    print(f"  {row['worker_id']}: {row['log_status']}, {row['tasks_found']} tasks, SR={row['sr']}")
print(f"\nStale artifacts excluded: {len(stale_rows)}")
print(f"Output: {OUT}")
print("\nRe-run this script after workers complete for final SR.")
