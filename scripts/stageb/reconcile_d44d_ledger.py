#!/usr/bin/env python3
"""D4.4D ledger reconciliation: reconcile all capture roots against frozen manifest."""
import csv, os, sys, json
from collections import defaultdict

MANIFEST = "/data/liuyu/outputs/d5_120_training_selection/d5_120_state_manifest.csv"
LEDGERS = {
    "orig": "/data/liuyu/outputs/d5_120_privileged_capture/capture_ledger.csv",
    "gpu13": "/data/liuyu/outputs/d44d_balanced120_gpu13_r1/capture_ledger.csv",
    "gpu26": "/data/liuyu/outputs/d44d_balanced120_gpu26_r1/capture_ledger.csv",
    "gpu50": "/data/liuyu/outputs/d44d_balanced120_gpu50_r1/capture_ledger.csv",
}

manifest = {(r["task_key"], int(r["state_id"])): r for r in csv.DictReader(open(MANIFEST))}
all_entries = {}

for src, lp in LEDGERS.items():
    if not os.path.exists(lp):
        continue
    for r in csv.DictReader(open(lp)):
        key = (r["task"], int(r["state_id"]))
        if key not in all_entries:
            all_entries[key] = {"attempts": [], "best_status": "UNSEEN"}
        entry = all_entries[key]
        status = r.get("status", "?")
        entry["attempts"].append({
            "src": src, "status": status,
            "steps": r.get("n_steps", ""), "gpu": r.get("gpu", ""),
            "rc": r.get("rc", ""), "priv": r.get("priv_valid", ""),
        })
        if status == "OK":
            entry["best_status"] = "OK"

# Classify
ok_list = []
fail_list = []
unattempted = []
for key in sorted(manifest.keys()):
    entry = all_entries.get(key, {"best_status": "UNSEEN", "attempts": []})
    if entry["best_status"] == "OK":
        ok_list.append(key)
    elif entry["attempts"]:
        fail_list.append(key)
    else:
        unattempted.append(key)

extra = [k for k in all_entries if k not in manifest]

print("=== D4.4D LEDGER RECONCILIATION ===")
print("Manifest states: 120")
print("OK (unique): {}".format(len(ok_list)))
print("Tried but FAIL: {}".format(len(fail_list)))
for k in fail_list:
    e = all_entries[k]
    print("  {}_s{} ({} attempts):".format(k[0], k[1], len(e["attempts"])))
    for a in e["attempts"]:
        print("    src={} status={} rc={} gpu={} priv={}".format(
            a["src"], a["status"], a["rc"], a["gpu"], a["priv"]))

print("Unattempted: {}".format(len(unattempted)))
print("Extra (not in manifest): {}".format(len(extra)))
for k in extra:
    print("  {}_s{}".format(k[0], k[1]))

print("")
print("Terminal completed: {} OK + {} FAIL = {}".format(
    len(ok_list), len(fail_list), len(ok_list) + len(fail_list)))
print("Unattempted: {}".format(len(unattempted)))
print("Maximum possible OK: {}".format(120 - len(fail_list)))

# Per-task breakdown
from collections import Counter
task_ok = Counter()
task_fail = Counter()
task_un = Counter()
for k in ok_list:
    task_ok[k[0]] += 1
for k in fail_list:
    task_fail[k[0]] += 1
for k in unattempted:
    task_un[k[0]] += 1

print("\nPer-task:")
for tk in sorted(set(list(task_ok) + list(task_fail) + list(task_un))):
    o = task_ok.get(tk, 0)
    f = task_fail.get(tk, 0)
    u = task_un.get(tk, 0)
    print("  {}: OK={} FAIL={} UNATTEMPTED={} (total={})".format(tk, o, f, u, o+f+u))

# Distinguish: 12/12 means OK count, not attempted count
print("\nKey: 12/12 in status report refers to OK count, not attempted count.")
print("If FAIL exists in a task with 12 OK, that task has >12 total states — LEAK.")
for tk in task_fail:
    if task_ok.get(tk, 0) + task_fail.get(tk, 0) > 12:
        print("  WARNING: {} has {} OK + {} FAIL = {} > 12".format(
            tk, task_ok.get(tk, 0), task_fail.get(tk, 0),
            task_ok.get(tk, 0) + task_fail.get(tk, 0)))
