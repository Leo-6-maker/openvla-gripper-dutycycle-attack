#!/usr/bin/env python3
"""Phase S0: Seal Spatial 100 evidence — immutable manifest."""
import csv, json, hashlib, os, re
from collections import defaultdict

SPATIAL = "/data/liuyu/outputs/libero_spatial_clean100_20260617_r1"
OUT = os.path.join(SPATIAL, "analysis")


def sha256_file(p):
    if not os.path.isfile(p):
        return ""
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


rows = []
task_states = defaultdict(set)
for d in sorted(os.listdir(SPATIAL)):
    dp = os.path.join(SPATIAL, d)
    if not os.path.isdir(dp):
        continue
    st = os.path.join(dp, "step_trace.csv")
    if not os.path.exists(st):
        continue
    m = re.match(r"(.+)_s(\d+)_shadow_attempt(\d+)", d)
    if not m:
        continue
    task = m.group(1)
    sid = int(m.group(2))
    attempt = int(m.group(3))
    if attempt != 1:
        continue

    rows_csv = list(csv.DictReader(open(st)))
    n = len(rows_csv)
    succ_d = rows_csv[-1].get("success_done", "0") if rows_csv else "0"
    succ_c = rows_csv[-1].get("success_check", "0") if rows_csv else "0"
    primary_succ = 1 if (succ_d == "1" or succ_c == "1") else 0

    de = os.path.join(dp, "detector_emission.json")
    emit = json.load(open(de)).get("emit_step", -1) if os.path.exists(de) else -1

    eef_ok = all(float(r.get("eef_valid", "0") or 0) == 1 for r in rows_csv)
    qpos_ok = all(float(r.get("qpos_valid", "0") or 0) == 1 for r in rows_csv)

    rows.append({
        "task": task, "state_id": sid, "attempt": attempt,
        "steps": n, "success_done": succ_d, "success_check": succ_c,
        "primary_success": primary_succ,
        "d5_emit": emit, "qpos_valid": int(qpos_ok), "eef_valid": int(eef_ok),
        "step_trace_sha256": sha256_file(st),
        "action_identity_sha256": sha256_file(os.path.join(dp, "action_identity.csv")),
        "candidates_sha256": sha256_file(os.path.join(dp, "detector_candidates.csv")),
        "emission_sha256": sha256_file(os.path.join(dp, "detector_emission.json")),
    })
    task_states[task].add(sid)

# Write manifest
with open(os.path.join(OUT, "spatial_clean100_episode_manifest.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

# Task-state matrix
with open(os.path.join(OUT, "spatial_clean100_task_state_matrix.csv"), "w", newline="") as f:
    f.write("task,states_present,n_states\n")
    for t in sorted(task_states.keys()):
        f.write(t + "," + str(len(task_states[t])) + "," + str(sorted(task_states[t])) + "\n")

# Artifact hashes
with open(os.path.join(OUT, "spatial_clean100_artifact_hashes.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["file", "sha256"])
    for r in rows:
        base = "Spatial/" + r["task"] + "_s" + str(r["state_id"])
        w.writerow([base + "/step_trace.csv", r["step_trace_sha256"]])
        w.writerow([base + "/action_identity.csv", r["action_identity_sha256"]])
        w.writerow([base + "/detector_candidates.csv", r["candidates_sha256"]])
        w.writerow([base + "/detector_emission.json", r["emission_sha256"]])

n_total = len(rows)
n_succ = sum(1 for r in rows if r["primary_success"])
n_emit = sum(1 for r in rows if r["d5_emit"] >= 0)
n_eef_ok = sum(1 for r in rows if r["eef_valid"])
n_qpos_ok = sum(1 for r in rows if r["qpos_valid"])
print("S0 SEAL: " + str(n_total) + " primary episodes")
print("  Success: " + str(n_succ) + "/" + str(n_total))
print("  Emit: " + str(n_emit) + "/" + str(n_total))
print("  All EEF valid: " + str(n_eef_ok) + "/" + str(n_total))
print("  All Qpos valid: " + str(n_qpos_ok) + "/" + str(n_total))
print("  Tasks: " + str(len(task_states)))
for t in sorted(task_states.keys()):
    ss = task_states[t]
    ok = "OK" if len(ss) == 10 else "WARNING: " + str(len(ss))
    print("  " + t[:60] + ": " + str(len(ss)) + "/10 states " + ok)
print("Output: " + OUT + "/")
