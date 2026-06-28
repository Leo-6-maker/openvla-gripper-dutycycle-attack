#!/usr/bin/env python3
"""Generate immutable 1500-job manifest for CLEAN1500 collection."""
import json, hashlib, sys, os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PROTO_PATH = REPO / "configs" / "cross_suite_clean1500_protocol_v1.json"
REG_PATH = REPO / "configs" / "cross_suite_object_target_registry_v1.json"
COL_PATH = REPO / "scripts" / "stageb" / "run_cross_suite_clean_v3.py"

with open(PROTO_PATH, "rb") as f: proto_raw = f.read()
with open(REG_PATH, "rb") as f: reg_raw = f.read()
with open(COL_PATH, "rb") as f: col_raw = f.read()
proto_sha = hashlib.sha256(proto_raw).hexdigest()
reg_sha = hashlib.sha256(reg_raw).hexdigest()
col_sha = hashlib.sha256(col_raw).hexdigest()

protocol = json.loads(proto_raw)
registry = json.loads(reg_raw)
output_root = protocol["output_root"]

jobs = []
seen = set()
for suite_name, sc in protocol["suites"].items():
    gpu = sc["gpu"]
    for t in range(10):
        entry = registry[suite_name][str(t)]
        eligible = entry["teacher_eligible"]
        for s in range(50):
            key = "%s_t%02d_s%02d_seed0" % (suite_name, t, s)
            assert key not in seen, "Duplicate key: %s" % key
            seen.add(key)

            job = {
                "job_key": key,
                "suite": suite_name,
                "task_idx": t,
                "state_id": s,
                "eval_seed": 0,
                "gpu": gpu,
                "max_steps": 400,
                "condition": "CLEAN",
                "teacher_eligible": eligible,
                "output_dir": "%s/%s/task_%02d/state_%02d" % (output_root, suite_name, t, s),
                "protocol_sha256": proto_sha,
                "registry_sha256": reg_sha,
                "collector_sha256": col_sha,
                "status": "PENDING",
                "attempt": 0,
            }
            jobs.append(job)

# Write JSONL manifest
manifest_dir = REPO / "manifests"
manifest_dir.mkdir(exist_ok=True)
manifest_path = manifest_dir / "cross_suite_clean1500_jobs_v1.jsonl"
with open(manifest_path, "w") as f:
    for job in jobs:
        f.write(json.dumps(job) + "\n")

# SHA256
with open(manifest_path, "rb") as f:
    manifest_sha = hashlib.sha256(f.read()).hexdigest()
with open(str(manifest_path) + ".sha256", "w") as f:
    f.write("%s  cross_suite_clean1500_jobs_v1.jsonl\n" % manifest_sha)

# Summary
counts = {}
for j in jobs:
    counts[j["suite"]] = counts.get(j["suite"], 0) + 1

print("Manifest: %s (%d jobs)" % (manifest_path, len(jobs)))
print("SHA256: %s" % manifest_sha)
for s, c in sorted(counts.items()):
    print("  %s: %d" % (s, c))
print("Unique keys: %d" % len(seen))
print("Protocol SHA: %s" % proto_sha[:16])
print("Registry SHA: %s" % reg_sha[:16])
print("Collector SHA: %s" % col_sha[:16])
