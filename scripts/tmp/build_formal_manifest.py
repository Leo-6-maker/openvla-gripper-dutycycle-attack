#!/usr/bin/env python3
"""Merge 8 GPU-shard manifests into formal_162.jsonl with validation."""
import json, os, sys, hashlib

def build(input_dir, output_path):
    shards = sorted(f for f in os.listdir(input_dir) if f.startswith("manifest_gpu") and f.endswith(".jsonl"))
    if len(shards) != 8:
        print(f"FATAL: expected 8 shards, got {len(shards)}: {shards}")
        sys.exit(1)

    all_jobs = []
    seen_keys = set()
    seen_dirs = set()
    seen_jobkeys = set()
    dupes = []
    for sf in shards:
        for line in open(os.path.join(input_dir, sf)):
            j = json.loads(line.strip())
            key = (j["fold"], str(j["state_id"]), str(j["detector_seed"]), str(j["perturbation_seed"]))
            d = j["output_dir"]
            jk = j.get("job_key", "")
            if key in seen_keys: dupes.append(f"key:{key}")
            if d in seen_dirs: dupes.append(f"dir:{d}")
            if jk in seen_jobkeys: dupes.append(f"job_key:{jk}")
            seen_keys.add(key); seen_dirs.add(d); seen_jobkeys.add(jk)
            all_jobs.append(j)

    errors = []
    if len(all_jobs) != 162:
        errors.append(f"total jobs={len(all_jobs)} != 162")
    if len(seen_keys) != 162:
        errors.append(f"unique keys={len(seen_keys)} != 162")
    if len(seen_dirs) != 162:
        errors.append(f"unique dirs={len(seen_dirs)} != 162")
    if len(seen_jobkeys) != 162:
        errors.append(f"unique job_keys={len(seen_jobkeys)} != 162")
    if dupes:
        errors.append(f"duplicates: {dupes}")

    if errors:
        for e in errors: print(f"FATAL: {e}")
        sys.exit(1)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        for j in all_jobs:
            f.write(json.dumps(j) + "\n")

    sha = hashlib.sha256(open(output_path, "rb").read()).hexdigest()
    print(f"Formal manifest: {len(all_jobs)} jobs, {len(seen_keys)} unique keys, SHA={sha[:16]}")
    return sha

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: build_formal_manifest.py <shards_dir> <output_path>")
        sys.exit(1)
    build(sys.argv[1], sys.argv[2])
