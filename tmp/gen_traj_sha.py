#!/usr/bin/env python3
import os, csv, hashlib
from pathlib import Path

def sha256_file(p):
    with open(p, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()

def trajectory_hash(tel_path, ep_path):
    tel_bytes = open(tel_path, "rb").read()
    ep_bytes = open(ep_path, "rb").read()
    return hashlib.sha256(tel_bytes + b"\x00" + ep_bytes).hexdigest()

corpus = Path("/mnt/sdc/dty_user/openvla_attack/evidence/m1c/object_clean_corpus")
out_dir = Path("/mnt/sdc/dty_user/openvla_attack/evidence/m1c/object_clean_corpus_audit_preflight_20260624")
out_dir.mkdir(parents=True, exist_ok=True)

rows = []
for pool in ["train", "validation"]:
    for cell_dir in sorted((corpus / pool).iterdir()):
        if not cell_dir.is_dir():
            continue
        try:
            parts = cell_dir.name.split("_")
            task = int(parts[0].replace("task", ""))
            state = int(parts[1].replace("state", ""))
        except (ValueError, IndexError):
            continue
        tel = cell_dir / "step_telemetry.csv"
        ep = cell_dir / "episode_summary.json"
        if tel.exists() and ep.exists():
            rows.append({
                "pool": pool, "task": task, "state": state,
                "telemetry_sha256": sha256_file(tel),
                "summary_sha256": sha256_file(ep),
                "trajectory_content_sha256": trajectory_hash(tel, ep),
            })

with open(out_dir / "trajectory_sha256_sidecar.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader()
    w.writerows(rows)

hashes = [r["trajectory_content_sha256"] for r in rows]
unique = len(set(hashes))
dup = len(hashes) - unique
print(f"Generated {len(rows)} trajectory hashes")
print(f"Unique: {unique}, Duplicates: {dup}")
for r in rows[:3]:
    print(f"  {r['pool']}/task{r['task']}_state{r['state']}: {r['trajectory_content_sha256'][:16]}...")
