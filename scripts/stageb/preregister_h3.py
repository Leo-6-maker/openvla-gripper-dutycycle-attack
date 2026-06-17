#!/usr/bin/env python3
"""P0: H3 preregistration — freeze all steps, queues, SHAs, GPU bindings."""
import csv, hashlib, json, os, subprocess, sys, yaml
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT_DIR = REPO / "tables"
CFG_DIR = REPO / "configs"
RPT_DIR = REPO / "reports"
ART_DIR = REPO / "artifacts"
for d in [OUT_DIR, CFG_DIR, RPT_DIR, ART_DIR]: d.mkdir(parents=True, exist_ok=True)

WINDOWS = {
    "butter_s11": {"task": "butter", "state_id": 11, "anchor": 60, "start": 57, "end": 63},
    "tomato_sauce_s23": {"task": "tomato_sauce", "state_id": 23, "anchor": 141, "start": 138, "end": 144},
    "salad_dressing_s11": {"task": "salad_dressing", "state_id": 11, "anchor": 59, "start": 56, "end": 62},
}

EXISTING = {
    ("butter", 11, 58), ("butter", 11, 60),
    ("tomato_sauce", 23, 139), ("tomato_sauce", 23, 141),
    ("salad_dressing", 11, 57), ("salad_dressing", 11, 59),
}

QUEUE_A = [("butter", 11, 57), ("butter", 11, 59), ("butter", 11, 61), ("butter", 11, 62), ("butter", 11, 63),
           ("tomato_sauce", 23, 138), ("tomato_sauce", 23, 140), ("tomato_sauce", 23, 142)]

QUEUE_B = [("salad_dressing", 11, 56), ("salad_dressing", 11, 58), ("salad_dressing", 11, 60),
           ("salad_dressing", 11, 61), ("salad_dressing", 11, 62),
           ("tomato_sauce", 23, 143), ("tomato_sauce", 23, 144)]


def sha256_file(p):
    if not os.path.isfile(p): return ""
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""): h.update(chunk)
    return h.hexdigest()


# ── All preregistered steps ──
all_steps = []
for pid, w in WINDOWS.items():
    for s in range(w["start"], w["end"] + 1):
        t = (w["task"], w["state_id"], s)
        exists = t in EXISTING
        if t in QUEUE_A:
            worker = "Worker_A_GPU15"
        elif t in QUEUE_B:
            worker = "Worker_B_GPU26"
        elif exists:
            worker = "existing_reuse"
        else:
            worker = "UNASSIGNED"
        all_steps.append({
            "parent_id": pid, "task": w["task"], "state_id": w["state_id"],
            "step": s, "is_anchor": s == w["anchor"],
            "existing": exists, "capture_worker": worker,
            "relation_to_anchor": s - w["anchor"],
        })

n_existing = sum(1 for r in all_steps if r["existing"])
n_missing = sum(1 for r in all_steps if not r["existing"])
print("Preregistered steps: {} ({} existing, {} missing)".format(len(all_steps), n_existing, n_missing))

with open(OUT_DIR / "l3_h3_preregistered_steps.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(all_steps[0].keys()))
    w.writeheader(); w.writerows(all_steps)

# Capture queue
cap_a = [r for r in all_steps if r["capture_worker"] == "Worker_A_GPU15"]
cap_b = [r for r in all_steps if r["capture_worker"] == "Worker_B_GPU26"]
print("Worker A capture: {} steps".format(len(cap_a)))
print("Worker B capture: {} steps".format(len(cap_b)))

with open(OUT_DIR / "l3_h3_capture_queue.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["worker", "task", "state_id", "step", "parent_id"])
    w.writeheader()
    for r in cap_a: w.writerow({"worker": "A_GPU15", "task": r["task"], "state_id": r["state_id"], "step": r["step"], "parent_id": r["parent_id"]})
    for r in cap_b: w.writerow({"worker": "B_GPU26", "task": r["task"], "state_id": r["state_id"], "step": r["step"], "parent_id": r["parent_id"]})

# ── Source hashes ──
hashes = {
    "production_tag": "l12-d5-v1-production-20260617",
    "production_tag_target": "593ffadba7c7d64eadc4305fa818cd5d2c570507",
    "starting_commit": "bcd945f5e9ff4e4f85479032755ed770226b64e6",
    "runner_sha256": sha256_file(REPO / "scripts/stageb/run_m3_step78_true_pgd_fixed_frame.py"),
    "attack_adapter_sha256": sha256_file(REPO / "src/gripper_attack/attack_adapter.py"),
    "v4_config_template_sha256": sha256_file(REPO / "configs/m3_butter_s11_step60_v4.yaml"),
    "frozen_params": {
        "arm_preserve_weight": 0.5, "epsilon": 0.023529411764705882,
        "target_token": 31744, "num_steps": 20, "arm_gate": 5,
        "seeds": [81, 82], "candidate_count": 21,
    },
}
with open(ART_DIR / "l3_h3_source_hashes.json", "w") as f:
    json.dump(hashes, f, indent=2)

# ── GPU bindings ──
gpu_out = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,name", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.strip().split("\n")
gpu_uuids = {}
for line in gpu_out:
    parts = [p.strip() for p in line.split(",")]
    if len(parts) >= 2: gpu_uuids[int(parts[0])] = parts[1]

xid_cmd = subprocess.run("dmesg", capture_output=True, text=True).stdout
xid_count = len([l for l in xid_cmd.split("\n") if "NVRM" in l and "Xid" in l])

bindings = {
    "worker_a": {"gpu_pair": [1, 5], "uuids": [gpu_uuids.get(1), gpu_uuids.get(5)]},
    "worker_b": {"gpu_pair": [2, 6], "uuids": [gpu_uuids.get(2), gpu_uuids.get(6)]},
    "xid_baseline": xid_count,
}
with open(ART_DIR / "l3_h3_gpu_bindings.json", "w") as f:
    json.dump(bindings, f, indent=2)

# ── H3 window config ──
window_cfg = {"stage": "L3_H3_ATTACK_WINDOW_MAPPING", "parents": {}}
for pid, w in WINDOWS.items():
    window_cfg["parents"][pid] = {
        "task": w["task"], "state_id": w["state_id"],
        "anchor": w["anchor"], "range": list(range(w["start"], w["end"] + 1)),
    }
with open(CFG_DIR / "l3_h3_window_v1.yaml", "w") as f:
    yaml.dump(window_cfg, f, default_flow_style=False, sort_keys=False)

# ── Report ──
with open(RPT_DIR / "L3_H3_PREREGISTRATION.md", "w") as f:
    f.write("# H3 Preregistration\n\n")
    f.write("## Windows\n\n")
    for pid, w in WINDOWS.items():
        f.write("- **{}**: anchor={}, range=[{},{})\n".format(pid, w["anchor"], w["start"], w["end"] + 1))
    f.write("\n## Capture Queue\n\n")
    f.write("- Worker A (GPU 1,5): {} steps\n".format(len(cap_a)))
    f.write("- Worker B (GPU 2,6): {} steps\n".format(len(cap_b)))
    f.write("\n## Frozen Contract\n\n")
    f.write("- Runner SHA: {}\n".format(hashes["runner_sha256"][:16]))
    f.write("- Adapter SHA: {}\n".format(hashes["attack_adapter_sha256"][:16]))
    f.write("- Xid baseline: {}\n".format(xid_count))

print("\nP0 complete. Files written to tables/, configs/, artifacts/, reports/")
