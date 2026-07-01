#!/bin/bash
# CHAIN Stage 5: TABLE1_SOTA_V1 aggregation — UNVERIFIED preview only (not freeze)
set -uo pipefail
EXEC=/mnt/sdc/dty_user/table1_sota_execution_v1
LOGDIR=$EXEC/logs
log() { echo "$(date -Iseconds) [STAGE5] $*" | tee -a $LOGDIR/chain.log; }

log "=== TABLE1_SOTA_V1 AGGREGATION (UNVERIFIED PREVIEW) ==="

python3 << 'PYEOF'
import os, json, time

EVID = "/mnt/sdc/dty_user/openvla_attack/evidence/sc5_object_privileged_loto_v1/vis_heldout_formal_v1"
OUT = "/mnt/sdc/dty_user/openvla_attack/evidence/TABLE1_SOTA_V1"
os.makedirs(OUT, exist_ok=True)

CONDITIONS = {
    "CLEAN": "CLEAN/formal_v1", "TRUE_T10": "TRUE_T10/formal_v1",
    "RANDOM_TIME_V3": "RANDOM_TIME_V3/formal_v1", "RAND_T10": "RAND_T10/formal_v1",
    "EARLY_SHIFT_T10": "EARLY_SHIFT_T10/formal_v1",
    "COMMAND_OPEN_ORACLE": "COMMAND_OPEN_ORACLE_T10/formal_v1",
    "TMA_STUDENT": "TMA/formal_v1", "TMA_RANDOM_TIME": "TMA_RANDOM_TIME/formal_v1",
    "UMA": "UMA/formal_v1", "SHUFFLED": "SHUFFLED/formal_v1",
}

def collect(path):
    if not os.path.isdir(path): return {}
    data = {}
    for root, dirs, files in os.walk(path):
        for f in files:
            if f == "episode_summary.json":
                d = json.load(open(os.path.join(root, f)))
                key = os.path.relpath(root, path)
                data[key] = {"success": d.get("task_success", False),
                             "n_steps": d.get("n_steps", 0),
                             "attack_frames": d.get("attack_frames", 0),
                             "condition": d.get("condition", "")}
    return data

rows = []
for name, rel_path in CONDITIONS.items():
    data = collect(os.path.join(EVID, rel_path))
    n = len(data)
    expected = 162 if name != "COMMAND_OPEN_ORACLE" else 141
    sr = round(sum(1 for v in data.values() if v["success"]) / max(1, n), 4)
    atk = sum(v["attack_frames"] for v in data.values())
    rows.append({"condition": name, "n": n, "expected": expected,
                 "complete": n >= expected,
                 "success_rate": sr if n > 0 else None,
                 "total_attack_frames": atk})
    status = "OK" if n >= expected else "INCOMPLETE" if n > 0 else "MISSING"
    print(f"  {name}: n={n}/{expected} SR={sr} atk={atk} [{status}]")

envelope = {
    "gate": "TABLE1_SOTA_V1",
    "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "status": "UNVERIFIED_PREVIEW",
    "panel": rows,
}
with open(os.path.join(OUT, "TABLE1_SOTA_PREVIEW.json"), "w") as f:
    json.dump(envelope, f, indent=2)
with open(os.path.join(OUT, "TABLE1_SOTA_PANEL.csv"), "w") as f:
    f.write("condition,n,expected,complete,success_rate,total_attack_frames\n")
    for r in rows:
        f.write(f"{r['condition']},{r['n']},{r['expected']},{r['complete']},{r['success_rate']},{r['total_attack_frames']}\n")
print(f"TABLE1_SOTA_V1 preview saved to {OUT} (status: UNVERIFIED)")
PYEOF

log "=== ALL DONE (Preview only — formal freeze requires separate validator) ==="
