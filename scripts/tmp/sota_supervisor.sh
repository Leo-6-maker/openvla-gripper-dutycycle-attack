#!/bin/bash
# TABLE1 SOTA 24-Hour Queue Supervisor
# Stages: Oracle audit → TMA canary → TMA full → UMA+SHUFFLED → aggregation
set -uo pipefail

BASE=/mnt/sdc/dty_user
EVID=$BASE/openvla_attack/evidence/sc5_object_privileged_loto_v1/vis_heldout_formal_v1
EXEC=$BASE/table1_sota_execution_v1
LOGDIR=$EXEC/logs
STATEDIR=$EXEC/state
PYTHON=$BASE/openvla_attack/envs/openvla-official-a800/bin/python
SOTA_WORKER=$EXEC/commands/run_sota_worker.py
ORACLE_WORKER=$EXEC/commands/run_oracle_env_gripper_worker.py

DEADLINE=$(date -d '+24 hours' +%s)
START_TIME=$(date +%s)
mkdir -p $STATEDIR $LOGDIR

log() { echo "$(date -Iseconds) $*" | tee -a $LOGDIR/supervisor.log; }
write_state() {
    python3 -c "import json; json.dump({'stage':'$1','updated':$(date +%s),'progress':'$2'}, open('$STATEDIR/queue_state.json','w'), indent=2)"
}

log "===== SOTA SUPERVISOR PID=$$ START ====="
log "Oracle currently running, waiting for completion..."

# ═══════════════════════════════════════════
# STAGE 0: Wait for Oracle completion
# ═══════════════════════════════════════════
write_state wait_oracle monitoring

ORACLE_DIR=$EVID/COMMAND_OPEN_ORACLE_T10/formal_v1
while true; do
    ORACLE_EP=$(find $ORACLE_DIR -name 'episode_summary.json' 2>/dev/null | wc -l)
    ORACLE_WORKERS=$(ps aux | grep -c 'run_oracle_env_gripper_worker' 2>/dev/null || echo 0)
    ORACLE_FAILED=$(grep -r 'FAILED' $LOGDIR/oracle_full_gpu*.log 2>/dev/null | wc -l) || ORACLE_FAILED=0

    log "Oracle: $ORACLE_EP/141, workers=$ORACLE_WORKERS, failed=$ORACLE_FAILED"

    if [ $ORACLE_WORKERS -le 2 ] && [ $ORACLE_EP -ge 141 ]; then
        log "Oracle COMPLETE: $ORACLE_EP/141"
        break
    fi

    NOW=$(date +%s)
    if [ $NOW -ge $DEADLINE ]; then
        log "DEADLINE at oracle=$ORACLE_EP/141"
        write_state deadline "oracle=$ORACLE_EP/141"
        exit 0
    fi
    sleep 120
done

# ═══════════════════════════════════════════
# STAGE 1: Audit Oracle
# ═══════════════════════════════════════════
write_state audit_oracle running
log "Stage 1: Auditing Oracle results"

python3 << 'PYEOF'
import os, json, csv, hashlib, time
from collections import defaultdict

ORACLE = "/mnt/sdc/dty_user/openvla_attack/evidence/sc5_object_privileged_loto_v1/vis_heldout_formal_v1/COMMAND_OPEN_ORACLE_T10/formal_v1"
TRUE_T10 = "/mnt/sdc/dty_user/openvla_attack/evidence/sc5_object_privileged_loto_v1/vis_heldout_formal_v1/TRUE_T10/formal_v1"
OUT = "/mnt/sdc/dty_user/openvla_attack/evidence/COMMAND_OPEN_ORACLE_V1"
os.makedirs(OUT, exist_ok=True)

def collect(base):
    data = {}
    for fold in sorted(os.listdir(base)):
        fp = os.path.join(base, fold)
        if not os.path.isdir(fp): continue
        fold_id = fold.split("_")[1]
        for sd in sorted(os.listdir(fp)):
            sid = int(sd.split("_")[1])
            sp = os.path.join(fp, sd)
            for dd in sorted(os.listdir(sp)):
                did = int(dd.split("_")[2])
                dp = os.path.join(sp, dd)
                for pd in sorted(os.listdir(dp)):
                    pid = int(pd.split("_")[2])
                    ep = os.path.join(dp, pd, "episode_summary.json")
                    if not os.path.exists(ep): continue
                    d = json.load(open(ep))
                    k = (fold_id, sid, did, pid)
                    tel = os.path.join(dp, pd, "step_telemetry.csv")
                    atk = 0; env_open = 0
                    if os.path.exists(tel):
                        for row in csv.DictReader(open(tel)):
                            if row.get("attack_this") == "True":
                                atk += 1
                                if float(row.get("env_gripper", 0)) < 0:
                                    env_open += 1
                    data[k] = {"success": d.get("task_success", False), "n_steps": d.get("n_steps", 0),
                               "attack_frames": atk, "env_open_frames": env_open,
                               "oracle_override_active": d.get("oracle_protocol", "") == "env_gripper_force_open_continuous"}
    return data

oracle = collect(ORACLE)
tt = collect(TRUE_T10)

tt_emit = {k: v for k, v in tt.items() if v.get("attack_frames", 0) > 0}
tt_keys = set(tt_emit.keys())
oracle_keys = set(oracle.keys())

common = tt_keys & oracle_keys
oracle_sr = sum(1 for k in common if oracle[k]["success"]) / max(1, len(common))
oracle_atk_rate = sum(1 for k in common if oracle[k]["attack_frames"] > 0) / max(1, len(common))

print(f"Oracle audit: {len(common)} common keys, SR={oracle_sr:.3f}, atk_exec={oracle_atk_rate:.3f}")

envelope = {
    "gate": "COMMAND_OPEN_ORACLE_V1",
    "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "total_episodes": len(oracle), "itt_emission_matched": len(common),
    "oracle_success_rate": round(oracle_sr, 4),
    "oracle_attack_execution_rate": round(oracle_atk_rate, 4),
    "oracle_protocol": "env_gripper_force_open_continuous",
    "status": "FROZEN",
}
with open(os.path.join(OUT, "FREEZE_ENVELOPE.json"), "w") as f:
    json.dump(envelope, f, indent=2)

# Per-episode ledger
with open(os.path.join(OUT, "ORACLE_LEDGER.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["fold","state_id","det_seed","pert_seed","oracle_success","oracle_n_steps","oracle_atk_frames","oracle_env_open_frames","tt_emit_step"])
    w.writeheader()
    for k in sorted(common):
        w.writerow({"fold": k[0], "state_id": k[1], "det_seed": k[2], "pert_seed": k[3],
                     "oracle_success": oracle[k]["success"], "oracle_n_steps": oracle[k]["n_steps"],
                     "oracle_atk_frames": oracle[k]["attack_frames"],
                     "oracle_env_open_frames": oracle[k]["env_open_frames"],
                     "tt_emit_step": tt_emit.get(k, {}).get("n_steps", -1)})

print(f"Oracle freeze: {OUT}")
print(json.dumps(envelope, indent=2))
PYEOF

log "Oracle audit complete"
write_state audit_oracle done

# ═══════════════════════════════════════════
# STAGE 2: TMA Canary Wave (4 conditions × 9-fold)
# ═══════════════════════════════════════════
NOW=$(date +%s)
if [ $NOW -ge $DEADLINE ]; then log "DEADLINE, exiting"; exit 0; fi

write_state tma_canary launching
log "Stage 2: Launching TMA/UMA/SHUFFLED canary wave"

# Kill any stale oracle processes
pkill -f 'run_oracle_env_gripper' 2>/dev/null || true
sleep 5

CANARY=$EXEC/canary

# Launch 4 canaries on GPUs 0-3 (1 per condition, serial execution of 9 jobs)
declare -A CANARY_GPU=([0]="TMA:tma_student" [1]="TMA_RANDOM_TIME:tma_random" [2]="UMA:uma" [3]="SHUFFLED:shuffled")

for gpu in 0 1 2 3; do
    IFS=':' read -r cond label <<< "${CANARY_GPU[$gpu]}"
    MF=$CANARY/$cond/manifest_canary.jsonl
    if [ -f "$MF" ]; then
        nohup $PYTHON -u $SOTA_WORKER $gpu $MF > $LOGDIR/canary_${label}_gpu${gpu}.log 2>&1 &
        log "  GPU $gpu ($cond canary) PID=$!"
    else
        log "  SKIP $cond: manifest missing"
    fi
done

log "Canary wave launched. Waiting for completion..."

# Wait for all 4 canaries (9 jobs each = 36 total)
CANARY_TOTAL=36
while true; do
    CANARY_DONE=0; CANARY_FAILED=0
    for label in tma_student tma_random uma shuffled; do
        for gpu in 0 1 2 3; do
            lf=$LOGDIR/canary_${label}_gpu${gpu}.log
            if [ -f "$lf" ]; then
                CANARY_DONE=$((CANARY_DONE + $(grep -c 'COMPLETE' $lf 2>/dev/null || echo 0)))
                CANARY_FAILED=$((CANARY_FAILED + $(grep -c 'FAILED' $lf 2>/dev/null || echo 0)))
            fi
        done
    done

    WORKERS=$(ps aux | grep -c 'run_sota_worker' 2>/dev/null || echo 0)
    log "Canary: $CANARY_DONE/$CANARY_TOTAL done, $CANARY_FAILED failed, $WORKERS workers"

    NOW=$(date +%s)
    if [ $NOW -ge $DEADLINE ]; then log "DEADLINE at canary=$CANARY_DONE/$CANARY_TOTAL"; exit 0; fi

    if [ $WORKERS -le 2 ] && [ $CANARY_DONE -ge $CANARY_TOTAL ]; then
        log "All canaries complete: $CANARY_DONE/$CANARY_TOTAL, failed=$CANARY_FAILED"
        break
    fi

    # Relaunch if workers died but not done
    if [ $WORKERS -le 2 ] && [ $CANARY_DONE -lt $CANARY_TOTAL ]; then
        log "Workers died, relaunching canaries"
        for gpu in 0 1 2 3; do
            IFS=':' read -r cond label <<< "${CANARY_GPU[$gpu]}"
            lf=$LOGDIR/canary_${label}_gpu${gpu}.log
            job_count=$(grep -c 'COMPLETE\|FAILED' $lf 2>/dev/null || echo 0)
            if [ "$job_count" -lt 9 ] 2>/dev/null; then
                MF=$CANARY/$cond/manifest_canary.jsonl
                nohup $PYTHON -u $SOTA_WORKER $gpu $MF > ${lf}_retry.log 2>&1 &
                log "  Relaunch GPU $gpu ($cond)"
            fi
        done
    fi
    sleep 180
done

# Canary verdict
if [ $CANARY_FAILED -gt 0 ]; then
    log "WARNING: $CANARY_FAILED canary failures — check logs"
fi
write_state tma_canary "done_${CANARY_DONE}_failed_${CANARY_FAILED}"

# ═══════════════════════════════════════════
# STAGE 3: TMA Full Wave — TMA Student + TMA Random-Time
# ═══════════════════════════════════════════
NOW=$(date +%s)
if [ $NOW -ge $DEADLINE ]; then log "DEADLINE, exiting"; exit 0; fi

write_state tma_full launching
log "Stage 3: TMA Full — Student (GPU 0-3) + Random-Time (GPU 4-7)"

SOTA_MF=$EXEC/manifests

for label in "TMA:tma_student" "TMA_RANDOM_TIME:tma_random"; do
    IFS=':' read -r cond label2 <<< "$label"
    MF_DIR=$SOTA_MF/$cond
    for gpu in 0 1 2 3 4 5 6 7; do
        MF=$MF_DIR/manifest_gpu${gpu}.jsonl
        if [ -f "$MF" ]; then
            nohup $PYTHON -u $SOTA_WORKER $gpu $MF > $LOGDIR/full_${label2}_gpu${gpu}.log 2>&1 &
            log "  GPU $gpu ($cond) PID=$!"
        fi
    done
done

log "TMA full launched. Waiting for completion (162 × 2 = 324 episodes)..."

while true; do
    DONE=0; FAILED=0
    for label2 in tma_student tma_random; do
        for gpu in 0 1 2 3 4 5 6 7; do
            lf=$LOGDIR/full_${label2}_gpu${gpu}.log
            if [ -f "$lf" ]; then
                DONE=$((DONE + $(grep -c 'COMPLETE' $lf 2>/dev/null || echo 0)))
                FAILED=$((FAILED + $(grep -c 'FAILED' $lf 2>/dev/null || echo 0)))
            fi
        done
    done
    WORKERS=$(ps aux | grep -c 'run_sota_worker' 2>/dev/null || echo 0)

    log "TMA Full: $DONE/324 done, $FAILED failed, $WORKERS workers"

    NOW=$(date +%s)
    if [ $NOW -ge $DEADLINE ]; then log "DEADLINE at TMA=$DONE/324"; exit 0; fi

    if [ $WORKERS -le 2 ] && [ $DONE -ge 324 ]; then
        log "TMA Full complete: $DONE/324"
        break
    fi

    # Relaunch dead workers
    if [ $WORKERS -le 2 ] && [ $DONE -lt 324 ]; then
        log "Workers died, relaunching TMA full"
        for label in "TMA:tma_student" "TMA_RANDOM_TIME:tma_random"; do
            IFS=':' read -r cond label2 <<< "$label"
            MF_DIR=$SOTA_MF/$cond
            for gpu in 0 1 2 3 4 5 6 7; do
                lf=$LOGDIR/full_${label2}_gpu${gpu}.log
                expected=$(wc -l < $MF_DIR/manifest_gpu${gpu}.jsonl 2>/dev/null || echo 0)
                done_count=$(grep -c 'COMPLETE\|FAILED' $lf 2>/dev/null || echo 0)
                if [ "$done_count" -lt "$expected" ] 2>/dev/null; then
                    MF=$MF_DIR/manifest_gpu${gpu}.jsonl
                    nohup $PYTHON -u $SOTA_WORKER $gpu $MF > ${lf}_retry.log 2>&1 &
                    log "  Relaunch GPU $gpu ($cond)"
                fi
            done
        done
    fi
    sleep 300
done

write_state tma_full "done_${DONE}_failed_${FAILED}"

# ═══════════════════════════════════════════
# STAGE 4: UMA + SHUFFLED Full Wave
# ═══════════════════════════════════════════
NOW=$(date +%s)
if [ $NOW -ge $DEADLINE ]; then log "DEADLINE, exiting"; exit 0; fi

write_state uma_shuffled launching
log "Stage 4: UMA (GPU 0-3) + SHUFFLED (GPU 4-7)"

for label in "UMA:uma" "SHUFFLED:shuffled"; do
    IFS=':' read -r cond label2 <<< "$label"
    MF_DIR=$SOTA_MF/$cond
    for gpu in 0 1 2 3 4 5 6 7; do
        MF=$MF_DIR/manifest_gpu${gpu}.jsonl
        if [ -f "$MF" ]; then
            nohup $PYTHON -u $SOTA_WORKER $gpu $MF > $LOGDIR/full_${label2}_gpu${gpu}.log 2>&1 &
            log "  GPU $gpu ($cond) PID=$!"
        fi
    done
done

log "UMA+SHUFFLED launched. Waiting for completion..."

while true; do
    DONE=0; FAILED=0
    for label2 in uma shuffled; do
        for gpu in 0 1 2 3 4 5 6 7; do
            lf=$LOGDIR/full_${label2}_gpu${gpu}.log
            if [ -f "$lf" ]; then
                DONE=$((DONE + $(grep -c 'COMPLETE' $lf 2>/dev/null || echo 0)))
                FAILED=$((FAILED + $(grep -c 'FAILED' $lf 2>/dev/null || echo 0)))
            fi
        done
    done
    WORKERS=$(ps aux | grep -c 'run_sota_worker' 2>/dev/null || echo 0)

    log "UMA+SHUFFLED: $DONE/324 done, $FAILED failed, $WORKERS workers"

    NOW=$(date +%s)
    if [ $NOW -ge $DEADLINE ]; then log "DEADLINE at U+S=$DONE/324"; exit 0; fi

    if [ $WORKERS -le 2 ] && [ $DONE -ge 324 ]; then
        log "UMA+SHUFFLED complete: $DONE/324"
        break
    fi

    if [ $WORKERS -le 2 ] && [ $DONE -lt 324 ]; then
        log "Workers died, relaunching UMA+SHUFFLED"
        for label in "UMA:uma" "SHUFFLED:shuffled"; do
            IFS=':' read -r cond label2 <<< "$label"
            MF_DIR=$SOTA_MF/$cond
            for gpu in 0 1 2 3 4 5 6 7; do
                lf=$LOGDIR/full_${label2}_gpu${gpu}.log
                expected=$(wc -l < $MF_DIR/manifest_gpu${gpu}.jsonl 2>/dev/null || echo 0)
                done_count=$(grep -c 'COMPLETE\|FAILED' $lf 2>/dev/null || echo 0)
                if [ "$done_count" -lt "$expected" ] 2>/dev/null; then
                    MF=$MF_DIR/manifest_gpu${gpu}.jsonl
                    nohup $PYTHON -u $SOTA_WORKER $gpu $MF > ${lf}_retry.log 2>&1 &
                    log "  Relaunch GPU $gpu ($cond)"
                fi
            done
        done
    fi
    sleep 300
done

write_state uma_shuffled "done_${DONE}_failed_${FAILED}"

# ═══════════════════════════════════════════
# STAGE 5: Final Aggregation
# ═══════════════════════════════════════════
NOW=$(date +%s)
if [ $NOW -ge $DEADLINE ]; then log "DEADLINE, exiting"; exit 0; fi

write_state aggregation running
log "Stage 5: TABLE1_SOTA_V1 final aggregation"

python3 << 'PYEOF'
import os, json, csv, hashlib, time
from collections import defaultdict

EVID = "/mnt/sdc/dty_user/openvla_attack/evidence/sc5_object_privileged_loto_v1/vis_heldout_formal_v1"
OUT = "/mnt/sdc/dty_user/openvla_attack/evidence/TABLE1_SOTA_V1"
os.makedirs(OUT, exist_ok=True)

CONDITIONS = {
    "CLEAN": "CLEAN/formal_v1",
    "TRUE_T10": "TRUE_T10/formal_v1",
    "RANDOM_TIME_V3": "RANDOM_TIME_V3/formal_v1",
    "RAND_T10": "RAND_T10/formal_v1",
    "EARLY_SHIFT_T10": "EARLY_SHIFT_T10/formal_v1",
    "COMMAND_OPEN_ORACLE_T10": "COMMAND_OPEN_ORACLE_T10/formal_v1",
    "TMA": "TMA/formal_v1",
    "TMA_RANDOM_TIME": "TMA_RANDOM_TIME/formal_v1",
    "UMA": "UMA/formal_v1",
    "SHUFFLED": "SHUFFLED/formal_v1",
}

def collect(base_path):
    if not os.path.isdir(base_path): return {}
    data = {}
    for fold in sorted(os.listdir(base_path)):
        fp = os.path.join(base_path, fold)
        if not os.path.isdir(fp): continue
        fold_id = fold.split("_")[1] if "_" in fold else fold
        for sd in sorted(os.listdir(fp)):
            sp = os.path.join(fp, sd)
            if not os.path.isdir(sp): continue
            for dd in sorted(os.listdir(sp)):
                dp = os.path.join(sp, dd)
                if not os.path.isdir(dp): continue
                for pd in sorted(os.listdir(dp)):
                    ep = os.path.join(dp, pd, "episode_summary.json")
                    if not os.path.exists(ep): continue
                    d = json.load(open(ep))
                    k = (fold_id, sd, dd, pd)
                    data[k] = {"success": d.get("task_success", False),
                               "n_steps": d.get("n_steps", 0),
                               "attack_frames": d.get("attack_frames", 0)}
    return data

all_data = {}
for name, rel_path in CONDITIONS.items():
    full_path = os.path.join(EVID, rel_path)
    all_data[name] = collect(full_path)
    print(f"{name}: {len(all_data[name])} episodes")

# Write per-condition summary
rows = []
for name, data in all_data.items():
    if not data: continue
    n = len(data)
    sr = sum(1 for v in data.values() if v["success"]) / max(1, n)
    atk = sum(v["attack_frames"] for v in data.values())
    rows.append({"condition": name, "n": n, "success_rate": round(sr, 4),
                 "total_attack_frames": atk})

with open(os.path.join(OUT, "TABLE1_SOTA_PANEL_A.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["condition", "n", "success_rate", "total_attack_frames"])
    w.writeheader(); w.writerows(rows)

envelope = {
    "gate": "TABLE1_SOTA_V1",
    "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "conditions_evaluated": list(all_data.keys()),
    "panel_a": rows,
}
with open(os.path.join(OUT, "FREEZE_ENVELOPE.json"), "w") as f:
    json.dump(envelope, f, indent=2)

print(f"\nTABLE1_SOTA_V1 saved to {OUT}")
for r in rows:
    print(f"  {r['condition']}: n={r['n']} SR={r['success_rate']} atk={r['total_attack_frames']}")
PYEOF

log "Aggregation complete"
write_state done "completed"

# ═══════════════════════════════════════════
# DONE
# ═══════════════════════════════════════════
ELAPSED=$(($(date +%s) - START_TIME))
log "===== SOTA SUPERVISOR COMPLETE in ${ELAPSED}s ====="
echo "ALL DONE"
