#!/usr/bin/env bash
# Phase 8 Worker V2 — Persistent polling mode
# Exits only on STOP file or ALL_DONE file (set by controller)
# Usage: bash phase8_worker_v2.sh <GPU_ID> <WORKER_ID>
set -euo pipefail

GPU=$1; WID=$2
BASE=/mnt/sdc/dty_user/openvla_attack
QDIR=$BASE/evidence/phase8_cross_suite_v1/queue_v2
RUNS=$BASE/evidence/phase8_cross_suite_v1/runs_v2
BRIDGE=$BASE/scripts/stageb/run_v2_vis_sc5_mlp_bridge_telemetry_v2.py
REGISTRY=$BASE/configs/phase8_primary_object_sites.json

export MUJOCO_GL=egl HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export OPENVLA_DTYPE=bfloat16 OPENVLA_ATTN_IMPLEMENTATION=eager TF_FORCE_GPU_ALLOW_GROWTH=true
export HOME=$BASE/sandbox_home/phase8_$WID TMPDIR=$BASE/tmp/phase8/$WID
export CUDA_VISIBLE_DEVICES=$GPU

mkdir -p $HOME $HOME/.libero $TMPDIR $RUNS
mkdir -p $QDIR/{pending,running,done,failed,claims,heartbeats,logs,gates}
mkdir -p $QDIR/holding_after_p1

# Auto-create LIBERO config
[ -f $HOME/.libero/config.yaml ] || cp /home/dty_user/.libero/config.yaml $HOME/.libero/config.yaml

PY=$BASE/envs/openvla-official-a800/bin/python3
DETECTOR=$BASE/outputs/sc5_v2_seed42/sc5_mlp_v2.pt
COMMIT=$(git -C $BASE rev-parse HEAD)

declare -A MODEL_PATH UK_MAP SN_MAP
MODEL_PATH[libero_spatial]="$BASE/models/libero-spatial/spatial_c8f03f4_20260620"
MODEL_PATH[libero_goal]="$BASE/models/libero-goal"
MODEL_PATH[libero_10]="$BASE/models/libero-10/openvla-7b-finetuned-libero-10"
UK_MAP[libero_spatial]=libero_spatial
UK_MAP[libero_goal]=libero_goal
UK_MAP[libero_10]=libero_10
SN_MAP[libero_spatial]=libero_spatial
SN_MAP[libero_goal]=libero_goal
SN_MAP[libero_10]=libero_10

echo "$(date -Iseconds) Worker $WID GPU$GPU started (persistent mode)"

idle_rounds=0

run_job() {
    local JID=$1 JSON=$2 OUT=$RUNS/$JID
    [ -f "$OUT/.done" ] && return 0

    local SUITE=$(echo "$JSON" | python3 -c "import sys,json; print(json.load(sys.stdin)['suite'])")
    local TASK=$(echo "$JSON" | python3 -c "import sys,json; print(json.load(sys.stdin)['task_idx'])")
    local SEED=$(echo "$JSON" | python3 -c "import sys,json; print(json.load(sys.stdin)['evaluation_seed'])")
    local COND=$(echo "$JSON" | python3 -c "import sys,json; print(json.load(sys.stdin)['condition'])")
    local AL=$(echo "$JSON" | python3 -c "import sys,json; print(json.load(sys.stdin)['arm_lock'])")
    local OBJ=$(echo "$JSON" | python3 -c "import sys,json; print(json.load(sys.stdin)['objective_id'])")
    local ATK=$(echo "$JSON" | python3 -c "import sys,json; print(json.load(sys.stdin)['attack_enabled'])")

    local MODEL=${MODEL_PATH[$SUITE]:-}
    local UK=${UK_MAP[$SUITE]:-}
    local SN=${SN_MAP[$SUITE]:-}
    [ -z "$MODEL" ] && { echo "FATAL: unknown suite=$SUITE"; return 1; }

    local SID=0
    [ "$SEED" = "123" ] && SID=1
    [ "$SEED" = "456" ] && SID=2

    local ARGS="--task_idx $TASK --state_id $SID --eval_seed 0 --seed_id $SEED"
    ARGS="$ARGS --output_dir $OUT --render_gpu $GPU --mlp_path $DETECTOR"
    ARGS="$ARGS --libero_preprocess_backend upstream_tf_jpeg --anchor 0"
    ARGS="$ARGS --unnorm_key $UK --suite_name $SN"
    ARGS="$ARGS --object_site_registry $REGISTRY"
    ARGS="$ARGS --source_commit $COMMIT"
    ARGS="$ARGS --save_video --video_fps 10 --frame_stride 2"

    if [ "$COND" = "RANDOM" ]; then
        ARGS="$ARGS --condition RAND_T10 --attack_objective ''"
    elif [ "$ATK" = "True" ] || [ "$ATK" = "true" ]; then
        ARGS="$ARGS --condition TRUE_T10 --attack_objective $OBJ"
        [ "$AL" = "True" ] || [ "$AL" = "true" ] && ARGS="$ARGS --arm_lock"
    else
        ARGS="$ARGS --condition CLEAN --attack_objective ''"
    fi

    mkdir -p "$OUT"
    echo "$(date -Iseconds) GPU$GPU $WID: $JID $COND $SUITE t$TASK s$SID seed$SEED" | tee -a $QDIR/logs/${WID}.log

    env OPENVLA_MODEL_PATH=$MODEL $PY -u $BRIDGE $ARGS > "$OUT/stdout.log" 2> "$OUT/stderr.log"
    local EC=$?
    echo "$(date -Iseconds) GPU$GPU $WID: $JID exit=$EC" | tee -a $QDIR/logs/${WID}.log
    [ $EC -eq 0 ] && touch "$OUT/.done"
    return $EC
}

while true; do
    # Exit conditions (controller only)
    if [ -f "$QDIR/STOP" ]; then
        echo "$(date -Iseconds) $WID: STOP received, exiting"
        exit 0
    fi
    if [ -f "$QDIR/ALL_DONE" ]; then
        echo "$(date -Iseconds) $WID: ALL_DONE received, exiting"
        exit 0
    fi

    # Scan pending
    shopt -s nullglob
    CANDIDATES=("$QDIR"/pending/*.json)
    shopt -u nullglob

    CLAIMED=""
    CLAIMED_FILE=""

    if [ ${#CANDIDATES[@]} -gt 0 ]; then
        # Shuffle to avoid 10 workers hitting same first file
        mapfile -t CANDIDATES < <(printf '%s\n' "${CANDIDATES[@]}" | shuf)

        for JOBF in "${CANDIDATES[@]}"; do
            JID=$(basename "$JOBF" .json)
            LOCKDIR=$QDIR/claims/$JID.lock

            if mkdir "$LOCKDIR" 2>/dev/null; then
                # Double-check pending file still exists
                if [ ! -f "$JOBF" ]; then
                    rmdir "$LOCKDIR" 2>/dev/null || true
                    continue
                fi

                CLAIMED=$JID
                CLAIMED_FILE=$JOBF

                cat > "$LOCKDIR/owner.json" << EOF
{"job_id":"$JID","worker_id":"$WID","gpu_id":$GPU,"pid":$$,"hostname":"$(hostname)","claimed_at":"$(date -Iseconds)","git_commit":"$COMMIT"}
EOF

                mv "$JOBF" "$QDIR/running/${JID}.${WID}.json"
                break
            fi
        done
    fi

    if [ -z "$CLAIMED" ]; then
        idle_rounds=$((idle_rounds + 1))
        [ $((idle_rounds % 10)) -eq 0 ] && \
            echo "$(date -Iseconds) $WID GPU$GPU: idle round=$idle_rounds"
        sleep $((5 + RANDOM % 8))
        continue
    fi

    idle_rounds=0

    # Heartbeat subprocess
    (
        while kill -0 $$ 2>/dev/null; do
            echo "$(date -Iseconds)" > "$QDIR/heartbeats/${CLAIMED}.json.tmp"
            mv "$QDIR/heartbeats/${CLAIMED}.json.tmp" "$QDIR/heartbeats/${CLAIMED}.json" 2>/dev/null || true
            sleep 30
        done
    ) &
    HB_PID=$!

    RUNNING_JSON=$QDIR/running/${CLAIMED}.${WID}.json
    JOB_JSON=$(cat "$RUNNING_JSON")
    run_job "$CLAIMED" "$JOB_JSON" && RC=0 || RC=$?

    kill $HB_PID 2>/dev/null || true
    wait $HB_PID 2>/dev/null || true
    rm -f "$QDIR/heartbeats/${CLAIMED}.json"
    rm -rf "$LOCKDIR"

    if [ $RC -eq 0 ] && [ -f "$RUNS/$CLAIMED/.done" ]; then
        mv "$RUNNING_JSON" "$QDIR/done/${CLAIMED}.json" 2>/dev/null || true
        echo "$(date -Iseconds) $WID: DONE $CLAIMED"
    else
        mv "$RUNNING_JSON" "$QDIR/failed/${CLAIMED}.${WID}.json" 2>/dev/null || true
        cat > "$QDIR/failed/${CLAIMED}.${WID}.failure.json" << EOF
{"job_id":"$CLAIMED","worker_id":"$WID","gpu_id":$GPU,"exit_code":$RC,"failed_at":"$(date -Iseconds)"}
EOF
        echo "$(date -Iseconds) $WID: FAILED $CLAIMED exit=$RC"
    fi
done
