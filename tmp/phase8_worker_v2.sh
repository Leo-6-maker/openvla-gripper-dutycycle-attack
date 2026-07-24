#!/bin/bash
# Phase 8 Cross-Suite Worker V2 — Suite-aware, fixed RANDOM dispatch
# Usage: bash phase8_worker_v2.sh <GPU_ID> <WORKER_ID>
set -e
GPU=$1; WID=${2:-0}
BASE=/mnt/sdc/dty_user/openvla_attack
QDIR=$BASE/evidence/phase8_cross_suite_v1/queue
RUNS=$BASE/evidence/phase8_cross_suite_v1/runs

export MUJOCO_GL=egl HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export OPENVLA_DTYPE=bfloat16 OPENVLA_ATTN_IMPLEMENTATION=eager TF_FORCE_GPU_ALLOW_GROWTH=true
export HOME=$BASE/sandbox_home/phase8_$WID TMPDIR=$BASE/tmp/phase8/$WID
mkdir -p $HOME $TMPDIR $RUNS

PY=$BASE/envs/openvla-official-a800/bin/python3
BRIDGE=$BASE/scripts/stageb/run_v2_vis_sc5_mlp_bridge_telemetry_v2.py
DETECTOR=$BASE/outputs/sc5_v2_seed42/sc5_mlp_v2.pt
COMMIT=$(git -C $BASE rev-parse HEAD)

declare -A MODEL_PATH
declare -A UNNORM_KEY_MAP
declare -A SUITE_NAME_MAP

MODEL_PATH[libero_spatial]="$BASE/models/libero-spatial/spatial_c8f03f4_20260620"
MODEL_PATH[libero_goal]="$BASE/models/libero-goal"
MODEL_PATH[libero_10]="$BASE/models/libero-10"

UNNORM_KEY_MAP[libero_spatial]=libero_spatial
UNNORM_KEY_MAP[libero_goal]=libero_goal
UNNORM_KEY_MAP[libero_10]=libero_10

SUITE_NAME_MAP[libero_spatial]=libero_spatial
SUITE_NAME_MAP[libero_goal]=libero_goal
SUITE_NAME_MAP[libero_10]=libero_10

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

    local MODEL=${MODEL_PATH[$SUITE]}
    local UK=${UNNORM_KEY_MAP[$SUITE]}
    local SN=${SUITE_NAME_MAP[$SUITE]}
    [ -z "$MODEL" ] && { echo "Unknown suite: $SUITE"; return 1; }

    local SID=0
    [ "$SEED" = "123" ] && SID=1
    [ "$SEED" = "456" ] && SID=2

    local ARGS="--task_idx $TASK --state_id $SID --eval_seed 0 --seed_id $SEED"
    ARGS="$ARGS --output_dir $OUT --render_gpu $GPU --mlp_path $DETECTOR"
    ARGS="$ARGS --libero_preprocess_backend upstream_tf_jpeg --anchor 0"
    ARGS="$ARGS --source_commit $COMMIT --save_video --video_fps 10 --frame_stride 2"
    ARGS="$ARGS --unnorm_key $UK --suite_name $SN"

    # FIXED dispatch order: RANDOM checked before attack_enabled
    if [ "$COND" = "RANDOM" ]; then
        ARGS="$ARGS --condition RAND_T10 --attack_objective ''"
    elif [ "$ATK" = "True" ] || [ "$ATK" = "true" ]; then
        ARGS="$ARGS --condition TRUE_T10 --attack_objective $OBJ"
        [ "$AL" = "True" ] || [ "$AL" = "true" ] && ARGS="$ARGS --arm_lock"
    else
        ARGS="$ARGS --condition CLEAN --attack_objective ''"
    fi

    mkdir -p "$OUT"
    echo "$(date -Iseconds) GPU$GPU W$WID: $JID $COND $SUITE" | tee -a $QDIR/logs/worker_${WID}.log

    env CUDA_VISIBLE_DEVICES=$GPU OPENVLA_MODEL_PATH=$MODEL $PY -u $BRIDGE $ARGS \
        > "$OUT/stdout.log" 2> "$OUT/stderr.log"
    local EC=$?
    echo "$(date -Iseconds) GPU$GPU W$WID: $JID exit=$EC" | tee -a $QDIR/logs/worker_${WID}.log
    [ $EC -eq 0 ] && touch "$OUT/.done"
    return $EC
}

# Setup queue
mkdir -p $QDIR/pending $QDIR/running $QDIR/done $QDIR/failed $QDIR/locks $QDIR/heartbeats $QDIR/logs

MANIFEST=$BASE/evidence/phase8_cross_suite_v1/manifests/ALL_630_JOBS.jsonl
if [ ! -f "$QDIR/.seeded" ]; then
    while IFS= read -r line; do
        JID=$(echo "$line" | python3 -c "import sys,json; print(json.load(sys.stdin)['job_id'])")
        echo "$line" > "$QDIR/pending/$JID.json"
    done < "$MANIFEST"
    touch "$QDIR/.seeded"
    echo "Seeded $(ls $QDIR/pending/*.json 2>/dev/null | wc -l) jobs"
fi

while true; do
    CLAIMED=0
    for JOBF in $(ls $QDIR/pending/*.json 2>/dev/null | sort -R); do
        JID=$(basename "$JOBF" .json)
        if mkdir "$QDIR/locks/$JID.lock" 2>/dev/null; then
            echo "{\"worker_id\":\"$WID\",\"gpu_id\":$GPU,\"pid\":$$,\"hostname\":\"$(hostname)\",\"claim_time\":\"$(date -Iseconds)\",\"git_commit\":\"$COMMIT\"}" > "$QDIR/locks/$JID.lock/claim.json"
            mv "$JOBF" "$QDIR/running/${JID}.${WID}.json"
            JOB_JSON=$(cat "$QDIR/running/${JID}.${WID}.json")
            run_job "$JID" "$JOB_JSON" && RC=0 || RC=$?
            rm -rf "$QDIR/locks/$JID.lock"
            if [ $RC -eq 0 ]; then
                mv "$QDIR/running/${JID}.${WID}.json" "$QDIR/done/${JID}.json" 2>/dev/null || true
            else
                mv "$QDIR/running/${JID}.${WID}.json" "$QDIR/failed/${JID}.json" 2>/dev/null || true
            fi
            CLAIMED=1; break
        fi
    done
    [ $CLAIMED -eq 0 ] && { echo "$(date -Iseconds) W$WID: queue empty"; break; }
done
echo "Worker $WID done."
