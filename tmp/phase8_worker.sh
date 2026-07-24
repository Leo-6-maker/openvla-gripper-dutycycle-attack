#!/bin/bash
# Phase 8 Cross-Suite Worker — Atomic mkdir dispatch
# Usage: bash phase8_worker.sh <GPU_ID> <WORKER_ID>
set -e
GPU=$1; WID=${2:-0}
BASE=/mnt/sdc/dty_user/openvla_attack
QDIR=$BASE/evidence/phase8_cross_suite_v1/queue
RUNS=$BASE/evidence/phase8_cross_suite_v1/runs

export MUJOCO_GL=egl HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export OPENVLA_DTYPE=bfloat16 OPENVLA_ATTN_IMPLEMENTATION=eager TF_FORCE_GPU_ALLOW_GROWTH=true
export HOME=$BASE/sandbox_home/$WID TMPDIR=$BASE/tmp/phase8/$WID
mkdir -p $HOME $TMPDIR $RUNS

PY=$BASE/envs/openvla-official-a800/bin/python3
BRIDGE=$BASE/scripts/stageb/run_v2_vis_sc5_mlp_bridge_telemetry_v2.py
DETECTOR=$BASE/outputs/sc5_v2_seed42/sc5_mlp_v2.pt
COMMIT=$(git -C $BASE rev-parse HEAD)

MODEL=$BASE/models/libero-spatial/spatial_c8f03f4_20260620
UNNORM_KEY=libero_spatial
SUITE_NAME=libero_spatial

run_job() {
    local JOB_ID=$1 JOB_JSON=$2 OUT=$RUNS/$JOB_ID
    [ -f "$OUT/.done" ] && return 0

    local TASK=$(echo "$JOB_JSON" | python3 -c "import sys,json; print(json.load(sys.stdin)['task_idx'])")
    local SEED=$(echo "$JOB_JSON" | python3 -c "import sys,json; print(json.load(sys.stdin)['evaluation_seed'])")
    local COND=$(echo "$JOB_JSON" | python3 -c "import sys,json; print(json.load(sys.stdin)['condition'])")
    local AL=$(echo "$JOB_JSON" | python3 -c "import sys,json; print(json.load(sys.stdin)['arm_lock'])")
    local OBJ=$(echo "$JOB_JSON" | python3 -c "import sys,json; print(json.load(sys.stdin)['objective_id'])")
    local ATK=$(echo "$JOB_JSON" | python3 -c "import sys,json; print(json.load(sys.stdin)['attack_enabled'])")

    # Map evaluation_seed -> state_id (0,1,2)
    local SID=0
    [ "$SEED" = "123" ] && SID=1
    [ "$SEED" = "456" ] && SID=2

    # Build args
    local ARGS="--task_idx $TASK --state_id $SID --eval_seed 0 --seed_id $SEED"
    ARGS="$ARGS --output_dir $OUT --render_gpu $GPU --mlp_path $DETECTOR"
    ARGS="$ARGS --libero_preprocess_backend upstream_tf_jpeg --anchor 0"
    ARGS="$ARGS --source_commit $COMMIT --save_video --video_fps 10 --frame_stride 2"
    ARGS="$ARGS --unnorm_key $UNNORM_KEY --suite_name $SUITE_NAME"

    if [ "$ATK" = "True" ] || [ "$ATK" = "true" ]; then
        ARGS="$ARGS --condition TRUE_T10 --attack_objective $OBJ"
        [ "$AL" = "True" ] || [ "$AL" = "true" ] && ARGS="$ARGS --arm_lock"
    elif [ "$COND" = "RANDOM" ]; then
        ARGS="$ARGS --condition RAND_T10 --attack_objective ''"
    else
        ARGS="$ARGS --condition CLEAN --attack_objective ''"
    fi

    mkdir -p "$OUT"
    echo "$(date -Iseconds) GPU$GPU W$WID: $JOB_ID $COND" | tee -a $QDIR/logs/worker_${WID}.log

    env CUDA_VISIBLE_DEVICES=$GPU OPENVLA_MODEL_PATH=$MODEL $PY -u $BRIDGE $ARGS \
        > "$OUT/stdout.log" 2> "$OUT/stderr.log"
    local EC=$?

    echo "$(date -Iseconds) GPU$GPU W$WID: $JOB_ID exit=$EC" | tee -a $QDIR/logs/worker_${WID}.log
    [ $EC -eq 0 ] && touch "$OUT/.done"
    return $EC
}

# Setup queue
mkdir -p $QDIR/pending $QDIR/running $QDIR/done $QDIR/failed $QDIR/locks $QDIR/heartbeats $QDIR/logs

MANIFEST=$BASE/evidence/phase8_cross_suite_v1/manifests/ALL_SPATIAL_210_JOBS.jsonl
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
            echo "{\"worker_id\":\"$WID\",\"gpu_id\":$GPU,\"pid\":$$,\"hostname\":\"$(hostname)\",\"claim_time\":\"$(date -Iseconds)\"}" > "$QDIR/locks/$JID.lock/claim.json"
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
