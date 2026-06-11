#!/bin/bash
# S18 Overnight GPU worker — reads manifest, runs jobs for one GPU group with timeout
# Usage: bash run_s18_overnight_gpu.sh <gpu_group> <cuda_visible_devices>
# Example: bash run_s18_overnight_gpu.sh gpu10 "1,0"
set +e
GPU_GROUP=$1
CUDA_DEV=$2
if [ -z "$GPU_GROUP" ] || [ -z "$CUDA_DEV" ]; then
    echo "Usage: $0 <gpu_group> <cuda_visible_devices>"
    exit 1
fi

export MUJOCO_GL=egl; export PYOPENGL_PLATFORM=egl; unset DISPLAY
export CUDA_VISIBLE_DEVICES=$CUDA_DEV

# Trap: kill all child processes on script exit to prevent orphans
cleanup_children() {
    echo "[$(date +%H:%M:%S)] [$GPU_GROUP] cleanup: killing child processes"
    jobs -p | xargs -r kill 2>/dev/null
    # Also kill any remaining python runners for this GPU group
    pgrep -f "run_s9b_phase1_runner_attack_port.*s18_overnight_census" | xargs -r kill 2>/dev/null
}
trap cleanup_children EXIT

ROOT=/data/liuyu/outputs/stageb_v1_1_k5c_targeted_expansion_rc1a_ca3a97e
OUT=$ROOT/s18_overnight_census
MANIFEST=$ROOT/../s18_jobs_manifest.csv
LOCK_DIR=$OUT/locks; DONE_DIR=$OUT/done; FAIL_DIR=$OUT/failed; TIMEOUT_DIR=$OUT/timeout
mkdir -p $OUT $LOCK_DIR $DONE_DIR $FAIL_DIR $TIMEOUT_DIR

PY=/home/liuyu/.conda/envs/openvla_official_libero_20260525/bin/python
S=/data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607/scripts/stageb/run_s9b_phase1_runner_attack_port.py
JOB_TIMEOUT=2700  # 45 minutes

TOTAL=0; DONE_C=0; FAIL_C=0; SKIP_C=0; TO_C=0

while IFS=, read -r gpu task sid ws we cond aseed jid juid puid rseed eps pgd od; do
    [ "$gpu" = "gpu_group" ] && continue
    [ "$gpu" != "$GPU_GROUP" ] && continue

    TOTAL=$((TOTAL+1))
    RETRYFILE=$TIMEOUT_DIR/${juid}.retried

    if [ -f "$DONEFILE" ]; then SKIP_C=$((SKIP_C+1)); continue; fi
    if [ -f "$FAILFILE" ]; then SKIP_C=$((SKIP_C+1)); continue; fi
    if [ -f "$TOFILE" ] && [ -f "$RETRYFILE" ]; then
        SKIP_C=$((SKIP_C+1)); echo "[$(date +%H:%M:%S)] SKIP $juid (timed out twice)"; continue
    fi
    if [ -f "$TOFILE" ]; then
        echo "[$(date +%H:%M:%S)] RETRY_TIMEOUT $juid (attempt 2)"
    fi

    echo $$ > "$LOCKFILE"
    echo "[$(date +%H:%M:%S)] [$GPU_GROUP] START $juid"

    timeout $JOB_TIMEOUT $PY -u $S \
        --gpu_pair 0,1 --task $task --state-id $sid \
        --window_start $ws --window_end $we \
        --condition $cond --open_duration $od \
        --attack_seed $aseed --pgd_steps $pgd --eps_raw_pixels $eps \
        --job_id $jid --pair_id ${juid%_${cond}_seed*} \
        --output_dir $OUT

    RC=$?
    rm -f "$LOCKFILE"

    if [ $RC -eq 0 ]; then
        EXPECTED_SUMMARY=$(ls $OUT/summary_*_job${jid}.json 2>/dev/null | head -1)
        if [ -f "$EXPECTED_SUMMARY" ]; then
            touch "$DONEFILE"; DONE_C=$((DONE_C+1)); rm -f "$TOFILE" "$RETRYFILE"
            echo "[$(date +%H:%M:%S)] [$GPU_GROUP] DONE $juid"
        else
            touch "$FAILFILE"; FAIL_C=$((FAIL_C+1))
            echo "[$(date +%H:%M:%S)] [$GPU_GROUP] FAIL $juid (summary missing)"
        fi
    elif [ $RC -eq 124 ]; then
        if [ -f "$TOFILE" ]; then
            touch "$RETRYFILE"; TO_C=$((TO_C+1))
            echo "[$(date +%H:%M:%S)] [$GPU_GROUP] TIMEOUT_FINAL $juid"
        else
            touch "$TOFILE"; TO_C=$((TO_C+1))
            echo "[$(date +%H:%M:%S)] [$GPU_GROUP] TIMEOUT $juid (will retry once)"
        fi
    else
        touch "$FAILFILE"; FAIL_C=$((FAIL_C+1)); rm -f "$TOFILE" "$RETRYFILE"
        echo "[$(date +%H:%M:%S)] [$GPU_GROUP] FAIL $juid (rc=$RC)"
    fi
done < "$MANIFEST"

echo "[$(date +%H:%M:%S)] [$GPU_GROUP] SUMMARY: total=$TOTAL done=$DONE_C failed=$FAIL_C timeout=$TO_C skipped=$SKIP_C"
