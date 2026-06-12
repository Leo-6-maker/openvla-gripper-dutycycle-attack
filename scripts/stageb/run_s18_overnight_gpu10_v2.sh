#!/bin/bash
# S18 Overnight GPU10 — reads manifest, runs 40 jobs with timeout
set +e
export MUJOCO_GL=egl; export PYOPENGL_PLATFORM=egl; unset DISPLAY
export CUDA_VISIBLE_DEVICES=1,0

ROOT=/data/liuyu/outputs/stageb_v1_1_k5c_targeted_expansion_rc1a_ca3a97e
OUT=$ROOT/s18_overnight_census
MANIFEST=$ROOT/../s18_jobs_manifest.csv
LOCK_DIR=$OUT/locks; DONE_DIR=$OUT/done; FAIL_DIR=$OUT/failed; TIMEOUT_DIR=$OUT/timeout
mkdir -p $OUT $LOCK_DIR $DONE_DIR $FAIL_DIR $TIMEOUT_DIR

PY=/home/liuyu/.conda/envs/openvla_official_libero_20260525/bin/python
S=/data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607/scripts/stageb/run_s9b_phase1_runner_attack_port.py
JOB_TIMEOUT=2700  # 45 minutes per job

run_one() {
    local TASK=$1; local SID=$2; local WS=$3; local WE=$4
    local COND=$5; local ASEED=$6; local JID=$7; local JUID=$8
    local EPS=$9; local PGD=${10}; local L=${11}

    local LOCKFILE=$LOCK_DIR/${JUID}.lock
    local DONEFILE=$DONE_DIR/${JUID}.done
    local FAILFILE=$FAIL_DIR/${JUID}.failed
    local TIMEOUTFILE=$TIMEOUT_DIR/${JUID}.timeout

    # Skip if already done
    [ -f "$DONEFILE" ] && { echo "[$(date +%H:%M:%S)] SKIP $JUID (already done)"; return 0; }
    [ -f "$FAILFILE" ] && { echo "[$(date +%H:%M:%S)] SKIP $JUID (previously failed)"; return 0; }
    [ -f "$TIMEOUTFILE" ] && { echo "[$(date +%H:%M:%S)] SKIP $JUID (previously timeout)"; return 0; }

    # Acquire lock
    echo $$ > "$LOCKFILE"

    echo "[$(date +%H:%M:%S)] START $JUID"
    timeout $JOB_TIMEOUT $PY -u $S \
        --gpu_pair 0,1 --task $TASK --state-id $SID \
        --window_start $WS --window_end $WE \
        --condition $COND --open_duration $L \
        --attack_seed $ASEED --pgd_steps $PGD --eps_raw_pixels $EPS \
        --job_id $JID --pair_id ${JUID%_${COND}_seed*} \
        --output_dir $OUT

    local RC=$?
    rm -f "$LOCKFILE"

    if [ $RC -eq 0 ]; then
        touch "$DONEFILE"
        echo "[$(date +%H:%M:%S)] DONE $JUID"
    elif [ $RC -eq 124 ]; then
        touch "$TIMEOUTFILE"
        echo "[$(date +%H:%M:%S)] TIMEOUT $JUID"
    else
        touch "$FAILFILE"
        echo "[$(date +%H:%M:%S)] FAIL $JUID (rc=$RC)"
    fi
}

# Read manifest and run GPU10 jobs
while IFS=, read -r gpu task sid ws we cond aseed jid juid puid rseed eps pgd od; do
    # Skip header
    [ "$gpu" = "gpu_group" ] && continue
    [ "$gpu" != "gpu10" ] && continue
    run_one "$task" "$sid" "$ws" "$we" "$cond" "$aseed" "$jid" "$juid" "$eps" "$pgd" "$od"
done < "$MANIFEST"

echo "[$(date +%H:%M:%S)] S18 GPU10 ALL DONE"
