#!/bin/bash
# S20b GPU26 — ketchup seed76 + tomato_sauce seed24 × clean/RAND (5 full episodes)
set +e
export MUJOCO_GL=egl; export PYOPENGL_PLATFORM=egl; unset DISPLAY
export CUDA_VISIBLE_DEVICES=2,6
OUT=/data/liuyu/outputs/stageb_v1_1_k5c_targeted_expansion_rc1a_ca3a97e/s20b_full_episodes
VID=/data/liuyu/outputs/stageb_v1_1_k5c_targeted_expansion_rc1a_ca3a97e/s20b_videos
PY=/home/liuyu/.conda/envs/openvla_official_libero_20260525/bin/python
S=/data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607/scripts/stageb/run_s9b_phase1_runner_attack_port.py
EPS=6; PGD=20; L=10; MAX=400; JOB=958100

# ketchup seed76 × 3
T=ketchup; SID=0; WS=150; WE=160; SEED=76
echo "[$(date +%H:%M:%S)] S20b ${T} w${WS}-${WE} seed${SEED} CLEAN"
$PY -u $S --gpu_pair 0,1 --task $T --state-id $SID --window_start $WS --window_end $WE --condition clean --open_duration $L --attack_seed $SEED --job_id $((JOB++)) --pair_id ${T}_s${SID}_w${WS}_${WE}_s20b_seed${SEED} --output_dir $OUT --full_episode --max_steps $MAX --save_video_dir $VID/${T}_s${SID}_w${WS}_${WE}_seed${SEED}_clean || echo "FAIL_${T}_s${SEED}_CLEAN"

echo "[$(date +%H:%M:%S)] S20b ${T} w${WS}-${WE} seed${SEED} RAND"
$PY -u $S --gpu_pair 0,1 --task $T --state-id $SID --window_start $WS --window_end $WE --condition random_linf --open_duration $L --attack_seed $SEED --eps_raw_pixels $EPS --random_control_seed $SEED --job_id $((JOB++)) --pair_id ${T}_s${SID}_w${WS}_${WE}_s20b_seed${SEED} --output_dir $OUT --full_episode --max_steps $MAX --save_video_dir $VID/${T}_s${SID}_w${WS}_${WE}_seed${SEED}_rand || echo "FAIL_${T}_s${SEED}_RAND"

echo "[$(date +%H:%M:%S)] S20b ${T} w${WS}-${WE} seed${SEED} VIS"
$PY -u $S --gpu_pair 0,1 --task $T --state-id $SID --window_start $WS --window_end $WE --condition vis_pgd --open_duration $L --attack_seed $SEED --pgd_steps $PGD --eps_raw_pixels $EPS --job_id $((JOB++)) --pair_id ${T}_s${SID}_w${WS}_${WE}_s20b_seed${SEED} --output_dir $OUT --full_episode --max_steps $MAX --save_video_dir $VID/${T}_s${SID}_w${WS}_${WE}_seed${SEED}_vis || echo "FAIL_${T}_s${SEED}_VIS"

# tomato_sauce seed24 × clean + RAND
T=tomato_sauce; SID=0; WS=70; WE=80; SEED=24
echo "[$(date +%H:%M:%S)] S20b ${T} w${WS}-${WE} seed${SEED} CLEAN"
$PY -u $S --gpu_pair 0,1 --task $T --state-id $SID --window_start $WS --window_end $WE --condition clean --open_duration $L --attack_seed $SEED --job_id $((JOB++)) --pair_id ${T}_s${SID}_w${WS}_${WE}_s20b_seed${SEED} --output_dir $OUT --full_episode --max_steps $MAX --save_video_dir $VID/${T}_s${SID}_w${WS}_${WE}_seed${SEED}_clean || echo "FAIL_${T}_s${SEED}_CLEAN"

echo "[$(date +%H:%M:%S)] S20b ${T} w${WS}-${WE} seed${SEED} RAND"
$PY -u $S --gpu_pair 0,1 --task $T --state-id $SID --window_start $WS --window_end $WE --condition random_linf --open_duration $L --attack_seed $SEED --eps_raw_pixels $EPS --random_control_seed $SEED --job_id $((JOB++)) --pair_id ${T}_s${SID}_w${WS}_${WE}_s20b_seed${SEED} --output_dir $OUT --full_episode --max_steps $MAX --save_video_dir $VID/${T}_s${SID}_w${WS}_${WE}_seed${SEED}_rand || echo "FAIL_${T}_s${SEED}_RAND"

echo "[$(date +%H:%M:%S)] S20b GPU26 DONE"
