#!/bin/bash
# S20b-0 smoke: ketchup seed74 full-episode clean + RAND + VIS
set +e
export MUJOCO_GL=egl; export PYOPENGL_PLATFORM=egl; unset DISPLAY
OUT=/data/liuyu/outputs/stageb_v1_1_k5c_targeted_expansion_rc1a_ca3a97e/s20b_smoke
mkdir -p $OUT/videos
PY=/home/liuyu/.conda/envs/openvla_official_libero_20260525/bin/python
S=/data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607/scripts/stageb/run_s9b_phase1_runner_attack_port.py
T=ketchup; SID=0; WS=150; WE=160; SEED=74; EPS=6; PGD=20

echo "[$(date +%H:%M:%S)] S20b-0 CLEAN"
CUDA_VISIBLE_DEVICES=1,0 $PY -u $S --gpu_pair 0,1 --task $T --state-id $SID --window_start $WS --window_end $WE --condition clean --open_duration 10 --attack_seed $SEED --job_id 957000 --pair_id ketchup_s0_w150_160_s20b_smoke_seed74 --output_dir $OUT --full_episode --max_steps 400 --save_video_dir $OUT/videos/clean_s74 || echo "FAIL_CLEAN"

echo "[$(date +%H:%M:%S)] S20b-0 RAND"
CUDA_VISIBLE_DEVICES=2,6 $PY -u $S --gpu_pair 0,1 --task $T --state-id $SID --window_start $WS --window_end $WE --condition random_linf --open_duration 10 --attack_seed $SEED --eps_raw_pixels $EPS --random_control_seed $SEED --job_id 957001 --pair_id ketchup_s0_w150_160_s20b_smoke_seed74 --output_dir $OUT --full_episode --max_steps 400 --save_video_dir $OUT/videos/rand_s74 || echo "FAIL_RAND"

echo "[$(date +%H:%M:%S)] S20b-0 VIS"
CUDA_VISIBLE_DEVICES=4,5 $PY -u $S --gpu_pair 0,1 --task $T --state-id $SID --window_start $WS --window_end $WE --condition vis_pgd --open_duration 10 --attack_seed $SEED --pgd_steps $PGD --eps_raw_pixels $EPS --job_id 957002 --pair_id ketchup_s0_w150_160_s20b_smoke_seed74 --output_dir $OUT --full_episode --max_steps 400 --save_video_dir $OUT/videos/vis_s74 || echo "FAIL_VIS"

echo "[$(date +%H:%M:%S)] S20b-0 SMOKE DONE"
