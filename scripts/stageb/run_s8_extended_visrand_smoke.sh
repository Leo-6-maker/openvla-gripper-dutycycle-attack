#!/bin/bash
# S8 Phase 2 extended VIS/RAND smoke — milk only, 8 jobs, GPU 1,0
# Runner: run_extended_visrand_physical.py (v2, S6-attack-aligned)
set +e
export CUDA_VISIBLE_DEVICES=1,0
OUT=/data/liuyu/outputs/stageb_v1_1_k5c_targeted_expansion_rc1a_ca3a97e/s8_extended_visrand_diagnostic/smoke
mkdir -p $OUT
PY=/home/liuyu/.conda/envs/openvla_official_libero_20260525/bin/python
S=/data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607/scripts/stageb/run_extended_visrand_physical.py

echo "[$(date +%H:%M:%S)] S8 VISRAND SMOKE START (milk only, 8 jobs, v2 S6-aligned)"

# ── milk short (original ws=70, we=80) ──
# ORACLE ref: milk_s0_w70_80_L10 pos_area = 0.261452
echo "  VIS milk short atk=9"
$PY -u $S --gpu_pair 0,1 --task milk --state-id 0 --window_start 70 --window_end 80 --condition vis_pgd --pgd_steps 20 --eps_raw_pixels 6 --attack_seed 9 --env_seed 0 --job_id 950000 --pair_id milk_s0_w70_80_short__atk9 --original_window_start 70 --original_window_end 80 --oracle_ref_L10_pos_area 0.261452 --length_mode short --output_dir $OUT || echo "FAIL_VIS_milk_short_atk9"
echo "  RAND milk short atk=9"
$PY -u $S --gpu_pair 0,1 --task milk --state-id 0 --window_start 70 --window_end 80 --condition random_linf --eps_raw_pixels 6 --attack_seed 9 --env_seed 0 --job_id 950001 --pair_id milk_s0_w70_80_short__atk9 --original_window_start 70 --original_window_end 80 --oracle_ref_L10_pos_area 0.261452 --length_mode short --output_dir $OUT || echo "FAIL_RAND_milk_short_atk9"
echo "  VIS milk short atk=10"
$PY -u $S --gpu_pair 0,1 --task milk --state-id 0 --window_start 70 --window_end 80 --condition vis_pgd --pgd_steps 20 --eps_raw_pixels 6 --attack_seed 10 --env_seed 0 --job_id 950002 --pair_id milk_s0_w70_80_short__atk10 --original_window_start 70 --original_window_end 80 --oracle_ref_L10_pos_area 0.261452 --length_mode short --output_dir $OUT || echo "FAIL_VIS_milk_short_atk10"
echo "  RAND milk short atk=10"
$PY -u $S --gpu_pair 0,1 --task milk --state-id 0 --window_start 70 --window_end 80 --condition random_linf --eps_raw_pixels 6 --attack_seed 10 --env_seed 0 --job_id 950003 --pair_id milk_s0_w70_80_short__atk10 --original_window_start 70 --original_window_end 80 --oracle_ref_L10_pos_area 0.261452 --length_mode short --output_dir $OUT || echo "FAIL_RAND_milk_short_atk10"

# ── milk extended20 (ws-10=60, we+10=90) ──
# ORACLE ref still: milk_s0_w70_80_L10 pos_area = 0.261452 (original window!)
echo "  VIS milk ext20 atk=9"
$PY -u $S --gpu_pair 0,1 --task milk --state-id 0 --window_start 60 --window_end 90 --condition vis_pgd --pgd_steps 20 --eps_raw_pixels 6 --attack_seed 9 --env_seed 0 --job_id 950004 --pair_id milk_s0_w60_90_extended20__atk9 --original_window_start 70 --original_window_end 80 --oracle_ref_L10_pos_area 0.261452 --length_mode extended20 --output_dir $OUT || echo "FAIL_VIS_milk_ext20_atk9"
echo "  RAND milk ext20 atk=9"
$PY -u $S --gpu_pair 0,1 --task milk --state-id 0 --window_start 60 --window_end 90 --condition random_linf --eps_raw_pixels 6 --attack_seed 9 --env_seed 0 --job_id 950005 --pair_id milk_s0_w60_90_extended20__atk9 --original_window_start 70 --original_window_end 80 --oracle_ref_L10_pos_area 0.261452 --length_mode extended20 --output_dir $OUT || echo "FAIL_RAND_milk_ext20_atk9"
echo "  VIS milk ext20 atk=10"
$PY -u $S --gpu_pair 0,1 --task milk --state-id 0 --window_start 60 --window_end 90 --condition vis_pgd --pgd_steps 20 --eps_raw_pixels 6 --attack_seed 10 --env_seed 0 --job_id 950006 --pair_id milk_s0_w60_90_extended20__atk10 --original_window_start 70 --original_window_end 80 --oracle_ref_L10_pos_area 0.261452 --length_mode extended20 --output_dir $OUT || echo "FAIL_VIS_milk_ext20_atk10"
echo "  RAND milk ext20 atk=10"
$PY -u $S --gpu_pair 0,1 --task milk --state-id 0 --window_start 60 --window_end 90 --condition random_linf --eps_raw_pixels 6 --attack_seed 10 --env_seed 0 --job_id 950007 --pair_id milk_s0_w60_90_extended20__atk10 --original_window_start 70 --original_window_end 80 --oracle_ref_L10_pos_area 0.261452 --length_mode extended20 --output_dir $OUT || echo "FAIL_RAND_milk_ext20_atk10"

echo "[$(date +%H:%M:%S)] S8 VISRAND SMOKE DONE"
