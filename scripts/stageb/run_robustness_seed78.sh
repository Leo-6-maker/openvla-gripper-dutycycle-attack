#!/bin/bash
set -e; export CUDA_VISIBLE_DEVICES=1,0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
OUT=/data/liuyu/outputs/stageb_v1_1_k5c_targeted_expansion_rc1a_ca3a97e/pipeline_v0_3_robustness_seed78
mkdir -p $OUT
PY=/home/liuyu/.conda/envs/openvla_official_libero_20260525/bin/python
S=/data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607/scripts/run_stageb_vis_labeling.py
JID=703000
echo "[$(date +%H:%M:%S)] ROBUSTNESS SEED78 START (24 jobs)"

run_job() { local j=$1 p=$2 t=$3 s=$4 w=$5 e=$6 c=$7 a=$8
  echo "  job=$j $p $t [$w,$e] $c atk=$a"
  $PY -u $S --gpu_pair 0,1 --task $t --state-id $s --window_start $w --window_end $e --condition $c --pgd_steps 20 --eps_raw_pixels 6 --max_steps 400 --seed 0 --env_seed $s --attack_seed $a --job_id $j --pair_id $p --output_dir $OUT --image_preprocess official_rot180
  echo "  DONE job=$j"; sleep 1; }

# ═══ Phase 1A: A-group seeds 7,8 (16 jobs) ═══
echo "=== PHASE 1A: A-group atk=7,8 ==="
for atk in 7 8; do
  run_job $((JID++)) rob_A1_milk_w70_80       milk           0 70 80  vis_pgd     $atk
  run_job $((JID++)) rob_A1_milk_w70_80       milk           0 70 80  random_linf $atk
done
for atk in 7 8; do
  run_job $((JID++)) rob_A2_butter_w80_90     butter         0 80 90  vis_pgd     $atk
  run_job $((JID++)) rob_A2_butter_w80_90     butter         0 80 90  random_linf $atk
done
for atk in 7 8; do
  run_job $((JID++)) rob_A3_cream_w50_60      cream_cheese   2 50 60  vis_pgd     $atk
  run_job $((JID++)) rob_A3_cream_w50_60      cream_cheese   2 50 60  random_linf $atk
done
for atk in 7 8; do
  run_job $((JID++)) rob_A4_tomato_w150_160   tomato_sauce   2 150 160 vis_pgd     $atk
  run_job $((JID++)) rob_A4_tomato_w150_160   tomato_sauce   2 150 160 random_linf $atk
done

# ═══ Phase 1B: FP/FN seeds 7,8 (8 jobs) ═══
echo "=== PHASE 1B: FP/FN atk=7,8 ==="
for atk in 7 8; do
  run_job $((JID++)) rob_FP_tomato_w55_65    tomato_sauce   0 55 65  vis_pgd     $atk
  run_job $((JID++)) rob_FP_tomato_w55_65    tomato_sauce   0 55 65  random_linf $atk
done
for atk in 7 8; do
  run_job $((JID++)) rob_FN_salad_w70_80     salad_dressing 2 70 80  vis_pgd     $atk
  run_job $((JID++)) rob_FN_salad_w70_80     salad_dressing 2 70 80  random_linf $atk
done

echo "[$(date +%H:%M:%S)] ROBUSTNESS SEED78 DONE ($((JID-703000)) jobs)"
