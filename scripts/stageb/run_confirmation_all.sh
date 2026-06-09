#!/bin/bash
set -e  # Stop on first error to avoid OOM cascade
export CUDA_VISIBLE_DEVICES=1,0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
OUT=/data/liuyu/outputs/stageb_v1_1_k5c_targeted_expansion_rc1a_ca3a97e/pipeline_v0_3_confirmation
mkdir -p $OUT
PY=/home/liuyu/.conda/envs/openvla_official_libero_20260525/bin/python
S=/data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607/scripts/run_stageb_vis_labeling.py
JID=702000
echo "[$(date +%H:%M:%S)] PIPELINE v0.3 CONFIRMATION ALL (48 jobs on GPU 1,0)"
echo "  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"

run_job() {
  local jid=$1 pair=$2 task=$3 sid=$4 ws=$5 we=$6 cond=$7 atk=$8
  echo "  job=$jid pair=$pair $task w=[$ws,$we] $cond atk=$atk"
  $PY -u $S --gpu_pair 0,1 --task $task --state-id $sid --window_start $ws --window_end $we \
    --condition $cond --pgd_steps 20 --eps_raw_pixels 6 --max_steps 400 --seed 0 \
    --env_seed $sid --attack_seed $atk --job_id $jid --pair_id $pair \
    --output_dir $OUT --image_preprocess official_rot180
  echo "  DONE job=$jid"
  sleep 1  # brief gap to let GPU memory fully release
}

# ═══ Group A: CleanRand-pass (4 windows × 4 jobs = 16) ═══
echo "=== GROUP A: CleanRand-pass ==="
for atk in 5 6; do
  run_job $((JID++)) conf_A1_milk_w70_80 milk 0 70 80 vis_pgd $atk
  run_job $((JID++)) conf_A1_milk_w70_80 milk 0 70 80 random_linf $atk
done
for atk in 5 6; do
  run_job $((JID++)) conf_A2_butter_w80_90 butter 0 80 90 vis_pgd $atk
  run_job $((JID++)) conf_A2_butter_w80_90 butter 0 80 90 random_linf $atk
done
for atk in 5 6; do
  run_job $((JID++)) conf_A3_cream_w50_60 cream_cheese 2 50 60 vis_pgd $atk
  run_job $((JID++)) conf_A3_cream_w50_60 cream_cheese 2 50 60 random_linf $atk
done
for atk in 5 6; do
  run_job $((JID++)) conf_A4_tomato_w150_160 tomato_sauce 2 150 160 vis_pgd $atk
  run_job $((JID++)) conf_A4_tomato_w150_160 tomato_sauce 2 150 160 random_linf $atk
done

# ═══ Group B: TaskOnly baseline (4 windows × 4 jobs = 16) ═══
echo "=== GROUP B: TaskOnly baseline ==="
for atk in 5 6; do
  run_job $((JID++)) conf_B1_tomato_w55_65 tomato_sauce 0 55 65 vis_pgd $atk
  run_job $((JID++)) conf_B1_tomato_w55_65 tomato_sauce 0 55 65 random_linf $atk
done
for atk in 5 6; do
  run_job $((JID++)) conf_B2_milk_w75_85 milk 0 75 85 vis_pgd $atk
  run_job $((JID++)) conf_B2_milk_w75_85 milk 0 75 85 random_linf $atk
done
for atk in 5 6; do
  run_job $((JID++)) conf_B3_cream_w85_95 cream_cheese 0 85 95 vis_pgd $atk
  run_job $((JID++)) conf_B3_cream_w85_95 cream_cheese 0 85 95 random_linf $atk
done
for atk in 5 6; do
  run_job $((JID++)) conf_B4_salad_w70_80 salad_dressing 2 70 80 vis_pgd $atk
  run_job $((JID++)) conf_B4_salad_w70_80 salad_dressing 2 70 80 random_linf $atk
done

# ═══ Group C: High-risk abstained (4 windows × 4 jobs = 16) ═══
echo "=== GROUP C: High-risk abstained ==="
for atk in 5 6; do
  run_job $((JID++)) conf_C1_milk_w80_90 milk 0 80 90 vis_pgd $atk
  run_job $((JID++)) conf_C1_milk_w80_90 milk 0 80 90 random_linf $atk
done
for atk in 5 6; do
  run_job $((JID++)) conf_C2_butter_w95_105 butter 0 95 105 vis_pgd $atk
  run_job $((JID++)) conf_C2_butter_w95_105 butter 0 95 105 random_linf $atk
done
for atk in 5 6; do
  run_job $((JID++)) conf_C3_alphabet_w60_70 alphabet_soup 0 60 70 vis_pgd $atk
  run_job $((JID++)) conf_C3_alphabet_w60_70 alphabet_soup 0 60 70 random_linf $atk
done
for atk in 5 6; do
  run_job $((JID++)) conf_C4_tomato_w115_125 tomato_sauce 2 115 125 vis_pgd $atk
  run_job $((JID++)) conf_C4_tomato_w115_125 tomato_sauce 2 115 125 random_linf $atk
done

echo "[$(date +%H:%M:%S)] PIPELINE v0.3 CONFIRMATION ALL DONE ($((JID-702000)) jobs)"
