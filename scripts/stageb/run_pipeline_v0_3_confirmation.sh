#!/bin/bash
# Pipeline v0.3 Fresh Top-K Confirmation
# 12 windows × 2 fresh seeds (5,6) × VIS/RAND = 48 jobs
# GPU pairs: worker_10 (1,0) and worker_45 (4,5)
# GPU 2,6 health check passed but reserved pending full validation

set +e
export CUDA_VISIBLE_DEVICES=1,0
OUT=/data/liuyu/outputs/stageb_v1_1_k5c_targeted_expansion_rc1a_ca3a97e/pipeline_v0_3_confirmation
mkdir -p $OUT
PY=/home/liuyu/.conda/envs/openvla_official_libero_20260525/bin/python
S=/data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607/scripts/run_stageb_vis_labeling.py
JID=700000
echo "[$(date +%H:%M:%S)] PIPELINE v0.3 CONFIRMATION START (48 jobs)"

# ═══════════════════════════════════════════════════════
# Group A: CleanRand-pass selected (4 windows — detector says "safe")
# ═══════════════════════════════════════════════════════
GRP=A
echo "=== GROUP A: CleanRand-pass selected ==="

# A1: milk[70,80] env=0 — GOLD cmd anchor
for atk in 5 6; do
  echo "  VIS A1 milk[70,80] atk=$atk"
  $PY -u $S --gpu_pair 0,1 --task milk --state-id 0 --window_start 70 --window_end 80 --condition vis_pgd --pgd_steps 20 --eps_raw_pixels 6 --max_steps 400 --seed 0 --env_seed 0 --attack_seed $atk --job_id $((JID++)) --pair_id conf_A1_milk_w70_80 --output_dir $OUT --image_preprocess official_rot180 || echo "VIS_FAIL conf_A1 atk=$atk"
  echo "  RAND A1 milk[70,80] atk=$atk"
  $PY -u $S --gpu_pair 0,1 --task milk --state-id 0 --window_start 70 --window_end 80 --condition random_linf --eps_raw_pixels 6 --max_steps 400 --seed 0 --env_seed 0 --attack_seed $atk --job_id $((JID++)) --pair_id conf_A1_milk_w70_80 --output_dir $OUT --image_preprocess official_rot180 || echo "RAND_FAIL conf_A1 atk=$atk"
done

# A2: butter[80,90] env=0 — K5c new cmd
for atk in 5 6; do
  echo "  VIS A2 butter[80,90] atk=$atk"
  $PY -u $S --gpu_pair 0,1 --task butter --state-id 0 --window_start 80 --window_end 90 --condition vis_pgd --pgd_steps 20 --eps_raw_pixels 6 --max_steps 400 --seed 0 --env_seed 0 --attack_seed $atk --job_id $((JID++)) --pair_id conf_A2_butter_w80_90 --output_dir $OUT --image_preprocess official_rot180 || echo "VIS_FAIL conf_A2 atk=$atk"
  echo "  RAND A2 butter[80,90] atk=$atk"
  $PY -u $S --gpu_pair 0,1 --task butter --state-id 0 --window_start 80 --window_end 90 --condition random_linf --eps_raw_pixels 6 --max_steps 400 --seed 0 --env_seed 0 --attack_seed $atk --job_id $((JID++)) --pair_id conf_A2_butter_w80_90 --output_dir $OUT --image_preprocess official_rot180 || echo "RAND_FAIL conf_A2 atk=$atk"
done

# A3: cream_cheese[50,60] env=2 — cmd, cream
for atk in 5 6; do
  echo "  VIS A3 cream[50,60] atk=$atk"
  $PY -u $S --gpu_pair 0,1 --task cream_cheese --state-id 2 --window_start 50 --window_end 60 --condition vis_pgd --pgd_steps 20 --eps_raw_pixels 6 --max_steps 400 --seed 0 --env_seed 2 --attack_seed $atk --job_id $((JID++)) --pair_id conf_A3_cream_w50_60 --output_dir $OUT --image_preprocess official_rot180 || echo "VIS_FAIL conf_A3 atk=$atk"
  echo "  RAND A3 cream[50,60] atk=$atk"
  $PY -u $S --gpu_pair 0,1 --task cream_cheese --state-id 2 --window_start 50 --window_end 60 --condition random_linf --eps_raw_pixels 6 --max_steps 400 --seed 0 --env_seed 2 --attack_seed $atk --job_id $((JID++)) --pair_id conf_A3_cream_w50_60 --output_dir $OUT --image_preprocess official_rot180 || echo "RAND_FAIL conf_A3 atk=$atk"
done

# A4: tomato[150,160] env=2 — cmd tomato
for atk in 5 6; do
  echo "  VIS A4 tomato[150,160] atk=$atk"
  $PY -u $S --gpu_pair 0,1 --task tomato_sauce --state-id 2 --window_start 150 --window_end 160 --condition vis_pgd --pgd_steps 20 --eps_raw_pixels 6 --max_steps 400 --seed 0 --env_seed 2 --attack_seed $atk --job_id $((JID++)) --pair_id conf_A4_tomato_w150_160 --output_dir $OUT --image_preprocess official_rot180 || echo "VIS_FAIL conf_A4 atk=$atk"
  echo "  RAND A4 tomato[150,160] atk=$atk"
  $PY -u $S --gpu_pair 0,1 --task tomato_sauce --state-id 2 --window_start 150 --window_end 160 --condition random_linf --eps_raw_pixels 6 --max_steps 400 --seed 0 --env_seed 2 --attack_seed $atk --job_id $((JID++)) --pair_id conf_A4_tomato_w150_160 --output_dir $OUT --image_preprocess official_rot180 || echo "RAND_FAIL conf_A4 atk=$atk"
done

# ═══════════════════════════════════════════════════════
# Group B: TaskOnly baseline (4 windows — naive task-prior)
# ═══════════════════════════════════════════════════════
GRP=B
echo "=== GROUP B: TaskOnly baseline ==="

# B1: tomato[55,65] env=0 — GOLD cmd+phys, but detector FALSE POSITIVE (oof_rand=0.97!)
for atk in 5 6; do
  echo "  VIS B1 tomato[55,65] atk=$atk"
  $PY -u $S --gpu_pair 0,1 --task tomato_sauce --state-id 0 --window_start 55 --window_end 65 --condition vis_pgd --pgd_steps 20 --eps_raw_pixels 6 --max_steps 400 --seed 0 --env_seed 0 --attack_seed $atk --job_id $((JID++)) --pair_id conf_B1_tomato_w55_65 --output_dir $OUT --image_preprocess official_rot180 || echo "VIS_FAIL conf_B1 atk=$atk"
  echo "  RAND B1 tomato[55,65] atk=$atk"
  $PY -u $S --gpu_pair 0,1 --task tomato_sauce --state-id 0 --window_start 55 --window_end 65 --condition random_linf --eps_raw_pixels 6 --max_steps 400 --seed 0 --env_seed 0 --attack_seed $atk --job_id $((JID++)) --pair_id conf_B1_tomato_w55_65 --output_dir $OUT --image_preprocess official_rot180 || echo "RAND_FAIL conf_B1 atk=$atk"
done

# B2: milk[75,85] env=0 — rand, detector correctly flags (oof_rand=0.75)
for atk in 5 6; do
  echo "  VIS B2 milk[75,85] atk=$atk"
  $PY -u $S --gpu_pair 0,1 --task milk --state-id 0 --window_start 75 --window_end 85 --condition vis_pgd --pgd_steps 20 --eps_raw_pixels 6 --max_steps 400 --seed 0 --env_seed 0 --attack_seed $atk --job_id $((JID++)) --pair_id conf_B2_milk_w75_85 --output_dir $OUT --image_preprocess official_rot180 || echo "VIS_FAIL conf_B2 atk=$atk"
  echo "  RAND B2 milk[75,85] atk=$atk"
  $PY -u $S --gpu_pair 0,1 --task milk --state-id 0 --window_start 75 --window_end 85 --condition random_linf --eps_raw_pixels 6 --max_steps 400 --seed 0 --env_seed 0 --attack_seed $atk --job_id $((JID++)) --pair_id conf_B2_milk_w75_85 --output_dir $OUT --image_preprocess official_rot180 || echo "RAND_FAIL conf_B2 atk=$atk"
done

# B3: cream_cheese[85,95] env=0 — borderline cmd, detector flags (oof_rand=0.74)
for atk in 5 6; do
  echo "  VIS B3 cream[85,95] atk=$atk"
  $PY -u $S --gpu_pair 0,1 --task cream_cheese --state-id 0 --window_start 85 --window_end 95 --condition vis_pgd --pgd_steps 20 --eps_raw_pixels 6 --max_steps 400 --seed 0 --env_seed 0 --attack_seed $atk --job_id $((JID++)) --pair_id conf_B3_cream_w85_95 --output_dir $OUT --image_preprocess official_rot180 || echo "VIS_FAIL conf_B3 atk=$atk"
  echo "  RAND B3 cream[85,95] atk=$atk"
  $PY -u $S --gpu_pair 0,1 --task cream_cheese --state-id 0 --window_start 85 --window_end 95 --condition random_linf --eps_raw_pixels 6 --max_steps 400 --seed 0 --env_seed 0 --attack_seed $atk --job_id $((JID++)) --pair_id conf_B3_cream_w85_95 --output_dir $OUT --image_preprocess official_rot180 || echo "RAND_FAIL conf_B3 atk=$atk"
done

# B4: salad[70,80] env=2 — rand BUT detector FALSE NEGATIVE (oof_rand=0.09!)
for atk in 5 6; do
  echo "  VIS B4 salad[70,80] atk=$atk"
  $PY -u $S --gpu_pair 0,1 --task salad_dressing --state-id 2 --window_start 70 --window_end 80 --condition vis_pgd --pgd_steps 20 --eps_raw_pixels 6 --max_steps 400 --seed 0 --env_seed 2 --attack_seed $atk --job_id $((JID++)) --pair_id conf_B4_salad_w70_80 --output_dir $OUT --image_preprocess official_rot180 || echo "VIS_FAIL conf_B4 atk=$atk"
  echo "  RAND B4 salad[70,80] atk=$atk"
  $PY -u $S --gpu_pair 0,1 --task salad_dressing --state-id 2 --window_start 70 --window_end 80 --condition random_linf --eps_raw_pixels 6 --max_steps 400 --seed 0 --env_seed 2 --attack_seed $atk --job_id $((JID++)) --pair_id conf_B4_salad_w70_80 --output_dir $OUT --image_preprocess official_rot180 || echo "RAND_FAIL conf_B4 atk=$atk"
done

# ═══════════════════════════════════════════════════════
# Group C: High-risk abstained (4 windows — detector says "don't attack")
# ═══════════════════════════════════════════════════════
GRP=C
echo "=== GROUP C: High-risk abstained ==="

# C1: milk[80,90] env=0 — rand, oof_rand=0.98
for atk in 5 6; do
  echo "  VIS C1 milk[80,90] atk=$atk"
  $PY -u $S --gpu_pair 0,1 --task milk --state-id 0 --window_start 80 --window_end 90 --condition vis_pgd --pgd_steps 20 --eps_raw_pixels 6 --max_steps 400 --seed 0 --env_seed 0 --attack_seed $atk --job_id $((JID++)) --pair_id conf_C1_milk_w80_90 --output_dir $OUT --image_preprocess official_rot180 || echo "VIS_FAIL conf_C1 atk=$atk"
  echo "  RAND C1 milk[80,90] atk=$atk"
  $PY -u $S --gpu_pair 0,1 --task milk --state-id 0 --window_start 80 --window_end 90 --condition random_linf --eps_raw_pixels 6 --max_steps 400 --seed 0 --env_seed 0 --attack_seed $atk --job_id $((JID++)) --pair_id conf_C1_milk_w80_90 --output_dir $OUT --image_preprocess official_rot180 || echo "RAND_FAIL conf_C1 atk=$atk"
done

# C2: butter[95,105] env=0 — rand, oof_rand=0.98
for atk in 5 6; do
  echo "  VIS C2 butter[95,105] atk=$atk"
  $PY -u $S --gpu_pair 0,1 --task butter --state-id 0 --window_start 95 --window_end 105 --condition vis_pgd --pgd_steps 20 --eps_raw_pixels 6 --max_steps 400 --seed 0 --env_seed 0 --attack_seed $atk --job_id $((JID++)) --pair_id conf_C2_butter_w95_105 --output_dir $OUT --image_preprocess official_rot180 || echo "VIS_FAIL conf_C2 atk=$atk"
  echo "  RAND C2 butter[95,105] atk=$atk"
  $PY -u $S --gpu_pair 0,1 --task butter --state-id 0 --window_start 95 --window_end 105 --condition random_linf --eps_raw_pixels 6 --max_steps 400 --seed 0 --env_seed 0 --attack_seed $atk --job_id $((JID++)) --pair_id conf_C2_butter_w95_105 --output_dir $OUT --image_preprocess official_rot180 || echo "RAND_FAIL conf_C2 atk=$atk"
done

# C3: alphabet_soup[60,70] env=0 — rand, oof_rand=0.92
for atk in 5 6; do
  echo "  VIS C3 alphabet[60,70] atk=$atk"
  $PY -u $S --gpu_pair 0,1 --task alphabet_soup --state-id 0 --window_start 60 --window_end 70 --condition vis_pgd --pgd_steps 20 --eps_raw_pixels 6 --max_steps 400 --seed 0 --env_seed 0 --attack_seed $atk --job_id $((JID++)) --pair_id conf_C3_alphabet_w60_70 --output_dir $OUT --image_preprocess official_rot180 || echo "VIS_FAIL conf_C3 atk=$atk"
  echo "  RAND C3 alphabet[60,70] atk=$atk"
  $PY -u $S --gpu_pair 0,1 --task alphabet_soup --state-id 0 --window_start 60 --window_end 70 --condition random_linf --eps_raw_pixels 6 --max_steps 400 --seed 0 --env_seed 0 --attack_seed $atk --job_id $((JID++)) --pair_id conf_C3_alphabet_w60_70 --output_dir $OUT --image_preprocess official_rot180 || echo "RAND_FAIL conf_C3 atk=$atk"
done

# C4: tomato[115,125] env=2 — rand confounded, oof_rand=0.97
for atk in 5 6; do
  echo "  VIS C4 tomato[115,125] atk=$atk"
  $PY -u $S --gpu_pair 0,1 --task tomato_sauce --state-id 2 --window_start 115 --window_end 125 --condition vis_pgd --pgd_steps 20 --eps_raw_pixels 6 --max_steps 400 --seed 0 --env_seed 2 --attack_seed $atk --job_id $((JID++)) --pair_id conf_C4_tomato_w115_125 --output_dir $OUT --image_preprocess official_rot180 || echo "VIS_FAIL conf_C4 atk=$atk"
  echo "  RAND C4 tomato[115,125] atk=$atk"
  $PY -u $S --gpu_pair 0,1 --task tomato_sauce --state-id 2 --window_start 115 --window_end 125 --condition random_linf --eps_raw_pixels 6 --max_steps 400 --seed 0 --env_seed 2 --attack_seed $atk --job_id $((JID++)) --pair_id conf_C4_tomato_w115_125 --output_dir $OUT --image_preprocess official_rot180 || echo "RAND_FAIL conf_C4 atk=$atk"
done

echo "[$(date +%H:%M:%S)] PIPELINE v0.3 CONFIRMATION DONE"
