#!/bin/bash
R=/mnt/sdc/dty_user/openvla_attack_evidence/c2f/table1_candidate_gpu17_20260709_235106
O=$R/output
REPO=/mnt/sdc/dty_user/openvla_attack
V=$REPO/envs/openvla-official-a800/bin/python
CKPT=/mnt/sdc/dty_user/openvla_attack_evidence/c2f/siglip_full_final/train_D/c2f_rgb_lang_temporal_detector_v0.pt
cd $REPO
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
echo "=== Goal all fill $(date) ==="

idx=0
while IFS='|' read -r parent cond; do
  [ -z "$parent" ] && continue
  meta="$O/${parent}/${cond}/episode_metadata.json"
  [ -s "$meta" ] && { echo "SKIP $parent $cond"; idx=$((idx+1)); continue; }
  echo "[$((idx+1))/36] $parent $cond $(date +%H:%M:%S)"
  CUDA_VISIBLE_DEVICES=7 PYTHONPATH=$REPO:$REPO/src:$REPO/scripts \
    $V scripts/stageb/run_c2f_canary_worker.py \
    --parent-key "$parent" --condition "$cond" --checkpoint "$CKPT" \
    --gpu 0 --output-dir "$O" --git-commit 1c181f8 \
    > "$R/log_goal2_${parent//\//_}_${cond}.log" 2>&1
  idx=$((idx+1))
done < /tmp/all_goal_jobs.txt
echo "=== DONE $(date) ==="
python3 -c "import json,glob; print('Total:', len(sorted(glob.glob('$R/output/**/episode_metadata.json', recursive=True))))"
