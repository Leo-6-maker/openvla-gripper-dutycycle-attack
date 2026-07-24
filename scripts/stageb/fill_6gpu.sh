#!/bin/bash
R=/mnt/sdc/dty_user/openvla_attack_evidence/c2f/table1_candidate_gpu17_20260709_235106
O=$R/output
REPO=/mnt/sdc/dty_user/openvla_attack
V=$REPO/envs/openvla-official-a800/bin/python
CKPT=/mnt/sdc/dty_user/openvla_attack_evidence/c2f/siglip_full_final/train_D/c2f_rgb_lang_temporal_detector_v0.pt
GPUS=(0 1 2 3 5 7)
cd $REPO
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl

echo "=== 6-GPU fill $(date) ==="

python3 -c "
import json
m = json.load(open('$R/postrun_audit.json'))['missing']
with open('/tmp/fill6_list.txt', 'w') as f:
    for p, c in m: f.write(p + '|' + c + '\n')
print(str(len(m)) + ' jobs')
"

run_queue() {
  local gpu=$1
  while IFS='|' read -r parent cond; do
    [ -z "$parent" ] && continue
    local meta="$O/${parent}/${cond}/episode_metadata.json"
    [ -s "$meta" ] && continue
    echo "[GPU$gpu] $parent $cond $(date +%H:%M:%S)"
    CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=$REPO:$REPO/src:$REPO/scripts \
      $V scripts/stageb/run_c2f_canary_worker.py \
      --parent-key "$parent" --condition "$cond" --checkpoint "$CKPT" \
      --gpu 0 --output-dir "$O" --git-commit 172b78d \
      > "$R/log_6gpu_${parent//\//_}_${cond}.log" 2>&1
  done < "$R/fill_jobs_gpu${gpu}.txt"
}

idx=0
rm -f "$R"/fill_jobs_gpu*.txt
while IFS='|' read -r parent cond; do
  [ -z "$parent" ] && continue
  gpu=${GPUS[$((idx % ${#GPUS[@]}))]}
  echo "${parent}|${cond}" >> "$R/fill_jobs_gpu${gpu}.txt"
  idx=$((idx+1))
done < /tmp/fill6_list.txt
echo "Jobs/GPU:"; wc -l "$R"/fill_jobs_gpu*.txt

for gpu in "${GPUS[@]}"; do
  run_queue $gpu &
done
wait
echo "=== DONE $(date) ==="
python3 -c "import json,glob; print('Total:', len(sorted(glob.glob('$R/output/**/episode_metadata.json', recursive=True))))"
