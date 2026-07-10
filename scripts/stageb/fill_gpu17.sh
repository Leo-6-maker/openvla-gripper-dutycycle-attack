#!/bin/bash
R=/mnt/sdc/dty_user/openvla_attack_evidence/c2f/table1_candidate_gpu17_20260709_235106
O=$R/output
REPO=/mnt/sdc/dty_user/openvla_attack
V=$REPO/envs/openvla-official-a800/bin/python
CKPT=/mnt/sdc/dty_user/openvla_attack_evidence/c2f/siglip_full_final/train_D/c2f_rgb_lang_temporal_detector_v0.pt
cd $REPO
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
echo "=== GPU1+7 fill $(date) ==="

python3 -c "
import json
m = json.load(open('$R/postrun_audit.json'))['missing']
with open('/tmp/fill_list.txt', 'w') as f:
    for p, c in m: f.write(p + '|' + c + '\n')
print(str(len(m)) + ' jobs')
"

idx=0
while IFS='|' read -r parent cond; do
  [ -z "$parent" ] && continue
  gpu=$([ $((idx % 2)) -eq 0 ] && echo 1 || echo 7)
  meta="$O/${parent}/${cond}/episode_metadata.json"
  if [ -s "$meta" ]; then
    echo "SKIP $parent $cond"
    idx=$((idx+1))
    continue
  fi
  echo "[$((idx+1))] GPU$gpu $parent $cond $(date +%H:%M:%S)"
  CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=$REPO:$REPO/src:$REPO/scripts \
    $V scripts/stageb/run_c2f_canary_worker.py \
    --parent-key "$parent" --condition "$cond" --checkpoint "$CKPT" \
    --gpu 0 --output-dir "$O" --git-commit 1616f52 \
    > "$R/log_s17_${parent//\//_}_${cond}.log" 2>&1 &
  idx=$((idx+1))
  if [ $((idx % 2)) -eq 0 ]; then wait; fi
done < /tmp/fill_list.txt
wait
echo "=== DONE $(date) ==="
python3 -c "import json,glob; print('Total:', len(sorted(glob.glob('$R/output/**/episode_metadata.json', recursive=True))))"
