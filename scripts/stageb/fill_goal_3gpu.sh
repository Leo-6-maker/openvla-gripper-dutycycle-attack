#!/bin/bash
R=/mnt/sdc/dty_user/openvla_attack_evidence/c2f/table1_candidate_gpu17_20260709_235106
O=$R/output
REPO=/mnt/sdc/dty_user/openvla_attack
V=$REPO/envs/openvla-official-a800/bin/python
CKPT=/mnt/sdc/dty_user/openvla_attack_evidence/c2f/siglip_full_final/train_D/c2f_rgb_lang_temporal_detector_v0.pt
GPUS=(1 3 5)
cd $REPO
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
echo "=== Goal 3-GPU $(date) ==="

# Rebuild remaining jobs
python3 -c "
import json, os
parents = [json.loads(l) for l in open('$R/parent_manifest.jsonl')]
goal = [p for p in parents if p['suite']=='libero_goal']
conds = ['CLEAN','TRUE_T10','RAND_T10']
remaining = []
for p in goal:
    for c in conds:
        meta = '$R/output/' + p['parent_key'] + '/' + c + '/episode_metadata.json'
        if not os.path.exists(meta):
            remaining.append((p['parent_key'], c))
print(str(len(remaining)) + ' jobs remaining')
# Write per-GPU job lists
for i, (parent, cond) in enumerate(remaining):
    gpu = [1,3,5][i % 3]
    with open('/tmp/goal_gpu' + str(gpu) + '.txt', 'a') as f:
        f.write(parent + '|' + cond + '\n')
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
      --gpu 0 --output-dir "$O" --git-commit 1c181f8 \
      > "$R/log_goal3_${parent//\//_}_${cond}.log" 2>&1
  done < "/tmp/goal_gpu${gpu}.txt"
}

for gpu in "${GPUS[@]}"; do
  run_queue $gpu &
done
wait
echo "=== DONE $(date) ==="
python3 -c "import json,glob; print('Total:', len(sorted(glob.glob('$R/output/**/episode_metadata.json', recursive=True))))"
