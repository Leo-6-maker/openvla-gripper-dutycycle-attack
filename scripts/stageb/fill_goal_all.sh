#!/bin/bash
R=/mnt/sdc/dty_user/openvla_attack_evidence/c2f/table1_candidate_gpu17_20260709_235106
O=$R/output
CODE_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
V="${A800_PY:-/mnt/sdc/dty_user/openvla_attack/envs/openvla-official-a800/bin/python}"
CKPT=/mnt/sdc/dty_user/openvla_attack_evidence/c2f/siglip_full_final/train_D/c2f_rgb_lang_temporal_detector_v0.pt
GOAL_MODEL_MANIFEST="$CODE_REPO/artifacts/goal_model_manifest.json"
cd "$CODE_REPO"
COMMIT=$(git rev-parse HEAD)
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
echo "=== Goal all fill $(date) ==="

idx=0
while IFS='|' read -r parent cond; do
  [ -z "$parent" ] && continue
  meta="$O/${parent}/${cond}/episode_metadata.json"
  [ -s "$meta" ] && "$V" scripts/stageb/audit_c2f_track_a_run.py --metadata-complete "$meta" >/dev/null 2>&1 && { echo "SKIP $parent $cond"; idx=$((idx+1)); continue; }
  [ -s "$meta" ] && "$V" scripts/stageb/audit_c2f_track_a_run.py --archive-invalid-output-root "$O" --parent-key "$parent" --condition "$cond" --invalid-archive-root "$R/invalid_attempts"
  echo "[$((idx+1))/36] $parent $cond $(date +%H:%M:%S)"
  CUDA_VISIBLE_DEVICES=7 PYTHONPATH=$CODE_REPO:$CODE_REPO/src:$CODE_REPO/scripts \
    $V scripts/stageb/run_c2f_canary_worker.py \
    --parent-key "$parent" --condition "$cond" --checkpoint "$CKPT" \
    --gpu 0 --output-dir "$O" --expected-git-commit "$COMMIT" --policy-model-manifest "$GOAL_MODEL_MANIFEST" \
    > "$R/log_goal2_${parent//\//_}_${cond}.log" 2>&1
  idx=$((idx+1))
done < /tmp/all_goal_jobs.txt
echo "=== DONE $(date) ==="
python3 -c "import json,glob; print('Total:', len(sorted(glob.glob('$R/output/**/episode_metadata.json', recursive=True))))"
