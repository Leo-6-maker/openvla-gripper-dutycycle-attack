#!/bin/bash
# Full 363K SigLIP materialization: 20 shards × 100ep, 3 GPUs parallel
# Each round: 3 shards on GPU 4,6,7 simultaneously
set -e
BASE="/mnt/sdc/dty_user/openvla_attack_evidence/c2f"
ROOT="$BASE/clean2000_v1.1_caveat"
REPO="/mnt/sdc/dty_user/openvla_attack"
VENV="$REPO/envs/openvla-official-a800/bin/python"
MODEL="$REPO/models/openvla-7b-finetuned-libero-object"
COMMIT="68f9a3a"
GPUS=(4 6 7)
EP_PER_SHARD=100
NUM_SHARDS=20

echo "=== Full SigLIP: $NUM_SHARDS shards × $EP_PER_SHARD ep on GPUs ${GPUS[*]} ==="
echo "Start: $(date)"

run_shard() {
  local gpu=$1 offset=$2 outdir=$3
  rm -rf "$outdir"; mkdir -p "$outdir"
  cd "$REPO"
  CUDA_VISIBLE_DEVICES=$gpu OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
    PYTHONPATH="$REPO:$REPO/src:$REPO/scripts" \
    $VENV tools/multisuite_detector/materialize_c2f_frozen_embeddings.py \
    --c2f-root "$ROOT" --output-dir "$outdir" \
    --backend openvla_siglip --openvla-model-path "$MODEL" \
    --device cuda --window 16 --seed 42 \
    --max-episodes $EP_PER_SHARD --episode-offset $offset \
    --git-commit "$COMMIT" \
    > "${outdir}/run.log" 2>&1
  if [ -f "$outdir/c2f_materialization_report.json" ]; then
    python3 -c "import json; r=json.load(open('$outdir/c2f_materialization_report.json')); print('  GPU${gpu} offset=${offset}: ep={} win={} rt={:.0f}s'.format(r['n_episodes'], r['n_windows'], r['runtime_seconds']))"
  else
    echo "  GPU${gpu} offset=${offset}: FAILED"
  fi
}

ROUNDS=$(( (NUM_SHARDS + 2) / 3 ))
for ((r=0; r<ROUNDS; r++)); do
  echo "[$(date +%H:%M:%S)] Round $((r+1))/$ROUNDS"
  pids=()
  for ((j=0; j<3; j++)); do
    idx=$((r * 3 + j))
    if [ $idx -ge $NUM_SHARDS ]; then break; fi
    gpu=${GPUS[$j]}
    offset=$((idx * EP_PER_SHARD))
    outdir="$BASE/siglip_full_shard_${offset}"
    run_shard $gpu $offset "$outdir" &
    pids+=($!)
  done
  for pid in "${pids[@]}"; do wait $pid; done
  echo "  Round $((r+1)) done"
done

echo "=== All shards done: $(date) ==="
