#!/bin/bash
# C2f online canary: Object 12 + L10 12 parents, TRUE_CMDOPEN_T10_C2F vs RAND_ACTION_NOISE_T10_C2F paired
# Queue-based: 1 worker per GPU, sequential per-GPU queue (no 48 concurrent jobs)
set -e
CANARY_ROOT=/mnt/sdc/dty_user/openvla_attack_evidence/c2f/canary_v1
CHECKPOINT=/mnt/sdc/dty_user/openvla_attack_evidence/c2f/siglip_full_final/train_D/c2f_rgb_lang_temporal_detector_v0.pt
REPO=/mnt/sdc/dty_user/openvla_attack
VENV=$REPO/envs/openvla-official-a800/bin/python
COMMIT=$(cd $REPO && git rev-parse --short HEAD)
PARENT_CSV=/mnt/sdc/dty_user/openvla_attack_evidence/condition_matrix/d7_table1_manifest/d7_table1_preregistered_parent_keys.csv

# ── Create output dir FIRST ──
mkdir -p $CANARY_ROOT/output

echo "=== C2f Online Canary v1 ==="
echo "Detector: D Full SigLIP, full gate (emit/suppress/abstain/primary)"
echo "Checkpoint: $CHECKPOINT"
echo "Parents: Object 12 + L10 12"
echo "Git: $COMMIT"
echo "Start: $(date)"

# ── Select and FREEZE parents ──
MANIFEST=$CANARY_ROOT/canary_parents.jsonl
if [ ! -f "$MANIFEST" ]; then
  python3 -c "
import csv, random, json, hashlib
random.seed(42)
parents = list(csv.DictReader(open('$PARENT_CSV')))
obj = [p for p in parents if p['suite'] == 'libero_object']
l10 = [p for p in parents if p['suite'] == 'libero_10']
selected = random.sample(obj, 12) + random.sample(l10, 12)
with open('$MANIFEST', 'w') as f:
    for p in selected: f.write(json.dumps(p) + '\n')
# Write provenance
csv_sha = hashlib.sha256(open('$PARENT_CSV','rb').read()).hexdigest()
with open('$CANARY_ROOT/parent_selection_provenance.json', 'w') as f:
    json.dump({'source_csv': '$PARENT_CSV', 'csv_sha256': csv_sha, 'seed': 42,
               'n_object': 12, 'n_l10': 12, 'total': 24}, f, indent=2)
print(f'Frozen {len(selected)} parents to $MANIFEST')
"
fi

# ── Generate per-GPU job queues ──
CONDITIONS=(TRUE_CMDOPEN_T10_C2F RAND_ACTION_NOISE_T10_C2F)
GPUS=(0 1 2 3)
rm -f $CANARY_ROOT/jobs_gpu*.txt
idx=0
while IFS= read -r line; do
  parent_key=$(echo "$line" | python3 -c "import json,sys; print(json.loads(sys.stdin.read())['parent_key'])")
  for cond in "${CONDITIONS[@]}"; do
    gpu=${GPUS[$((idx % ${#GPUS[@]}))]}
    echo "$parent_key $cond $gpu" >> $CANARY_ROOT/jobs_gpu${gpu}.txt
    idx=$((idx + 1))
  done
done < "$MANIFEST"

TOTAL_JOBS=$idx
echo "Total jobs: $TOTAL_JOBS across ${#GPUS[@]} GPUs"
for gpu in "${GPUS[@]}"; do
  nj=$(wc -l < $CANARY_ROOT/jobs_gpu${gpu}.txt 2>/dev/null || echo 0)
  echo "  GPU $gpu: $nj jobs"
done

# ── Per-GPU sequential queue workers ──
run_queue() {
  local gpu=$1
  local jobfile=$CANARY_ROOT/jobs_gpu${gpu}.txt
  while IFS=' ' read -r parent_key cond _; do
    [ -z "$parent_key" ] && continue
    local logname="${parent_key//\//_}_${cond}"
    echo "[$(date +%H:%M:%S)] GPU$gpu: $parent_key $cond"
    CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=$REPO:$REPO/src:$REPO/scripts \
      $VENV scripts/stageb/run_c2f_canary_worker.py \
      --parent-key "$parent_key" --condition "$cond" \
      --checkpoint "$CHECKPOINT" --gpu 0 \
      --output-dir "$CANARY_ROOT/output" \
      --git-commit "$COMMIT" \
      > "$CANARY_ROOT/output/log_${logname}.log" 2>&1
  done < "$jobfile"
  echo "[GPU$gpu] ALL DONE $(date +%H:%M:%S)"
}

# Launch all GPU queues in parallel
for gpu in "${GPUS[@]}"; do
  run_queue $gpu &
done

wait
echo "=== CANARY COMPLETE: $(date) ==="

# ── Quick audit ──
python3 -c "
import json, os, glob
base = '$CANARY_ROOT/output'
reports = sorted(glob.glob(base + '/*/*/episode_metadata.json'))
print(f'Completed episodes: {len(reports)}')
from collections import Counter
conds = Counter()
for rp in reports:
    conds[json.load(open(rp))['condition']] += 1
print(f'Per condition: {dict(conds)}')
"
