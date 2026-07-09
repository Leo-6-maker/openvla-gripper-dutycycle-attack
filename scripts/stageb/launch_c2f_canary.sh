#!/bin/bash
# C2f online canary: Object 12 + L10 12 parents, TRUE_T10 vs RAND_T10 paired
# Uses D Full SigLIP detector (default gate: tau_emit=0.33, tau_suppress=0.67)
set -e
CANARY_ROOT=/mnt/sdc/dty_user/openvla_attack_evidence/c2f/canary_v1
CHECKPOINT=/mnt/sdc/dty_user/openvla_attack_evidence/c2f/siglip_full_final/train_D/c2f_rgb_lang_temporal_detector_v0.pt
REPO=/mnt/sdc/dty_user/openvla_attack
VENV=$REPO/envs/openvla-official-a800/bin/python
COMMIT=$(cd $REPO && git rev-parse --short HEAD)
PARENT_CSV=/mnt/sdc/dty_user/openvla_attack_evidence/condition_matrix/d7_table1_manifest/d7_table1_preregistered_parent_keys.csv

echo "=== C2f Online Canary v1 ==="
echo "Detector: D Full SigLIP, tau_emit=0.33, tau_suppress=0.67"
echo "Checkpoint: $CHECKPOINT"
echo "Parents: Object 12 + L10 12"
echo "Git: $COMMIT"
echo "Start: $(date)"

# Select parents: 12 Object + 12 L10
python3 -c "
import csv, random, json
random.seed(42)
parents = list(csv.DictReader(open('$PARENT_CSV')))
obj = [p for p in parents if p['suite'] == 'libero_object']
l10 = [p for p in parents if p['suite'] == 'libero_10']
selected = random.sample(obj, 12) + random.sample(l10, 12)
with open('$CANARY_ROOT/canary_parents.jsonl', 'w') as f:
    for p in selected: f.write(json.dumps(p) + '\n')
print(f'Selected {len(selected)} parents')
"

mkdir -p $CANARY_ROOT

# Use GPUs 0,1,2,3 in parallel
GPUS=(0 1 2 3)
CONDITIONS=(TRUE_T10 RAND_T10)
PARENTS=($(python3 -c "import json; [print(json.loads(l)['parent_key']) for l in open('$CANARY_ROOT/canary_parents.jsonl')]"))

TOTAL=$((${#PARENTS[@]} * ${#CONDITIONS[@]}))
echo "Total jobs: $TOTAL (${#PARENTS[@]} parents × ${#CONDITIONS[@]} conditions)"
DONE=0

for parent in "${PARENTS[@]}"; do
  for cond in "${CONDITIONS[@]}"; do
    gpu=${GPUS[$((DONE % ${#GPUS[@]}))]}
    echo "[$DONE/$TOTAL] GPU$gpu: $parent $cond"
    CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=$REPO:$REPO/src:$REPO/scripts \
      $VENV scripts/stageb/run_c2f_canary_worker.py \
      --parent-key "$parent" --condition "$cond" \
      --checkpoint "$CHECKPOINT" --gpu 0 \
      --output-dir "$CANARY_ROOT/output" \
      --git-commit "$COMMIT" \
      > "$CANARY_ROOT/log_${parent//\//_}_${cond}.log" 2>&1 &
    DONE=$((DONE + 1))
    # Stagger launches to avoid GPU memory contention
    sleep 2
  done
done

echo "All launched. Waiting..."
wait
echo "=== CANARY COMPLETE: $(date) ==="
