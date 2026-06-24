#!/bin/bash
# Train reserve 80 launcher — verifies asset lock before collection.
set -e

REPO="${REPO:-/mnt/sdc/dty_user/openvla_attack}"
PY="$REPO/envs/openvla-official-a800/bin/python3"
COLLECTOR="$REPO/scripts/stageb/run_v6_perturbed_collector.py"
CKPT="$REPO/artifacts/detector/sc5_mlp_s2.pt"
MANIFEST="$REPO/migration_audit/m1c/sc5_v2_data/T2_TRAIN_RESERVE_V2_EXACT_MANIFEST.csv"
OUTBASE="$REPO/evidence/m1c/sc5_v2_primary"
LOCK="$REPO/migration_audit/m1c/sc5_v2_data/RESERVE_ASSET_LOCK.json"

export MUJOCO_GL=egl HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HOME="$REPO/sandbox_home" TMPDIR="$REPO/tmp"
export OPENVLA_DTYPE=bfloat16 OPENVLA_ATTN_IMPLEMENTATION=eager
export OPENVLA_MODEL_PATH="$REPO/models/openvla-7b-finetuned-libero-object"

# Pre-compute VLA manifest SHA
VLA_SHA=$($PY -c "import hashlib; from pathlib import Path; d=Path('$OPENVLA_MODEL_PATH'); lines=sorted('%s %s'%(f.relative_to(d),hashlib.sha256(f.read_bytes()).hexdigest()) for f in d.rglob('*') if f.is_file()); print(hashlib.sha256('\n'.join(lines).encode()).hexdigest())")

echo "=== TRAIN RESERVE 80 $(date) ==="
echo "VLA SHA: $VLA_SHA"

# Asset lock verification
COLLECTOR_SHA=$(sha256sum "$COLLECTOR" | awk '{print $1}')
CKPT_SHA=$(sha256sum "$CKPT" | awk '{print $1}')
echo "Collector: $COLLECTOR_SHA"
echo "Checkpoint: $CKPT_SHA"

$PY -c "
import json, hashlib
lock = json.load(open('$LOCK'))['assets']
# Verify collector
actual_col = '$COLLECTOR_SHA'
exp_col = lock['collector_sha256']
assert actual_col == exp_col, f'COLLECTOR MISMATCH: {actual_col} != {exp_col}'
# Verify checkpoint
actual_ckpt = '$CKPT_SHA'
exp_ckpt = lock['checkpoint_sha256']
assert actual_ckpt == exp_ckpt, f'CHECKPOINT MISMATCH: {actual_ckpt} != {exp_ckpt}'
# Verify VLA manifest
actual_vla = '$VLA_SHA'
exp_vla = lock['vla_model_manifest_sha256']
assert actual_vla == exp_vla, f'VLA_MANIFEST MISMATCH: {actual_vla} != {exp_vla}'
print('Asset lock: ALL 3 core assets VERIFIED')
# Verify other assets (lazy: check file exists and SHA matches expected)
for path, expected in [
    ('migration_audit/object_checkpoint_migration/m1_runtime/teacher_config_frozen.json', lock['teacher_config_sha256']),
    ('scripts/migration/label_m1c_object_teacher.py', lock['target_resolver_sha256']),
    ('src/gripper_attack/v5_perturbation.py', lock['perturbation_generator_sha256']),
]:
    actual = hashlib.sha256(open(path,'rb').read()).hexdigest()
    assert actual == expected, f'ASSET MISMATCH {path}: {actual[:16]} != {expected[:16]}'
print('Asset lock: ALL assets VERIFIED')
"

echo "Asset lock verified. Starting collection."

# Collection loop
total=0
pids=()
while IFS=, read -r task parent_state template seed dx_m dy_m dyaw_rad selection_source; do
    out_name="train/task${task}_state${parent_state}_${template}_seed${seed}"
    echo "[GPU$((total%5+1))] cell$total: $out_name"
    CUDA_VISIBLE_DEVICES=$(( total % 5 + 1 )) $PY $COLLECTOR \
        --task_idx "$task" --state_id "$parent_state" --seed_id "$seed" \
        --perturbation_template "$template" --pool train \
        --output_dir "$OUTBASE/$out_name" --render_gpu $(( total % 5 + 1 )) \
        --mlp_path "$CKPT" --vla_manifest_sha256 "$VLA_SHA" \
        > "$OUTBASE/${out_name}.stdout.log" 2>"$OUTBASE/${out_name}.stderr.log" &
    pids+=($!)
    total=$((total + 1))
    if [ $((total % 5)) -eq 0 ]; then
        for pid in "${pids[@]}"; do wait "$pid" || true; done
        pids=()
        echo "=== BATCH $((total/5)) DONE ($total/80) $(date) ==="
    fi
done < <(tail -n +2 "$MANIFEST")

for pid in "${pids[@]}"; do wait "$pid" || true; done
echo "=== TRAIN RESERVE DONE $(date) ==="
echo "Done: $(find $OUTBASE/train -name '.done' -path '*_seed*' | wc -l)"
