#!/bin/bash
# Dev reserve 40 launcher — Latin-square, states 23-27 only
set -e

REPO="${REPO:-/mnt/sdc/dty_user/openvla_attack}"
PY="$REPO/envs/openvla-official-a800/bin/python3"
COLLECTOR="$REPO/scripts/stageb/run_v6_perturbed_collector.py"
CKPT="$REPO/artifacts/detector/sc5_mlp_s2.pt"
MANIFEST="$REPO/migration_audit/m1c/sc5_v2_data/V2_DEV_RESERVE_V2_EXACT_MANIFEST.csv"
OUTBASE="$REPO/evidence/m1c/sc5_v2_primary"
LOCK="$REPO/migration_audit/m1c/sc5_v2_data/RESERVE_ASSET_LOCK.json"

export MUJOCO_GL=egl HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HOME="$REPO/sandbox_home" TMPDIR="$REPO/tmp"
export OPENVLA_DTYPE=bfloat16 OPENVLA_ATTN_IMPLEMENTATION=eager
export OPENVLA_MODEL_PATH="$REPO/models/openvla-7b-finetuned-libero-object"

VLA_SHA=$($PY -c "import hashlib; from pathlib import Path; d=Path('$OPENVLA_MODEL_PATH'); lines=sorted('%s %s'%(f.relative_to(d),hashlib.sha256(f.read_bytes()).hexdigest()) for f in d.rglob('*') if f.is_file()); print(hashlib.sha256('\n'.join(lines).encode()).hexdigest())")

echo "=== DEV RESERVE 40 $(date) ==="
echo "VLA SHA: $VLA_SHA"

# Asset lock
$PY -c "
import json, hashlib
lock = json.load(open('$LOCK'))['asset_specs']
# Collector
actual = hashlib.sha256(open('$COLLECTOR','rb').read()).hexdigest()
assert actual == lock['collector_sha256']['expected_sha256'], f'COLLECTOR: {actual[:16]} != {lock[\"collector_sha256\"][\"expected_sha256\"][:16]}'
# Checkpoint
actual = hashlib.sha256(open('$CKPT','rb').read()).hexdigest()
assert actual == lock['checkpoint_sha256']['expected_sha256'], f'CKPT: {actual[:16]} != expected'
# VLA
actual = '$VLA_SHA'
assert actual == lock['vla_model_manifest_sha256']['expected_sha256'], f'VLA: {actual[:16]} != expected'
# Teacher config
actual = hashlib.sha256(open('migration_audit/object_checkpoint_migration/m1_runtime/teacher_config_frozen.json','rb').read()).hexdigest()
assert actual == lock['teacher_config_sha256']['expected_sha256'], f'TEACHER_CFG: mismatch'
# Target resolver
actual = hashlib.sha256(open('scripts/migration/label_m1c_object_teacher.py','rb').read()).hexdigest()
assert actual == lock['target_resolver_sha256']['expected_sha256'], f'RESOLVER: mismatch'
# Perturbation generator
actual = hashlib.sha256(open('src/gripper_attack/v5_perturbation.py','rb').read()).hexdigest()
assert actual == lock['perturbation_generator_sha256']['expected_sha256'], f'PERT_GEN: mismatch'
print('Asset lock: ALL VERIFIED')
"

echo "Starting collection (40 cells, 5 GPUs, ~70 min ETA)"

total=0; pids=()
while IFS=, read -r task parent_state template seed dx_m dy_m dyaw_rad selection_source; do
    out_name="dev/task${task}_state${parent_state}_${template}_seed${seed}"
    gpu=$(( total % 5 + 1 ))
    echo "[GPU$gpu] cell$total: $out_name"
    CUDA_VISIBLE_DEVICES=$gpu $PY $COLLECTOR \
        --task_idx "$task" --state_id "$parent_state" --seed_id "$seed" \
        --perturbation_template "$template" --pool dev \
        --output_dir "$OUTBASE/$out_name" --render_gpu "$gpu" \
        --mlp_path "$CKPT" --vla_manifest_sha256 "$VLA_SHA" \
        > "$OUTBASE/${out_name}.stdout.log" 2>"$OUTBASE/${out_name}.stderr.log" &
    pids+=($!); total=$((total + 1))
    if [ $((total % 5)) -eq 0 ]; then
        for pid in "${pids[@]}"; do wait "$pid" || true; done
        pids=()
        echo "=== BATCH $((total/5)) DONE ($total/40) $(date) ==="
    fi
done < <(tail -n +2 "$MANIFEST")

for pid in "${pids[@]}"; do wait "$pid" || true; done
echo "=== DEV RESERVE DONE $(date) ==="
echo "Done: $(find $OUTBASE/dev -name '.done' -path '*_seed*' | wc -l)"
