#!/bin/bash
# FORMAL_SMOKE_22_LAUNCHER_FROZEN.sh
# Launches 22-cell formal smoke using V6 collector.
# Must be run from REPO root with V6 collector deployed.
set -e

REPO="${REPO:-/mnt/sdc/dty_user/openvla_attack}"
PY="$REPO/envs/openvla-official-a800/bin/python3"
COLLECTOR="$REPO/scripts/stageb/run_v6_perturbed_collector.py"
CKPT="$REPO/artifacts/detector/sc5_mlp_s2.pt"
MANIFEST="$REPO/migration_audit/m1c/sc5_v2_data/FORMAL_SMOKE_22_EXACT_MANIFEST.csv"
AUDITOR="$REPO/scripts/stageb/audit_formal_smoke_22.py"
OUTBASE="$REPO/evidence/m1c/formal_smoke_22"

export MUJOCO_GL=egl HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HOME="$REPO/sandbox_home"
export TMPDIR="$REPO/tmp"
export OPENVLA_DTYPE=bfloat16 OPENVLA_ATTN_IMPLEMENTATION=eager
export OPENVLA_MODEL_PATH="$REPO/models/openvla-7b-finetuned-libero-object"

# Pre-compute VLA manifest SHA once
VLA_SHA=$($PY -c "import hashlib; from pathlib import Path; d=Path('$OPENVLA_MODEL_PATH'); lines=sorted('%s %s'%(f.relative_to(d),hashlib.sha256(f.read_bytes()).hexdigest()) for f in d.rglob('*') if f.is_file()); print(hashlib.sha256('\n'.join(lines).encode()).hexdigest())")

echo "=== FORMAL SMOKE 22 $(date) ==="
echo "VLA manifest SHA: $VLA_SHA"

rm -rf "$OUTBASE"
mkdir -p "$OUTBASE"

total=0
tail -n +2 "$MANIFEST" | while IFS=, read -r task state template seed pool rg dx dy dyaw output; do
    gpu=$(( (total % 5) + 1 ))
    echo "[GPU$gpu] cell$total: $output (task=$task state=$state $template)"
    CUDA_VISIBLE_DEVICES=$gpu $PY $COLLECTOR \
        --task_idx "$task" --state_id "$state" --seed_id "$seed" \
        --perturbation_template "$template" --pool smoke \
        --output_dir "$OUTBASE/$output" --render_gpu "$gpu" \
        --mlp_path "$CKPT" --vla_manifest_sha256 "$VLA_SHA" \
        > "$OUTBASE/${output}.stdout.log" 2>"$OUTBASE/${output}.stderr.log" &
    total=$((total + 1))
    if [ $((total % 5)) -eq 0 ]; then
        wait
        echo "=== BATCH $((total/5)) DONE $(date) ==="
    fi
done
wait

echo "=== COLLECTION DONE $(date) ==="
echo "Running auditor..."
$PY $AUDITOR --manifest "$MANIFEST" --output_base "$OUTBASE"
echo "=== SMOKE COMPLETE $(date) ==="
