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

# Fail-closed: refuse to overwrite completed evidence
if [ -f "$OUTBASE/.smoke_complete" ]; then
    echo "FATAL: .smoke_complete exists — refusing to overwrite completed evidence"
    exit 2
fi
if [ -d "$OUTBASE" ] && [ ! -f "$OUTBASE/.smoke_complete" ]; then
    STALE_DIR="${OUTBASE}.stale.$(date +%Y%m%d_%H%M%S)"
    echo "Moving stale partial to $STALE_DIR"
    mv "$OUTBASE" "$STALE_DIR"
fi
mkdir -p "$OUTBASE"

# Pre-compute VLA manifest SHA once
VLA_SHA=$($PY -c "import hashlib; from pathlib import Path; d=Path('$OPENVLA_MODEL_PATH'); lines=sorted('%s %s'%(f.relative_to(d),hashlib.sha256(f.read_bytes()).hexdigest()) for f in d.rglob('*') if f.is_file()); print(hashlib.sha256('\n'.join(lines).encode()).hexdigest())")

echo "=== FORMAL SMOKE 22 $(date) ==="
echo "VLA manifest SHA: $VLA_SHA"

# Collect all cells (process substitution avoids subshell PID loss)
total=0
pids=()
while IFS=, read -r task state template seed pool role rg dx dy dyaw output; do
    gpu=$(( (total % 5) + 1 ))
    echo "[GPU$gpu] cell$total: $output (task=$task state=$state $template seed=$seed role=$role)"
    CUDA_VISIBLE_DEVICES=$gpu $PY $COLLECTOR \
        --task_idx "$task" --state_id "$state" --seed_id "$seed" \
        --perturbation_template "$template" --pool smoke \
        --output_dir "$OUTBASE/$output" --render_gpu "$gpu" \
        --mlp_path "$CKPT" --vla_manifest_sha256 "$VLA_SHA" \
        > "$OUTBASE/${output}.stdout.log" 2>"$OUTBASE/${output}.stderr.log" &
    pids+=($!)
    total=$((total + 1))
    if [ $((total % 5)) -eq 0 ]; then
        for pid in "${pids[@]}"; do wait "$pid" || true; done
        pids=()
        echo "=== BATCH $((total/5)) DONE $(date) ==="
    fi
done < <(tail -n +2 "$MANIFEST")

# Wait for remaining cells
for pid in "${pids[@]}"; do wait "$pid" || true; done
echo "=== COLLECTION DONE $(date) ==="

# Negative completion test: re-run one completed cell, verify rejection
echo "=== NEGATIVE COMPLETION TEST ==="
TEST_CELL="$OUTBASE/smoke_P0_t0_s3"
TEST_DONE_SHA=$(sha256sum "$TEST_CELL/.done" | awk '{print $1}')
TEST_TEL_SHA=$(sha256sum "$TEST_CELL/step_telemetry.csv" | awk '{print $1}')
TEST_EP_SHA=$(sha256sum "$TEST_CELL/episode_summary.json" | awk '{print $1}')

# Attempt duplicate
set +e
CUDA_VISIBLE_DEVICES=1 $PY $COLLECTOR \
    --task_idx 0 --state_id 3 --seed_id 42 \
    --perturbation_template P0 --pool smoke \
    --output_dir "$TEST_CELL" --render_gpu 1 \
    --mlp_path "$CKPT" --vla_manifest_sha256 "$VLA_SHA" \
    > "$OUTBASE/negative_test.stdout.log" 2>"$OUTBASE/negative_test.stderr.log"
NEG_RC=$?
set -e

NEG_STDERR=$(cat "$OUTBASE/negative_test.stderr.log")
NEG_HAS_CELL=$(echo "$NEG_STDERR" | grep -c "CELL_ALREADY_COMPLETE" || true)

NEW_DONE_SHA=$(sha256sum "$TEST_CELL/.done" | awk '{print $1}')
NEW_TEL_SHA=$(sha256sum "$TEST_CELL/step_telemetry.csv" | awk '{print $1}')
NEW_EP_SHA=$(sha256sum "$TEST_CELL/episode_summary.json" | awk '{print $1}')

FILES_UNCHANGED=false
if [ "$TEST_DONE_SHA" = "$NEW_DONE_SHA" ] && [ "$TEST_TEL_SHA" = "$NEW_TEL_SHA" ] && [ "$TEST_EP_SHA" = "$NEW_EP_SHA" ]; then
    FILES_UNCHANGED=true
fi

$PY -c "
import json
with open('$OUTBASE/negative_duplicate_test.json','w') as f:
    json.dump({
        'collector_nonzero_exit': $NEG_RC != 0,
        'collector_exit_code': $NEG_RC,
        'stderr_contains_CELL_ALREADY_COMPLETE': bool($NEG_HAS_CELL > 0),
        'files_unchanged': $FILES_UNCHANGED,
        'original_done_sha': '$TEST_DONE_SHA',
        'original_tel_sha': '$TEST_TEL_SHA',
        'original_ep_sha': '$TEST_EP_SHA',
        'final_done_sha': '$NEW_DONE_SHA',
        'final_tel_sha': '$NEW_TEL_SHA',
        'final_ep_sha': '$NEW_EP_SHA',
    }, f, indent=2)
"
echo "Negative test: rc=$NEG_RC CELL_MSG=$NEG_HAS_CELL files_unchanged=$FILES_UNCHANGED"

# Run auditor
echo "=== AUDITOR $(date) ==="
$PY $AUDITOR --manifest "$MANIFEST" --output_base "$OUTBASE"
AUDIT_RC=$?

# Create .smoke_complete marker on PASS
if [ $AUDIT_RC -eq 0 ]; then
    MARKER_TMP="$OUTBASE/.smoke_complete.tmp"
    $PY -c "
import json, hashlib
m = {
    'completed_at': '$(date --iso-8601=seconds)',
    'manifest_sha256': hashlib.sha256(open('$MANIFEST','rb').read()).hexdigest(),
    'collector_sha256': hashlib.sha256(open('$COLLECTOR','rb').read()).hexdigest(),
    'auditor_sha256': hashlib.sha256(open('$AUDITOR','rb').read()).hexdigest(),
    'launcher_sha256': hashlib.sha256(open('$0','rb').read()).hexdigest(),
    'audit_result': 'PASS',
}
with open('$MARKER_TMP','w') as f:
    json.dump(m, f, indent=2)
"
    sync "$MARKER_TMP"
    mv "$MARKER_TMP" "$OUTBASE/.smoke_complete"
    # fsync parent dir
    $PY -c "import os; fd=os.open('$OUTBASE', os.O_RDONLY); os.fsync(fd); os.close(fd)"
    echo "=== .smoke_complete CREATED ==="
fi

echo "=== FORMAL SMOKE 22 COMPLETE $(date) ==="
exit $AUDIT_RC
