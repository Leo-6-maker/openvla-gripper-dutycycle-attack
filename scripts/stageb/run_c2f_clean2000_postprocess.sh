#!/bin/bash
# C2f Clean2000 post-run pipeline
# Execute AFTER clean2000 collection completes.
# Order: merge → hygiene → audit → stats materialization
# CLIP and training must NOT run if hygiene/audit fail.
set -euo pipefail

COMMIT="3a22cf6"
SHARDED="/mnt/sdc/dty_user/openvla_attack_evidence/c2f/clean2000_obs_clean_36712cc"
MERGED="/mnt/sdc/dty_user/openvla_attack_evidence/c2f/clean2000_merged_${COMMIT:0:7}"
STATS="/mnt/sdc/dty_user/openvla_attack_evidence/c2f/clean2000_emb_stats_${COMMIT:0:7}"
CLIP="/mnt/sdc/dty_user/openvla_attack_evidence/c2f/clean2000_emb_clip_${COMMIT:0:7}"
ABL="/mnt/sdc/dty_user/openvla_attack_evidence/c2f/clean2000_ablation_${COMMIT:0:7}"
RUNS="/mnt/sdc/dty_user/openvla_attack_evidence/c2f/c2f_ablation_runs_${COMMIT:0:7}"

echo "=== C2f Clean2000 Post-Process Pipeline ==="
echo "Commit: $COMMIT"
echo ""

# ── Step 1: Merge shards ──
echo "--- Step 1: merge ---"
python scripts/stageb/merge_c2f_sharded_collection.py \
  --sharded-root "$SHARDED" \
  --output-root "$MERGED" \
  --mode hardlink \
  --git-commit "$COMMIT"
echo "Merge done: $MERGED"

# ── Step 2: Hygiene check ──
echo "--- Step 2: hygiene ---"
python scripts/stageb/check_c2f_collection_hygiene.py \
  --c2f-root "$MERGED" \
  --expected-episodes 2000
HYGIENE_RC=$?
if [ $HYGIENE_RC -ne 0 ]; then
    echo "FATAL: Hygiene FAIL. Aborting pipeline."
    exit 1
fi

# ── Step 3: Observation audit ──
echo "--- Step 3: audit ---"
python scripts/stageb/audit_c2f_observation_collection.py \
  --c2f-root "$MERGED" \
  --expected-episodes 2000 \
  --mode pilot200 \
  --strict-primary-nondegenerate
AUDIT_RC=$?
if [ $AUDIT_RC -ne 0 ]; then
    echo "FATAL: Audit FAIL. Check primary label distribution before proceeding."
    echo "Pipeline stops here. Manual review required for CLIP materialization decision."
    exit 1
fi

# ── Step 4: Stats materialization (always safe) ──
echo "--- Step 4: stats materialization ---"
python tools/multisuite_detector/materialize_c2f_frozen_embeddings.py \
  --c2f-root "$MERGED" \
  --output-dir "$STATS" \
  --window 16 \
  --backend stats \
  --device cpu \
  --git-commit "$COMMIT"
echo "Stats done: $STATS"

# ── Step 5: CLIP materialization (GPU required) ──
# UNCOMMENT only after Steps 1-4 ALL PASS:
# echo "--- Step 5: CLIP materialization ---"
# CUDA_VISIBLE_DEVICES=4 python tools/multisuite_detector/materialize_c2f_frozen_embeddings.py \
#   --c2f-root "$MERGED" \
#   --output-dir "$CLIP" \
#   --window 16 \
#   --backend clip \
#   --model-name openai/clip-vit-base-patch32 \
#   --device cuda \
#   --git-commit "$COMMIT"

# ── Step 6: Ablation datasets ──
# UNCOMMENT only after CLIP materialization PASS:
# echo "--- Step 6: ablation datasets ---"
# python tools/multisuite_detector/make_c2f_ablation_datasets.py \
#   --dataset "$CLIP/c2f_w16_clip_dataset.npz" \
#   --output-dir "$ABL" \
#   --git-commit "$COMMIT"

# ── Step 7: A/B/C/D Training ──
# UNCOMMENT only after ablation datasets created:
# for NAME in A_25d_only B_25d_language C_25d_rgb D_full_rgb_language; do
#   CUDA_VISIBLE_DEVICES=4 python tools/multisuite_detector/train_c2f_rgb_lang_temporal_detector_v0.py \
#     --dataset "$ABL/${NAME}.npz" \
#     --output-dir "$RUNS/${NAME}" \
#     --device cuda \
#     --epochs 30 --batch-size 256 \
#     --git-commit "$COMMIT"
# done

echo ""
echo "=== Pipeline complete ==="
echo "Merged: $MERGED"
echo "Stats:  $STATS"
# echo "CLIP:   $CLIP"
# echo "Ablation: $ABL"
# echo "Runs:  $RUNS"
