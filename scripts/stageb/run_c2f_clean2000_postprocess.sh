#!/bin/bash
# C2f Clean2000 post-run pipeline
# Execute AFTER clean2000 collection completes.
# Order: merge → hygiene → audit → stats materialization
# CLIP and training must NOT run if hygiene/audit fail.
set -euo pipefail

# Override from the shell when needed:
#   COMMIT=b11241c83ee46cca59719b9a61a0cc43403fa3b5 \
#   SHARDED=/path/to/sharded/root \
#   bash scripts/stageb/run_c2f_clean2000_postprocess.sh
COMMIT="${COMMIT:-$(git rev-parse --short HEAD 2>/dev/null || echo unknown)}"
SHARDED="${SHARDED:-/mnt/sdc/dty_user/openvla_attack_evidence/c2f/clean2000_obs_clean_36712cc}"
MERGED="${MERGED:-/mnt/sdc/dty_user/openvla_attack_evidence/c2f/clean2000_merged_${COMMIT:0:7}}"
STATS="${STATS:-/mnt/sdc/dty_user/openvla_attack_evidence/c2f/clean2000_emb_stats_${COMMIT:0:7}}"
CLIP="${CLIP:-/mnt/sdc/dty_user/openvla_attack_evidence/c2f/clean2000_emb_clip_${COMMIT:0:7}}"
ABL="${ABL:-/mnt/sdc/dty_user/openvla_attack_evidence/c2f/clean2000_ablation_${COMMIT:0:7}}"
RUNS="${RUNS:-/mnt/sdc/dty_user/openvla_attack_evidence/c2f/c2f_ablation_runs_${COMMIT:0:7}}"
EXPECTED_EPISODES="${EXPECTED_EPISODES:-2000}"

fail() {
  echo "FATAL: $*" >&2
  exit 1
}

echo "=== C2f Clean2000 Post-Process Pipeline ==="
echo "Commit:  $COMMIT"
echo "Sharded: $SHARDED"
echo "Merged:  $MERGED"
echo "Stats:   $STATS"
echo "Expected episodes: $EXPECTED_EPISODES"
echo ""

# ── Step 0: Preflight ──
echo "--- Step 0: preflight ---"
[ -d "$SHARDED" ] || fail "SHARDED root does not exist: $SHARDED"
META_COUNT=$(find "$SHARDED" -path '*/episodes/*/*/episode_metadata.json' -type f | wc -l | tr -d ' ')
echo "Found shard episode_metadata.json files: $META_COUNT"
if [ "$META_COUNT" -ne "$EXPECTED_EPISODES" ]; then
  fail "Expected $EXPECTED_EPISODES completed shard episodes, found $META_COUNT. Do not postprocess until collection is complete."
fi
if grep -R "Traceback" "$SHARDED"/logs >/dev/null 2>&1; then
  fail "Traceback found in shard logs. Inspect $SHARDED/logs before merge."
fi

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
if ! python scripts/stageb/check_c2f_collection_hygiene.py \
  --c2f-root "$MERGED" \
  --expected-episodes "$EXPECTED_EPISODES"; then
    fail "Hygiene FAIL. Aborting pipeline. Do not run audit/stats/CLIP."
fi

# ── Step 3: Observation audit ──
echo "--- Step 3: audit ---"
if ! python scripts/stageb/audit_c2f_observation_collection.py \
  --c2f-root "$MERGED" \
  --expected-episodes "$EXPECTED_EPISODES" \
  --mode pilot200 \
  --strict-primary-nondegenerate; then
    fail "Audit FAIL. Check label distribution before proceeding. CLIP/training remain unauthorized."
fi

# ── Step 4: Stats materialization ──
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
# UNCOMMENT only after Steps 1-4 ALL PASS and label distribution is accepted:
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
