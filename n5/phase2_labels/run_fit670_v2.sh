#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 || $# -gt 4 ]]; then
  echo "usage: $0 <canary|formal> <transition_root> <new_output_root> [gpu_csv]" >&2
  exit 64
fi

MODE="$1"
TRANSITION_ROOT="$2"
OUTPUT_ROOT="$3"
GPU_CSV="${4:-0,1,2,3,4,5,6,7}"

if [[ "$MODE" != "canary" && "$MODE" != "formal" ]]; then
  echo "mode must be canary or formal" >&2
  exit 64
fi
if [[ -e "$OUTPUT_ROOT" ]]; then
  echo "output root already exists: $OUTPUT_ROOT" >&2
  exit 73
fi

WORKTREE="${FIT670_WORKTREE:-/tmp/fresh670_v5_worktree}"
PYTHON="${FIT670_PYTHON:-/mnt/sdc/dty_user/openvla_attack/envs/openvla-official-a800/bin/python}"
MODEL="${FIT670_MODEL:-/mnt/sdc/dty_user/openvla_attack/models/libero-10/openvla-7b-finetuned-libero-10}"
OFFICIAL_WORKER="${FIT670_OFFICIAL_WORKER:-/mnt/sdc/dty_user/openvla_attack_official_v3_20260716/scripts/official_clean_worker.py}"
ALLOWLIST="${FIT670_ALLOWLIST:-/mnt/sdc/dty_user/openvla_attack_outputs/n5/phase3_student/fit670_allowlist/FIT670_IDENTITY_ALLOWLIST.json}"
SHARD_PLAN="${FIT670_SHARD_PLAN:?export FIT670_SHARD_PLAN to the sealed FIT670_GPU_SHARD_PLAN.json}"
REGISTRY_ROOT="${FIT670_REGISTRY_ROOT:-/mnt/sdc/dty_user/openvla_attack_outputs/n5/phase3_student/c1_v2_r7/run_A/per_task}"
ALIAS_LEDGER="${FIT670_ALIAS_LEDGER:-/mnt/sdc/dty_user/openvla_attack_outputs/n5/phase3_student/c1_v2_r7/run_A/ALIAS_LEDGER.json}"
UPSTREAM_ROOT="${FIT670_UPSTREAM_ROOT:-/mnt/sdc/dty_user/openvla_attack}"
LIBERO_ROOT="${FIT670_LIBERO_ROOT:-/mnt/sdc/dty_user/pi0_openpi/third_party/libero}"

exec "$PYTHON" -u "$WORKTREE/n5/phase2_labels/run_fit670_supervisor_v2.py" \
  --mode "$MODE" \
  --gpus "$GPU_CSV" \
  --transition-receipt "$TRANSITION_ROOT" \
  --output-root "$OUTPUT_ROOT" \
  --identity-allowlist "$ALLOWLIST" \
  --shard-plan "$SHARD_PLAN" \
  --model-path "$MODEL" \
  --official-worker "$OFFICIAL_WORKER" \
  --registry-root "$REGISTRY_ROOT" \
  --alias-ledger "$ALIAS_LEDGER" \
  --upstream-root "$UPSTREAM_ROOT" \
  --libero-root "$LIBERO_ROOT" \
  --seed 20260717
