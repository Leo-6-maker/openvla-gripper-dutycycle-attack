#!/usr/bin/env bash
set -euo pipefail

MODEL_FAMILY="$1"
SUITE="$2"
GPU_ID="$3"
ROOT="/mnt/sdc/dty_user/openvla_attack_worktrees/stage-ac2-clean-screen-c217acfc-lf"
OUT_ROOT="/mnt/sdc/dty_user/openvla_attack_outputs/STAGE_AC_AC3_G2_PHYSICAL_V1"

export CUDA_VISIBLE_DEVICES="$GPU_ID"
export XLA_PYTHON_CLIENT_PREALLOCATE="false"
exec /mnt/sdc/dty_user/openvla_attack/envs/openvla-official-a800/bin/python \
  "$ROOT/scripts/stage_ac/run_stage_ac3_g2_model_suite.py" \
  --model-family "$MODEL_FAMILY" \
  --suite "$SUITE" \
  --gpu-id "$GPU_ID" \
  --output-dir "$OUT_ROOT/receipts" \
  --video-dir "$OUT_ROOT/videos" \
  --protocol "$ROOT/configs/STAGE_AC_AC3_AC4_AC5_PROGRAM_PROTOCOL_V1.json" \
  --g0-root "$ROOT/reports/STAGE_AC_AC3_G0_ROOT_SEAL_V1.json" \
  --g1-root "$ROOT/reports/STAGE_AC_AC3Q_G1_ROOT_SEAL_V1.json" \
  --runtime-authority "$ROOT/reports/STAGE_AC_AC3Q_RUNTIME_SOURCE_AUTHORITY_V3.json" \
  --manifest "$ROOT/reports/STAGE_AC_AC3_G0_LAUNCH_MANIFEST_V1.json" \
  --blind-sample "$ROOT/reports/STAGE_AC_AC4_BLIND_AUDIT_SAMPLE_V1.json" \
  --config "$ROOT/configs/STAGE_Z_Z1_RUNTIME_PROTOCOL_V11.json"
