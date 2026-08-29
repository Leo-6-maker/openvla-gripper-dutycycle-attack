#!/usr/bin/env bash
set -euo pipefail

root=/mnt/sdc/dty_user/openvla_attack_worktrees/stage-ac2-clean-screen-c217acfc-lf
out=/mnt/sdc/dty_user/openvla_attack_outputs/STAGE_AC_AC3Q_ENGINEERING_V1
family=$1
key=$2
gpu=$3
attempt=${4:-1}
tag=${family}__${key//\//__}__attempt_${attempt}

export CUDA_VISIBLE_DEVICES="$gpu"
export PYTHONUNBUFFERED=1
if [[ "$family" == M2_PI05_LIBERO ]]; then
  export XLA_PYTHON_CLIENT_PREALLOCATE=false
fi
exec /mnt/sdc/dty_user/openvla_attack/envs/openvla-official-a800/bin/python \
  "$root/scripts/stage_ac/run_stage_ac3q_engineering_canary.py" \
  --model-family "$family" --canonical-parent-key "$key" --gpu-id "$gpu" \
  --output "$out/receipts/$tag.json" \
  --point-ledger "$out/points/$tag.json" \
  --video-dir "$out/videos/$tag"
