#!/usr/bin/env bash
set -euo pipefail

# Future-use wrapper only. Do not run until fresh V5.1 capture is explicitly
# authorized after Layer2 shutdown and GPU requalification.

: "${CONFIRM_V5_1_FRESH_CAPTURE:?set CONFIRM_V5_1_FRESH_CAPTURE=YES}"
[[ "${CONFIRM_V5_1_FRESH_CAPTURE}" == "YES" ]] || { echo "confirmation mismatch"; exit 2; }

: "${EXPECTED_COMMIT:?required}"
: "${EXPECTED_BRANCH:?required}"
: "${EXPECTED_CONFIG_SHA256:?required}"
: "${EXPECTED_LEDGER_SHA256:?required}"
: "${EXPECTED_POOL_CSV_SHA256:?required}"
: "${CUDA_VISIBLE_DEVICES:?required physical index list, e.g. 2,6}"
: "${EXPECTED_GPU_UUIDS:?required ordered UUID list matching CUDA_VISIBLE_DEVICES}"
: "${OUTPUT_DIR:?required new output root}"
: "${RENDER_GPU_DEVICE_ID:?required physical render GPU id}"

PY="/home/liuyu/.conda/envs/openvla_official_libero_20260525/bin/python"
FORBIDDEN_PY="/data/aviary/envs/openvla_official_libero_20260525/bin/python"
[[ -x "${PY}" ]] || { echo "authorized Python env missing: ${PY}"; exit 2; }
[[ "${PY}" != "${FORBIDDEN_PY}" ]] || { echo "forbidden fallback env selected"; exit 2; }
[[ ! -e "${OUTPUT_DIR}" ]] || { echo "OUTPUT_DIR must not already exist"; exit 2; }

test "$(git rev-parse HEAD)" = "${EXPECTED_COMMIT}"
test "$(git rev-parse --abbrev-ref HEAD)" = "${EXPECTED_BRANCH}"
test -z "$(git status --porcelain)"

nvidia-smi --query-gpu=index,uuid,memory.used,memory.total,temperature.gpu --format=csv,noheader > "${OUTPUT_DIR}.gpu_before.txt"

PYTHONHASHSEED=0 \
CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUBLAS_WORKSPACE_CONFIG=:4096:8 \
TOKENIZERS_PARALLELISM=false \
MUJOCO_GL=egl \
PYTHONUNBUFFERED=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
OPENVLA_ATTN_IMPLEMENTATION=eager \
OPENVLA_CUDA_MAX_MEMORY="${OPENVLA_CUDA_MAX_MEMORY:-10000MiB}" \
"${PY}" -u scripts/stageb/run_m3_arm_v5_clean_capture.py \
  --mode capture_clean_pool \
  --config configs/m3_arm_v5_clean_close_event_panel.yaml \
  --output_dir "${OUTPUT_DIR}" \
  --model_gpu_device_id -1 \
  --render_gpu_device_id "${RENDER_GPU_DEVICE_ID}" \
  --max_steps 280 \
  --num_steps_wait 10 \
  --expected_commit "${EXPECTED_COMMIT}" \
  --expected_branch "${EXPECTED_BRANCH}" \
  --expected_config_sha256 "${EXPECTED_CONFIG_SHA256}" \
  --expected_ledger_sha256 "${EXPECTED_LEDGER_SHA256}" \
  --expected_pool_csv_sha256 "${EXPECTED_POOL_CSV_SHA256}" \
  --expected_cuda_visible_devices "${CUDA_VISIBLE_DEVICES}" \
  --expected_gpu_uuids "${EXPECTED_GPU_UUIDS}"

nvidia-smi --query-gpu=index,uuid,memory.used,memory.total,temperature.gpu --format=csv,noheader > "${OUTPUT_DIR}/gpu_after.txt"
