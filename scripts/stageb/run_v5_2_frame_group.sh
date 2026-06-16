#!/usr/bin/env bash
set -euo pipefail

# Future-use wrapper. This script intentionally fails unless explicit V5.2 GPU
# authorization is supplied after V5.1 exact inputs are frozen and reviewed.

: "${CONFIRM_V5_2_FRAME_GROUP:?set CONFIRM_V5_2_FRAME_GROUP=YES after authorization}"
[[ "${CONFIRM_V5_2_FRAME_GROUP}" == "YES" ]] || { echo "confirmation mismatch"; exit 2; }

: "${EXPECTED_COMMIT:?required}"
: "${EXPECTED_BRANCH:?required}"
: "${FROZEN_INPUT_CSV:?required V5.1 frozen eight-frame CSV}"
: "${OUTPUT_DIR:?required new output root}"
: "${CUDA_VISIBLE_DEVICES:?required}"
: "${EXPECTED_GPU_UUIDS:?required ordered UUID list}"
: "${FRAME_IDS:?required exact frozen frame ids}"
: "${ATTACK_SEED:?required}"

[[ "${ATTACK_SEED}" == "428198" ]] || { echo "V5.2 seed must be 428198"; exit 2; }
[[ ! -e "${OUTPUT_DIR}" ]] || { echo "OUTPUT_DIR must not already exist"; exit 2; }
test "$(git rev-parse HEAD)" = "${EXPECTED_COMMIT}"
test "$(git rev-parse --abbrev-ref HEAD)" = "${EXPECTED_BRANCH}"
test -z "$(git status --porcelain)"

PY="/home/liuyu/.conda/envs/openvla_official_libero_20260525/bin/python"
[[ -x "${PY}" ]] || { echo "authorized Python env missing: ${PY}"; exit 2; }

echo "V5.2 real GPU runner is intentionally not wired in this seal commit."
echo "Use this wrapper as the fail-closed gate template when V5.2 is authorized."
exit 3
