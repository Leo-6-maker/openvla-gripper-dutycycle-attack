#!/usr/bin/env bash
set -euo pipefail

# CPU/mock only. This does not run OpenVLA, PGD, RAND, shuffled-gradient, GPU,
# or LIBERO. It validates the V5.2 artifact layout with synthetic candidates.

: "${OUTPUT_DIR:?required new output root}"
: "${FRAME_IDS:?required comma-separated development frame ids}"
PY="${PYTHON:-python}"
[[ ! -e "${OUTPUT_DIR}" ]] || { echo "OUTPUT_DIR must not already exist"; exit 2; }

PYTHONPATH=. "${PY}" scripts/stageb/run_m3_arm_v5_frame_group.py \
  --mode mock_zero_perturbation \
  --output_dir "${OUTPUT_DIR}" \
  --frame_ids "${FRAME_IDS}" \
  --seed 428198

PYTHONPATH=. "${PY}" scripts/stageb/audit_m3_arm_v5_frame_group.py \
  --artifact_root "${OUTPUT_DIR}" \
  --frame_ids "${FRAME_IDS}" \
  --seed 428198 \
  --audit_output "${OUTPUT_DIR}/independent_audit.json"
