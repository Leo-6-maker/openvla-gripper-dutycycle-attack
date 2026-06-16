#!/usr/bin/env bash
set -euo pipefail

: "${ARTIFACT_ROOT:?required}"
: "${FRAME_IDS:?required comma-separated frozen frame ids}"
: "${AUDIT_OUTPUT:?required}"
: "${ATTACK_SEED:?required}"

[[ "${ATTACK_SEED}" == "428198" ]] || { echo "V5.2 seed must be 428198"; exit 2; }
PY="${PYTHON:-python}"

PYTHONPATH=. "${PY}" scripts/stageb/audit_m3_arm_v5_frame_group.py \
  --artifact_root "${ARTIFACT_ROOT}" \
  --frame_ids "${FRAME_IDS}" \
  --seed "${ATTACK_SEED}" \
  --audit_output "${AUDIT_OUTPUT}"
