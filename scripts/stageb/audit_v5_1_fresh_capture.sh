#!/usr/bin/env bash
set -euo pipefail

: "${CAPTURE_ROOT:?required}"
: "${EXPECTED_COMMIT:?required capture commit}"
: "${AUDIT_OUTPUT_DIR:?required}"

PY="/home/liuyu/.conda/envs/openvla_official_libero_20260525/bin/python"
[[ -x "${PY}" ]] || { echo "authorized Python env missing: ${PY}"; exit 2; }
mkdir -p "${AUDIT_OUTPUT_DIR}"

"${PY}" scripts/stageb/audit_m3_arm_v5_clean_capture.py \
  --capture_root "${CAPTURE_ROOT}" \
  --config configs/m3_arm_v5_clean_close_event_panel.yaml \
  --expected_commit "${EXPECTED_COMMIT}" \
  --audit_output "${AUDIT_OUTPUT_DIR}/m3_arm_v5_clean_capture_external_audit.json"

find "${AUDIT_OUTPUT_DIR}" -type f -print0 | sort -z | xargs -0 sha256sum > "${AUDIT_OUTPUT_DIR}/recursive_sha256_manifest.txt"
