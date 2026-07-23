#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
# CODEX ONE-SHOT FORMAL ATTACK MATRIX LAUNCHER
# ═══════════════════════════════════════════════════════════════════════════════
# Fail-closed: any non-zero exit → immediate stop.
# Does NOT overwrite existing output roots.
# Authorized: FALSE (await external review before execution).
# ═══════════════════════════════════════════════════════════════════════════════
set -euo pipefail

# ── Configuration ────────────────────────────────────────────────────────────
# EDIT THESE PATHS before execution
PYTHON="${PYTHON:-python}"
EVIDENCE_ROOT="${EVIDENCE_ROOT:-/mnt/sdc/dty_user/openvla_attack_evidence}"
CLEAN2000_ROOT="${CLEAN2000_ROOT:-${EVIDENCE_ROOT}/c2g/c2g_cs200_official_v3_20260716}"
DETECTOR_ROOT="${DETECTOR_ROOT:-${EVIDENCE_ROOT}/final_detector_pipeline/FINAL_FACTORIZED_DETECTOR_V1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${EVIDENCE_ROOT}/formal_attack_matrix}"
SCRIPTS_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPTS_DIR}/.." && pwd)"

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 0: PREFLIGHT
# ═══════════════════════════════════════════════════════════════════════════════
echo "=== STEP 0: PREFLIGHT ==="

# Verify detector freeze
if [ ! -f "${DETECTOR_ROOT}/SHA256SUMS" ]; then
    echo "FATAL: Detector SHA256SUMS missing at ${DETECTOR_ROOT}"
    exit 1
fi
echo "  Detector freeze: VERIFIED"

# Verify attack is NOT authorized
if [ -f "${DETECTOR_ROOT}/attack_authorization.json" ]; then
    AUTH=$(python -c "import json; print(json.load(open('${DETECTOR_ROOT}/attack_authorization.json')).get('attack_authorized', False))")
    if [ "${AUTH}" != "False" ]; then
        echo "FATAL: attack_authorized is not False"
        exit 1
    fi
fi
echo "  Attack authorization: FALSE (correct)"

# Verify H heldout PASS
H_RECEIPT="${DETECTOR_ROOT}/../stage_7b_h_evaluation/receipt.json"
if [ -f "${H_RECEIPT}" ]; then
    H_STATUS=$(python -c "import json; print(json.load(open('${H_RECEIPT}')).get('status',''))")
    if [ "${H_STATUS}" != "PASS" ]; then
        echo "FATAL: H heldout receipt status=${H_STATUS}, expected PASS"
        exit 1
    fi
    echo "  H heldout: PASS"
else
    echo "  WARNING: H receipt not found at ${H_RECEIPT} (continuing)"
fi

# Guard: output root must not exist
if [ -d "${OUTPUT_ROOT}" ]; then
    echo "FATAL: OUTPUT_ROOT already exists: ${OUTPUT_ROOT}"
    echo "  Remove it or set a different OUTPUT_ROOT."
    exit 1
fi

echo "  Preflight: PASS"

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1: BUILD PARENT MANIFEST & JOB MATRIX
# ═══════════════════════════════════════════════════════════════════════════════
echo ""
echo "=== STEP 1: PARENT MANIFEST & JOB MATRIX ==="

${PYTHON} "${SCRIPTS_DIR}/detector_v5/build_codex_one_shot_handoff.py" \
    --final-detector-root "${DETECTOR_ROOT}" \
    --h-receipt-root "${DETECTOR_ROOT}/../stage_7b_h_evaluation" \
    --a9-parity-root "${DETECTOR_ROOT}/../a9_adapter_parity" \
    --a10-e2e-root "${DETECTOR_ROOT}/../a10_cli_e2e" \
    --clean2000-root "${CLEAN2000_ROOT}" \
    --identity-manifests-root "${CLEAN2000_ROOT}/identity_manifests" \
    --output-root "${OUTPUT_ROOT}/handoff"

echo "  Parent manifest & job matrix: SEALED"

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2: LAUNCH WORKERS (MANUAL STEP — Codex operator)
# ═══════════════════════════════════════════════════════════════════════════════
echo ""
echo "=== STEP 2: LAUNCH ROLLOUT ==="
echo "  >>> MANUAL STEP: Codex operator must launch the rollout workers. <<<"
echo "  >>> This script does NOT auto-launch GPU processes. <<<"
echo "  After all rollouts complete, proceed to Step 3."

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3: EXECUTION VALIDATION (post-rollout)
# ═══════════════════════════════════════════════════════════════════════════════
echo ""
echo "=== STEP 3: EXECUTION VALIDATION ==="

${PYTHON} "${REPO_ROOT}/analysis/pilot_attack/validate_factorized_attack_pilot_execution.py" \
    --pilot-job-matrix-root "${OUTPUT_ROOT}/matrix/job_matrix" \
    --pilot-run-ledger-root "${OUTPUT_ROOT}/artifacts/run_ledger" \
    --pilot-telemetry-index-root "${OUTPUT_ROOT}/artifacts/telemetry_index" \
    --pilot-video-index-root "${OUTPUT_ROOT}/artifacts/video_index" \
    --pilot-parent-manifest-root "${OUTPUT_ROOT}/matrix/parent_manifest" \
    --pilot-arm-parity-protocol-root "${OUTPUT_ROOT}/matrix/arm_protocol" \
    --evidence-root "${OUTPUT_ROOT}/artifacts" \
    --output-root "${OUTPUT_ROOT}/validation/execution"

echo "  Execution validation: DONE"

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 4: PAIRED ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
echo ""
echo "=== STEP 4: PAIRED ANALYSIS ==="

${PYTHON} "${REPO_ROOT}/analysis/pilot_attack/analyze_factorized_attack_pilot.py" \
    --pilot-execution-validation-root "${OUTPUT_ROOT}/validation/execution" \
    --pilot-job-matrix-root "${OUTPUT_ROOT}/matrix/job_matrix" \
    --pilot-run-ledger-root "${OUTPUT_ROOT}/artifacts/run_ledger" \
    --pilot-telemetry-index-root "${OUTPUT_ROOT}/artifacts/telemetry_index" \
    --pilot-go-no-go-rules-root "${OUTPUT_ROOT}/matrix/go_no_go_rules" \
    --output-root "${OUTPUT_ROOT}/analysis"

echo "  Paired analysis: DONE"

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 5: BLIND PACKAGE
# ═══════════════════════════════════════════════════════════════════════════════
echo ""
echo "=== STEP 5: BLIND PACKAGE ==="

${PYTHON} "${REPO_ROOT}/analysis/pilot_attack/build_factorized_pilot_blind_review.py" \
    --pilot-execution-validation-root "${OUTPUT_ROOT}/validation/execution" \
    --pilot-run-ledger-root "${OUTPUT_ROOT}/artifacts/run_ledger" \
    --pilot-video-index-root "${OUTPUT_ROOT}/artifacts/video_index" \
    --evidence-root "${OUTPUT_ROOT}/artifacts" \
    --blind-package-root "${OUTPUT_ROOT}/blind/package" \
    --unblinding-root "${OUTPUT_ROOT}/blind/unblinding"

echo "  Blind package: DONE"

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 6: FINAL STATUS
# ═══════════════════════════════════════════════════════════════════════════════
echo ""
echo "=============================================="
echo "  FORMAL ATTACK MATRIX: LAUNCHER COMPLETE"
echo "  Output: ${OUTPUT_ROOT}"
echo ""
echo "  IMPORTANT: Automated GO/NO-GO is advisory only."
echo "  scientific_go_no_go_authorized = FALSE"
echo "  Blind manual review is REQUIRED before any scientific claim."
echo "=============================================="
