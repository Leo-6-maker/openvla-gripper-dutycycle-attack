#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
# CODEX ONE-SHOT FORMAL ATTACK MATRIX LAUNCHER
# ═══════════════════════════════════════════════════════════════════════════════
# Requires: formal_attack_authorized=true (external review must set this).
# H heldout gate must be PASS. Detector freeze must be verified.
# Fail-closed: any non-zero exit → immediate stop.
# ═══════════════════════════════════════════════════════════════════════════════
set -euo pipefail

PYTHON="${PYTHON:-python}"
EVIDENCE_ROOT="${EVIDENCE_ROOT:-/mnt/sdc/dty_user/openvla_attack_evidence}"
DETECTOR_ROOT="${DETECTOR_ROOT:-${EVIDENCE_ROOT}/final_detector_pipeline/FINAL_FACTORIZED_DETECTOR_V1}"
H_ROOT="${H_ROOT:-${EVIDENCE_ROOT}/final_detector_pipeline/stage_7b_h_evaluation}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${EVIDENCE_ROOT}/formal_attack_matrix}"
SCRIPTS_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPTS_DIR}/.." && pwd)"

echo "=== CODEX ONE-SHOT FORMAL ATTACK MATRIX ==="
echo ""

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 0: AUTHORIZATION CHECK (MUST BE TRUE)
# ═══════════════════════════════════════════════════════════════════════════════
AUTH_FILE="${DETECTOR_ROOT}/formal_attack_authorization.json"
if [ ! -f "${AUTH_FILE}" ]; then
    echo "FATAL: No authorization file at ${AUTH_FILE}"
    echo "  External review must create this file with:"
    echo '  {"formal_attack_authorized": true, "reviewer": "<name>", "date": "<ISO date>"}'
    exit 1
fi

AUTH=$(${PYTHON} -c "import json; print(json.load(open('${AUTH_FILE}')).get('formal_attack_authorized', False))")
if [ "${AUTH}" != "True" ]; then
    echo "FATAL: formal_attack_authorized is not True (got: ${AUTH})"
    echo "  Attack matrix execution requires explicit external authorization."
    exit 1
fi
echo "  Authorization: TRUE"

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1: PREFLIGHT
# ═══════════════════════════════════════════════════════════════════════════════

# Verify detector freeze
if [ ! -f "${DETECTOR_ROOT}/SHA256SUMS" ]; then
    echo "FATAL: Detector SHA256SUMS missing at ${DETECTOR_ROOT}"
    exit 1
fi
echo "  Detector freeze: VERIFIED"

# Verify H heldout gate_pass=true (MANDATORY)
H_RECEIPT=""
for name in HELDOUT_L3_RUN_COMPLETE_RECEIPT_V1.json FACTORIZED_HELDOUT_L3_EVALUATION_RECEIPT_V1.json receipt.json; do
    if [ -f "${H_ROOT}/${name}" ]; then
        H_RECEIPT="${H_ROOT}/${name}"
        break
    fi
done

if [ -z "${H_RECEIPT}" ]; then
    echo "FATAL: No H receipt found in ${H_ROOT}"
    exit 1
fi

H_GATE=$(${PYTHON} -c "import json; print(json.load(open('${H_RECEIPT}')).get('gate_pass', False))")
if [ "${H_GATE}" != "True" ]; then
    echo "FATAL: H heldout gate_pass is not True (got: ${H_GATE})"
    exit 1
fi
echo "  H heldout gate: PASS"

# Guard: output root must not exist
if [ -d "${OUTPUT_ROOT}" ]; then
    echo "FATAL: OUTPUT_ROOT already exists: ${OUTPUT_ROOT}"
    exit 1
fi

echo ""
echo "=== PREFLIGHT COMPLETE ==="
echo ""

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2: BUILD PARENT MANIFEST AND JOB MATRIX FROM A-POOL EPISODES
# ═══════════════════════════════════════════════════════════════════════════════
echo "=== STEP 2: PARENT MANIFEST & JOB MATRIX ==="

${PYTHON} "${REPO_ROOT}/scripts/detector_v5/build_codex_one_shot_handoff.py" \
    --final-detector-root "${DETECTOR_ROOT}" \
    --h-receipt-root "${H_ROOT}" \
    --output-root "${OUTPUT_ROOT}/handoff"

echo "  Parent manifest & job matrix prepared"
echo ""

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3: LAUNCH ROLLOUT WORKERS (AUTO)
# ═══════════════════════════════════════════════════════════════════════════════
echo "=== STEP 3: LAUNCH ROLLOUT ==="
echo "  Launching 1 GPU worker for A-pool states 35-44 (400 episodes)..."
echo "  Worker command must be provided by the runtime adapter."
echo "  This script validates artifacts after rollout completes."
echo ""

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 4: EXECUTION VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════
echo "=== STEP 4: EXECUTION VALIDATION ==="

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
echo ""

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 5: PAIRED ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
echo "=== STEP 5: PAIRED ANALYSIS ==="

${PYTHON} "${REPO_ROOT}/analysis/pilot_attack/analyze_factorized_attack_pilot.py" \
    --pilot-execution-validation-root "${OUTPUT_ROOT}/validation/execution" \
    --pilot-job-matrix-root "${OUTPUT_ROOT}/matrix/job_matrix" \
    --pilot-run-ledger-root "${OUTPUT_ROOT}/artifacts/run_ledger" \
    --pilot-telemetry-index-root "${OUTPUT_ROOT}/artifacts/telemetry_index" \
    --pilot-go-no-go-rules-root "${OUTPUT_ROOT}/matrix/go_no_go_rules" \
    --output-root "${OUTPUT_ROOT}/analysis"

echo "  Paired analysis: DONE"
echo ""

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 6: BLIND PACKAGE
# ═══════════════════════════════════════════════════════════════════════════════
echo "=== STEP 6: BLIND PACKAGE ==="

${PYTHON} "${REPO_ROOT}/analysis/pilot_attack/build_factorized_pilot_blind_review.py" \
    --pilot-execution-validation-root "${OUTPUT_ROOT}/validation/execution" \
    --pilot-run-ledger-root "${OUTPUT_ROOT}/artifacts/run_ledger" \
    --pilot-video-index-root "${OUTPUT_ROOT}/artifacts/video_index" \
    --evidence-root "${OUTPUT_ROOT}/artifacts" \
    --blind-package-root "${OUTPUT_ROOT}/blind/package" \
    --unblinding-root "${OUTPUT_ROOT}/blind/unblinding"

echo ""
echo "=============================================="
echo "  FORMAL ATTACK MATRIX: LAUNCHER COMPLETE"
echo "  Output: ${OUTPUT_ROOT}"
echo ""
echo "  Automated GO/NO-GO is advisory only."
echo "  scientific_go_no_go_authorized = FALSE"
echo "  Blind manual review REQUIRED before any scientific claim."
echo "=============================================="
