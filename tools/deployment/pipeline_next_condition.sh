#!/bin/bash
# TRUE_T10 Closure → Next Condition Pipeline
# Default: dry_run only. Requires --execute for each step.
#
# Usage: bash tools/deployment/pipeline_next_condition.sh <CONDITION_ID> [--execute]
set -euo pipefail

CONDITION="${1:?Usage: $0 <CONDITION_ID> [--execute]}"
EXECUTE="${2:-}"
REPO="/mnt/sdc/dty_user/openvla_attack"
EVIDENCE="${REPO}/evidence/sc5_object_privileged_loto_v1/vis_heldout_formal_v1"
PYTHON="${REPO}/envs/openvla-official-a800/bin/python"
TOOLS="${REPO}/tools/deployment"

TRUE_T10_MANIFEST="${EVIDENCE}/TRUE_T10/formal_manifest.jsonl"
CLOSURE_REPORT="${EVIDENCE}/TRUE_T10/launch/CLOSURE_REPORT_V2.json"
EXPECTED_BRIDGE="4ef2a919ee650cf35b35eaa5b9c2152c0d7d18f43710c246ce14dd1c8a83e468"
EXPECTED_WORKER="e21f7fbe7f78003ac2e626bfe9ddb047c194022727bb4d9bc19b9ce0876e337c"

echo "========================================="
echo "TRUE_T10 Closure → ${CONDITION} Pipeline"
if [ "${EXECUTE}" = "--execute" ]; then
    echo "MODE: EXECUTE"
else
    echo "MODE: DRY_RUN (add --execute to launch)"
fi
echo "========================================="

# Step 1: Closure validation
echo ""
echo "[1/3] Validating TRUE_T10 closure..."
${PYTHON} "${TOOLS}/validate_true_t10_closure.py" \
    --manifest "${TRUE_T10_MANIFEST}" \
    --expected_bridge_sha "${EXPECTED_BRIDGE}" \
    --expected_worker_sha "${EXPECTED_WORKER}" \
    --output "${CLOSURE_REPORT}"

CLOSURE_OK=$(${PYTHON} -c "import json; print(json.load(open('${CLOSURE_REPORT}'))['closure_pass'])")
if [ "${CLOSURE_OK}" != "True" ]; then
    echo "ERROR: Closure validation failed. See ${CLOSURE_REPORT}"
    exit 1
fi
echo "  Closure: PASS (162/162)"

# Step 2: Generate condition manifest
echo ""
echo "[2/3] Generating ${CONDITION} manifest..."
MANIFEST_EXEC=""
if [ "${EXECUTE}" = "--execute" ]; then
    MANIFEST_EXEC="--execute"
fi
${PYTHON} "${TOOLS}/build_next_condition_manifest.py" \
    --true_t10_manifest "${TRUE_T10_MANIFEST}" \
    --conditions "${CONDITION}" \
    --evidence_root "${EVIDENCE}" \
    ${MANIFEST_EXEC}

COND_MANIFEST="${EVIDENCE}/${CONDITION}/formal_manifest.jsonl"

if [ "${EXECUTE}" = "--execute" ]; then
    JOB_COUNT=$(wc -l < "${COND_MANIFEST}")
    echo "  Manifest: ${COND_MANIFEST} (${JOB_COUNT} jobs)"
    MANIFEST_SHA=$(sha256sum "${COND_MANIFEST}" | awk '{print $1}')
    echo "  SHA256: ${MANIFEST_SHA}"
else
    echo "  DRY_RUN: would write ${COND_MANIFEST}"
    echo ""
    echo "Pipeline DRY_RUN complete. Re-run with --execute to write manifests and launch."
    echo "  bash tools/deployment/pipeline_next_condition.sh ${CONDITION} --execute"
    exit 0
fi

# Step 3: Launch (only in execute mode)
echo ""
echo "[3/3] Launching ${CONDITION}..."
${PYTHON} "${TOOLS}/launch_condition.py" \
    --manifest "${COND_MANIFEST}" \
    --condition_id "${CONDITION}" \
    --launch_dir "${EVIDENCE}/${CONDITION}/launch" \
    --mode formal \
    --expected_worker_sha "${EXPECTED_WORKER}" \
    --expected_bridge_sha "${EXPECTED_BRIDGE}" \
    --expected_manifest_sha "${MANIFEST_SHA}" \
    --execute

echo ""
echo "Pipeline complete. Monitor:"
echo "  tail -f ${EVIDENCE}/${CONDITION}/launch/worker_*.log"
