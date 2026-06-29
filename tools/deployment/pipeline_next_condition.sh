#!/bin/bash
# TRUE_T10 Closure → Next Condition Pipeline
# Run AFTER TRUE_T10 reaches 162/162.
#
# Usage: bash tools/deployment/pipeline_next_condition.sh <CONDITION_ID>
#   CONDITION_ID: RANDOM_TIME | RAND_LINF | SHUFFLED | TMA | UMA | EARLY_SHIFT
#
# Steps:
#   1. Validate TRUE_T10 closure (162/162 check)
#   2. Generate condition manifest from TRUE_T10 template
#   3. Launch condition across idle GPUs

set -euo pipefail

CONDITION="${1:?Usage: $0 <CONDITION_ID>}"
REPO="/mnt/sdc/dty_user/openvla_attack"
EVIDENCE="${REPO}/evidence/sc5_object_privileged_loto_v1/vis_heldout_formal_v1"
PYTHON="${REPO}/envs/openvla-official-a800/bin/python"
TOOLS="${REPO}/tools/deployment"

TRUE_T10_MANIFEST="${EVIDENCE}/TRUE_T10/formal_manifest.jsonl"
CLOSURE_REPORT="${EVIDENCE}/TRUE_T10/launch/CLOSURE_REPORT.json"

echo "========================================="
echo "TRUE_T10 Closure → ${CONDITION} Pipeline"
echo "========================================="

# Step 1: Closure validation
echo ""
echo "[1/3] Validating TRUE_T10 closure..."
${PYTHON} "${TOOLS}/validate_true_t10_closure.py" \
    --manifest "${TRUE_T10_MANIFEST}" \
    --output "${CLOSURE_REPORT}" \
    --fail_on_missing

MISSING=$(python3 -c "import json; print(json.load(open('${CLOSURE_REPORT}'))['missing'])")
if [ "${MISSING}" != "0" ]; then
    echo "ERROR: ${MISSING} episodes missing. Cannot proceed."
    exit 1
fi
echo "  Closure: PASS (162/162)"

# Step 2: Generate condition manifest
echo ""
echo "[2/3] Generating ${CONDITION} manifest..."
COND_MANIFEST="${EVIDENCE}/${CONDITION}/formal_manifest.jsonl"
${PYTHON} "${TOOLS}/build_next_condition_manifest.py" \
    --true_t10_manifest "${TRUE_T10_MANIFEST}" \
    --conditions "${CONDITION}" \
    --output_root "${EVIDENCE}"

JOB_COUNT=$(wc -l < "${COND_MANIFEST}")
echo "  Manifest: ${COND_MANIFEST} (${JOB_COUNT} jobs)"

# Step 3: Launch
echo ""
echo "[3/3] Launching ${CONDITION}..."
LAUNCH_DIR="${EVIDENCE}/${CONDITION}/launch"
${PYTHON} "${TOOLS}/launch_condition.py" \
    --manifest "${COND_MANIFEST}" \
    --condition_id "${CONDITION}" \
    --launch_dir "${LAUNCH_DIR}"

echo ""
echo "Pipeline complete. Monitor:"
echo "  tail -f ${LAUNCH_DIR}/worker_*.log"
echo "  watch -n 30 'find ${EVIDENCE}/${CONDITION}/formal_v1/ -name episode_summary.json | wc -l'"
