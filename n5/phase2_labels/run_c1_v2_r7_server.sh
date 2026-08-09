#!/bin/bash
# [DeepSeek] C1-V2-R7 Server Execution Script
# Run on Linux server: bash n5/phase2_labels/run_c1_v2_r7_server.sh
set -euo pipefail

PYTHON=/mnt/sdc/dty_user/openvla_attack/envs/openvla-official-a800/bin/python
REPO=/mnt/sdc/dty_user/openvla_attack
BRANCH=deepseek/detector-grec-r3-20260727
OUT_BASE=/mnt/sdc/dty_user/openvla_attack_outputs/n5/phase3_student/c1_v2_r7
TIMESTAMP=$(date -u +%Y%m%dT%H%M%SZ)
LOGDIR="${OUT_BASE}/logs_${TIMESTAMP}"

echo "============================================================"
echo "[DeepSeek] C1-V2-R7 Server Execution"
echo "  branch: ${BRANCH}"
echo "  timestamp: ${TIMESTAMP}"
echo "  python: ${PYTHON}"
echo "============================================================"

# ── Stage 0: Pull latest code ──
echo ""
echo "--- Stage 0: Code sync ---"
cd "${REPO}"
git fetch origin
git checkout "${BRANCH}"
git pull origin "${BRANCH}"
COMMIT=$(git rev-parse HEAD)
TREE=$(git rev-parse HEAD^{tree})
echo "  commit: ${COMMIT}"
echo "  tree: ${TREE}"
echo "  clean: $(git status --porcelain -- n5/ | wc -l) untracked"

mkdir -p "${LOGDIR}"

# ── Stage 1: Full test suite (62 tests required) ──
echo ""
echo "--- Stage 1: Full test suite ---"

echo "  [1/4] test_r5_c1_contract.py (pure + integration)..."
${PYTHON} -m pytest n5/phase3_student/tests/test_r5_c1_contract.py -v 2>&1 | tee "${LOGDIR}/test_r5_c1_contract.log"
echo "  EXIT: $?"

echo "  [2/4] test_c1_v2_resolver.py..."
${PYTHON} -m pytest n5/phase3_student/tests/test_c1_v2_resolver.py -v 2>&1 | tee "${LOGDIR}/test_c1_v2_resolver.log"
echo "  EXIT: $?"

echo "  [3/4] test_comparator_mutations.py..."
${PYTHON} n5/phase2_labels/test_comparator_mutations.py 2>&1 | tee "${LOGDIR}/test_comparator_mutations.log"
echo "  EXIT: $?"

echo "  [4/4] py_compile all changed scripts..."
for f in \
    n5/phase2_labels/run_grec_fit_geometry_fallback_canary.py \
    n5/phase2_labels/run_r5e_same_live_gate.py \
    n5/phase2_labels/run_r5f_full40_materialize.py \
    n5/phase2_labels/compare_r5_canonical.py \
    n5/phase3_student/t2rc1_v2_registry.py \
    n5/phase3_student/tests/test_r5_c1_contract.py \
    n5/phase3_student/tests/test_c1_v2_resolver.py \
    n5/phase2_labels/test_comparator_mutations.py \
    n5/phase2_labels/run_c1_v2_r7_server.sh; do
    ${PYTHON} -c "import py_compile; py_compile.compile('${f}', doraise=True)" && echo "  COMPILE OK: ${f}" || echo "  COMPILE FAIL: ${f}"
done | tee "${LOGDIR}/py_compile.log"

# Count test results
echo ""
echo "--- Test summary ---"
TOTAL=$(grep -c "PASSED\|PASS\|OK\|passed" "${LOGDIR}/test_r5_c1_contract.log" "${LOGDIR}/test_c1_v2_resolver.log" 2>/dev/null || echo "COUNT_FAILED")
echo "  See ${LOGDIR}/ for full logs"

# ── Stage 2: C1-V2-R7 Run A ──
echo ""
echo "--- Stage 2: C1-V2-R7 Run A ---"
RUN_A="${OUT_BASE}/run_A"
if [ -d "${RUN_A}" ]; then
    echo "  FATAL: run_A exists: ${RUN_A}"
    exit 1
fi
C1_V2_OUT="${RUN_A}" ${PYTHON} n5/phase3_student/t2rc1_v2_registry.py 2>&1 | tee "${LOGDIR}/c1_v2_r7_run_A.log"
echo "  Run A exit: $?"

# ── Stage 3: C1-V2-R7 Run B ──
echo ""
echo "--- Stage 3: C1-V2-R7 Run B ---"
RUN_B="${OUT_BASE}/run_B"
if [ -d "${RUN_B}" ]; then
    echo "  FATAL: run_B exists: ${RUN_B}"
    exit 1
fi
C1_V2_OUT="${RUN_B}" ${PYTHON} n5/phase3_student/t2rc1_v2_registry.py 2>&1 | tee "${LOGDIR}/c1_v2_r7_run_B.log"
echo "  Run B exit: $?"

# ── Stage 4: C1 comparator ──
echo ""
echo "--- Stage 4: C1 comparator ---"
${PYTHON} n5/phase2_labels/compare_r5_canonical.py \
    --root-a "${RUN_A}" \
    --root-b "${RUN_B}" \
    --gate c1 2>&1 | tee "${LOGDIR}/c1_comparator.log"
COMP_EXIT=$?
echo "  Comparator exit: ${COMP_EXIT}"

# ── Final report ──
echo ""
echo "============================================================"
echo "[DeepSeek] C1-V2-R7 Execution Complete"
echo "  commit: ${COMMIT}"
echo "  tree: ${TREE}"
echo "  run_A: ${RUN_A}"
echo "  run_B: ${RUN_B}"
echo "  logs: ${LOGDIR}"
echo "  comparator_exit: ${COMP_EXIT}"
echo "============================================================"
echo ""
echo "SHA256SUMS verification:"
sha256sum -c "${RUN_A}/SHA256SUMS.sha256" 2>&1 || echo "  RUN_A SEAL FAIL"
sha256sum -c "${RUN_B}/SHA256SUMS.sha256" 2>&1 || echo "  RUN_B SEAL FAIL"

exit ${COMP_EXIT}
