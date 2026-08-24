#!/bin/bash
# Full external K10 pipeline for states 35-49 (600 identities)
# Physics Teacher V2.1C → K10 labeler → Factorized Teacher builder
# NO INTERNAL_SIMPLIFIED_V1 fallback.
set -euo pipefail

cd /mnt/sdc/dty_user/openvla_attack
PY=/mnt/sdc/dty_user/openvla_attack/envs/openvla-official-a800/bin/python
export PYTHONPATH=/mnt/sdc/dty_user/openvla_attack/src:$PYTHONPATH

EVIDENCE=/mnt/sdc/dty_user/openvla_attack_evidence/c2g/c2g_cs200_official_v3_20260716
OPS=$EVIDENCE/ops
STAMP=$(date +%Y%m%d)

# Stage 0: Filter registry CSV to states 35-49
echo "=== Stage 0: Filtering registry CSV ==="
REG_CSV=$OPS/OFFICIAL_V3_CAMPAIGN_REGISTRY_V1_d31187f/OFFICIAL_V3_FORMAL_REGISTRY_V1.csv
REG_ROOT=$OPS/OFFICIAL_V3_CAMPAIGN_REGISTRY_V1_d31187f
FILTERED_CSV=/tmp/registry_states_35_49.csv

head -1 "$REG_CSV" > "$FILTERED_CSV"
grep -E 'state_(3[5-9]|4[0-9])/' "$REG_CSV" >> "$FILTERED_CSV" || true
N=$(wc -l < "$FILTERED_CSV")
echo "Filtered registry: $((N-1)) identities"

# Stage 1: Physics Teacher V2.1C for states 35-49
echo "=== Stage 1: Physics Teacher V2.1C ==="
PHYSICS_OUT=$OPS/OFFICIAL_V3_DETECTOR_V5_TEACHER_PHYSICS_V21C_STATES_35_49_${STAMP}
rm -rf "$PHYSICS_OUT" 2>/dev/null || true

$PY scripts/detector_v5/build_v5_physics_teacher.py \
  --registry-csv "$FILTERED_CSV" \
  --registry-root "$REG_ROOT" \
  --decoder-root $OPS/OFFICIAL_V3_PHYSICS_TASK_DECODER_V1_3c53bcd_20260719 \
  --physics-audit-root $OPS/OFFICIAL_V3_PRIVILEGED_PHYSICS_TEACHER_AUDIT_V1_20260718_01 \
  --protocol /mnt/sdc/dty_user/openvla_attack/configs/DETECTOR_V5_FACTORIZED_TEACHER_PROTOCOL_V1.json \
  --output-root "$PHYSICS_OUT" \
  --expected-source-commit 097db75ab50952ba29b99f71b810d91333e293c9

echo "Physics Teacher: $PHYSICS_OUT"

# Stage 2: External K10 labeler
echo "=== Stage 2: External K10 Labeler ==="
K10_OUT=$OPS/OFFICIAL_V3_R7_K10_V122_V21C_STATES_35_49_${STAMP}
rm -rf "$K10_OUT" 2>/dev/null || true

$PY scripts/detector_v4/label_k10_v122_v21c.py \
  --teacher-root "$PHYSICS_OUT" \
  --output-root "$K10_OUT"

echo "K10 Labels: $K10_OUT"

# Stage 3: Factorized Teacher builder (external K10, fail-closed)
echo "=== Stage 3: Factorized Teacher Builder ==="
FACTORIZED_OUT=$OPS/OFFICIAL_V3_DETECTOR_V5_FACTORIZED_TEACHER_STATES_35_49_${STAMP}
rm -rf "$FACTORIZED_OUT" 2>/dev/null || true

$PY scripts/detector_v5/build_v5_factorized_teacher.py \
  --registry-csv "$FILTERED_CSV" \
  --registry-root "$REG_ROOT" \
  --decoder-root $OPS/OFFICIAL_V3_PHYSICS_TASK_DECODER_V1_3c53bcd_20260719 \
  --physics-audit-root $OPS/OFFICIAL_V3_PRIVILEGED_PHYSICS_TEACHER_AUDIT_V1_20260718_01 \
  --protocol /mnt/sdc/dty_user/openvla_attack/configs/DETECTOR_V5_FACTORIZED_TEACHER_PROTOCOL_V1.json \
  --output-root "$FACTORIZED_OUT" \
  --expected-source-commit 097db75ab50952ba29b99f71b810d91333e293c9 \
  --k10-root "$K10_OUT" \
  --expected-k10-schema R7_K10_OPPORTUNITY_LABELER_V1_2_2_V21C_CANONICAL

echo "Factorized Teacher: $FACTORIZED_OUT"

# Stage 4: Validation
echo "=== Stage 4: Validation ==="
$PY -c "
import json, os
labels_dir = '$FACTORIZED_OUT/labels'
count = 0
for suite in os.listdir(labels_dir):
    sp = os.path.join(labels_dir, suite)
    if not os.path.isdir(sp): continue
    for task in os.listdir(sp):
        tp = os.path.join(sp, task)
        if not os.path.isdir(tp): continue
        for state in os.listdir(tp):
            stp = os.path.join(tp, state)
            if os.path.isfile(os.path.join(stp, 'factorized_teacher_v1.jsonl')):
                count += 1
print(f'Factorized teacher labels: {count}/600 identities')
"

echo "=== DONE ==="
echo "Outputs:"
echo "  Physics:   $PHYSICS_OUT"
echo "  K10:       $K10_OUT"
echo "  Factorized: $FACTORIZED_OUT"
