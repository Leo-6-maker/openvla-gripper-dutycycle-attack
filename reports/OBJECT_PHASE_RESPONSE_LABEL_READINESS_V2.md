# Object Phase Response Label Readiness V2

Status: **NOT GENERATED FROM FULL SOURCES YET**

This placeholder records the current readiness boundary after patching the label builder. It is not a label-generation result and should not be used as detector training evidence.

## Current State

| Item | Status |
|---|---:|
| Label builder supports Batch3b/c CLI inputs | Yes |
| `candidate_role` preservation | Yes |
| `control_type` preservation | Yes |
| `denominator_type` preservation | Yes |
| `action_bridge_confounded` preservation | Yes |
| `manual_review` excluded from train | Yes |
| Duplicate conflict hard fail | Yes |
| Candidate metadata join | Yes |
| Summary/candidate role conflict hard fail | Yes |
| Batch3c missing role excluded from train | Yes |
| Synthetic tests | Pass |
| Full server-source `tables/object_phase_response_labels_v2.csv` | Missing |
| Detector v2 training | BLOCKED |

## Required Next Step

After the server is synced to the reviewed branch/worktree, DeepSeek should run:

```bash
PY=/home/liuyu/.conda/envs/openvla_official_libero_20260525/bin/python
$PY scripts/diagnostics/finalize_phase_response_labels.py \
  --batch1-merged tables/object_teacher_delay50_vis_smoke_merged_summary.csv \
  --batch2b-vis tables/object_phase_response_batch2b_vis_summary.csv \
  --batch3-vis tables/object_phase_response_batch3_vis_summary.csv \
  --batch3b-vis tables/object_phase_response_batch3b_vis_summary.csv \
  --batch3c-vis tables/object_phase_response_batch3c_vis_summary.csv \
  --batch2b-candidates tables/object_phase_response_batch2b_candidates.csv \
  --batch3-candidates tables/object_phase_response_batch3_candidates.csv \
  --batch3b-candidates tables/object_phase_response_batch3b_candidates.csv \
  --batch3c-candidates tables/object_phase_response_batch3c_candidates.csv \
  --output-labels tables/object_phase_response_labels_v2.csv \
  --output-readiness reports/OBJECT_PHASE_RESPONSE_LABEL_READINESS_V2.md \
  --output-conflicts tables/object_phase_response_label_conflicts_v2.csv

$PY scripts/diagnostics/audit_label_schema.py \
  --labels-csv tables/object_phase_response_labels_v2.csv
```

Detector v2 training remains blocked until this full-source run and schema audit pass.
