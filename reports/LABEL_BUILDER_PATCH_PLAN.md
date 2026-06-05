# Label Builder Patch Status

Audit target: `scripts/diagnostics/finalize_phase_response_labels.py`

## Verdict

The label builder has been patched for v2 multi-source label building, but detector v2 training remains blocked until the server is synced and a real `object_phase_response_labels_v2.csv` is generated from Batch3b/Batch3c sources.

It now supports:

- `--batch1-merged`
- `--batch2b-vis`
- `--batch3-vis`
- `--batch3b-vis`
- `--batch3c-vis`
- `--output-labels`
- `--output-readiness`
- `--output-conflicts`

## Implemented

| Requirement | Status | Notes |
|---|---:|---|
| Multi-source CSV reading | DONE | Batch1, Batch2b, Batch3, Batch3b, Batch3c. |
| `source_batch` preservation | DONE | Output includes `source_batch`. |
| `candidate_role` preservation | DONE | Output includes `candidate_role`. |
| `control_type` preservation | DONE | Output includes `control_type`. |
| `denominator_type` preservation | DONE | Output includes `denominator_type`. |
| `action_bridge_confounded` preservation | DONE | Output includes `action_bridge_confounded`. |
| Role-specific taxonomy support | DONE | Uses `role_specific_gates.py` for stable_post_lock, far_too_early, and pre_lock controls. |
| Duplicate conflict hard fail | DONE | Writes conflict CSV and exits nonzero. |
| Train/ignore/manual_review separation | DONE | Only positive/negative rows get `label_use=train`; manual_review does not enter train. |
| v2 fixed 9-label assertions removed | DONE | No hardcoded Batch2b count assertions in normal v2 mode. |

## Current Blocker

No full-source v2 label file has been generated in this local review because:

- Server checkout is still on `exp/vis-payload-upgrade-validation-20260601`.
- Server does not yet have the reviewed commit.
- Real Batch3b/Batch3c source CSVs are not available in the local reviewed checkout.

Therefore, `tables/object_phase_response_labels_v2.csv` should still be treated as missing for detector v2 readiness.

## Intended Server Command

```bash
PY=/home/liuyu/.conda/envs/openvla_official_libero_20260525/bin/python
$PY scripts/diagnostics/finalize_phase_response_labels.py \
  --batch1-merged tables/object_teacher_delay50_vis_smoke_merged_summary.csv \
  --batch2b-vis tables/object_phase_response_batch2b_vis_summary.csv \
  --batch3-vis tables/object_phase_response_batch3_vis_summary.csv \
  --batch3b-vis tables/object_phase_response_batch3b_vis_summary.csv \
  --batch3c-vis tables/object_phase_response_batch3c_vis_summary.csv \
  --output-labels tables/object_phase_response_labels_v2.csv \
  --output-readiness reports/OBJECT_PHASE_RESPONSE_LABEL_READINESS_V2.md \
  --output-conflicts tables/object_phase_response_label_conflicts_v2.csv

$PY scripts/diagnostics/audit_label_schema.py \
  --labels-csv tables/object_phase_response_labels_v2.csv
```

## Synthetic Tests

Added `tests/diagnostics/test_finalize_phase_response_labels_v2.py`.

Covered cases:

- Batch3b/Batch3c CLI source wiring.
- stable_post_lock done=False becomes manual_review and not train.
- far_too_early done=True/strong becomes negative.
- pre_lock done=False/strong becomes positive.
- duplicate task/state/window conflicting labels hard fail and write conflict CSV.
