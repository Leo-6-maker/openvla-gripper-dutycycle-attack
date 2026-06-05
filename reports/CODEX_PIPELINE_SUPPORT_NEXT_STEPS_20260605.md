# Codex Pipeline Support Next Steps — 2026-06-05

## Scope

Codex added CPU-only support for receiving Phase E qpos cache outputs, Batch4 full-VIS outputs, labels_v3 candidates, detector v3 diagnostic gating, and status reporting.

No GPU, VIS, rollout, watcher, detector training, or server output mutation was run.

## Scripts Ready For DeepSeek

### Phase E qpos cache

Run after the GPU2,6 cache export finishes:

```bash
python scripts/diagnostics/audit_phase_e_qpos_cache.py \
  --cache-root /data/liuyu/outputs/phaseE_mujoco_qpos_cache_20260605 \
  --candidates tables/fast_vis_calibration_candidates_v0.csv \
  --output-csv tables/phaseE_qpos_cache_audit_v0.csv \
  --output-report reports/PHASE_E_QPOS_CACHE_AUDIT_V0.md \
  --dry-run
```

Only `cache_status=ok` rows with MuJoCo qpos are safe for Phase E aligned-window generation. Obs-only or all-zero obs qpos remains audit-only.

### Phase E aligned-window audit

Run before any Phase E GPU canary:

```bash
python scripts/diagnostics/audit_phase_e_aligned_windows.py \
  --aligned-windows tables/phaseE_aligned_windows_v0_server.csv \
  --output-csv tables/phaseE_aligned_windows_audit_v0.csv \
  --output-report reports/PHASE_E_ALIGNED_WINDOWS_AUDIT_V0.md
```

Canary is ready only if recommended rows use MuJoCo qpos and include at least one positive and one negative recommended row.

### Batch4 closeout

Run after Batch4 full VIS traces arrive:

```bash
python scripts/diagnostics/finalize_batch4_vis.py \
  --candidates tables/object_phase_response_batch4_candidates.csv \
  --precheck tables/object_phase_response_batch4_precheck_summary.csv \
  --vis-root /data/liuyu/outputs/object_phase_response_batch4_fullVIS_20260605 \
  --output-summary tables/object_phase_response_batch4_vis_summary.csv \
  --output-provenance tables/object_phase_response_batch4_vis_provenance.csv \
  --output-report reports/OBJECT_PHASE_RESPONSE_BATCH4_VIS_SUMMARY.md \
  --dry-run
```

Full VIS only is gold. Phase D/E outputs must remain excluded.

### labels_v3 candidate

Run only after Batch4 closeout has a usable full-VIS summary:

```bash
python scripts/diagnostics/build_labels_v3_candidate.py \
  --labels-v2 tables/object_phase_response_labels_v2.csv \
  --batch4-summary tables/object_phase_response_batch4_vis_summary.csv \
  --batch4-candidates tables/object_phase_response_batch4_candidates.csv \
  --output-labels tables/object_phase_response_labels_v3_candidate.csv \
  --output-conflicts tables/object_phase_response_labels_v3_conflicts.csv \
  --output-readiness reports/OBJECT_PHASE_RESPONSE_LABEL_READINESS_V3_CANDIDATE.md \
  --dry-run
```

Detector v3 remains blocked unless readiness says `READY_FOR_DETECTOR_V3`.

### Detector v3 diagnostic scaffold

```bash
python scripts/diagnostics/run_detector_v3_diagnostic.py --dry-run
```

This scaffold does not train. It reports `BLOCKED_NOT_READY` unless labels_v3 readiness passes.

## Current Local Blockers

- Phase E qpos cache is missing locally, so the qpos cache audit reports `BLOCKED_MISSING_QPOS_CACHE`.
- Batch4 full VIS outputs are missing locally, so Batch4 closeout reports `BLOCKED_MISSING_BATCH4_OUTPUTS`.
- labels_v3 candidate is not ready without labels_v2 plus Batch4 full-VIS gold rows.
- Detector v3 diagnostic is blocked until labels_v3 readiness passes.

## Safety Boundaries

- Do not use GPU3/GPU7.
- Do not merge Phase D/E into labels_v2/v3.
- Do not mark Phase E as gold.
- Do not use obs-only qpos for automatic Phase E recommendation.
- Do not count infra_failed or missing trace rows as negatives.
