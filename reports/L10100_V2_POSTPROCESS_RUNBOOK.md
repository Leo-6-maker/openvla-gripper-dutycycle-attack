# L10-100 V2 Postprocess Runbook

**Output root**: `/data/liuyu/outputs/milestone_2e5_l10100_parser_v2_privileged_rerun_20260527`

## Prerequisites
- [ ] Both L10-A (tasks 0-4) and L10-B (tasks 5-9) workers exit cleanly
- [ ] 100 episodes (10 tasks × 10 states)
- [ ] No stale/quarantined GPU3 shards included

## Sequence

### 1. Aggregation
```bash
python scripts/aggregate_artifact_rich_manifests.py \
  --input_root /data/liuyu/outputs/milestone_2e5_l10100_parser_v2_privileged_rerun_20260527 \
  --output_table tables/official_clean_artifact_rich_manifest.csv
```
Check: row count = 100.

### 2. Artifact Completeness Audit
Standard completeness check.

### 3. Visual Feature Manifest
Build then extract on GPU7.

### 4. No-Timestep Export
Standard export with leakage check.

### 5. Privileged State Audit
**Key L10-specific checks**:
- Multi-object tasks must have `segments` list populated.
- `selected_segment_id` must reference a valid segment.
- Selected object/target pose must be available from obs or sim.
- `mechanism_type` = `multi_object_transfer` for put-both tasks.
- Pick-place L10 tasks (book+caddy) should have `pick_place_transfer`.
- Articulated L10 tasks (open+put) should abstain.

### 6. Teacher V2 Detector
```bash
python src/utils/detect_contact_window_from_clean.py \
  --step_records_root /data/liuyu/outputs/milestone_2e5_l10100_parser_v2_privileged_rerun_20260527/runs \
  --output tables/l10100_teacher_window_labels.csv
```
Check: segment-level window detection. Multi-object tasks may have windows on each segment.

### 7. Labeled Export
Standard labeled export.

### 8. Segment-Level Mechanism Audit
- Per-segment privileged coverage.
- Per-segment window detection rate.
- Abstain reasons (articulated, multi-object with unresolvable segment, etc.).
- Failed episode FP check.

## Expected Results
- **multi-object tasks** (put both X and Y in Z): privileged coverage for at least one segment, teacher may detect window on first object or both
- **pick-place L10 tasks** (book+caddy): privileged coverage ~100%, window coverage ~success rate
- **articulated L10 tasks**: teacher abstains
- **Failed episodes**: no high-confidence windows

## GPU3 Risk
- GPU3 (PCI:0000:0f:00) had Xid 31 on May 27.
- If fresh Xid appears during L10-B run: quarantine affected shard, note in audit, rerun on stable GPU.
- Check `tables/shard_quarantine_log.csv` before aggregation.
