# Goal-100 V2 Postprocess Runbook

**Output root**: `/data/liuyu/outputs/milestone_2e5_goal100_parser_v2_privileged_rerun_20260527`

## Prerequisites
- [ ] Worker exits cleanly
- [ ] 100 episodes (10 tasks × 10 states)
- [ ] run_manifest.json per episode present
- [ ] step_records.jsonl per episode present
- [ ] RGB frames saved per step (valid PNG files)

## Sequence

### 1. Aggregation
```bash
python scripts/aggregate_artifact_rich_manifests.py \
  --input_root /data/liuyu/outputs/milestone_2e5_goal100_parser_v2_privileged_rerun_20260527 \
  --output_table tables/official_clean_artifact_rich_manifest.csv
```
Check: row count = 100, no missing step_records.

### 2. Artifact Completeness Audit
```bash
python scripts/audit_artifact_completeness.py \
  --manifest tables/official_clean_artifact_rich_manifest.csv \
  --output tables/artifact_completeness_summary.csv
```
Check: rgb_coverage ≥ 0.95, step_records_coverage = 1.0.

### 3. Visual Feature Manifest
Build image list from artifact manifest, then extract on GPU7.
```bash
python scripts/build_visual_feature_manifest.py \
  --manifest tables/official_clean_artifact_rich_manifest.csv \
  --output tables/visual_feature_manifest.csv
```
Then: GPU7 extraction (separate step).

### 4. No-Timestep Export
```bash
python scripts/export_no_timestep_visual_proprio_dataset.py \
  --manifest tables/official_clean_artifact_rich_manifest.csv \
  --output tables/no_timestep_visual_proprio_student_dataset.csv
```
Check: no TEACHER_ONLY_FIELDS in columns.

### 5. Privileged State Audit
**Key expectation**: articulated tasks (drawer, turn-on) should show `gripper_duty_eligible=false` / `mechanism=articulated_object`. Pick-place tasks should have `object_pose_json` and `target_pose_json` populated.
- Count privileged_state_available=True by mechanism.
- Verify articulated tasks abstain (privileged_state_error contains "mechanism=articulated").
- Verify failed episodes don't differ in privileged coverage.

### 6. Teacher V2 Detector
```bash
python src/utils/detect_contact_window_from_clean.py \
  --step_records_root /data/liuyu/outputs/milestone_2e5_goal100_parser_v2_privileged_rerun_20260527/runs \
  --output tables/goal100_teacher_window_labels.csv
```
Check: window_detected=true only on clean_success=true episodes. Articulated tasks should have no windows (or low-confidence abstain only).

### 7. Labeled Export
```bash
python scripts/export_labeled_no_timestep_dataset.py \
  --no_timestep tables/no_timestep_visual_proprio_student_dataset.csv \
  --teacher_windows tables/goal100_teacher_window_labels.csv \
  --output tables/no_timestep_visual_proprio_student_dataset_labeled.csv
```

### 8. Mechanism-Aware Failure Audit
- Coverage by mechanism (pick_place vs articulated).
- Abstain reasons for each non-window episode.
- Verify 0 false positives on failed episodes.

## Expected Results
- **pick_place tasks** (bowl, butter, etc.): privileged coverage ~100%, teacher window coverage ~success rate
- **articulated tasks** (drawer, turn-on): teacher abstains, `gripper_duty_eligible=false`
- **planar tasks** (push): teacher abstains
- **Failed episodes**: no high-confidence windows
