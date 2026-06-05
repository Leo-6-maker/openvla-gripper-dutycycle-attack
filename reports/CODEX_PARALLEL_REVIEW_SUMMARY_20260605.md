# Codex Parallel Review Summary 2026-06-05

Scope: design, audit, scripts, and CPU-only feasibility validation. No GPU, rollout, VIS experiment, watcher, or server output mutation was run.

## Checked Files

- `reports/HANDOFF_20260605_WINDOW_COMPRESSION_AND_DETECTOR.md`
- `scripts/train_vulnerability_ready_detector_v1.py`
- `scripts/diagnostics/finalize_phase_response_labels.py`
- `scripts/diagnostics/role_specific_gates.py`
- `tables/object_phase_response_batch3_vis_summary.csv`
- `tables/object_phase_response_labels_v1.csv`
- `tables/object_phase_response_labels_v2.csv` (missing)

## Created / Updated Files

- `scripts/diagnostics/audit_label_schema.py`
- `scripts/diagnostics/generate_window_compression_candidates.py`
- `scripts/train_vulnerability_ready_detector_v1.py`
- `tables/label_schema_audit_v2.csv`
- `tables/object_window_compression_candidates.csv`
- `reports/LABEL_SCHEMA_AUDIT_V2.md`
- `reports/OBJECT_WINDOW_COMPRESSION_CANDIDATE_PLAN.md`
- `reports/HANDOFF_CONSISTENCY_AUDIT_20260605.md`
- `reports/DETECTOR_TRAINING_SCRIPT_AUDIT.md`
- `reports/DETECTOR_V2_AND_VISUAL_TRANSFER_DESIGN.md`
- `reports/LABEL_BUILDER_PATCH_PLAN.md`
- `reports/CODEX_PARALLEL_REVIEW_SUMMARY_20260605.md`

## Blocking Issues

1. `tables/object_phase_response_labels_v2.csv` is missing locally.
2. `finalize_phase_response_labels.py` does not support `--batch3b-vis` / `--batch3c-vis`.
3. The label builder still contains hardcoded 9-label assertions and is not safe for v2 multi-source merging.
4. Batch3c labels are not safe to merge until role metadata, denominator type, and manual_review separation are preserved.
5. Detector v2 should not train until schema audit passes on `object_phase_response_labels_v2.csv`.
6. Server read-only verification found the server repo on `exp/vis-payload-upgrade-validation-20260601`, not the handoff branch `exp/vis-prefix-margin-repair-20260603`; local Codex commit `3a35239` is not present on server.

## Non-Blocking Warnings

1. Handoff duplicates the Codex parallel plan section.
2. Handoff still contains older hardcoded commit IDs; use `git log -1 -- reports/HANDOFF_20260605_WINDOW_COMPRESSION_AND_DETECTOR.md` for current committed revision.
3. Local Batch2b/Batch3b/Batch3c final CSVs are absent, so server-side state must be verified by DeepSeek before merge/training.
4. Local sklearn is unavailable, so model-training smoke could not complete on this Windows machine. Compile and source-level hardening passed.

## Server Read-Only Verification

SSH route checked:

```bash
ssh -J scene@10.60.133.3 liuyu@10.60.133.4
```

Server repo:

```bash
/data/liuyu/repos/openvla-gripper-dutycycle-attack-clean-main-20260524
```

Result:

- Current server branch: `exp/vis-payload-upgrade-validation-20260601`.
- Current server HEAD: `653ed33d78578aa0f0af96539a9c8b4c2a6d4c08`.
- Expected handoff branch `exp/vis-prefix-margin-repair-20260603` was not listed in the server local branch check.
- Local Codex audit commit `3a35239` is not present on server.
- Official env exists: `/home/liuyu/.conda/envs/openvla_official_libero_20260525/bin/python`, `Python 3.10.13`.
- Server has Batch3/Batch3b output directories, but labels v2 and this review package are missing.

No GPU command, rollout, VIS job, watcher, training job, or server output mutation was run.

## Validation Run

Passed:

```bash
python -m py_compile scripts/diagnostics/audit_label_schema.py scripts/diagnostics/generate_window_compression_candidates.py scripts/train_vulnerability_ready_detector_v1.py scripts/diagnostics/finalize_phase_response_labels.py
```

Ran schema dry-run:

```bash
python scripts/diagnostics/audit_label_schema.py --labels-csv tables/object_phase_response_labels_v2.csv
```

Result: FAIL as expected because labels v2 is missing. Outputs were written to `tables/label_schema_audit_v2.csv` and `reports/LABEL_SCHEMA_AUDIT_V2.md`.

Ran compression candidate generation:

```bash
python scripts/diagnostics/generate_window_compression_candidates.py \
  --summary-csv tables/object_phase_response_batch3_vis_summary.csv \
  --output-csv tables/object_window_compression_candidates.csv
```

Result: PASS. Generated 15 candidates: five parent windows times L12/L10/L8.

## DeepSeek Readiness

Can DeepSeek safely train detector v2 now: **No.**

Required first:

1. Sync/check out the intended reviewed branch on server.
2. Patch or replace label builder for Batch3b/Batch3c.
3. Generate `tables/object_phase_response_labels_v2.csv`.
4. Run `scripts/diagnostics/audit_label_schema.py` and require PASS.
5. Confirm class balance, controls, and task split warnings before interpreting v2 metrics.

Are compression candidates ready: **Yes, as candidate windows only.**

They are ready for DeepSeek planning, but not evidence. They require matched VIS/random and denominator audit before any claim.

Are Batch3c labels safe to merge: **No.**

They need role-specific preservation and manual_review handling first.

## Next Recommended Action

DeepSeek should first finish/audit Batch3b/Batch3c source CSVs, then patch label builder minimally for multi-source v2 merge. After `object_phase_response_labels_v2.csv` exists, run the schema audit before training. Window compression can stay queued as a CPU-generated plan and should not block Batch3b/c labels.
