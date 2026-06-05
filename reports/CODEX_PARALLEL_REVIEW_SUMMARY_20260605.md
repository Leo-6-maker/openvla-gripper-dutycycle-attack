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
- `scripts/diagnostics/finalize_phase_response_labels.py`
- `scripts/train_vulnerability_ready_detector_v1.py`
- `tables/label_schema_audit_v2.csv`
- `tables/object_window_compression_candidates.csv`
- `reports/LABEL_SCHEMA_AUDIT_V2.md`
- `reports/OBJECT_WINDOW_COMPRESSION_CANDIDATE_PLAN.md`
- `reports/OBJECT_PHASE_RESPONSE_LABEL_READINESS_V2.md`
- `reports/HANDOFF_CONSISTENCY_AUDIT_20260605.md`
- `reports/DETECTOR_TRAINING_SCRIPT_AUDIT.md`
- `reports/DETECTOR_V2_AND_VISUAL_TRANSFER_DESIGN.md`
- `reports/LABEL_BUILDER_PATCH_PLAN.md`
- `reports/SERVER_SYNC_PLAN_20260605.md`
- `reports/CODEX_PARALLEL_REVIEW_SUMMARY_20260605.md`

## Blocking Issues

1. Server read-only verification found the server repo on `exp/vis-payload-upgrade-validation-20260601`, not the reviewed branch `exp/vis-prefix-margin-repair-20260603`; the reviewed branch tip is not present on server.
2. `tables/object_phase_response_labels_v2.csv` is still missing as a full-source label artifact.
3. Batch3c labels are not safe to train on until the patched builder is run on synced server sources with candidate metadata joins and schema audit passes.
4. Detector v2 should not train until schema audit passes on `object_phase_response_labels_v2.csv`.

## Non-Blocking Warnings

1. Handoff duplicates the Codex parallel plan section.
2. Handoff still contains older hardcoded commit IDs; use `git log -1 -- reports/HANDOFF_20260605_WINDOW_COMPRESSION_AND_DETECTOR.md` for current committed revision.
3. Local Batch2b/Batch3b/Batch3c final CSVs are absent, so server-side state must be verified by DeepSeek before merge/training.
4. Local sklearn is unavailable, so model-training smoke could not complete on this Windows machine. Compile and source-level hardening passed.
5. Label builder synthetic tests pass, including candidate metadata joins, but they are not a substitute for full-source server label generation.

## Final Label-Builder Audit

- Remote visibility before final denominator patch: `git ls-remote origin exp/vis-prefix-margin-repair-20260603` showed branch tip `a99d6adc0566ee8ef50d295fab74da3829729f12`.
- `finalize_phase_response_labels.py` supports `--batch3b-vis` and `--batch3c-vis`.
- It supports candidate metadata inputs: `--batch2b-candidates`, `--batch3-candidates`, `--batch3b-candidates`, and `--batch3c-candidates`.
- Candidate metadata join fills missing `candidate_role`, `control_type`, `phase_bin_proxy`, `denominator_type`, `action_bridge_confounded`, and `reason_selected`.
- Summary/candidate `candidate_role` conflict hard fails and writes the conflict CSV.
- Batch3c rows with missing joined role become `manual_review` and do not enter train.
- Batch3b/Batch3c rows with missing explicit denominator fields become `manual_review` and do not enter train.
- Synthetic tests: 7 OK.
- Full-source `tables/object_phase_response_labels_v2.csv` is still missing.
- DeepSeek detector v2 training remains **BLOCKED** until server-generated labels v2 passes schema audit.

Warning: full-source label merge must only use summaries with explicit denominator status; missing denominator fields are dangerous and are now excluded from train for Batch3b/Batch3c.

## Remote Visibility

Local reviewed branch:

```text
exp/vis-prefix-margin-repair-20260603
```

Reviewed code-bearing HEAD:

```text
a74eaead95fc139548ee2e39b0ec1c40bf254c96
```

Remote visibility:

```text
PASS. Use `git ls-remote origin exp/vis-prefix-margin-repair-20260603` for the current branch tip.
```

`e086836` was confirmed pushed first. The branch then advanced with the label-builder patch and docs-only sync-plan commits. The current branch tip is intentionally not hard-coded here to avoid a report self-reference loop.

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
- Reviewed branch tip is not present on server.
- Official env exists: `/home/liuyu/.conda/envs/openvla_official_libero_20260525/bin/python`, `Python 3.10.13`.
- Server has Batch3/Batch3b output directories, but labels v2 and this review package are missing.

No GPU command, rollout, VIS job, watcher, training job, or server output mutation was run.

Recommended server sync: **Option A safe worktree** from `reports/SERVER_SYNC_PLAN_20260605.md`.

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

Can DeepSeek safely train detector v2 now: **No. BLOCKED.**

Required first:

1. Sync/check out the intended reviewed branch on server.
2. Run the patched label builder for Batch3b/Batch3c on real server sources.
3. Generate `tables/object_phase_response_labels_v2.csv`.
4. Run `scripts/diagnostics/audit_label_schema.py` and require PASS.
5. Confirm class balance, controls, and task split warnings before interpreting v2 metrics.

Label builder Batch3b/c support: **Yes, implemented locally and pushed to remote.**

Candidate metadata join support: **Yes.** The builder supports `--batch2b-candidates`, `--batch3-candidates`, `--batch3b-candidates`, and `--batch3c-candidates`; summary/candidate role conflicts hard fail; Batch3c rows with no joined role become manual_review and do not enter train.

Server synced to reviewed branch: **No.**

Full-source labels v2 present: **No.**

Are compression candidates ready: **Yes, as candidate windows only.**

They are ready for DeepSeek planning, but not evidence. They require matched VIS/random and denominator audit before any claim.

Are Batch3c labels safe to merge: **No.**

They need role-specific preservation and manual_review handling first.

## Next Recommended Action

DeepSeek should first sync a safe reviewed worktree, then run the patched label builder on Batch3b/Batch3c source CSVs. After `object_phase_response_labels_v2.csv` exists, run the schema audit before training. Window compression can stay queued as a CPU-generated plan and should not block Batch3b/c labels.
