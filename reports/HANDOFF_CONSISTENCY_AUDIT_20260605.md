# Handoff Consistency Audit 2026-06-05

Audit target: `reports/HANDOFF_20260605_WINDOW_COMPRESSION_AND_DETECTOR.md`

Scope: local repo consistency only. No server rollout, VIS job, GPU job, or server output mutation was run.

## Verdict

**Conditionally usable, but not sufficient for detector v2 training as-is.**

The handoff correctly states the current scientific boundary and the Codex/DeepSeek division of labor, but the local repo has several blocking gaps for a safe v2 label merge:

- `tables/object_phase_response_labels_v2.csv` is not present locally.
- `scripts/diagnostics/finalize_phase_response_labels.py` does not accept `--batch3b-vis` or `--batch3c-vis`.
- The label builder still has strict 9-row v0/v1 assertions, so it is not v2-ready.
- Batch3b/Batch3c status is described as in progress or ready, but the relevant output CSVs are not locally present for verification.

## Path Audit

| Item | Handoff Reference | Local Status | Notes |
|---|---|---:|---|
| Repo branch | `exp/vis-prefix-margin-repair-20260603` | OK | Current branch matches. |
| Handoff file | `reports/HANDOFF_20260605_WINDOW_COMPRESSION_AND_DETECTOR.md` | OK | File exists locally. |
| Batch3 summary | `tables/object_phase_response_batch3_vis_summary.csv` | OK | 18 data rows present locally. |
| Labels v1 | `tables/object_phase_response_labels_v1.csv` | OK | 20 data rows: 19 train rows expected by v1. |
| Labels v2 | `tables/object_phase_response_labels_v2.csv` | MISSING | Blocks detector v2 training. |
| Batch2b VIS summary | `tables/object_phase_response_batch2b_vis_summary.csv` | MISSING locally | Handoff may refer to remote/server copy. |
| Batch3b VIS summary | `tables/...batch3b...` | MISSING locally | Handoff says precheck launched, not final merged labels. |
| Batch3c VIS summary | `tables/...batch3c...` | MISSING locally | Controls are not locally mergeable yet. |
| Role gates | `scripts/diagnostics/role_specific_gates.py` | OK | Exists locally; runtime integration still needs verification. |
| Label builder | `scripts/diagnostics/finalize_phase_response_labels.py` | PARTIAL | Missing Batch3b/3c args and v2 assertions. |
| Detector trainer | `scripts/train_vulnerability_ready_detector_v1.py` | OK after audit fix | CSV path and hardcoded freeze are distinct. |

## Branch / Commit Audit

| Check | Result |
|---|---|
| Current branch | `exp/vis-prefix-margin-repair-20260603` |
| Handoff current committed revision | `c28d9f0 docs: refine handoff - fix date, commit fields, label merge, compression gates` |
| Handoff body still mentions older commits | Yes: `Handoff commit: 05dfea2`, `Experimental base commit: ba6f860`, checklist `Commit pushed (05dfea2)` |
| Self-referential commit loop risk | Partially mitigated by the line recommending `git log -1 -- reports/HANDOFF_20260605_WINDOW_COMPRESSION_AND_DETECTOR.md`; still present where older hardcoded commit IDs remain. |

Recommendation: do not create a standalone commit only to refresh the handoff SHA. If the handoff is edited for substantive reasons, replace checklist-style hardcoded "latest" fields with the `git log -1 -- ...` command.

## Python / Environment Audit

The handoff correctly points server execution to:

- Conda env: `openvla_official_libero_20260525`
- Python: `/home/liuyu/.conda/envs/openvla_official_libero_20260525/bin/python`

This matches the official runtime boundary for VIS evidence. Local Windows Python is only suitable for CPU-only code audits and dry-runs.

## Output Directory Audit

The server output roots named in the handoff are plausible and clearly separated:

- `/data/liuyu/outputs/nightly_object_batch3_20260604`
- `/data/liuyu/outputs/nightly_object_batch3b_20260604`
- `/data/liuyu/outputs/object_phase_response_batch3_VIS_20260604`

They were not modified. Local verification cannot establish final Batch3b/3c completion because the corresponding local CSVs are absent.

## Batch3 / Batch3b / Batch3c Consistency

| Batch | Handoff State | Audit Result |
|---|---|---|
| Batch3 | 11 VIS completed, 7 claim_usable, 4 task-negative | Consistent with local Batch3 summary after excluding non-VIS/no-action rows. |
| Batch3b | 14 candidates, precheck launched | Not contradicted locally, but final audit/VIS-ready outputs are not present. |
| Batch3c | 11 controls, role-specific gates implemented, VIS blocked until verification | Scientifically sound boundary. Local label builder does not yet preserve enough Batch3c role metadata for safe merge. |

No direct contradiction was found, but Batch3b/Batch3c are not locally complete enough to train v2.

## Detector v1 / v2 Status

The handoff correctly prevents overclaiming:

- Detector v1 is underpowered.
- Always-positive matches LR task-key F1.
- No model beats prevalence.
- v2 requires Batch3b/c label expansion and controls.

The corrected trainer now hard-fails on too few rows and single-class data, reports prevalence/random/control metrics, and checks feature names for forbidden leakage.

## Responsibility Split

Clear and consistent:

- DeepSeek: server execution, watcher, audits, labels, training, compression rollout.
- Codex: design docs, schema audit, training script audit, candidate generators, feasibility review, no GPU.

This remains valid.

## Blocking Issues

1. `tables/object_phase_response_labels_v2.csv` is missing locally.
2. Label builder lacks `--batch3b-vis` and `--batch3c-vis`.
3. Label builder still has v0/v1 hardcoded 9-row assertions.
4. Batch3c labels are not safe to merge until `candidate_role`, `denominator_type`, and role-specific taxonomy are preserved end to end.

## Non-Blocking Warnings

1. Handoff duplicates the "Codex Parallel Plan" section.
2. Handoff still contains older hardcoded commit IDs; use `git log -1 -- reports/HANDOFF_20260605_WINDOW_COMPRESSION_AND_DETECTOR.md` for current revision.
3. Local repo has many unrelated untracked files; keep commits scoped.

## Recommendation

DeepSeek should not train detector v2 until:

1. Label builder supports Batch3b/Batch3c inputs.
2. `object_phase_response_labels_v2.csv` exists.
3. `scripts/diagnostics/audit_label_schema.py --labels-csv tables/object_phase_response_labels_v2.csv` passes.
4. Batch3c role/control labels are verified as train/ignore/manual_review separated.
