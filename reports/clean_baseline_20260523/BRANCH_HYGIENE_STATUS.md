# Branch Hygiene Status

**Timestamp**: 2026-05-23T22:25Z
**Branch**: `fix/protocol-schema-and-condition-config-20260523`
**Latest commit**: `f5acd9b` — Fix protocol validators and test imports

## Working Tree

| Check | Status |
|-------|--------|
| Modified tracked files | NONE (clean) |
| Staged changes | NONE |
| Untracked files | 13 safe files + 1 stray |
| Artifacts/videos/HDF5/models | NONE |
| Large files (>1MB) | NONE |

Untracked files are evidence freeze reports and tables from a read-only postprocess
task — all small text files, safe to commit or gitignore:

```
reports/NEXT_ACTION_STATUS.md
reports/STATE5_DENOMINATOR_CORRECTION.md
reports/STATE5_EXACT_CODEX_EVIDENCE_FREEZE.md
reports/STATE5_LINF_METRIC_HARMONIZATION.md
reports/STATE5_WINDOW_PHASE_SENSITIVITY.md
tables/state5_claim_boundary.csv
tables/state5_condition_outcome_compact.csv
tables/state5_linf_metric_harmonized.csv
tables/state5_quarantined_runs.csv
tables/state5_valid_repeat_accounting.csv
tables/state5_window_phase_table.csv
nul  (stray Windows file, should not be committed)
```

## Branch Position

| Metric | Value |
|--------|-------|
| Current branch | `fix/protocol-schema-and-condition-config-20260523` |
| Ahead of origin/main | 5 commits |
| Behind origin/main | 8 commits |
| Total diff from main | +1173 / -4996 (43 files) |
| Remote push status | Up to date with origin |

Branch commits (ahead of main):

```
f5acd9b Fix protocol validators and test imports
1b9b037 Fix protocol imports and fail-fast deprecated condition configs
8fe45c3 Fix protocol schema and clarify attack mechanism definitions
de98d53 Standardize task identity metadata and fix matched condition config
e81b6b3 fix: pass --force_open_raw_gripper 1.0 for gripper-logit-margin conditions
```

Main is ahead by 8 commits (full4 denominator PR merge + supporting changes).
These are unrelated to protocol schema changes and should not block merge.

## File Sizes

All new/modified source files are small text:

| File | Lines | Type |
|------|-------|------|
| src/utils/condition_protocols.py | ~230 | source |
| src/utils/protocol_validation.py | ~180 | source |
| src/utils/task_identity.py | ~160 | source |
| tests/v4/test_task_identity.py | ~180 | test |
| tests/v4/test_protocol_validation.py | ~170 | test |
| tests/v4/test_metadata_schema.py | ~80 | test |
| tests/v4/test_clean_protocol.py | ~60 | test |
| docs/claim_and_evidence.md | ~30 lines added | docs |
| reports/TASK_IDENTITY_PATCH_STATUS.md | ~70 | report |

No HDF5, pickle, video, model weights, or dataset files.

## Verdict

Working tree is hygienic for a clean baseline commit.
No blocking hygiene issues.
