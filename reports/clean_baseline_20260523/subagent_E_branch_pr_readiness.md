# Subagent E — Branch / PR Readiness Audit

## Summary

- **Branch**: `fix/protocol-schema-and-condition-config-20260523` (current HEAD: `f5acd9b`)
- **Remote tracking**: `origin/fix/protocol-schema-and-condition-config-20260523` exists (pushed)
- **Purpose**: Consolidate scattered protocol/condition config code into `src/utils/`, removing old experiment scripts, configs, and test files. 5 unique commits on this branch vs main.
- **Divergence**: 8 commits behind `origin/main` (merge base `53b0112`). No in-progress PR found (gh CLI unauthenticated).
- **Verdict**: NEEDS_REBASE — branch has diverged significantly from `origin/main`; rebase is required for any merge or baseline use. Rebase carries moderate but manageable conflict risk on 2 files. After rebase, could serve as a clean baseline.

---

## Rebase Analysis

### Branch topology (from merge-base `53b0112`)

```
* f5acd9b (HEAD -> fix/protocol-schema-and-condition-config-20260523)
* 1b9b037  Fix protocol imports and fail-fast deprecated condition configs
* 8fe45c3  Fix protocol schema and clarify attack mechanism definitions
* de98d53  Standardize task identity metadata and fix matched condition config
* e81b6b3  fix: pass --force_open_raw_gripper 1.0 for gripper-logit-margin conditions
| *   20a1da2 (origin/main) Merge pull request #1
| |\
| | * faa7fc7 (duplicate of e81b6b3, different hash)
| | * aeb7d05  Add LIBERO full4 clean denominator tooling
| |/
|/|
| * 8275e99 (remote: refactor/generic-autowindow-cqv2-20260517)
| * 8f3d2fd  fix: add contact-quality v2 metric...
| * b023949  test: isolate Black Bowl reference-window evaluation
| * 9f51c83  refactor: add task-agnostic generic auto-window detector
| * 1e2d71d  feat: add action-clamp decomposition...
|/
* 53b0112  (merge base — PGD/guard/condition fixes)
|
* 33bdf59 (local main)
```

### Rationale: YES, this branch SHOULD be rebased onto `origin/main`

1. **8 commits behind**: `origin/main` has moved forward with substantial additions (LIBERO full4 denominator tooling, generic auto-window detector, CQv2 metrics, Moka experiments).
2. **Branch goal is cleanup/consolidation**: A rebase produces the correct result — the consolidated protocol code sits on top of main's latest state, making it a true superset.
3. **No merge commit risk**: The branch is a simple linear chain (no merge commits on the branch side), making rebase straightforward.

### Rebase conflict risk assessment

| File | Branch change (e81b6b3) | Main change (faa7fc7) | Conflict risk |
|---|---|---|---|
| `scripts/run_attack_pipeline.py` | +8 lines: `--force_open_raw_gripper 1.0` | Same logical change (different parent base) | MODERATE — same patch applied to different base; likely auto-resolves but could conflict |
| `tests/v4/test_public_pipeline.py` | +21 lines: tests for force_open_raw_gripper | Same logical change | MODERATE — same considerations |
| All other files | New files only (src/utils/*, tests/v4/test_*) | Different files entirely | NONE — no overlap |

The remaining 41 files shown in `git diff --stat origin/main..HEAD` are DELETIONS of files that main has added (configs, scripts, docs) — these are expected since the branch intentionally removes old scattered files.

---

## Conflict Risk with Main

### 8 commits on `origin/main` NOT on this branch:

| # | Commit | Description | Files changed | Overlap with branch? |
|---|---|---|---|---|
| 1 | `1e2d71d` | Action-clamp decomposition, Moka two-pot | 13 files (src/gripper_attack/*, scripts/moka_*, tests/v4/test_*) | NO — touches different src/ subdirs and different scripts |
| 2 | `9f51c83` | Generic auto-window detector | 5 files (configs, scripts, tests) | NO |
| 3 | `b023949` | Black Bowl reference-window eval | 2 files (config, script) | NO |
| 4 | `8f3d2fd` | CQv2 metric, docs, scripts | 7 files (docs, scripts, tests, .gitignore) | NO |
| 5 | `8275e99` | Code submission summary docs | 1 file (docs/code_submission_summary_*.md) | NO |
| 6 | `aeb7d05` | LIBERO full4 clean denominator | 7 files (configs, scripts, docs) | NO |
| 7 | `faa7fc7` | force_open_raw_gripper (duplicate) | run_attack_pipeline.py, test_public_pipeline.py | YES — same logical change as e81b6b3 |
| 8 | `20a1da2` | Merge PR #1 | Merge commit | NO — structure only |

### Overlap detail

Commits `faa7fc7` (main) and `e81b6b3` (branch) have identical commit messages ("fix: pass --force_open_raw_gripper 1.0") but different hashes and different parent trees. Both modify `scripts/run_attack_pipeline.py` and `tests/v4/test_public_pipeline.py`. During rebase, Git may see:
- If the change is already applied (faa7fc7 is in the new base), the patch may apply cleanly as a no-op
- Or it could conflict if contextual lines differ

**Recommendation**: After rebase, verify the final state of these two files to ensure no double-application or regression.

---

## Old Branch Cleanup Checklist

### Branches with stale/wrong protocol configs that should be cleaned up:

| Branch | Status | Reason to deprecate |
|---|---|---|
| `deprecated/task-identity-metadata-20260523` | LOCAL only | Tip `de98d53` is ancestor of current branch. Old name for same work. **Should be deleted.** |
| `experiment/strict-after-close-guard` | LOCAL + REMOTE | Tip `e81b6b3` is ancestor of current branch. All its work is subsumed. **Mark SUPERSEDED.** |
| `experiment/moka-twopot-window-theory` | LOCAL + REMOTE | Tip `1e2d71d` is base of refactor chain. Has old protocol code in `src/gripper_attack/*`. **Mark SUPERSEDED by new protocol consolidation.** |
| `experiment/moka-crossdataset-diagnostic` | LOCAL only | Tip `33bdf59` (stale main). No unique value. **Consider deleting.** |
| `origin/refactor/generic-autowindow-cqv2-20260517` | REMOTE only | Tip `8275e99`. Pre-protocol-consolidation code in `src/gripper_attack/` — **does NOT have the new `src/utils/` structure.** **Mark SUPERSEDED.** |
| `origin/denominator/full4-clean-20260519` | REMOTE only | Tip `faa7fc7`. Contains old LIBERO full4 configs and scripts that this branch actively removes. **Mark SUPERSEDED.** |
| `origin/evidence/nonbb-bowl-plate-targeted-vis-20260522` | REMOTE only | Tip `bd9c922`. Evidence branch, different purpose. Retain unless proven stale. |

### No branches found with "state5" in the name.

---

## Large File / Artifact Scan

### Binary/model artifact check (`git diff --name-only origin/main..HEAD -- *.hdf5 *.h5 *.pt *.pth *.safetensors *.ckpt *.tar *.tar.gz *.zip *.mp4 *.avi *.mov *.webm`)
- **No binary files detected** in the diff.

### File size check (threshold 100 KB)
- **No files over 100 KB** in the diff. Largest checked file: `scripts/v4_run_eval_openvla.py` at ~84 KB.

### Diff composition
- **43 files** changed in two-dot diff (branch tip vs main tip):
  - 12 files are this branch's actual changes (per three-dot diff from merge-base)
  - 31 files are deletions of files that main added (configs, scripts, docs, tests)
- **+1173 insertions, -4996 deletions** — net removal of 3823 lines (cleanup/consolidation)

### .gitignore coverage gap

The current branch's `.gitignore` is MISSING 8 patterns that exist in `origin/main`'s `.gitignore`:

| Missing pattern | Present in main? |
|---|---|
| `manual_review_packages/` | YES |
| `videos/` | YES |
| `*.mp4` | YES |
| `*.avi` | YES |
| `*.mov` | YES |
| `*.mkv` | YES |
| `*.pkl` | YES |
| `*.npy` | YES |

This gap is because the branch's `.gitignore` (blob hash `4a741cf`) predates main's additions (blob `26c8673`). After rebase onto `origin/main`, the more complete `.gitignore` would be inherited. **No action needed before rebase, but verify afterward.**

### Credentials / secrets scan
- **No credentials, secrets, API keys, or SSH keys detected** in any tracked files on this branch.
- Matches for "token" in `archive/` files are NLP/ML tokenizer references, not credentials.

---

## PR Status

- **`gh` CLI**: Not authenticated — cannot query GitHub for PR status.
- **Remote branch exists**: `origin/fix/protocol-schema-and-condition-config-20260523` has been pushed.
- **Old branch name**: `fix/task-identity-metadata-20260523` (now at `deprecated/task-identity-metadata-20260523` locally) — any PR opened under the old name would reference the old commits only (up to `de98d53`). A new PR would be needed under the current branch name.

---

## Overall Verdict: NEEDS_REBASE

**Why not CLEAN_BASELINE_READY:**
1. Branch is 8 commits behind `origin/main` — cannot serve as a baseline without catching up.
2. `.gitignore` is missing 8 artifact patterns that main has added.
3. Two files (`run_attack_pipeline.py`, `test_public_pipeline.py`) have duplicate logical changes applied to different base states — rebase verification is required.

**Why not BLOCKED:**
1. No binary artifacts in the diff.
2. No credentials or secrets found.
3. Conflict risk is limited to 2 files with moderate risk.
4. Branch purpose (protocol consolidation, cleanup) is sound and complementary to main's direction.
5. All new code is in `src/utils/` and `tests/v4/` — no overlap with main's files.

**Recommended path forward:**
1. Rebase onto `origin/main` (expected conflicts: moderate on 2 files).
2. Verify the rebased `.gitignore` has the 8 missing patterns.
3. Verify `run_attack_pipeline.py` and `test_public_pipeline.py` are correct post-rebase.
4. Delete deprecated local branch (`deprecated/task-identity-metadata-20260523`).
5. Mark these remote branches as SUPERSEDED: `origin/refactor/generic-autowindow-cqv2-20260517`, `origin/denominator/full4-clean-20260519`, `origin/experiment/strict-after-close-guard`.
