# Official V3 R7.1 K10 Opportunity V1.2.1 Audit Closure

Date: 2026-07-19  
PR: #87, Draft  
Scope: FIT states 0–19 only

## Review disposition

The submitted V1.2.1 label logic is accepted for a final server rerun, but the previously reported server root beginning `3886522c...` is not yet accepted as the formal R7.1 root because it predates PR-head integration and the independent auditor added during review.

The following code is now on the actual PR head branch:

- `scripts/detector_v4/label_k10_v121.py`
- `tests/test_r7_k10_v121.py`
- `scripts/detector_v4/audit_k10_v121_artifact.py`
- `tests/test_r7_k10_v121_auditor.py`
- `protocols/R7_K10_OPPORTUNITY_LABELER_V1_2_1.md`

## Accepted scientific contract

The clean-only label is a `gripper-critical opportunity`, not a VIS-vulnerability label. The fixed dense target is a K=10 start whose ten steps satisfy the frozen Physics V2.1 conjunction and remain in one candidate-close segment.

The last submitted development geometry was:

```text
FIT identities          = 800
feasible K10 episodes   = 109/800
feasible starts         = 7,399
```

These numbers are retained as a pre-integration reference only. The formal root must be regenerated at the final PR head and independently audited.

## Required final server execution

DeepSeek must use a new clean worktree at the final PR head and perform exactly:

1. verify `git status --porcelain` is empty;
2. run `label_k10_v121.py` against the sealed Physics Teacher V2.1 root into a new, previously nonexistent output root;
3. run `audit_k10_v121_artifact.py` against the Teacher root, new label root, and the same clean repository worktree;
4. preserve the independent audit JSON outside the immutable label root, then seal an audit bundle containing that JSON, the exact command lines, stdout/stderr, environment census, and checksums;
5. post the full, untruncated `SHA256SUMS` digest for both the label root and independent audit bundle.

Recommended command shape:

```bash
python scripts/detector_v4/label_k10_v121.py \
  --teacher-root <PHYSICS_V21_ROOT> \
  --output-root <NEW_R7_K10_V121_ROOT>

python scripts/detector_v4/audit_k10_v121_artifact.py \
  --teacher-root <PHYSICS_V21_ROOT> \
  --label-root <NEW_R7_K10_V121_ROOT> \
  --repo-root <CLEAN_PR_HEAD_WORKTREE> \
  --output <NEW_AUDIT_BUNDLE>/INDEPENDENT_AUDIT.json
```

## Required handoff fields

```text
BRANCH
HEAD
PR
CI workflow runs
SERVER_LABEL_ROOT
SERVER_LABEL_SHA256SUMS_FULL
SERVER_AUDIT_ROOT
SERVER_AUDIT_SHA256SUMS_FULL
INDEPENDENT_AUDIT_STATUS
IDENTITIES_READ
SOURCE_ROWS_READ
FEASIBLE_EPISODES
TOTAL_FEASIBLE_STARTS
PER_SUITE_FEASIBLE_EPISODES
PER_SUITE_FEASIBLE_STARTS
PROTECTED_SPLIT_READS
ATTACK_OR_MANUAL_OUTCOME_READS
SOURCE_ARTIFACT_MUTATION
WORKTREE_DIRTY_FILES
```

The full component funnel, per-task geometry, K10 union source categories, and component co-occurrence counts must be attached as sealed tables, not only pasted into a PR comment.

## Current authorization

```text
R7_R1_V121_LABEL_LOGIC          = PASS FOR FINAL RERUN
R7_R1_V121_GITHUB_INTEGRATION   = IN PROGRESS
R7_R1_V121_INDEPENDENT_AUDITOR  = ADDED
R7_R1_FORMAL_ARTIFACT           = HOLD PENDING RERUN

R7_R2_OFFLINE_REPLAY            = NOT YET AUTHORIZED
R7_R3_TRAINING                  = HOLD
R7_R4_EXACT_PREFIX              = HOLD
R7_R5_ATTACK_CANARY             = HOLD
FIT_DEV / CAL / CHECK           = NOT READ
CS200_ATTACK                    = NOT STARTED
```

After the final root and independent audit both pass at the integrated PR head, R7.2 may be authorized as a read-only offline replay only.
