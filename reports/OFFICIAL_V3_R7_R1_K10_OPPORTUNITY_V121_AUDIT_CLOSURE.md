# Official V3 R7.1 K10 Opportunity V1.2.1 Audit Closure

Date: 2026-07-19  
PR: #87, Draft  
Scope: FIT states 0–19 only

## Review disposition

The submitted V1.2.1 label logic is accepted. The previously reported server root beginning `3886522c...` may be retained because the labeler bytes at its bound commit `1353e3b4190b2bf2d8842d42c42aef0bbb8ae420` are unchanged by the subsequent GitHub audit-integration commits.

The root is not yet the formal R7.1 artifact because its full checksum digest and an independent audit result have not been supplied. Regeneration is unnecessary unless the existing root fails the independent audit.

The following code is now on the actual PR head branch:

- `scripts/detector_v4/label_k10_v121.py`
- `tests/test_r7_k10_v121.py`
- `scripts/detector_v4/audit_k10_v121_artifact.py`
- `tests/test_r7_k10_v121_auditor.py`
- `protocols/R7_K10_OPPORTUNITY_LABELER_V1_2_1.md`

All three PR workflows pass at the integrated head.

## Accepted scientific contract

The clean-only label is a `gripper-critical opportunity`, not a VIS-vulnerability label. The fixed dense target is a K=10 start whose ten steps satisfy the frozen Physics V2.1 conjunction and remain in one candidate-close segment.

The submitted development geometry is:

```text
FIT identities          = 800
feasible K10 episodes   = 109/800
feasible starts         = 7,399
```

These values remain provisional until the independent auditor recomputes them from the sealed root.

## Required final server audit

DeepSeek must not regenerate the label root unless this audit fails. Use:

1. a clean detached worktree at `1353e3b4190b2bf2d8842d42c42aef0bbb8ae420`, matching the root's `SOURCE_BINDING.json`;
2. the independent auditor script from the current PR head;
3. the existing immutable V1.2.1 label root beginning `3886522c...`;
4. a new audit-bundle directory outside the label root.

Required checks:

```bash
git -C <CLEAN_1353_WORKTREE> status --porcelain

python <CURRENT_PR_HEAD>/scripts/detector_v4/audit_k10_v121_artifact.py \
  --teacher-root <PHYSICS_V21_ROOT> \
  --label-root <EXISTING_R7_K10_V121_ROOT> \
  --repo-root <CLEAN_1353_WORKTREE> \
  --output <NEW_AUDIT_BUNDLE>/INDEPENDENT_AUDIT.json
```

The audit bundle must also contain:

- exact command line;
- stdout/stderr;
- current auditor commit;
- bound labeler commit `1353e3b...`;
- worktree status;
- environment census;
- a recursive checksum list and checksum sidecar.

If the existing root fails due to an actual label/root inconsistency, preserve it unchanged and regenerate a new root at the current PR head. An infrastructure/path mistake may be corrected without changing the existing label root.

## Required handoff fields

```text
PR_HEAD
PR_CI_RUNS
BOUND_LABELER_HEAD
CURRENT_AUDITOR_HEAD
SERVER_LABEL_ROOT
SERVER_LABEL_SHA256SUMS_FULL
SERVER_AUDIT_ROOT
SERVER_AUDIT_SHA256SUMS_FULL
INDEPENDENT_AUDIT_STATUS
SOURCE_IDENTITIES_READ
SOURCE_ROWS_READ
OUTPUT_IDENTITIES_READ
FEASIBLE_EPISODES_RECOMPUTED
TOTAL_FEASIBLE_STARTS_RECOMPUTED
PER_SUITE_FEASIBLE_EPISODES
PER_SUITE_FEASIBLE_STARTS
PROTECTED_SPLIT_READS
ATTACK_OR_MANUAL_OUTCOME_READS
SOURCE_ARTIFACT_MUTATION
WORKTREE_DIRTY_FILES
```

The full component funnel, per-task geometry, K10 union source categories, and component co-occurrence counts must remain in the sealed label root or audit bundle, not only in a PR comment.

## Current authorization

```text
R7_R1_V121_LABEL_LOGIC          = PASS
R7_R1_V121_GITHUB_INTEGRATION   = PASS
R7_R1_V121_CI                   = PASS
R7_R1_V121_INDEPENDENT_AUDITOR  = PASS CODE / NOT YET RUN ON SERVER ROOT
R7_R1_FORMAL_ARTIFACT           = HOLD PENDING INDEPENDENT AUDIT

R7_R2_OFFLINE_REPLAY            = NOT YET AUTHORIZED
R7_R3_TRAINING                  = HOLD
R7_R4_EXACT_PREFIX              = HOLD
R7_R5_ATTACK_CANARY             = HOLD
FIT_DEV / CAL / CHECK           = NOT READ
CS200_ATTACK                    = NOT STARTED
```

After the existing root passes the independent audit and full digests are posted, R7.2 may be authorized as a read-only offline replay only.
