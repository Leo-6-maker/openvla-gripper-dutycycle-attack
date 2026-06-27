# Table 1 Freeze Status V2

```
TABLE1_PANEL_A_NUMERICAL_FREEZE  = PASS
TABLE1_PANEL_B_NUMERICAL_FREEZE  = PASS
TABLE1_CORE_NUMERICAL_FREEZE     = PASS
ARMLOCK_EXEC_ARM_INVARIANT       = PASS
RNAD_MECHANISM_FREEZE_V2         = PASS   (same-space, actual action stats)
TABLE1_ARTIFACT_FREEZE_V2        = PASS   (full SHAs, split commit roles)
CQFR_PACKAGE_READY_V2            = PASS   (68 unique, global shuffle, task instructions)
CQFR_HUMAN_REVIEW                = PENDING
TIMING_FINAL_FREEZE              = HOLD
GLOBAL_ACCEPTED_RUN_COUNT        = HOLD
TABLE1_PUBLICATION_FREEZE        = HOLD
```

## V2 Changes from c82e85c

1. **rNAD**: All mixed-space comparisons removed. Same-space only (policy-policy, env-env). Actual victim model action stats used instead of hardcoded approximate bounds.
2. **Artifact SHA**: Full 64-char hashes, no short prefixes, no MISSING entries. Commit roles split: runtime/experiment/analysis/final.
3. **CQFR**: 68 unique videos (from 108 runs), globally shuffled, task instructions included, label definitions and review protocol in package.
4. **Breadth manifest**: SHA conflict explained (JSONL re-serialized after sidecar seal; functional content identical).
5. **Timing**: Moved from Frozen Claims to Provisional Findings.
6. **Latency**: Corrected to report both mean and median ratios.
