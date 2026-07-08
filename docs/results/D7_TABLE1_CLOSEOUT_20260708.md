# D7 Table1 Closeout — 2026-07-08

**Status**: `D7_TABLE1_MAIN_RESULT = PASS_AUDITED_WITH_L10_STRATIFIED_CAVEAT`

**Commit**: `91f5937` | **Branch**: `plan/codex-gated-experiment-v1-c2e0`

## Pipeline Gates

| Gate | Status | Detail |
|---|---|---|
| D7B2 Rollout | COMPLETE | 716/716 episodes, 32 workers (4/GPU), C2e3 GRU detector |
| D7C Postrun Audit | **PASS** | 0 missing, 0 unpaired, 0 condition violations, 0 runtime contract violations, 0 SHA violations |
| D7D Aggregate | COMPLETE | Panel A per-suite, O/G/S pooled, all-suite summaries |
| D7E Render | COMPLETE | Markdown with COMMAND_OPEN_ORACLE naming, Detector Emit Rate / Attack Delivery Rate split |
| Paired McNemar | COMPLETE | All comparisons p < 0.001 |

## Evidence Roots

| Artifact | Path | Key SHA256 |
|---|---|---|
| Rollout | `/mnt/sdc/.../d7b2_table1_normalized_rollout/` | 716 summaries |
| Audit | `/mnt/sdc/.../d7b2_audit/` | `5a34a455...` |
| Aggregate | `/mnt/sdc/.../d7b2_aggregate/` | `9d0b572e...` |
| Render | `/mnt/sdc/.../d7b2_render/` | `e803fe63...` |
| Paired Stats | `/mnt/sdc/.../d7b2_paired_stats/` | `d08c09b0...` |

## Detector

- C2e3 GRU, W=16, H=128
- Checkpoint SHA: `3283f9492902f8cb...`
- τ_emit=0.33, τ_suppress=0.67
- Input: 25D proprio/action temporal features + 108D context
- Trained on clean-only C2e1 temporal dataset with teacher privilege labels
- No RGB, no task language, no attack outcome in student input

## Attack Protocol

- ε=6/255, K=10, PGD=20, MAX_STEPS=300
- TRUE_T10: targeted VIS perturbation at detector trigger
- RAND_T10: random-direction perturbation at detector trigger
- COMMAND_OPEN_ORACLE: command-layer open proxy at detector trigger
- CLEAN: no attack

## Statement

D7 Table1 result is frozen. C2f is post-D7 and does not modify D7 Table1.
