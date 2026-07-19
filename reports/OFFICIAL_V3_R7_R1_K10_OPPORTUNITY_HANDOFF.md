# R7.1 K10 Opportunity Labeler Handoff

## Binding

```text
BRANCH                  = agent/official-v3-r7-k10-opportunity-20260719
HEAD                    = b0a2574
PR                      = #87
SERVER_WORKTREE          = /mnt/sdc/dty_user/worktrees/official_v3_r7_k10_66f3604
SERVER_WORKTREE_HEAD     = 66f3604da1f178942ed3cfb17ec8db0675b5068b
SOURCE_TEACHER_ROOT      = OFFICIAL_V3_S1_FIT_V1_5e27d7c
SOURCE_TEACHER_SHA       = 15c97212fde19682a9e3042d6d051c51606b0989881d471cb8eb80f22354b0cf
OUTPUT_ROOT              = OFFICIAL_V3_R7_K10_OPPORTUNITY_LABELER_V1_66f3604_20260719
OUTPUT_SHA256SUMS        = 665c4f62cb17162de0739517ff260b6abe011512d191c4a93440179cadcf49d6
IDENTITIES_READ          = 800
PROTECTED_READS          = 0
SOURCE_MUTATION          = 0
ATTACK_OUTCOME_READS     = 0
```

## Field census (R7.1-A)

29 fields audited from `B3_OFFICIAL_V3_TEACHER_RECORD_V1`. All 7 required concept groups fully available:

| Concept | Fields | Status |
|---|---|---|
| candidate_close | `event_close_onset`, `event_end_step` | FULL |
| label_known | `retention_unknown_mask`, `event_evidence_valid` | FULL |
| stable_grasp | `event_support` | FULL |
| contact_evidence | `grasp_support` | FULL |
| manipulation_active | `retention_active` | FULL |
| release_safe | `release_imminent`, `event_release_onset` | FULL |
| segment_identity | `event_id`, `event_start_step`, `event_end_step` | FULL |

Notable: `event_opening_stable` is `None` (97,830 steps) or `True` (78,506 steps) — never `False`. Cannot detect instability from this field alone. `regrasp_or_instability` deferred to future Teacher versions.

## Label results

| Metric | Value |
|---|---|
| FIT identities | 800 |
| Candidate segments | 1,130 |
| Episodes with feasible K10 | 726 (90.75%) |
| Total feasible starts | 50,023 |
| No-corridor episodes | 74 |
| Non-gripper tasks | `libero_goal/t00` (1 close event), `libero_goal/t05` (0 close events) |

### Per-suite summary

| Suite | Feasible episodes | Total starts | No-corridor |
|---|---|---|---|
| libero_10 | 192/200 (96%) | 25,336 | 8 |
| libero_goal | 145/200 (72.5%) | 5,032 | 55 |
| libero_object | 192/200 (96%) | 10,917 | 8 |
| libero_spatial | 197/200 (98.5%) | 8,738 | 3 |

### Component hit rates (step-level, within close segments)

| Component | Steps | % of close steps |
|---|---|---|
| `candidate_close` | 78,491 | 100% |
| `stable_grasp` | 69,441 | 88.5% |
| `manipulation_active` | ~52,000 | ~66% |
| `release_safe` | 3,385 | 4.3% |
| `critical_t` | ~51,000 | ~65% |

### No-corridor reason breakdown

| Reason | Count |
|---|---|
| No close events (non-gripper task) | 40 (`libero_goal/t00`: 19, `libero_goal/t05`: 20, other: 1) |
| Has close events but no critical steps | ~15 |
| Has critical steps but no K=10 contiguous burst | ~19 |

### Feasible start distribution

Per-episode feasible start counts range from 1 to ~300. Median ~50 starts per feasible episode. Long transport tasks accumulate more starts (e.g., `libero_10/t05`: 3,308 starts across 20 episodes).

## Hard gates (all PASS)

```
segment_crossing              = 0
K10 out-of-bound              = 0
unknown in positive           = 0
release_safe in positive      = 0
unsupported forced positive   = 0
duplicate identity            = 0
identity closure              = 800/800
```

## Architecture decisions

1. `regrasp_or_instability` set to `False` — `event_opening_stable` provides only positive stability evidence, not instability detection
2. `manipulation_active` uses conjunction of `grasp_support AND retention_active AND in_close_segment` — requires both privileged grasp and retention evidence
3. Release-safe margin = ±3 steps around `event_release_onset` or `release_imminent`
4. `target_relevant` determined per-episode by presence of at least one close event
5. Dense `burst_feasible_t` mask (50K starts) — not just `first_feasible_start`
6. No final segment tier, global ranking, attack outcome, or privileged future in student view

## Known limitations

1. **Wide positive universe**: 90.75% feasible rate and 50K starts may be overly permissive. The conjunction relies heavily on `grasp_support AND retention_active` which are present in most close segments. Future audit should check whether `burst_feasible_t` windows truly represent gripper-critical opportunities or simply "any sustained close with grasp evidence."

2. **No instability detection**: `event_opening_stable` cannot indicate regrasping. Multi-close episodes may contain transitions that should be excluded but aren't.

3. **No lift/support_removed/target_progress**: These fields don't exist in V2.1. `manipulation_active` uses `grasp_support + retention_active` as a proxy, which may not capture task-phase-specific constraints.

4. **Suite skew**: `libero_goal` has only 72.5% feasible rate vs 96-98.5% for other suites. Some goal tasks involve non-gripper mechanisms (button push, door open) where K10 gripper intervention is not meaningful.

## Next steps (NOT AUTHORIZED — audit only)

```
R7_R2_OFFLINE_REPLAY         = NOT STARTED
R7_R3_TRAINING               = NOT STARTED
R7_R4_EXACT_PREFIX           = NOT STARTED
R7_R5_ATTACK_CANARY          = NOT STARTED
```
