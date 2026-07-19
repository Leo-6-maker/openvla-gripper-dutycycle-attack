# R7 K10 Gripper-Critical Opportunity Labeler V1

## Scope

Derives `critical_t` and `burst_feasible_t` labels from sealed S1 Physics Teacher V2.1 records. Does NOT modify, overwrite, or replace the source Teacher. This is a read-only label layer that maps existing clean physical evidence to K=10 attack-executor-aligned opportunity labels.

## Source

- **Physics Teacher root:** `OFFICIAL_V3_S1_FIT_V1_5e27d7c`
- **Schema:** `B3_OFFICIAL_V3_TEACHER_RECORD_V1`
- **Feature rebuilder SHA:** `d3a1aacacdffddc3ef1c0679f7ebf82159be0a37314e9eb867912c05e7ae23f1`
- **Identities:** 800 FIT (states 0–19), 4 suites × 10 tasks × 20 states

## Label definitions

### critical_t

```
critical_t =
    label_known_t
    AND candidate_close_t
    AND target_relevant_t
    AND stable_grasp_t
    AND manipulation_active_t
    AND NOT release_safe_t
```

Where:

| Component | Source field(s) | Notes |
|---|---|---|
| `label_known` | `retention_unknown_mask == False` AND `event_evidence_valid == True` | 7,200 steps masked |
| `candidate_close` | Step is within a close event segment (`event_close_onset` through `event_end_step`) | 1,130 segments across 800 eps |
| `target_relevant` | Episode has at least one close event | `libero_goal/t00` and `libero_goal/t05` excluded |
| `stable_grasp` | `event_support == True` | 69,441 steps |
| `manipulation_active` | `grasp_support == True` AND `retention_active == True` AND in close segment | Both fields unmasked |
| `NOT release_safe` | Step NOT within ±3 steps of `event_release_onset` or `release_imminent` | 3,385 steps marked release_safe |

`regrasp_or_instability` is set to `False` because `event_opening_stable` is either `None` (unknown) or `True` (stable) — it never indicates instability. This component is reserved for future Teacher versions with explicit instability evidence.

### burst_feasible_t

```
burst_feasible_t = AND_{j=0}^{K-1} critical_{t+j}
```

Constraints:
1. All K steps must be `critical_t == True`
2. All K steps must belong to the SAME candidate segment
3. No unknown gap crossing within the K-step window
4. No release_safe step within the K-step window
5. Window fully contained within episode (no horizon crossing)
6. K = 10 (fixed)

## Output schema

Per-identity step labels (`k10_labels.jsonl`):
- `step`, `episode_key`, `candidate_segment_id`
- `candidate_close`, `label_known`, `target_relevant`
- `stable_grasp`, `manipulation_active`, `release_safe`
- `critical_t`, `burst_feasible_t`, `is_feasible_start`
- `teacher_reason_code`, `teacher_source_sha`

Per-identity segment summary (`k10_segments.json`):
- `segment_id`, `onset`, `end`, `duration`
- `critical_step_count`, `feasible_start_count`
- `has_unknown`, `has_release_event`

Episode summary (`EPISODE_SUMMARY.csv`):
- `has_candidate_segment`, `has_feasible_k10`, `feasible_start_count`
- `first_feasible_start`, `last_feasible_start`, `no_feasible_reason`
- `mechanism_supported`

## Hard gates

| Gate | Requirement | Status |
|---|---|---|
| Identity closure | 800/800 | PASS |
| Segment crossing | 0 | PASS |
| K10 out-of-bound | 0 | PASS |
| Unknown in positive | 0 | PASS |
| Release-safe in positive | 0 | PASS |
| Unsupported task forced | 0 | PASS |
| Duplicate identity | 0 | PASS |
| Protected reads | 0 | PASS |
| Attack outcome reads | 0 | PASS |
| Source mutation | 0 | PASS |

## Server artifact

```
Root: OFFICIAL_V3_R7_K10_OPPORTUNITY_LABELER_V1_66f3604_20260719
SHA256SUMS: 665c4f62cb17162de0739517ff260b6abe011512d191c4a93440179cadcf49d6
```
