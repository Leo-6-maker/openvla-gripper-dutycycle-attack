# L12 E4B Final Summary

## Teacher-P Score Rank

- Unique top-1 by score: 4/10
- Competition-rank top-2: 7/10

## Local-Maximum Policy

- Coverage: 6/10 emitted decisions
- Overall near-correct: 3/10
- Conditional: 3/6
- No-decision: 4/10
- Avg delay (emitted): 51.3 steps

## Feature Discriminability (8 traces with non-P comparators)

| Feature | Type | P_high | P_low | tie | other | Status |
|---------|------|--------|-------|-----|-------|--------|
| total_score | discrete_score | 2 | 0 | 0 | 6 | ok |
| raw_crossing_bonus | candidate_definition | 0 | 0 | 8 | 0 | ok |
| close_streak_bonus | candidate_definition | 0 | 0 | 8 | 0 | ok |
| close_onset_qpos_bonus | qpos_conditioned_discrete | 0 | 0 | 8 | 0 | ok |
| eef_deceleration_bonus | discrete_score | 2 | 0 | 0 | 6 | ok |
| qpos_ready_bonus | qpos_conditioned_discrete | 0 | 0 | 8 | 0 | ok |
| eef_speed_now | continuous_dynamic | 5 | 0 | 1 | 2 | ok |
| eef_speed_prev | continuous_dynamic | 6 | 0 | 0 | 2 | ok |
| eef_deceleration_delta | continuous_dynamic | 0 | 4 | 0 | 4 | ok |
| close_streak | candidate_definition | 0 | 0 | 8 | 0 | ok |
| raw_crossing | candidate_definition | 0 | 0 | 8 | 0 | ok |
| close_onset | candidate_definition | 0 | 0 | 8 | 0 | ok |
| qpos | continuous_dynamic | 2 | 1 | 5 | 0 | ok |
| time_since_prev_close | temporal_context | 2 | 0 | 0 | 4 | ok |
| time_since_last_open | temporal_context | - | - | - | - | unavailable_all_missing |
| candidate_index | temporal_context | 2 | 2 | 0 | 4 | ok |

## Key Findings

1. Four saturated discrete score components (raw_crossing, close_streak, close_onset_qpos, qpos_ready, and their bonuses) produce identical values for ALL close-event candidates on all traces with non-P comparators — zero within-trace discrimination.

2. EEF-related continuous features (speed_now, speed_prev, deceleration_delta) vary across candidates and Teacher-P shows distinct dynamics (speed_now: P_higher=5/8, speed_prev: P_higher=6/8, decel_delta: P_lower=4/8), but their correct ranking direction and per-trace consistency have not been established.

3. The current scalar score reflects saturated discrete features, resulting in Teacher-P unique top-1 in only 4/10 P-available traces.

4. Causal peak-hold policies add delay without improving online near-correct rate.

5. The current scoring function only coarsely binarizes EEF dynamics into a 0/0.5 bonus and does not effectively exploit the continuous variation for candidate ranking.

6. Establishing discriminative deployment-safe features requires moving beyond saturated discrete signals toward continuous dynamic and temporal-context features with validated ranking direction.
