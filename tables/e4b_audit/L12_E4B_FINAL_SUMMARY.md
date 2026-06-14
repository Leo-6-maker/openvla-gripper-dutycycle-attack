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

## Feature Discriminability

Features classified by type:

| Feature | Type | P vs best non-P |
|---------|------|----------------|
| total_score | discrete_score | mixed: high=2 low=0 tie=0 |
| raw_crossing_bonus | candidate_definition | always tied (8/8) |
| close_streak_bonus | candidate_definition | always tied (8/8) |
| close_onset_qpos_bonus | candidate_definition | always tied (8/8) |
| eef_deceleration_bonus | discrete_score | mixed: high=2 low=0 tie=0 |
| qpos_ready_bonus | candidate_definition | always tied (8/8) |
| eef_speed_now | continuous_dynamic | mixed: high=5 low=0 tie=1 |
| eef_speed_prev | continuous_dynamic | mixed: high=6 low=0 tie=0 |
| eef_deceleration_delta | continuous_dynamic | mixed: high=0 low=4 tie=0 |
| close_streak | candidate_definition | always tied (8/8) |
| raw_crossing | candidate_definition | always tied (8/8) |
| close_onset | candidate_definition | always tied (8/8) |
| qpos | continuous_dynamic | mixed: high=2 low=1 tie=5 |
| time_since_prev_close | temporal_context | mixed: high=2 low=0 tie=0 |
| candidate_index | temporal_context | mixed: high=2 low=2 tie=0 |

## Key Findings

1. Four candidate-definition discrete score components (raw_crossing, close_streak, close_onset_qpos, qpos_ready) produce identical values for ALL close-event candidates on all traces with non-P comparators — zero within-trace discrimination.

2. EEF-related continuous features (speed_now, speed_prev, deceleration_delta) are among the few signals that vary across candidates. Their correct ranking direction and per-trace consistency have not been established.

3. The current scalar score predominantly reflects candidate-definition features, resulting in Teacher-P unique top-1 in only 4/10 P-available traces.

4. Causal peak-hold policies add delay without improving online near-correct rate.

5. Establishing discriminative deployment-safe features for critical-close identification requires moving beyond candidate-definition signals toward continuous dynamic and temporal-context features with validated direction.
