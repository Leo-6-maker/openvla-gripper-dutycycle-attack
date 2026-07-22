# Exact-W32 FSM Replay Erratum — 2026-07-22

**Status:** RETRACTED_DUE_TO_KNOWN_MASK_BUG → CORRECTED_HEAD_LEVEL_REPLAY

## Retracted Findings

The initial offline FSM replay (run ~2026-07-22 11:50 CST, script `tmp_offline_fsm_analysis.py`) contained a critical known-mask violation. The following claims are **RETRACTED**:

| Retracted Claim | Bug |
|-----------------|-----|
| o2_i0 BG any-head P50 = 0.516 | manipulation head included without known_mask check |
| 437-520 step consecutive background emit | manipulation don't-care outputs extended streaks |
| manipulation bg emit = 39.34% | manipulation_known_mask is False on ~all BG steps |
| "100-500 step consecutive bg emit" | streak counter not reset at mask gaps or episode boundaries |

## Bug Details

**Script:** `tmp_offline_fsm_analysis.py` (NOT committed, temporary diagnostic)

**Error:** The any-head union computation used:
```python
any_score = max(s['grasp_prob'], s['manipulation_prob'], s['release_prob'])
```
without requiring the corresponding `known_mask` to be True.

**Why this matters:** On background steps (`event_id < 0`), `manipulation_known_mask` is False for virtually all steps:
- o1_i2: 0/23760 BG steps have manipulation_known_mask=True
- o2_i0: 1/24467 BG steps have manipulation_known_mask=True

The manipulation head produces high probability values on background as "don't-care" outputs — the Teacher provides no manipulation supervision during background. Including these values in the any-head union inflated the false-emit rate by ~3x.

## Corrected Rules

For each head `h` ∈ {grasp, manipulation, release}:

```
emit_h = known_mask_h
         AND route_supported
         AND prob_h >= threshold_h
```

**Mandatory constraints:**

1. `known_mask_h` = False → head excluded from any-head union
2. Unknown is NOT negative — excluded from denominator
3. Unknown step does NOT extend continuous-emit streak
4. Episode boundary resets streak to 0
5. Valid-mask gap resets streak to 0
6. `route_supported` = False → all heads excluded (unsupported_route counter)
7. Streak counter operates per-episode, per-step within episode

## Verification

| Check | o1_i2 | o2_i0 |
|-------|-------|-------|
| BG steps | 23760 | 24467 |
| grasp_known on BG | 23760 (100%) | 24467 (100%) |
| manipulation_known on BG | 0 (0%) | 1 (~0%) |
| release_known on BG | 23760 (100%) | 24467 (100%) |
| any-head union effectively | grasp ∪ release | grasp ∪ release |
| Corrected any-emit step rate | 0.0928 | 0.1742 |
| Matches evaluator | YES (0.093 vs 0.0928) | YES (0.174 vs 0.1742) |

## Corrected Consecutive Emit (known-only, tau=0.5)

| Metric | o1_i2 | o2_i0 |
|--------|-------|-------|
| Max consecutive (known-heads only) | 106 | 292 |
| Episodes with >1 consecutive | 136/180 | 156/180 |
| Episodes with >2 consecutive | 112/180 | 147/180 |

These remain elevated but ~2-3x lower than the retracted values (437, 520).

## Corrected Pareto Frontier

420 points scanned (10 tau_g × 6 tau_r × 7 persistence).

**Cross-split (o1_i2 + o2_i0):** 0 viable operating points under constraints:
- episode false_rate ≤ 0.10 (both splits)
- release event recall ≥ 0.50 (both splits)

**Minimum achievable false_rate:** 0.133 (tau_g=0.94, k=8) with recall=0 (not a feasible point).

**Under false_rate ≤ 0.10:** maximum recall ≈ 0.0 (no point reaches this false_rate for o2_i0 with positive recall).

## Non-Retracted Findings

The following findings from the original analysis remain valid (verified against evaluator output):

- Release AUPRC stable across splits (0.832-0.884)
- Short-event AUPRC strong (0.927-0.985)
- Release overlap uniformly safe (0.007-0.017)
- Unsupported route emit universally 0.0
- Grasp head dominates background false emit
- o1_i1 and o1_i2 pass frozen raw any-head gate

## Artifact Binding

- Erratum author: Claude Opus 4.7 (Codex agent)
- Erratum timestamp: 2026-07-22 ~12:15 CST
- Corrected replay script: `tmp_fsm_v2_corrected.py` (SHA in commit)
- Evaluator verified against: `scripts/detector_v5/evaluate_factorized_v2_inner_cv.py` line 167
