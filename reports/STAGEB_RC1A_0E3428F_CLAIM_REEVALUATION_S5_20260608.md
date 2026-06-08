# Stage-B RC1a 0e3428f — Claim Re-Evaluation After S5 K5 Mid-Run

**Date**: 2026-06-08
**Commit**: 0e3428f
**Status**: Claims revised upward — detector path REOPENED

## Root Cause of Old 8/8 Instability

The old runner (`e33b5e4`) used a single `--seed` parameter that coupled:
1. Environment replay seed (`env.seed(seed)`)
2. VIS attacker initialization seed (`TokenPrefixPGDAttacker(seed=...)`)
3. RAND perturbation seed (`np.random.RandomState(seed + job_id)`)

When repeat seeds differed (to vary attack), the env replay also changed, producing different clean prefixes. Gate B prefix audit confirmed 6/8 parents had non-matching pre-window trajectories.

**The 8/8 instability was protocol-confounded, not proof of intrinsic attack stochasticity.**

## Revised Claims

### Retained (unchanged)
1. RC1a gripper semantics correction is correct
2. Corrected VIS produces command OPEN on selected windows
3. Random-sensitive behavior is real and must be modeled
4. S5 repeat-stability protocol is necessary

### Upgraded (previously downgraded, now restored)
1. **Detector training is feasible** — but ONLY on K-repeat stable labels, not single-shot
2. **72-pair pool is still valid** — as Bronze candidate discovery, not final labels
3. **Stable label structure exists** — K5 mid-run shows 4/6 parents with clear stable labels (cmd, neg, rand)
4. **Abstain head may be real** — pending recomputation on stable labels
5. **Visual sidecar was tested on protocol-bugged labels** — not a final rejection

### Downgraded (still)
1. ~~Single-shot labels can train detector~~ — NOT YET. K5 stable labels needed first.
2. ~~8/8 unstable proves attack is random~~ — REVERSED. It was protocol bug.
3. ~~Detector path is closed~~ — REVERSED. Path reopened under fixed-env K-repeat.

## K5 Mid-Run Evidence

| Parent | K | pV_cmd | pR_cmd | Stable Label |
|--------|---|--------|--------|-------------|
| milk [70,80] | 5 | 1.0 | 0.0 | stable_cmd_specific |
| milk [230,240] | 3 | 1.0 | 0.0 | stable_cmd_specific (was "confounded"!) |
| tomato [90,100] | 5 | 1.0 | 0.0 | stable_cmd+vis_phys (was "rand_phys"!) |
| bbq [100,110] | 5 | 0.0 | 0.0 | stable_negative (was "cmd+phys"!) |
| salad [120,130] | 4 | 0.0 | 0.0 | stable_negative |
| tomato [115,125] | 4 | 0.8 | 0.8 | stable_rand_sensitive (genuine confound!) |

**Conclusion**: Under fixed env_seed, windows show clear, interpretable stability patterns. The corrected VIS mechanism is real and repeatable. Detector training is feasible on K-repeat stable labels.

## New Detector Definition

Input: online-safe clean features (proprio, timing, task prior, optional action-hidden/visual)
Output: multi-head probability estimates

```
Head A: p(stable_cmd_specific | window)
Head B: p(stable_vis_specific_phys | window)
Head C: p(stable_rand_sensitive | window) — abstain
Head D: p(stable_negative | window)

attack_score = expected_VIS_yield - λ * random_risk - μ * uncertainty
```

## Forbidden Claims
- Detector is solved (still need K-repeat training labels)
- 72-pair single-shot labels as ground truth
- Old 8/8 unstable as evidence of attack stochasticity
