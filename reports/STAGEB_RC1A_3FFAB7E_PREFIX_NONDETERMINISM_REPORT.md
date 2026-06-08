# Stage-B RC1a — Prefix Nondeterminism Report (Gate B)

**Date**: 2026-06-08
**Gate**: B — FAIL (fixed)
**Runner fix commit**: [pending]

## Finding

**6/8 confirmation parents have non-deterministic clean rollout prefixes across repeat seeds.**

The runner's single `--seed` parameter coupled both environment replay determinism and attack perturbation initialization. When repeat seeds differed (to vary attack), the env replay also changed, producing different pre-window trajectories.

## Evidence

| Parent | r0 hash | r1 hash | Match? |
|--------|---------|---------|--------|
| salad_dressing s2 [120,130] | 79a18525 | b576a79a | ❌ |
| milk s0 [70,80] | e92cc9bb | e92cc9bb | ✅ |
| bbq_sauce s2 [200,210] | 19a6487f | 11dcdb10 | ❌ (edge) |
| bbq_sauce s2 [100,110] | e822c2a9 | 5cd300a6 | ❌ |
| milk s0 [230,240] | 388b84e5 | 1f86d248 | ❌ |
| cream_cheese s1 [145,155] | 55984870 | 55984870 | ✅ (edge) |
| tomato_sauce s2 [150,160] | ceba1e4d | 55099be4 | ❌ |
| tomato_sauce s2 [90,100] | 465e9103 | 378bb394 | ❌ |

## Root Cause

Runner code (before fix):
```python
ap.add_argument('--seed', type=int, default=0)      # single seed
...
env.seed(args.seed)                                    # env replay
attacker = TokenPrefixPGDAttacker(..., seed=args.seed) # attack init
rng = np.random.RandomState(args.seed + args.job_id)   # random perturbation
```

`--seed` controlled all three: env replay, VIS PGD init, and RAND perturbation. Making the attack stochastic also made the baseline trajectory different.

## Fix Applied

Added `--env_seed` and `--attack_seed` to the runner (backward compatible — both default to `--seed`):

```python
ap.add_argument('--env_seed', type=int, default=None)
ap.add_argument('--attack_seed', type=int, default=None)
_env_seed = args.env_seed if args.env_seed is not None else args.seed
_attack_seed = args.attack_seed if args.attack_seed is not None else args.seed
env.seed(_env_seed)                          # fixed for K-repeat
attacker = TokenPrefixPGDAttacker(..., seed=_attack_seed)  # varied for K-repeat
rng = np.random.RandomState(_attack_seed + args.job_id)
```

## Implications for 8/8 Unstable Finding

The original "8/8 unstable" finding from single-shot confirmation may be partially inflated by this coupling. The true attack stochasticity could be lower than the raw 8/8 suggests. This will be resolved by the K=5 round using:
- Fixed `--env_seed` (same clean prefix for all K repeats of a parent)
- Varied `--attack_seed` (different perturbation realizations)

## Gate B Status

**FAIL → RESOLVED**. Runner fixed. Proceed to Phase C with `--env_seed` and `--attack_seed` separation.
