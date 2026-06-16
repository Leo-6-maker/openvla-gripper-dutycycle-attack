# Layer 1/2 G4-R — Canonical Live Canary

**Date:** 2026-06-16/17 | **Commits:** `85bfda1`+

## Result: ALL 6/6 PASS

| Parent | OFF | ON | Raw act diff | Env act diff | D5 emit | Outcome |
|--------|------|-----|-------------|-------------|---------|---------|
| alphabet_soup_s2 | 144/1 | 144/1 | 0 | 0 | 55 | Match |
| bbq_sauce_s27 | 129/1 | 129/1 | 0 | 0 | 61 | Match |
| butter_s2 | 158/1 | 158/1 | 0 | 0 | 68 | Match |
| orange_juice_s8 | 111/1 | 111/1 | 0 | 0 | 42 | Match |
| tomato_sauce_s2 | 168/1 | 168/1 | 0 | 0 | 97 | Match |
| alphabet_soup_s17 | 280/0 | 280/0 | 0 | 0 | 49 | Match |

- 6/6 outcome match (5/6 success, 1/6 consistent failure)
- 6/6 action hash exact match (0 diffs)
- 6/6 env action hash exact match (0 diffs)
- 6/6 trace length match
- 0 abstained emissions
- 0 second emissions

## Detector Latency

| Parent | Steps | Mean | p50 | p99 | Max |
|--------|-------|------|-----|-----|-----|
| alphabet_soup_s2 | 144 | 38us | 24us | 58us | 1.9ms |
| bbq_sauce_s27 | 129 | 60us | 24us | 2.2ms | 2.4ms |
| butter_s2 | 158 | 95us | 24us | 3.4ms | 3.5ms |
| orange_juice_s8 | 111 | 40us | 24us | 61us | 1.7ms |
| tomato_sauce_s2 | 168 | 67us | 24us | 3.2ms | 3.8ms |
| alphabet_soup_s17 | 280 | 223us | 24us | 5.4ms | 5.5ms |

All p99 < 5.5ms. Detector overhead is negligible relative to VLA inference (~1.1s).

## GPU Pair Variance

Cross-GPU OFF/ON comparison FAILS (action hash divergence). Same-GPU retry PASSES.
All final results use GPU pair (2,6) for both OFF and ON.

Production rule: matched OFF/ON MUST use same GPU pair, same device map, same dtype, same attention implementation.

## GPU Health

| GPU | Status |
|-----|--------|
| 0 | FAULT (Xid 13+43) — excluded |
| 1 | HEALTHY |
| 2 | HEALTHY (M3 qual PASS) |
| 3 | QUARANTINED (new Xid 31 during G4) |
| 4 | FAULT (non-deterministic) — excluded |
| 5 | HEALTHY |
| 6 | HEALTHY (M3 qual PASS) |
| 7 | RENDER-ONLY (historical Xid 31, stable in G4) |

Authorized pairs: (2,6), (1,5). (5,7) forward-only. (1,3) NOT AUTHORIZED.
