# Stage-B RC1a 26da793 — S5 Claim Boundary

**Date**: 2026-06-08
**Anchor**: 26da793 / a080bee
**Protocol**: S5 repeat-stability-first stochastic window selection

## Allowed Claims

1. RC1a gripper semantics correction is necessary and correct
2. Corrected VIS objective can produce command-level OPEN on selected windows
3. Physical bridge/qpos transfer exists but shows seed-dependent instability
4. Random-sensitive and random-confounded behavior is real and must be modeled
5. Single-shot VIS/RAND labels are NOT reliable ground truth for detector training
6. 8/8 Silver confirmation parents failed repeat-stability testing
7. Future detector labels require K-repeat confirmation (K >= 5)
8. The 72-pair pool is Bronze single-shot exploratory — useful for taxonomy and candidate discovery
9. Clean timing + proprio features show exploratory signal for identifying perturbation-sensitive windows
10. S5 repeat-stability-first protocol is the correct next direction

## Forbidden Claims

1. ❌ Detector is solved or near-solved
2. ❌ 72-pair pool can train a final detector
3. ❌ abstain head AUROC=0.889 is detector evidence (single-shot readout only)
4. ❌ Single-shot labels can be used as ground truth
5. ❌ cmd_specific prediction from clean features is reliable
6. ❌ Global frozen visual embedding is permanently disqualified (tested on unstable labels)
7. ❌ Hard negative can be defined by clean heuristic alone
8. ❌ Visual sidecar conclusively proved visual features are useless
9. ❌ Random-sensitive equals negative (must be abstain)
10. ❌ Detector AUROC/P@K from single-shot labels supports any performance claim

## Downgraded Artifacts

| Artifact | Old Status | New Status |
|----------|-----------|-----------|
| 72-pair pool labels | Training labels | Bronze single-shot exploratory |
| Multi-head readout AUROC=0.761/0.917 | Detector evidence | Exploratory readout only |
| Visual sidecar negative result | Visual permanently disproven | Tested on unstable labels; re-evaluate on stable pool |
| abstain_any head | Strongest detector signal | Single-shot signal; pending stable-label verification |
| cmd_specific head | Task-biased | Labels themselves unstable; cannot assess |

## S5 Protocol

```
Stage 1: Bronze single-shot exploration ✅ (72 pairs)
Stage 2: Silver K-repeat confirmation ← CURRENT (K=5, 8 parents, 80 jobs)
Stage 3: Stability-gated label construction
Stage 4: Detector trained ONLY on stable-subset labels
```

## Gate A Status

- [PASS] 72-pair detector readout marked exploratory
- [PASS] Visual sidecar negative result marked exploratory
- [PASS] 8/8 unstable written in instability report
- [PASS] Forbidden claims updated
- [PASS] S5 protocol direction declared
