# VIS Correct Decode — Decision

**Date**: 2026-06-01 | **Phase**: F

## Decision: Case A — Strong No-Rollout PASS

### Gate Summary

| Gate | Criterion | Result | Status |
|------|-----------|--------|--------|
| S1 | Direction semantics correct | 0.0→0.996 = NEUTRAL→FULL OPEN | **PASS** |
| B | ketchup contact >=8/10 reproducible | 10/10 (100%) | **PASS** |
| B | Random baseline negative | 0/5 token flips | **PASS** |
| C | Multi-frame (>=2 frames) | 10+ frames across 3 tasks | **PASS** |
| C | High-sensitive frame positive | tomato_0130,0138,0142,0150; cream_cheese 0065,0070,0080,0085 | **PASS** |
| C | Robust frame not generic collapse | Random baseline proves specificity | **PASS** |
| E | Arm drift controlled | Best: cream_cheese_0070 armL2=0.11 | **PASS** |
| E | Not arm-drift dominated | cream_cheese_0075: armL2=1.07, grip delta=0 | **PASS** |

### Key Evidence

1. **Direction confirmed**: 0.0 (neutral) → 0.996 (fully open), correct for sustained-open attack
2. **Deterministic repeat**: 10/10 on ketchup_0098, 3/3 on multiple tomato and cream_cheese frames
3. **High-sensitive positive**: Both tomato_sauce and cream_cheese (high-sensitive tasks) show grip flips
4. **Random baseline**: 0/5 flips on all Phase B frames — effect is PGD-specific
5. **Arm specificity**: cream_cheese_0070 shows full grip flip with only 0.11 arm L2
6. **Arm-drift proof**: cream_cheese_0075 shows arm L2=1.07 but grip delta=0 — effect is not arm-drift

### Frame-Specificity Observation

Adjacent frames can behave completely differently (e.g., step_0098 works, step_0099 fails). This is expected for a token-boundary effect and suggests the PGD exploits specific visual states where the open/close decision boundary is near the current token's representation.

### Remaining Concerns

1. The effect is frame-specific — not all contact-phase frames show it
2. The arm L2 on ketchup (0.84) remains elevated compared to best frames
3. The effect has only been tested on single frames, not temporal sequences
4. GPU1-3 pair had Xid 31 crash during extended testing — hardware reliability is a concern

### Rollout Status

**Rollout remains BLOCKED.** This is a strong no-rollout payload signal but a controlled rollout proposal should be written and approved before any execution.

## Next Steps

1. Write controlled rollout proposal (Phase G) — do NOT execute
2. Backup and handoff (Phase H)
3. Proposal requires Leon approval before any rollout execution
