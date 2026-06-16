# Layer 1/2 D5 v1 — True Online Final Acceptance

**Date:** 2026-06-17
**Tag:** `l12-d5-v1-production-20260617`
**Commit:** `593ffad`
**Branch:** `exp/l12-production-streaming-adapter-20260615`

## Final Status

| Module | Status | Evidence |
|--------|--------|----------|
| LAYER1_CAPTURE_AND_LABELS | **PASS** | Teacher-P auditor 120/120, 0 disagreements |
| LAYER2_FROZEN_REPLAY | **PASS** | D5 tau=0.050, internal 76% |
| LAYER2_HISTORICAL_FUNCTIONAL_PARITY | **PASS 154/154** | Candidate count/step/abstain/emit 100% |
| LAYER2_LIVE_FEATURE_PARITY | **153/154 + 1 waiver** | alphabet_soup_s17 archival precision |
| LAYER2_CANONICAL_OFF_ON_CANARY | **PASS 6/6** | 0 action/env hash diffs |
| LAYER2_TRUE_ONLINE_STREAMING | **PASS** | Full pipeline verified |
| LAYER2_EXTERNAL_GENERALIZATION | MEASURED, NOT SOLVED | 34-state 59% in-window |
| LAYER2_TIMING | NOT FINAL | early 19% internal, 33% external |
| LAYER2_TO_LAYER3 | **NOT AUTHORIZED** | |

## Production Bundle

`configs/d5_v1_production_bundle.json` — v1.0.0

| Component | SHA256 |
|-----------|--------|
| Detector | `af6d58c7...` |
| Adapter | `81ee7fd3...` |
| Runtime | `d8621637...` |
| Collector | `d3dd1360...` |
| Checkpoint | `7eea609f...` |
| Config | `d6f6af61...` |
| Labels | `e731c273...` |
| Manifest | `fe125a55...` |

## Verification Gates

| Gate | Tests | Result |
|------|-------|--------|
| Adapter | 17/17 | PASS |
| Detector | 22/22 | PASS |
| Bundle verifier | 20/20 | PASS |
| G3 functional parity | 154/154 | PASS |
| G4 canonical canary | 6/6 | PASS |
| G5 regression | OFF 144/1 ON 144/1 | PASS |

## GPU Authorization

- Primary: (2,6)
- Fallback: (1,5)
- Forward-only: (5,7)
- Quarantined: GPU3 (Xid 31)
- Excluded: GPU0, GPU4

## Production Rules

1. Matched OFF/ON MUST use same GPU pair
2. Never compare across GPU pairs
3. Bundle verifier MUST pass before any production run
4. Detector is read-only — no action modification
5. Abstained candidates NEVER emit
