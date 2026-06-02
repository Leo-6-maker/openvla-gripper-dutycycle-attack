# Proprio + Visual Re-ranker — Longrun Handoff

**Date**: 2026-05-31 | **Branch**: `exp/proprio-visual-reranker-ablation-20260530` | **Status**: re-ranker ablation complete

## Summary

Investigated whether visual can serve as a re-ranker on ProprioNoStep's contact-timed trigger windows. **Result: Current visual models cannot — scores at contact phase are near-zero and non-discriminative.**

## Key Finding

Visual model scores have the wrong temporal profile:
- Highest at episode start (step 0-10): "scene novelty"
- Decay to near-zero by contact phase (step 100-200)
- ProprioNoStep fires at contact phase → visual scores already zero
- No discrimination between vulnerable and robust windows

## Complete Picture (3 Independent Lines of Evidence)

| Approach | Result | Reason |
|----------|--------|--------|
| Visual standalone (V6) | Pre-contact, non-selective | Scores highest at start |
| Visual standalone (v2) | Pre-contact, non-selective | Same — learns scene difficulty |
| Proprio + Visual re-ranker | No improvement | Scores zero at contact phase |

**Root cause convergence**: Visual models (DINOv2+SigLIP) encode static scene appearance. The temporal profile is wrong — scores peak at episode start and decay. Without temporal/motion features, visual cannot distinguish contact-phase vulnerability from scene-level difficulty.

## Production Line (Unchanged)

- **Detector**: ProprioNoStep
- **Attack**: sustained_command_open_proxy_30
- **Selectivity**: High 0/10, Robust 10/10
- **Status**: production_ready_for_group_meeting

## Future Directions

To make visual useful for attack-relevance:
1. **Motion features**: Use frame differences (delta features) instead of static appearance
2. **Contact-only training**: Train only on contact-phase frames (step 100+)
3. **Temporal contrastive**: Learn to detect "change at contact" vs "static scene"
4. **Optical flow**: Use pretrained flow features as additional input

All require new feature extraction and training design — not a quick fix.

## GPU Status

- All GPUs idle (GPU0 quarantined)
- No fresh Xid (last: 5/29)
- No active jobs

## Reports Written

- `reports/RERANKER_ABLATION_RESULTS.md`
- `reports/VISUAL_V2_BASELINE_SANITY_AUDIT.md`
- `reports/VISUAL_V2_FULL10_OFFLINE_REPLAY.md`
- `reports/VISUAL_V2_PHASE_C_HANDOFF_20260530.md`
- `reports/NEXT_PHASE_VISUAL_RERANKER_PLAN.md`

## Valid Claims

- ProprioNoStep is the production online detector.
- sustained_command_open_proxy_30 selectively causes failures on high oracle-sensitive tasks (0/10) while preserving robust controls (10/10).
- Visual signal encodes scene appearance; its temporal profile peaks at episode start and decays — it does not detect contact-phase vulnerability.
- Proprio + Visual re-ranker with current models does not improve selectivity.
- Visual information is not useless — but requires fundamentally different training (motion/temporal features) to be useful for contact-phase re-ranking.

## Forbidden Claims

- VIS attack successful / failed
- Visual information useless
- Visual production-ready
- Universal attack
- Detector oracle-optimal
