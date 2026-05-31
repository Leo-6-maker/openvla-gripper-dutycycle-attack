# Final Claims Audit After PR #5

**Date**: 2026-06-01

## Valid Claims

1. ProprioNoStep + sustained_command_open_proxy_30 selectively disrupts high oracle-sensitive Object tasks while preserving robust controls (High 0/10, Robust 10/10).
2. Effect is Object-suite validated.
3. Proprioceptive signal naturally encodes gripper-object contact dynamics — this is why ProprioNoStep fires at contact/transport/placement phase.
4. Static visual features (DINOv2+SigLIP) fail contact-phase timing across four independent attempts: V6 frozen, v2 trained, re-ranker, contact-aware delta.
5. VIS token-prefix PGD at ε≤8/255 does not flip decoded gripper token on verified contact/carry frames — CE drops but argmax unchanged.
6. Cross-suite zero-shot transfer is limited by workspace distribution shift (eef_z 5.7x). Spatial 65% partial, Goal 8% insufficient.

## Forbidden Claims

1. VIS attack successful — PGD mechanism works but no decoded gripper effect.
2. Universal attack — selectivity is task-dependent.
3. Detector oracle-optimal — ProprioNoStep is best practical, not proven optimal.
4. ProprioNoStep universal across LIBERO — Object only; Spatial/Goal transfer limited.
5. Cross-suite attack ready — Goal 8% trigger, sus30 blocked.
6. Command-layer sus30 equals VIS — sus30 is command-layer proxy, not visual perturbation.
7. Forced-window VIS works — not tested (blocked by VIS-1 gate).
8. Detector-triggered VIS works — not tested (blocked by VIS-1 gate).
9. Visual information useless — visual signal encodes task difficulty; just not contact timing.
