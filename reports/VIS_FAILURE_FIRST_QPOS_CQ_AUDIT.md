# VIS Failure-First Qpos/CQ Audit

**Date**: 2026-06-02

Status: pending rollout completion.

Audit requirements:

- Use post-step qpos and width deltas for phase-wise physical response.
- Treat `official_success` as an outcome field, not a VIS-specific failure explanation.
- Keep `cq_success`, `cq_failure`, and `manual_audit_needed` explicit even when CQ is unavailable.
- Mark any low-qpos failure or matched-random failure as needing manual review.
- Confirm whether failure occurs during or after an attacked phase.

The audit will be populated after parsing completed traces from `/data/liuyu/outputs/vis_failure_first_multiphase_20260602/`.
