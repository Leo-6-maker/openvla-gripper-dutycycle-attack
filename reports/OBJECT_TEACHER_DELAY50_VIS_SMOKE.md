# Object Teacher-Oracle Delay=-50 VIS Smoke

**Date**: 2026-06-04
**Method**: Teacher-oracle window [T_gform-50, T_gform-33]
**Objective**: prefix_locked_gripper_open_margin, eps_raw_pixels=6
**Status**: 3 VIS completed, 2/3 task-positive

---

## Results

| Task | Phase bin | Lead | VIS OPEN | qpos_delta | done | Denominator | Claim |
|------|-----------|------|----------|-----------|------|-------------|-------|
| **ketchup** | far_closed | -50 | 18/18 | 0.038 strong | False | clean | **YES** |
| **butter** | near_closed | -47 | 18/18 | 0.038 strong | False | manual merge* | **YES after merge** |
| alphabet_soup | far_closed | -50 | 18/18 | 0.028 weak | False | missing random | No |

*butter: clean/random traces and VIS trace were in different audit groups due to run_id mismatch.
Manual merge confirmed: task=butter, state=0, window=[29,46], eps=6, objective matches.
After merge, butter is claim_usable.

## Interpretation

Closed pre-grasp teacher-oracle probes produced:
- **2 task-positive** (ketchup, butter): 18/18 OPEN, strong physical qpos, timeout failure
- **1 weak/incomplete** (alphabet_soup): 18/18 OPEN but qpos=0.028 (weak), missing random denominator

This is **NOT** a simple "lead=-50 works" result. The phase-response is task-dependent:
- ketchup at far_closed (lead=-50) is vulnerable
- butter at near_closed (lead=-47) is vulnerable
- alphabet_soup at far_closed (lead=-50) shows weaker physical coupling

## Claim boundaries

**Allowed:**
- "Closed pre-grasp windows can be task-critical for Object tasks."
- "ketchup and butter produce 18/18 OPEN, strong qpos, and timeout under clean controls."
- "Task-dependent phase-response requires further mapping."

**Forbidden:**
- "far_closed is vulnerable"
- "lead=-50 works"
- "student target is ready"
- "Object generalization established"

## Pending

- alphabet_soup: rerun random to complete denominator
- phase-response mini-matrix: 3 far_closed + 3 near_closed + controls

## Provenance

- Traces recovered from global runs after reboot; localized copies in per-episode dirs.
- Butter manual merge: `tables/object_teacher_delay50_manual_trace_merge.csv`
- Audit: `tables/object_teacher_delay50_vis_smoke_summary.csv`
