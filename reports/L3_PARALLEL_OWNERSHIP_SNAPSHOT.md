# L3 Parallel Ownership Snapshot

## Timestamp
2026-06-17 (snapshot generated at workspace creation)

## DeepSeek Workspace
- **Branch:** exp/l3-independent-audit-prep-20260617
- **Head:** 50da442c1b033a780b802c6345c376b23d4833b1
- **Worktree:** D:/vla_attack/outputs/code/l3_independent_audit_/
- **Role:** Independent verification only — NO GPU execution

## Codex Workspace
- **Branch:** exp/l3-vis-handoff-contract-repair-20260617
- **Head:** 50da442c1b033a780b802c6345c376b23d4833b1
- **Worktree:** D:/vla_attack/outputs/code/v3_parity_pipeline_finalize_codex_20260614/
- **Role:** Exclusive Layer 3 GPU execution on GPU(1,5)
- **Active watcher PID:** NONE DETECTED (worktree clean, no running processes)
- **Current persisted gate:** H0 (contract repair — not yet committed)

## Production Tag
- **Tag:** l12-d5-v1-production-20260617
- **Target:** 593ffadba7c7d64eadc4305fa818cd5d2c570507
- **Status:** IMMUTABLE — not to be modified

## GPU Mapping
| GPU Pair | Owner | Status |
|----------|-------|--------|
| (1,5) | Codex | L3 VIS execution — exclusive |
| (2,6) | DeepSeek released | IDLE — no active collection |
| GPU0/4 | Excluded | Hardware exclusion |
| GPU3 | Quarantined | Hardware fault |
| GPU7 | Render-only | EGL rendering |

## Issue #28
- **Title:** "Repair VIS handoff contract and complete Layer3 autonomously on GPU1,5"
- **Access:** gh CLI not authenticated on this machine
- **State:** Read from local worktree only

## Known Closeout Contract Gaps (H0)
1. Old example command omitted --output_dir
2. Closeout runner exposed old canary rather than proven canary_v4
3. Frame directory naming did not match old step78 frozen-input loader
4. Old config remained hard-coded to Tomato state0 step78
5. EXACT_BOUND not yet supported by committed comparison audit
6. Captured processor tensor and attack runner preprocessing paths differed
7. Instruction / clean generation metadata incomplete
8. Full 71-frame recursive evidence manifest not committed

## Frozen Multi-Parent Denominator
- **3 selected parents:** butter_s11 (exact), tomato_sauce_s23 (early), salad_dressing_s11 (late)
- **10 frames:** 6 primary in-window + 4 diagnostic comparators
- **20 jobs:** 10 frames × 2 seeds (81, 82)
- **Lambda:** 2.0, Target token: 31744, Linf budget: 6/255

## Current Persisted Gate
- **Codex claimed:** H0 (contract repair in progress)
- **DeepSeek independent:** H0 NOT YET AUDITED
- **Next authorized phase:** H1 (after H0 PASS)

## Active Output Roots
- **Object frames:** /data/liuyu/outputs/l12_frame_handoff_v2_r1
- **Timing panel:** /data/liuyu/outputs/l12_timing_panel_v2
- **Labels:** /data/liuyu/outputs/d5_label_generation/d5_teacher_p_labels_v2.csv
- **Spatial clean:** /data/liuyu/outputs/libero_spatial_clean100_20260617_r1
