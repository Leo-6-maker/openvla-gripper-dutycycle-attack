# M1C Object Blind-v2 Protocol

**Status**: DESIGN_ONLY (no execution until model fully frozen)
**Date**: 2026-06-23

## Disposition of Original Blind

```text
states 38-47: BLIND_COMPROMISED_DIAGNOSTIC_ONLY
Cannot be used as formal blind.
Cannot be mixed with Blind-v2.
12 cells exposed in aggregate reports (2026-06-23).
```

## Blind-v2 Design

```text
Suite:    LIBERO Object
Tasks:    10 (all)
States:   20 new states per task (200 total)
Profile:  B0 (BF16 + eager)
Attack:   disabled
```

### Split

```text
B1 = first 100 cells (10 tasks × states 0a-9a)
B2 = second 100 cells (10 tasks × states 0b-9b)
```

If B1 denominator is sufficient (teacher-valid ≥ 30, no-corridor ≥ 30),
evaluate only B1. If insufficient, add B2. No per-cell cherry-picking.

### Pre-Execution Freeze

Before any blind rollout:
- Model checkpoint SHA
- Normalization (mean/std)
- FSM version and all thresholds
- Evaluation script SHA
- State generation seed and code SHA
- B1/B2 membership
- This document: FROZEN

### Monitoring Policy

During collection, only display:
- Completeness (done/total)
- Errors (non-zero RC, missing files)
- File integrity (SHA)
- GPU status
- Disk space

FORBIDDEN during collection:
- Success rate
- Emit/no-emit counts
- Teacher labels
- Any performance metrics

### Evaluation

After model and FSM are fully frozen:
1. Run evaluation exactly ONCE on B1
2. Report all six gates with numerators and denominators
3. If B1 insufficient, add B2 (pre-registered, not cherry-picked)
4. No parameter tuning after seeing blind results
