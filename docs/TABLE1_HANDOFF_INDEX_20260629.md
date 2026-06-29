# Table 1 and GPT Handoff Index — 2026-06-29

This index groups the documentation created to continue the OpenVLA Gripper Duty-Cycle Attack project in a new GPT session.

## Files

### 1. Scientific and execution plan

```text
docs/table1/TABLE1_SOTA_COMPARISON_AND_EXECUTION_PLAN_V1.md
```

Contains:

- claim hierarchy;
- SOTA comparison policy;
- mandatory baselines;
- TMA adapted-versus-original distinction;
- Table 1 panel design;
- metrics and denominators;
- statistics;
- execution batches;
- freeze requirements;
- GO/HOLD matrix.

### 2. Full project handoff

```text
docs/handoff/GPT_TABLE1_HANDOFF_20260629.md
```

Contains:

- project background;
- Layer 1/2/3 architecture;
- Phase B results and failure modes;
- VIS canary status;
- CLEAN1500 and data-management status;
- current Table 1 objective;
- immediate continuation procedure;
- required first audit response.

### 3. Ready-to-paste prompt for a new GPT conversation

```text
docs/handoff/NEW_GPT_START_PROMPT_20260629.md
```

Use this file verbatim to start a new project conversation.

### 4. Baseline preregistration draft

```text
docs/table1/LOTO_TABLE1_BASELINE_PREREGISTRATION_V1_DRAFT.json
```

This artifact is intentionally marked:

```text
DRAFT_NOT_AUTHORIZED
```

It does not authorize formal experiments. Exact paths, script hashes, manifests, metric implementations, and Formal CLEAN closure must be filled and frozen first.

## Recommended reading order

1. `GPT_TABLE1_HANDOFF_20260629.md`
2. `TABLE1_SOTA_COMPARISON_AND_EXECUTION_PLAN_V1.md`
3. `LOTO_TABLE1_BASELINE_PREREGISTRATION_V1_DRAFT.json`
4. `NEW_GPT_START_PROMPT_20260629.md`

## Current top-level gates

```text
Phase B detector evaluation       CLOSED
Post-Phase-B detector tuning      PROHIBITED
VIS engineering canary            PASS
Formal CLEAN                      AUDIT / COMPLETE-THEN-FREEZE
Object Table 1 attack conditions  HOLD until preregistration freeze
Cross-suite VIS                   HOLD
CLEAN1500                         GO in background
```
