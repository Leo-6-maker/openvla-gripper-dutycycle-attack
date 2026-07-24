# UADA Implementation Audit

## Scope
Audit whether the codebase contains a faithful Adapted UADA-DoF7 implementation.

## UADA Protocol Requirements (from user spec)

| Requirement | Description |
|-------------|-------------|
| Target DoF | Gripper only (DoF 7 / action_dim-1) |
| Target action | Farthest valid action bound from clean DoF7 action |
| Objective | Soft action discrepancy (not CE) |
| NOT | Ordinary untargeted CE |

## Codebase Audit

### Existing untargeted objectives (`route_contract.py:14`)
- `untargeted_clean_token_ce` — CE loss, untargeted token direction
- `untargeted_clean_ce` — CE loss variant

### Existing action discrepancy utilities (`metrics.py:9`)
- `normalized_action_discrepancy_cleanref()` — offline NAD computation utility
- NOT used as an attack objective

### Conclusion
**No UADA-compatible objective exists in the codebase.** The existing untargeted objectives use CE loss on tokens, not action-space discrepancy optimization. There is no "farthest bound" target selection logic.

## Recommendation

Per the user's protocol spec: **rename to "Adapted Action-Discrepancy PGD"**.

Two options:

### Option A: Implement true action discrepancy (recommended)
- Target: gripper DoF only
- Loss: `||adv_gripper - target_gripper||` where target = farthest bound from clean
- This requires modifying `attack_adapter.py` to add a new objective class

### Option B: Use existing best-match `untargeted_clean_token_ce`
- Already implemented and battle-tested
- Token-level untargeted CE
- Pros: zero implementation risk, immediately runnable
- Cons: not true action discrepancy

## Gate Decision
**BLOCKED** — UADA cannot proceed without either:
1. Implementing true action discrepancy objective in attack_adapter.py
2. Accepting `untargeted_clean_token_ce` as the closest available baseline (rename to "Adapted Untargeted CE PGD")

**Recommendation**: Proceed with Option B (`untargeted_clean_token_ce`) under the name "Adapted Untargeted CE PGD" to maintain forward momentum. Option A can be added later as a supplementary objective.
