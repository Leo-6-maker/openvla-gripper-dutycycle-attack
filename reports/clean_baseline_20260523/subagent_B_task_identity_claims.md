# Subagent B — Task Identity / Claim Boundary Audit

**Date**: 2026-05-23
**Mode**: Read-only audit (no modifications)
**Auditor**: Subagent B (task identity and claim boundaries)

## Summary

All 7 assertions PASS with strong evidence. The task identity mapping, claim boundaries, and legacy/deprecation handling are consistently enforced across source code, test files, documentation, and evidence-freeze reports. No files incorrectly label state5 as non-BB. Non-BB candidate tasks are not yet defined (scoped as future work). One minor observation: the constraint "true non-BB tasks must have is_black_bowl_related=false" is implied by documentation but not formalized as a schema validation rule.

## Per-Assertion Results

| # | Assertion | Verdict | Evidence |
|---|-----------|---------|----------|
| 1 | runner_task_id (`libero_spatial_black_bowl`) is SEPARATED from semantic_task_name (`goal_put_the_bowl_on_the_plate`) | **PASS** | `src/utils/task_identity.py` L14-L16 defines both as different values. L39 `RUNNER_TASK_ID` reads `runner_task_id`, L42 `RUN_ID_TASK_KEY` reads `semantic_task_name`. Docstring L3-5 explicitly documents the separation. Tests: `test_task_identity.py` L40-41 (`RUNNER_TASK_ID != RUN_ID_TASK_KEY`), `test_metadata_schema.py` L62-63. |
| 2 | libero_spatial_black_bowl has is_black_bowl_related=true | **PASS** | `src/utils/task_identity.py` L24: `"is_black_bowl_related": True`. Also L25: `"is_non_black_bowl_claim": False`. Test: `test_task_identity.py` L36-38. Documented: `docs/claim_and_evidence.md` L38, `tables/state5_claim_boundary.csv` L8. |
| 3 | state5 / bowl-on-plate spatial task is NOT documented as true non-BB | **PASS** | Multiple independent sources confirm: `task_identity.py` L25 (`is_non_black_bowl_claim: False`); `docs/claim_and_evidence.md` L47-53 ("NOT true non-black-bowl task validations"); `STATE5_EXACT_CODEX_EVIDENCE_FREEZE.md` L21 ("Is NOT: True non-Black-Bowl evidence"); `NEXT_ACTION_STATUS.md` L27, L30; `tables/state5_claim_boundary.csv` L9, L12. |
| 4 | True non-BB tasks must have is_black_bowl_related=false (constraint documented?) | **PARTIAL** | The concept is documented: `docs/claim_and_evidence.md` L51-53 says true non-BB requires a separate task with non-black-bowl objects, implying `is_black_bowl_related=false`. However, no schema/validation rule explicitly enforces "if `is_non_black_bowl_claim=True` then `is_black_bowl_related` must be false." No non-BB task identity dicts exist yet, so this is a latent gap, not a bug. |
| 5 | "nonbb" directory/file names are marked as legacy naming, not a claim | **PASS** | The only "nonbb" name in the repo is a remote branch `remotes/origin/evidence/nonbb-bowl-plate-targeted-vis-20260522`. No local files/directories contain "nonbb". `docs/claim_and_evidence.md` L55 states "Legacy directory names containing 'nonbb' are historical; the claim boundary is corrected here." `TASK_IDENTITY_PATCH_STATUS.md` L43 mirrors this. |
| 6 | TASK_IDENTITY is the primary black-bowl spatial task identity dict | **PASS** | `task_identity.py` L36: `TASK_IDENTITY = BOWL_ON_PLATE_SPATIAL`. Preceded by comment "# Convenience aliases for the primary experiment task". Docstring L1: "Canonical task identity mapping for experiment tasks." All tests import and use `TASK_IDENTITY` as the canonical source. |
| 7 | DEPRECATED_DEEPSEEK_DRIFT_MATCHED_CONDITIONS marked with explicit deprecation reasons | **PASS** | All 4 deprecated conditions in `task_identity.py` L63-128 have both `"deprecated": True` and a specific `"deprecation_reason"` explaining why the condition is wrong and what Codex actually used. E.g., L76-77: `"deprecation_reason": "rho=0 disables attack; Codex used rho=1.0 with actual perturbation (linf=0.10)"`. Tests in `test_task_identity.py` L114-128 verify rho=0 is indeed the bug. |

## Non-BB vs Black-Bowl Boundary Analysis

### Current boundary is clean and consistently enforced

1. **Every reference to state5 explicitly denies non-BB status**:
   - The claim boundary table (`tables/state5_claim_boundary.csv`) lists both `is_non_black_bowl_claim: false` and `is_true_nonbb_evidence: false` as confirmed.
   - The evidence freeze (`STATE5_EXACT_CODEX_EVIDENCE_FREEZE.md`) and next-action status (`NEXT_ACTION_STATUS.md`) both contain explicit "Is NOT" statements.
   - `task_identity.py` hard-codes `is_non_black_bowl_claim: False` on the only defined task identity.

2. **Diagnostic experiments are correctly scoped**:
   - `docs/claim_and_evidence.md` L47-51 clarifies that "non-BB bowl-on-plate diagnostic experiments" (e.g., `overnight_multilane_followup_20260522_fixed`) use the **same** `libero_spatial_black_bowl` task — they are mechanism diagnostics, not true non-BB validations.

3. **No files incorrectly label state5 as non-BB**: Confirmed — zero matches across all source files, documentation, and reports.

4. **Non-BB candidate tasks are not defined**: The concept exists only as a future-action item in status documents:
   - `NEXT_ACTION_STATUS.md` L52: "True non-BB candidate gated pilot — rerun with correct models and fresh autowindow"
   - `TASK_IDENTITY_PATCH_STATUS.md` L66: `next_action=review_patch_then_select_true_non_black_bowl_candidate`
   - No concrete task identity dicts, config files, or experiment drivers for non-BB candidates exist yet. Proper scoping: they are explicitly future work.

### Latent concern (low severity)

Assertion #4 reveals a gap: there is no formal schema validation rule that enforces `is_black_bowl_related=false` when `is_non_black_bowl_claim=true`. Currently this is a non-issue because no non-BB task identity exists. When one is created, it should be verified that `is_black_bowl_related` is set correctly. A schema-level enforcement (e.g., in `protocol_validation.py`) would prevent human error at that point.

## Legacy/Derived Naming Audit

| Artifact | Type | Status | Documentation |
|----------|------|--------|---------------|
| `remotes/origin/evidence/nonbb-bowl-plate-targeted-vis-20260522` | Git branch (remote only, no local checkout) | Legacy naming, no claim | `docs/claim_and_evidence.md` L55: "Legacy directory names containing 'nonbb' are historical; the claim boundary is corrected here." |
| `DEPRECATED_DEEPSEEK_DRIFT_MATCHED_CONDITIONS` | Python variable | Deprecated with reasons | Each entry has `"deprecated": True` + `"deprecation_reason"` string |
| `MATCHED_CONDITIONS` / `TRIAGE_MATCHED_CONDITIONS` | Python sentinel objects | Fail-fast deprecated | `_DeprecatedProtocolSentinel` raises `RuntimeError` on any access, pointing to `condition_protocols.LEGACY_CODEX_STATE5_MATCHED_CONDITIONS` |

### Key design decisions confirmed

- The sentinel pattern for `MATCHED_CONDITIONS` / `TRIAGE_MATCHED_CONDITIONS` (lines 135-148 of `task_identity.py`) is a thorough fail-fast: `__iter__`, `__getitem__`, `__len__`, AND `__bool__` all raise `RuntimeError`. Tests cover all four entry points plus the `TRIAGE_MATCHED_CONDITIONS` variant.
- The deprecation reasons specify both what was wrong AND what the correct Codex protocol used, making them useful for forensic reference.
- The `table1_task_key` in `task_identity.py` L30 is correctly mapped to `"goal_put_the_bowl_on_the_plate"`, matching `semantic_task_name`. This is consistent because Table1 uses the semantic task name as the cross-reference join key.

## Overall Verdict

**ALL CLEAR.** The task identity mapping and claim boundaries are consistent, well-documented, and enforced across all layers:
- **Source code**: `task_identity.py` defines the canonical mapping with clear separation of concerns.
- **Tests**: 20+ assertions in `test_task_identity.py` verify every aspect of the mapping, plus fail-fast tests for deprecated aliases.
- **Documentation**: `claim_and_evidence.md` explicitly documents the claim boundary, the separation of runner_task_id vs semantic_task_name, and the legacy naming convention.
- **Evidence freeze reports**: `STATE5_EXACT_CODEX_EVIDENCE_FREEZE.md`, `NEXT_ACTION_STATUS.md`, `TASK_IDENTITY_PATCH_STATUS.md`, and `tables/state5_claim_boundary.csv` all independently confirm the boundary.
- **Protocol validation**: `protocol_validation.py` and `test_protocol_validation.py` prevent misuse of deprecated configs and wrong objectives.

**One minor recommendation (informational, not blocking)**: When a true non-BB task identity is eventually defined, add a validation rule (e.g., in `protocol_validation.py`) that enforces `is_black_bowl_related=False` when `is_non_black_bowl_claim=True`, to prevent misconfiguration at schema level.
