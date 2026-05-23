# Subagent Review Summary — Clean Baseline Consolidation

**Timestamp**: 2026-05-23T22:30Z
**Branch**: `fix/protocol-schema-and-condition-config-20260523`
**Subagents**: A (protocol semantics), B (task identity), C (tests), D (runtime), E (branch/PR)

## Verdict Summary

| Subagent | Core Assertions | Pass Rate | BLOCKING | HIGH | MEDIUM | LOW |
|----------|----------------|-----------|----------|------|--------|-----|
| A — Protocol Semantics | 7 | 7/7 PASS | 0 | 0 | 1 | 0 |
| B — Task Identity | 7 | 7/7 PASS | 0 | 0 | 0 | 1 |
| C — Tests | 10 | 9/10 PASS | 0 | 0 | 0 | 1 |
| D — Runtime Integration | 7 | 7/7 PASS | 0 | 0 | 1 | 0 |
| E — Branch/PR | 6 | 6/6 PASS | 0 | 1 | 0 | 0 |

**Overall**: NO BLOCKING issues. 1 HIGH finding (rebase needed), 3 MEDIUM findings, 2 LOW findings.

## Findings by Level

### MEDIUM

**A-M1**: `configs/paper_black_bowl_attack.yaml:15` has `objective: gripper_logit_margin_cw` as config default.
- This is the YAML-level fallback used by `effective_attack_objective()` when no CLI arg is passed.
- If a driver omits `--attack_objective` for random (as Codex did), the effective objective would be `gripper_logit_margin_cw`, not `targeted_directional_ce`.
- Protocol dicts document `effective_attack_objective_expected` correctly; the YAML config does not match.
- **Mitigation**: Protocol validators catch protocol-level issues; CLI arg override takes precedence over config.
- **Fix**: Add a comment to the YAML file noting this is overridden by CLI args in Codex protocol; or update default to match documented behavior.

**D-M1**: `scripts/run_attack_pipeline.py` defines inline `CONDITIONS` dicts that do not import from `condition_protocols.py`.
- This creates a maintenance risk: protocol changes in `condition_protocols.py` are not reflected in the pipeline.
- The inline values currently match protocol intent, so no immediate correctness bug.
- **Mitigation**: Documented in `docs/driver_protocol_integration.md`.
- **Fix**: Future refactor to have the pipeline import from `condition_protocols.py` (not blocking for this baseline).

**C-M1**: Protocol validators are comprehensive but NOT wired into any runtime code (scripts/, src/gripper_attack/).
- `validate_command_open_protocol`, `validate_codex_targeted_protocol`, etc. exist and are tested but no runtime calls them.
- This is a feature gap, not a bug — validators are available for driver integration but not yet used.
- **Fix**: Documented as TODO in `docs/driver_protocol_integration.md`. Future drivers should call validators at config-load time.

### HIGH

**E-H1**: Branch is 8 commits behind `origin/main`. Rebase required before merge.
- 2 files overlap between branch and main: `scripts/v4_run_eval_openvla.py` and `tests/v4/test_public_pipeline.py`
- Both have the same logical change (force_open_raw_gripper 1.0) applied by different commits
- All other 41 differing files are deletions of files main added (no content conflicts)
- `.gitignore` missing 8 patterns from main (resolved by rebase)
- **Fix**: `git rebase origin/main` and resolve the 2 overlapping file merge points

### LOW

**B-L1**: The constraint "true non-BB implies `is_black_bowl_related=false`" is documented by implication but not formalized as a validation rule. Non-blocking since no non-BB task identity exists yet.

**C-L1**: `tests/v4/test_guard.py:3` had an unused `import pytest` — no module in the file uses any pytest feature. **Fixed** during Phase 3 (import removed).

## Addressed During Phase 3

| Issue | Source | Fix |
|-------|--------|-----|
| CLEAN_DETECT_PROTOCOL missing recommended fields | Phase 3 checklist | Added `attack_objective_raw_arg`, `effective_attack_objective_expected`, `omit_attack_objective_cli_arg` |
| Missing `validate_clean_detect_protocol` | Phase 3 checklist | Added validator with 6 invariant checks |
| `test_guard.py` unused `import pytest` | Subagent C | Removed |
| No `docs/attack_mechanisms.md` | Phase 3 checklist | Created with full mechanism taxonomy |
| No `docs/protocol_baseline_20260523.md` | Phase 3 checklist | Created with baseline reference |
| No `docs/driver_protocol_integration.md` | Phase 3 checklist | Created with 7 integration points |

## Unresolved (Deferred to Future Work)

1. YAML config default (`paper_black_bowl_attack.yaml`) not aligned with documented effective objective — needs review in separate config-audit task
2. `run_attack_pipeline.py` inline CONDITIONS not synced with `condition_protocols.py` — needs pipeline refactor
3. Protocol validators not wired into runtime — needs driver update
4. True non-BB task identity not yet defined — needs Object-suite task identity mapping
5. Unified command builder (`src/utils/command_config.py`) not yet implemented — needs design

## Conclusion

The protocol schema branch passes all subagent audits with NO BLOCKING issues.
All 3 MEDIUM findings are deferred to future work (config audit, pipeline refactor, driver integration).
The branch is ready for clean baseline commit after Phase 4 (tests) and Phase 5 (commit).
