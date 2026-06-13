# V3 Surrogate-Generation Parity Diagnostic Plan

**Date:** 2026-06-14
**Branch:** exp/vis-prefix-margin-repair-20260603

## Background

Smoke A (butter_s2, seed 811, PGD1) revealed:
- v3 surrogate `global_top_token_final = 31744` (disc 255, native extreme CLOSE)
- Full AR generation gripper token = 31872 (disc 127, native boundary)

Smoke B (cream_cheese_s2, seed 811, PGD20) showed:
- Surrogate and generation aligned on token 31872
- But global top was boundary, not OPEN
- Native OPEN vs CLOSE margin improved (3.0→4.5) but did not produce OPEN output

## Separated Hard-Fail Classification

### INFRA Hard-Fail (aborts episode)
- 7-token count mismatch
- 6-token prefix mismatch (surrogate vs full AR)
- Generation score argmax != generated token
- Missing telemetry, NaN, Inf
- Budget violation, fallback detected

### METHOD Diagnostic (recorded, not fatal)
- `SURROGATE_TO_GENERATION_TOP1_MISMATCH`: no-cache forward top ≠ generate top
- Token classification: NATIVE_OPEN/CLOSE/BOUNDARY/CLIP_MEDIATED_OPEN/CLOSE

## Generation Score Audit

Runner now captures via `generate(..., output_scores=True)`:
- Final step score argmax
- Top-1/Top-2 scores and gap
- Scores for diagnostic tokens 31744, 31872
- Best OPEN/CLOSE token scores from the official score distribution

## Four-Path Parity Diagnostic

`diagnose_v3_generation_parity.py` compares on a fixed replay frame:

| Path | Method | Purpose |
|------|--------|---------|
| A | `generate(use_cache=True)` | Source of truth |
| B | No-cache forward | Current v3 surrogate |
| C | Cache forward | Cache parity test |
| D | `generate(use_cache=False)` | Generation config test |

## Replay Bundle

`--v3_parity_dump_dir` saves:
- Prompt input IDs
- Adversarial pixel values (.pt tensor)
- Generated arm prefix tokens
- Full AR tokens
- Surrogate global top token
- Generation score argmax
- Provenance (runner SHA, task, seed, objective)

## Test Coverage

`test_v3_generation_parity.py`:
1. Token classification: NATIVE_OPEN/CLOSE/BOUNDARY/CLIP_MEDIATED
2. Prefix mismatch → hard-fail
3. Generation score argmax mismatch → hard-fail
4. Surrogate mismatch → diagnostic only (not fatal)
5. Legal 0.0 preserved
6. Replay bundle schema validation

## Next Steps

After code review approval:
1. Run butter seed811 + cream seed811 with `--v3_parity_dump_dir`
2. Run `diagnose_v3_generation_parity.py` on replay bundles
3. Determine whether surrogate parity can be achieved (Path B→A, C→A)
4. Only then decide on v3.1 objective (global-top native-OPEN)
