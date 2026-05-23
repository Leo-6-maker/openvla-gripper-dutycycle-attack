# State5 Exact-Codex Evidence Freeze

**Branch**: `evidence/state5-exact-codex-freeze-20260523`
**Based on**: `origin/main` @ `f02394f` (clean protocol baseline)
**Input root**: `/data/liuyu/outputs/state5_exact_codex_protocol_repair_20260523`

## Final State

```
final_state = state5_confirmatory_promising_valid_4_of_5_after_exact_codex_protocol_repair
```

## Accounting

```
valid_eligible_repeats = 5
valid_pass_repeats     = 4
pass_repeats           = r0, r1, r2, r18
valid_nonpass_repeats  = r13
quarantined_repeats    = r5
```

## Claim Boundary

- **Is**: Black-bowl-related spatial evidence. LIBERO-Spatial `libero_spatial_black_bowl` task.
- **Is NOT**: True non-Black-Bowl evidence.
- **Is NOT**: Full Table2.
- **Is NOT**: Broad attack benchmark.
- **Human review**: Pending.

## Protocol

Exact Codex r0/r1/r2 protocol verified via raw step_records audit:

| Parameter | Value |
|-----------|-------|
| attack_objective (targeted) | force_gripper_open_token_ce |
| epsilon (targeted) | 0.25 |
| step_size (targeted) | 0.050 |
| attack_steps (targeted) | 60 |
| force_open_raw_gripper (targeted) | 1.0 |
| rho (all conditions) | 1.0 |
| seed protocol | matched_seed == clean_seed |
| window source | clean_detect autowindow |

## Per-Repeat Summary

| Repeat | Window | Classification | Targeted SR | Pass |
|--------|--------|----------------|-------------|------|
| r0 | [62,71] | valid_positive | 0.0 | YES |
| r1 | [62,71] | valid_positive | 0.0 | YES |
| r2 | [62,71] | valid_positive | 0.0 | YES |
| r13 | [65,74] | valid_phase_robust | 1.0 | NO |
| r18 | [93,102] | valid_positive | 0.0 | YES |
| r5 | [93,102]* | quarantined_no_baseline | 1.0 | QUARANTINED |

* r5 window inherited from r18 autowindow; no independent clean_detect baseline.

## Evidence Files

### Reports (`evidence/state5_exact_codex_20260523/reports/`)

| File | Content |
|------|---------|
| STATE5_EXACT_CODEX_EVIDENCE_FREEZE.md | Main evidence document — protocol verification matrix, per-repeat summary |
| STATE5_DENOMINATOR_CORRECTION.md | Denominator correction — r5 quarantine, 6→5 eligible |
| STATE5_WINDOW_PHASE_SENSITIVITY.md | Window phase analysis — r13 robust vs r18 sensitive |
| STATE5_LINF_METRIC_HARMONIZATION.md | linf=2.12 explanation — semantic vs magnitude |
| NEXT_ACTION_STATUS.md | Open questions and recommended next steps |

### Tables (`evidence/state5_exact_codex_20260523/tables/`)

| File | Rows | Content |
|------|------|---------|
| state5_valid_repeat_accounting.csv | 6 | Per-repeat classification and metrics |
| state5_quarantined_runs.csv | 1 | r5 quarantine rationale |
| state5_window_phase_table.csv | 6 | Window positions and phase classifications |
| state5_condition_outcome_compact.csv | 14 | Per-condition raw metrics (linf, l2, SR, oracle) |
| state5_linf_metric_harmonized.csv | 4 | linf normalization by condition |
| state5_claim_boundary.csv | 16 | Claim boundary matrix |

## Integrity

- No raw rollout dirs
- No videos
- No HDF5
- No model/dataset files
- No secrets
- All files < 1MB (largest: ~4KB)
