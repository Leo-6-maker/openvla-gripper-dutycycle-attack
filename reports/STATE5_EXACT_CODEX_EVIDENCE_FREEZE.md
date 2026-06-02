# State5 Exact-Codex Protocol Evidence Freeze

**Generated**: 2026-05-23T17:30Z (read-only postprocess, no rollout)
**Input root**: `/data/liuyu/outputs/state5_exact_codex_protocol_repair_20260523`
**Referenced experiment**: `/data/liuyu/outputs/state5_three_lane_exact_protocol_repair_20260523` (r0/r1/r2 provenance only)

## Final State

```
valid_eligible_repeats = 5
valid_pass_repeats     = 4
pass_repeats           = r0, r1, r2, r18
valid_nonpass_repeats  = r13
quarantined_repeats    = r5
final_state = state5_confirmatory_promising_valid_4_of_5_after_exact_codex_protocol_repair
```

## Claim Boundary

- **Is**: Black-bowl-related spatial evidence. LIBERO-Spatial `libero_spatial_black_bowl` task. `goal_put_the_bowl_on_the_plate` semantic task.
- **Is NOT**: True non-Black-Bowl evidence. Not a non-BB candidate validation.
- **Is NOT**: Full Table2. Only state5 confirmatory repeats with exact Codex protocol.
- **Is NOT**: Broad attack benchmark. Single task, single state, single mechanism.
- **Human review**: Pending.

## Protocol Verified

The exact Codex r0/r1/r2 protocol has been independently verified on r18:

| Parameter | Codex r0/r1/r2 | This experiment (r13/r18/r5) | Match |
|-----------|---------------|------------------------------|-------|
| attack_objective (targeted) | force_gripper_open_token_ce | force_gripper_open_token_ce | YES |
| epsilon (targeted) | 0.25 | 0.25 | YES |
| step_size (targeted) | 0.050 | 0.050 | YES |
| attack_steps (targeted) | 60 | 60 | YES |
| force_open_raw_gripper (targeted) | 1.0 | 1.0 | YES |
| rho (all conditions) | 1.0 | 1.0 | YES |
| seed protocol | matched_seed == clean_seed | matched_seed == clean_seed | YES |
| window source | clean_detect autowindow | clean_detect autowindow | YES |
| linf range (targeted/VIS_current/random) | 2.12 | 2.12 | YES |
| attack_active steps per repeat | 10 | 10 | YES |

## Three-Lane vs Exact-Codex Protocol Drift

The three_lane experiment (`state5_three_lane_exact_protocol_repair_20260523`) passed
`attack_objective=gripper_logit_margin_cw` for ALL conditions, NOT `force_gripper_open_token_ce`.
This is the WRONG objective for targeted attacks and explains why the three_lane repair initially
failed (r13/r18/r5 all `repeat_not_pass`).

The exact_codex repair corrected this to `force_gripper_open_token_ce` and r18 passed,
confirming the objective difference is the critical protocol parameter.

## Per-Repeat Summary

| Repeat | Window | Protocol | Targeted SR | CmdOpen SR | Controls OK | Pass | Classification |
|--------|--------|----------|-------------|------------|-------------|------|----------------|
| r0 | [62,71] | three_lane (fixed obj=force_gripper_open_token_ce) | 0.0 | 1.0 | YES | PASS | valid_positive |
| r1 | [62,71] | three_lane (fixed obj=force_gripper_open_token_ce) | 0.0 | 1.0 | YES | PASS | valid_positive |
| r2 | [62,71] | three_lane (fixed obj=force_gripper_open_token_ce) | 0.0 | 1.0 | YES | PASS | valid_positive |
| r13 | [65,74] | exact_codex (obj=force_gripper_open_token_ce) | 1.0 | 1.0 | YES | NOT_PASS | valid_phase_robust |
| r18 | [93,102] | exact_codex (obj=force_gripper_open_token_ce) | 0.0 | 0.0* | YES | PASS | valid_positive |
| r5 | [93,102]* | exact_codex (no clean baseline) | 1.0 | 0.0 | NO | QUARANTINED | quarantined_no_baseline |

* r5 window [93,102] inherited from r18 autowindow — no independent clean_detect baseline.
  Random control also failed (SR=0), confirming control contamination.

## Evidence Quality

- r0/r1/r2: Summary-level evidence preserved in combined interpretation CSV.
  Raw step_records purged during export. Protocol cross-validated against r18.
- r13: Complete step_records preserved. Attack applied correctly but task was phase-robust.
- r18: Complete step_records preserved. Independent positive replication of Codex protocol.
- r5: Step_records preserved but quarantined. No valid denominator contribution.
