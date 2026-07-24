# M1C Runtime Repair: R0 / R1 / R2 Offline Replay Report

## Gate

```text
M1C_RUNTIME_REPLAY = COMPLETE
R0_REGRESSION_PARITY = PASS (60/60 cells)
EXECUTED_AT = 2026-06-23
A800_HOST = pm-364c0001
SCRIPT = scripts/migration/replay_sc5_v1r_fsm.py
```

## FSM Configurations

| Parameter | R0 (legacy_v1) | R1 (v1r_r1) | R2 (v1r_r2) |
|---|---|---|---|
| tau_corridor | 0.3 | 0.3 | — |
| tau_release | 0.3 | 0.3 | 0.3 |
| guard | 5 | 5 | 5 |
| tau_on | — | — | 0.5 |
| tau_off | — | — | 0.3 |
| n_candidate | — | — | 3 |
| max_arm_age | — | — | 50 |

All parameters are M1B frozen defaults. No threshold sweep was performed.
M1B 60-cell data is diagnostic-only; no parameter selection occurred.

## Six-Gate Metrics

### R0 (legacy_v1 — frozen baseline)

| Metric | B0 | D1 | Combined | Gate |
|---|---|---|---|---|
| Coverage | 23/23 = 1.000 | 26/26 = 1.000 | 49/49 = 1.000 | ≥0.80 PASS |
| False-early | 1/27 = 0.037 | 0/28 = 0.000 | 1/55 = 0.018 | ≤0.10 PASS |
| Post-release | 0/27 = 0.000 | 0/28 = 0.000 | 0/55 = 0.000 | ≤0.05 PASS |
| K10 containment | 22/23 = 0.957 | 26/26 = 1.000 | 48/49 = 0.980 | ≥0.85 PASS |
| Median anchor error | 2.0 | 2.0 | 0.0 | ≤8 PASS |
| No-corridor abstain | 3/7 = 0.429 | 2/4 = 0.500 | 5/11 = 0.455 | ≥0.90 FAIL |
| Silent ARM stalls | 1 | 0 | 1 | — |
| Feature-valid rate | >0.99 | >0.99 | >0.99 | ≥0.99 PASS |

### R1 (v1r_r1 — minimal disarm)

| Metric | B0 | D1 | Combined | Gate | Δ vs R0 |
|---|---|---|---|---|---|
| Coverage | 23/23 = 1.000 | 26/26 = 1.000 | 49/49 = 1.000 | PASS | 0 |
| False-early | 1/25 = 0.040 | 0/28 = 0.000 | 1/53 = 0.019 | PASS | +0.001* |
| Post-release | 0/25 = 0.000 | 0/28 = 0.000 | 0/53 = 0.000 | PASS | 0 |
| K10 containment | 22/23 = 0.957 | 26/26 = 1.000 | 48/49 = 0.980 | PASS | 0 |
| Median anchor error | 2.0 | 2.0 | 0.0 | PASS | 0 |
| No-corridor abstain | 5/7 = 0.714 | 2/4 = 0.500 | 7/11 = 0.636 | FAIL | +0.182 |
| Silent ARM stalls | 0 | 0 | 0 | — | -1 |
| Mean disarm count | 0.13 | 0.00 | 0.07 | — | +0.07 |

*False-early apparent change is a denominator artifact: 2 fewer triggered episodes
(sticky-arm false triggers eliminated) with the same 1 absolute early-trigger count.

### R2 (v1r_r2 — candidate machine, default parameters)

| Metric | Value | Gate |
|---|---|---|
| Coverage | 0/49 = 0.000 | FAIL |
| No-corridor abstain | 11/11 = 1.000 | PASS |
| All other metrics | N/A (zero triggers) | FAIL |

**R2_DEFAULT_CONFIGURATION = REJECTED**. tau_on=0.5 with n_candidate=3 is too
strict: zero teacher-valid episodes achieve 3 consecutive frames with cp>0.5.
The candidate-FSM method family is NOT rejected; parameter tuning on independent
validation is allowed. Tuning on M1B 60-cell data is FORBIDDEN.

## Known Case Detail

### B0 No-Corridor (7 episodes)

| Episode | R0 emit | R1 emit | R1 disarms | Result |
|---|---|---|---|---|
| butter_s1 | 112 | **None** | 2 (PHASE_EXIT, FEATURE_INVALID) | FIXED |
| chocolate_pudding_s1 | 42 | **None** | 1 (PHASE_EXIT) | FIXED |
| cream_cheese_s0 | 149 | 149 | 0 | RESIDUAL |
| butter_s2 | 33 | 33 | 0 | RESIDUAL |
| bbq_sauce_s2 | None | None | 0 | CORRECT |
| tomato_sauce_s0 | None | None | 0 | CORRECT |
| orange_juice_s2 | None (stall) | **None** | 1 (PHASE_EXIT) | FIXED (silent stall eliminated) |

### D1 No-Corridor (4 episodes)

| Episode | R0 emit | R1 emit | R1 disarms | Result |
|---|---|---|---|---|
| butter_s1 | 92 | 92 | 0 | RESIDUAL |
| bbq_sauce_s2 | 210 | 210 | 0 | RESIDUAL |
| alphabet_soup_s1 | None | None | 0 | CORRECT |
| butter_s2 | None | None | 0 | CORRECT |

### Summary

```text
B0: 2 sticky-arm FIXED + 1 silent stall FIXED + 2 model-error RESIDUAL + 2 correct
D1: 0 FIXED + 2 model-error RESIDUAL + 2 correct

Total: 3/6 false triggers FIXED (2 emit + 1 stall), 4/6 RESIDUAL
```

## Conclusions

1. **R0 regression parity confirmed** — legacy FSM reproduced exactly from telemetry scores.
2. **R1 fixes sticky-arm and silent-stall failures** — no coverage or K10 regression, abstain improves from 0.455 to 0.636, silent stall eliminated.
3. **R1 does NOT fix sustained model-selectivity errors** — 4/6 false triggers remain. No-corridor abstain=0.636 still below 0.90 threshold.
4. **R2 defaults are too strict** — zero coverage. Needs independent validation tuning.
5. **SC5-v2 retraining is indicated** — runtime repair alone cannot reach ≥0.90 abstain.

## Provenance

- M1B close commit: `9ab9f26`
- Replay script commit: `aea3f38`
- Runtime source commit: `07bd43b`
- Checkpoint: `artifacts/detector/sc5_mlp_s2.pt`
- A800 output: `/mnt/sdc/dty_user/openvla_attack/evidence/m1c/runtime_replay/`
