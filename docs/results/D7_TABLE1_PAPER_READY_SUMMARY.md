# D7 Table1 — Paper-Ready Summary

## Main Claim

**On Object, Goal, and Spatial, detector-triggered TRUE_T10 reduces task success relative to both CLEAN and RAND_T10, supporting direction-specific vulnerability under the unified D7 protocol. LIBERO-10 behaves differently and is treated as a distinct failure regime consistent with the 25D detector ceiling found in D8F.**

## Panel A — Per-Suite Results

| Suite | Condition | Success/N | SR | 95% CI | Attack Frames | Detector Emit Rate | Attack Delivery Rate |
|---|---|---|---|---|---|---|---|
| Object | CLEAN | 50/50 | 1.000 | [0.929, 1.000] | 0 | 1.000 | 0.000 |
| Object | TRUE_T10 | 30/50 | 0.600 | [0.462, 0.724] | 330 | 0.660 | 0.660 |
| Object | RAND_T10 | 49/50 | 0.980 | [0.895, 0.997] | 490 | 0.980 | 0.980 |
| Object | COMMAND_OPEN_ORACLE | 25/50 | 0.500 | [0.366, 0.634] | 500 | 1.000 | 1.000 |
| Goal | CLEAN | 28/33 | 0.848 | [0.691, 0.934] | 0 | 1.000 | 0.000 |
| Goal | TRUE_T10 | 17/33 | 0.515 | [0.352, 0.675] | 250 | 0.758 | 0.758 |
| Goal | RAND_T10 | 26/33 | 0.788 | [0.623, 0.893] | 330 | 1.000 | 1.000 |
| Goal | COMMAND_OPEN_ORACLE | 26/33 | 0.788 | [0.623, 0.893] | 330 | 1.000 | 1.000 |
| L10 | CLEAN | 18/50 | 0.360 | [0.241, 0.499] | 0 | 1.000 | 0.000 |
| L10 | TRUE_T10 | 15/50 | 0.300 | [0.191, 0.438] | 330 | 0.660 | 0.660 |
| L10 | RAND_T10 | 6/50 | 0.120 | [0.056, 0.238] | 500 | 1.000 | 1.000 |
| L10 | COMMAND_OPEN_ORACLE | 13/50 | 0.260 | [0.159, 0.396] | 500 | 1.000 | 1.000 |
| Spatial | CLEAN | 37/46 | 0.804 | [0.668, 0.894] | 0 | 1.000 | 0.000 |
| Spatial | TRUE_T10 | 27/46 | 0.587 | [0.443, 0.717] | 440 | 0.957 | 0.957 |
| Spatial | RAND_T10 | 39/46 | 0.848 | [0.718, 0.924] | 460 | 1.000 | 1.000 |
| Spatial | COMMAND_OPEN_ORACLE | 41/46 | 0.891 | [0.770, 0.953] | 460 | 1.000 | 1.000 |

## Panel A-Summary — O/G/S Pooled (Main Evidence, N=129)

| Condition | Success/N | SR | 95% CI |
|---|---|---|---|
| CLEAN | 115/129 | 0.891 | [0.826, 0.934] |
| TRUE_T10 | 74/129 | 0.574 | [0.488, 0.655] |
| RAND_T10 | 114/129 | 0.884 | [0.818, 0.928] |
| COMMAND_OPEN_ORACLE | 92/129 | 0.713 | [0.630, 0.784] |

TRUE_T10 vs RAND_T10: 4/44 discordant, McNemar p < 0.001.
TRUE_T10 vs CLEAN: 5/46 discordant, McNemar p < 0.001.

## Panel A-Summary — All Suites (N=179)

| Condition | Success/N | SR | 95% CI |
|---|---|---|---|
| CLEAN | 133/179 | 0.743 | [0.674, 0.801] |
| TRUE_T10 | 89/179 | 0.497 | [0.425, 0.570] |
| RAND_T10 | 120/179 | 0.670 | [0.599, 0.735] |
| COMMAND_OPEN_ORACLE | 105/179 | 0.587 | [0.513, 0.656] |

## Direction-Specific Delta (TRUE_T10 − RAND_T10)

| Suite | TRUE_T10 SR | RAND_T10 SR | Delta (pp) |
|---|---|---|---|
| Object | 60.0% | 98.0% | −38.0 |
| Goal | 51.5% | 78.8% | −27.3 |
| Spatial | 58.7% | 84.8% | −26.1 |
| O/G/S pooled | 57.4% | 88.4% | −31.0 |
| L10 | 30.0% | 12.0% | +18.0 |

## L10 Statement

LIBERO-10 exposes a distinct failure regime: low clean baseline (36%), multi-object long-horizon tasks, and weak 25D detector localization. In this regime, random perturbation can be more destructive than targeted OPEN-token perturbation. This is consistent with the 25D detector ceiling found in D8F and motivates C2f visual-language grounding.

## Paper-Ready Wording

> Across Object, Goal, and Spatial, detector-triggered TRUE_T10 attacks reduce success from 89.1% to 57.4% (O/G/S pooled, N=129), while RAND_T10 remains at 88.4%, showing strong direction-specific vulnerability (McNemar p < 0.001).

> LIBERO-10 behaves differently: its clean baseline is only 36%, and RAND_T10 is more destructive than TRUE_T10. We treat this as a separate long-horizon multi-object failure regime, consistent with the 25D detector ceiling and motivating C2f visual-language grounding.
