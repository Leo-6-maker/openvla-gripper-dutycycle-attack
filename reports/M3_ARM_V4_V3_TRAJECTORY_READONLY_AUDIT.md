# M3 arm-v4 read-only audit of v3 trajectory telemetry

## Scope

This is a read-only audit of existing arm-v3 seed81 and seed82 debug artifacts.
It checks whether the saved v3 telemetry suggests that intermediate optimization
prefixes could satisfy the arm gate before the final official decode.

This audit is explanatory only. It does not reclassify any v3 result.

## Inputs

| Seed | Source |
| ---: | --- |
| 81 | `/data/liuyu/outputs/m3_logratio_arm_v3_step78_290fb13_seed81/m3_step78_canary_debug.json` |
| 82 | `/data/liuyu/outputs/m3_logratio_arm_v3_step78_seed82_480c95f/m3_step78_canary_debug.json` |

Derived table:

`tables/m3_arm_v4_v3_trajectory_readonly_audit.csv`

## Key Limitation

The v3 debug JSON stores optimization-time `generated_arm_prefix_trajectory`
and margin telemetry, but it does not store an official decode for every
intermediate delta candidate. Therefore these rows are not valid v4 candidate
results.

Only the final official decode is available in the v3 artifacts.

## Findings

| Seed | Surrogate telemetry arm-gate steps | Surrogate arm-gate and positive-margin steps | Final official arm match | Final official margin | v3 result remains |
| ---: | --- | --- | ---: | ---: | --- |
| 81 | `1,2,3,4,7,9,11,13,19` | `2,3,4,7,9,11,13,19` | `6/6` | `21.499736785888672` | development success |
| 82 | `2,3,4,5,20` | `2,3,4,5,20` | `2/6` | `29.937013626098633` | `TARGET_ONLY_ARM_FAIL` |

Seed82 is the important case: optimization telemetry contains arm-feasible
surrogate prefixes, including step 20, but the final official decode is still
arm-nonselective (`2/6`). That means v4 cannot rely on training-time prefix
telemetry as a candidate selector. It must official-decode each of the 21
candidates and apply the hard feasible selection rule to those official
candidate records.

## Interpretation

The audit supports the v4 design change:

- v3 final-iterate selection is not robust enough;
- v3 telemetry suggests arm feasibility can appear during optimization;
- existing artifacts do not prove any seed82 intermediate official candidate
  passed both arm and token gates;
- v4 must materialize and official-decode all 21 candidates for TRUE_PGD,
  RAND21, and shuffled-gradient.

## Claim Boundary

Allowed:

- Existing v3 telemetry motivates hard feasible trajectory selection.
- Seed82 remains a v3 arm-selectivity failure.

Forbidden:

- Do not claim seed82 had an official successful intermediate candidate.
- Do not claim v3 can be salvaged without official 21-candidate decoding.
- Do not proceed to panel or LIBERO rollout from this audit.
