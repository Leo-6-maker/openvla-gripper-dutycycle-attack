# M3 arm-v4 fixed-frame panel closure

## Decision

```text
PANEL_DENOMINATOR_BLOCKED_BY_CLEAN_ALREADY_TARGET_6_OF_8
```

PR #24 is closed as protocol and infrastructure work plus a negative
denominator-feasibility finding. It is not a positive panel result and not an
attack failure result.

## Frozen Evidence

| Item | Status |
| --- | --- |
| Capture-only preflight | `ACCEPTED` |
| Step78 parity | `PASS` |
| Main clean eligible | `2/8` |
| Main already target | `6/8` |
| Seed85 used | `NO` |
| Seed86 used | `NO` |
| Attack execution | `NO` |
| LIBERO attack rollout | `NO` |
| Panel robustness conclusion | `NOT_ESTABLISHED` |

The accepted capture-only run is archived under:

```text
reports/artifacts/m3_arm_v4_panel_capture_f41ab1a_r2
```

## Main-Denominator Clean Eligibility

| Frame | Clean token | Status |
| ---: | ---: | --- |
| 70 | 31744 | `CLEAN_ALREADY_TARGET` |
| 72 | 31744 | `CLEAN_ALREADY_TARGET` |
| 74 | 31744 | `CLEAN_ALREADY_TARGET` |
| 76 | 31744 | `CLEAN_ALREADY_TARGET` |
| 80 | 31872 | `CLEAN_ELIGIBLE` |
| 82 | 31744 | `CLEAN_ALREADY_TARGET` |
| 84 | 31744 | `CLEAN_ALREADY_TARGET` |
| 86 | 31872 | `CLEAN_ELIGIBLE` |

Step78 remained a positive-control frame and is not part of the main
denominator.

## Why Seed85 Is Retired

The preregistered aggregate gate requires at most one clean-ineligible main
frame and at least six same-frame full selective passes. Capture-only preflight
proved that six of eight main frames are already target token `31744` under
clean execution. Therefore seed85 cannot produce a valid panel result under the
frozen denominator.

```text
SEED85_STATUS: UNUSED_AND_RETIRED_WITH_BLOCKED_PANEL
```

Running seed85 now would only create a two-frame result on step80 and step86,
which is not the preregistered panel.

## Allowed Claim

The fixed absolute-frame panel around Tomato state0 step70-86 is scientifically
non-executable for arm-v4 multi-frame robustness because the clean trajectory
contains only two main-denominator clean CLOSE contexts.

## Forbidden Claims

Do not claim:

- arm-v4 failed on the fixed-frame panel;
- arm-v4 succeeded on the fixed-frame panel;
- clean `31744` frames are attack successes;
- step80/86 alone establish multi-frame robustness;
- frames may be replaced while preserving this preregistration;
- seed85 or seed86 were run.

## Next Stage

The next valid design unit is a clean-CLOSE event, not an absolute frame.
Proceed only through a separate V5 preregistration and review.
