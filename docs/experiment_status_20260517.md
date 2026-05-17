# Experiment Status 2026-05-17

## Established

Black Bowl fixed-window mechanism evidence is strong.

Generic v4 non-hardcoded auto-window passed manual-audited Black Bowl replication:

- State5 VIS manual CQ failure: 5/5
- State7 VIS manual CQ failure: 4/5
- VIS total manual CQ failure: 9/10
- Random control manual CQ failure: 0/10

The earlier automated partial result was caused by CQ v1 under-detecting State5 cases where the object lands on the target after an uncontrolled drop.

CQ v2 fixes that Black Bowl recall issue and matches the 20-video manual review set.

## Boundaries

Moka remains helper/diagnostic evidence only. A40 Moka was strong, but 2080Ti strict replication did not reproduce cleanly and showed control fragility.

LIBERO clean denominator is currently insufficient with the configured four-task setup. The 2026-05-17 clean scan found only Open Middle Drawer state1/state2 clean successes, and those are articulated-object boundary cases rather than gripper-release mechanism-eligible runs.

## Next Bottleneck

The next experimental bottleneck is expanded LIBERO task inventory plus clean state discovery. Do not run attack breadth until there are enough non-Black-Bowl clean-success, mechanism-eligible, window-detected, CQ-computable tasks.
