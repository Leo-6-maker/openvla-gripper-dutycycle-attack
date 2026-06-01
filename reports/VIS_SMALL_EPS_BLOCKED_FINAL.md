# VIS Small-Epsilon Blocked Final

Date: 2026-05-31

VIS remains blocked before rollout.

## Blocking Condition

The diagnostic harness now enforces the correct interface, but the real OpenVLA re-decode path from `debug["adv_inputs"]` is not yet implemented.

Because of that, the branch cannot establish:

- decoded gripper token flip at epsilon <= 8/255
- decoded gripper action movement in the intended direction
- controlled arm drift
- targeted effect stronger than random same-norm control

## Action Taken

- Ran dry-run schema for token-flip diagnostics.
- Ran a real diagnostic probe only far enough to verify that it fails loudly instead of fabricating decoded actions.
- Wrote `tables/vis_token_flip_threshold_diagnostic.csv` with the missing-decode error.

## Gate Decision

VIS-1: FAIL.

Do not run:

- VIS rollout
- forced-window VIS micro
- detector-triggered VIS

Next required work is a real re-decode helper that consumes `debug["adv_inputs"]`.
