# VIS Small-Epsilon Blocked Final

Date: 2026-05-31

VIS remains blocked before rollout.

## Blocking Condition

The diagnostic harness now enforces the correct interface and the reusable OpenVLA re-decode helper is implemented. The remaining blocker is the real one-frame model/frame/attack-result loader that produces `debug["adv_inputs"]` for the diagnostic.

Because of that, the branch cannot establish:

- decoded gripper token flip at epsilon <= 8/255
- decoded gripper action movement in the intended direction
- controlled arm drift
- targeted effect stronger than random same-norm control

## Action Taken

- Ran dry-run schema for token-flip diagnostics.
- Ran a real diagnostic probe only far enough to verify that it fails loudly instead of fabricating decoded actions.
- Wrote `tables/vis_token_flip_threshold_diagnostic.csv` with the missing-decode error.
- Added `src/gripper_attack/openvla_redecode.py` and mock tests for OpenVLA action re-decode from prepared adversarial inputs.

## Gate Decision

VIS-1: FAIL.

Do not run:

- VIS rollout
- forced-window VIS micro
- detector-triggered VIS

Next required work is wiring a concrete one-frame loader that:

1. loads a real Object contact frame,
2. runs `TokenPrefixPGDAttacker`,
3. passes `attack_result.debug["adv_inputs"]` to `redecode_openvla_action_from_adv_inputs`,
4. records decoded clean/adv token and action metrics.
