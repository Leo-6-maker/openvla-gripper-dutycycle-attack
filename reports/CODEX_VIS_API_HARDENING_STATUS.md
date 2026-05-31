# Codex VIS API Hardening Status

Date: 2026-05-31

## Scope

This update hardens the TokenPrefixPGD interface without enabling VIS rollout or changing production command-layer behavior.

## Changes

- Added `get_adv_inputs_from_attack_result(result)`.
- Documented that `TokenPrefixPGDAttacker.attack()` returns `action_adv=None` by design.
- Documented that callers must re-decode OpenVLA from `debug["adv_inputs"]`.
- Added lightweight mock tests for dtype handling and `adv_inputs` validation.

## Safety Boundary

The helper validates that `debug["adv_inputs"]` exists and contains:

- `input_ids`
- `pixel_values`

If the field is absent or incomplete, the helper raises a clear error. It never returns zeros and does not infer an actuator action.

## Gate A Status

Expected pass condition:

- mock tests run without OpenVLA model weights
- production predicate and sustained proxy tests still pass
- no production runner behavior changes

This status file is updated by code review and test output, not by rollout.

## Follow-Up

- Consider a formal `adv_inputs` field in `AttackResult`.
- Add a re-decode helper once the OpenVLA decode interface is stable.
- Add a caller-side guard so `action_adv=None` cannot be silently interpreted as a zero action.
