# VIS Re-decode Helper Implementation Status

Date: 2026-05-31

## Status

Implemented reusable helper:

`src/gripper_attack/openvla_redecode.py`

The helper decodes OpenVLA continuous actions from `debug["adv_inputs"]` prepared by `TokenPrefixPGDAttacker`.

## Behavior

- Requires `adv_inputs["input_ids"]` and `adv_inputs["pixel_values"]`.
- Preserves tensor dtype/device when moving inputs to the model device.
- Appends the OpenVLA action-prefix token `29871` when needed, matching the existing evaluation runner.
- Calls `model.generate(..., max_new_tokens=action_dim, do_sample=False, return_dict_in_generate=True, output_scores=True)`.
- Decodes generated action tokens through `model.bin_centers` and `model.get_action_stats(unnorm_key)`.
- Rejects missing inputs, missing model parameters, missing stats, dimension mismatch, and NaN/Inf decoded actions.
- Never uses `action_adv`.
- Never falls back to zeros.

## Diagnostic Integration

Updated:

- `scripts/diagnostics/vis_token_flip_threshold.py`
- `scripts/diagnostics/vis_arm_drift_sweep.py`

Both scripts now import the shared re-decode helper. Real diagnostic mode still requires a model/frame/attack-result loader; when that loader is missing, the scripts write an error row and fail loudly rather than producing fake decoded actions.

## Tests

Added:

- `tests/v4/test_openvla_redecode.py`

Coverage:

- rejects missing `adv_inputs`
- rejects missing `pixel_values`
- rejects missing `input_ids`
- preserves `pixel_values` dtype through fake generation
- returns finite non-zero decoded action from fake OpenVLA generation
- rejects action-stat dimension mismatch

## Gate VIS-Decode

Status: partial pass / integration blocked.

The reusable re-decode helper is implemented and mock-tested. A real one-frame OpenVLA decode smoke is still blocked because the diagnostic harness does not yet have a concrete frame/model/attack-result loader that produces `debug["adv_inputs"]` from a real frame.

No rollout was launched.
