# VIS Token-Flip Threshold Diagnostic Run

Date: 2026-05-31

## Status

Dry-run schema passed. The reusable OpenVLA re-decode helper is now implemented, but real one-frame token-flip smoke remains blocked because the diagnostic script still needs a concrete model/frame/attack-result loader that creates `debug["adv_inputs"]` from a real frame.

## Commands

Dry-run:

```bash
python scripts/diagnostics/vis_token_flip_threshold.py --dry-run --print-schema --output_csv /tmp/vis_token_flip_schema.csv
```

Real diagnostic probe:

```bash
python scripts/diagnostics/vis_token_flip_threshold.py \
  --frame contact_frame_placeholder \
  --instruction "pick up the ketchup and place it in the basket" \
  --model_path /missing/openvla \
  --unnorm_key libero_object \
  --output_csv tables/vis_token_flip_threshold_diagnostic.csv
```

## Result

The harness failed loudly as intended:

```text
OpenVLA adversarial re-decode helper is implemented, but this diagnostic still needs a real model/frame/attack-result loader for one-frame smoke.
The real path must call redecode_openvla_action_from_adv_inputs(model, processor, debug['adv_inputs'], ...); do not use action_adv and do not fallback to zeros.
```

The CSV was written with the error field populated:

```text
tables/vis_token_flip_threshold_diagnostic.csv
```

## Gate VIS-1

Result: FAIL / BLOCKED.

Reason:

- decoded gripper token flip was not evaluated
- decoded action was not evaluated
- real model/frame/attack-result loader is missing

No VIS rollout is allowed from this state.
