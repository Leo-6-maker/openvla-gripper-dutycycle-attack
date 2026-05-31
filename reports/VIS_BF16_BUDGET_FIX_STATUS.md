# VIS bf16 Budget Fix Status

Date: 2026-05-31

## Status

Implemented bf16-safe TokenPrefixPGD budget accounting in:

```text
src/gripper_attack/attack_adapter.py
```

The PGD master perturbation is now maintained in fp32, then cast to the model
pixel dtype with a final budget guard. If bf16/fp16 quantization would round a
pixel outside the nominal `processor_pixel_values_linf` budget, that element is
reset to the original model-dtype pixel value.

## Added Debug Fields

```text
pixel_master_dtype
pixel_model_dtype
pixel_budget_master_linf
pixel_budget_adv_inputs_linf
pixel_budget_quantized_correction_count
pixel_budget_quantized_correction_rate
```

The reported `observation_perturb_linf` remains the actual Linf of
`debug["adv_inputs"]["pixel_values"]` relative to the original model-dtype
processor pixels.

## Regression Test

Added a mock bf16 multi-step budget test in:

```text
tests/v4/test_token_prefix_pgd_interface.py
```

The test verifies that final bf16 `adv_inputs["pixel_values"]` remain within
the nominal epsilon budget after quantization.

## Real One-frame Smoke

No rollout was run.

Command used physical GPUs `4,5` only:

```text
CUDA_VISIBLE_DEVICES=4,5
```

Input frame:

```text
/data/liuyu/outputs/milestone_2i_visual_fusion_online_detector_pilot_20260530/runs/libero_object/vis_ketchup_clean_ketchup_s0/frames/step_0000.png
```

Output:

```text
tables/vis_one_frame_loader_bf16_budget_fix_smoke.csv
```

Key result:

| Metric | Value |
| --- | ---: |
| objective | `target_action_ce` |
| eps | `4/255` |
| steps | `4` |
| perturbation_linf | `0.015625` |
| nominal eps | `0.015686` |
| clean gripper token | `31872` |
| adversarial gripper token | `31744` |
| token flip | `true` |
| clean gripper action | `0.0` |
| adversarial gripper action | `0.996078` |
| gripper delta | `0.996078` |
| arm L2 | `0.069530` |
| target CE | `32.2500 -> 0.0432` |
| open-bin mass | `9.04e-13 -> 0.957853` |
| close-bin mass | `0.999995 -> 6.47e-05` |

## Gate Interpretation

VIS-1 is now promising on this single frame under strict nominal budget:

- decoded gripper token flips at `eps=4/255`
- decoded gripper action changes strongly in the intended direction
- arm L2 is nonzero and still requires the next gate

This does not authorize rollout.

## Next Required Gate

Run a limited no-rollout arm-drift/random-baseline diagnostic before any VIS
micro rollout:

- gripper objective only
- random same-norm baseline
- report gripper delta, arm L2, gripper-to-arm ratio, and perturbation budget

No forced-window VIS micro, detector-triggered VIS, or rollout should run until
the arm-drift/random-baseline gate passes.
