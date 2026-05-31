# VIS Token-Flip Threshold Sweep Run

Date: 2026-05-31

## Scope

No rollout and no training were run.

The diagnostic used one saved Object frame:

```text
/data/liuyu/outputs/milestone_2i_visual_fusion_online_detector_pilot_20260530/runs/libero_object/vis_ketchup_clean_ketchup_s0/frames/step_0000.png
```

The process exposed only physical GPUs:

```text
CUDA_VISIBLE_DEVICES=2,6
```

## Output

Partial raw table:

```text
tables/vis_token_flip_threshold_sweep_full.csv
```

Summary:

```text
tables/vis_token_flip_threshold_sweep_summary.csv
```

## Result Summary

| Metric | Value |
| --- | ---: |
| rows total | 25 |
| successful rows | 24 |
| error rows | 1 |
| token-flip rows | 7 |
| nominal-budget-ok rows | 0 |
| token-flip and nominal-budget-ok rows | 0 |

The first observed token flip was:

| Field | Value |
| --- | --- |
| objective | `gripper_open_region_ce` |
| eps | `4/255` |
| steps | `20` |
| perturbation_linf | `0.01953125` |
| arm_action_l2 | `0.0067647104151546955` |

## Failure

The sweep stopped at:

| Field | Value |
| --- | --- |
| objective | `gripper_logit_margin_cw` |
| eps | `4/255` |
| steps | `10` |
| error | `CUDA error: an illegal memory access was encountered` |

Kernel log shows a contemporaneous Xid:

```text
Xid 31, pid=47697, name=python, MMU Fault
```

## Interpretation

This sweep does not pass VIS-1.

Although some rows produced decoded gripper token flips, every successful row
exceeded the nominal epsilon budget under strict processor-pixel Linf checking.
For example:

```text
eps = 4/255 = 0.015686
observed perturbation_linf = 0.01953125
```

The likely cause is bf16 quantization of processor-normalized `pixel_values` in
the real OpenVLA path. This means token flips from this partial sweep cannot be
used as rollout authorization evidence.

## Gate Decision

VIS-1: FAIL / blocked.

Do not run:

- VIS rollout
- forced-window VIS micro
- detector-triggered VIS
- heavy arm-drift sweep

## Next Required Work

Before another real threshold sweep:

1. Make TokenPrefixPGD budget accounting robust to bf16 model input
   quantization.
2. Ensure generated `adv_inputs["pixel_values"]` satisfy nominal
   `processor_pixel_values_linf` or explicitly report an approved quantized
   budget semantics.
3. Add a regression test for multi-step bf16 budget accounting.
4. Re-run a small no-rollout diagnostic before any broader threshold sweep.
