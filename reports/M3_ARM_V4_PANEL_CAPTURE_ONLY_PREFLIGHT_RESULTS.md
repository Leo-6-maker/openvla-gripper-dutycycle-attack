# M3 arm-v4 panel capture-only preflight results

## Decision

`CAPTURE_ONLY_PREFLIGHT_COMPLETE`

This run performed only the authorized clean single-replay capture. It did not
run PGD, RAND, shuffled-gradient, seed85 panel attack, seed86, or any LIBERO
attack rollout.

## Execution

| Field | Value |
| --- | --- |
| Commit | `f41ab1a27e5337b6aceedd104e81a24dd8ff2168` |
| Config | `configs/m3_step78_true_pgd_31744_logratio_arm_v4.yaml` |
| Config SHA256 | `2dcef93bf2decf742e0c98f321267ae665b57890f3ab03088dfda3686ae8a2a8` |
| Output dir | `/data/liuyu/outputs/m3_arm_v4_panel_capture_f41ab1a_r2` |
| Log | `/data/liuyu/outputs/m3_arm_v4_panel_capture_f41ab1a_r2.log` |
| CUDA_VISIBLE_DEVICES | `1,0` |
| Model GPU mode | `--model_gpu_device_id -1` |
| Render GPU | `--render_gpu_device_id 1` |

The first attempted capture used `--model_gpu_device_id 0` and failed during
model loading with CUDA OOM before any capture artifacts were written. The
successful capture used the same two-GPU mapping as the accepted arm-v4 seed83
and seed84 fixed-frame runs.

## Step78 Parity

Step78 matched the previously frozen step78 input on all preregistered parity
fields:

- raw image SHA;
- processed tensor SHA;
- prompt-token SHA;
- clean exact 7 tokens;
- clean arm prefix;
- clean gripper token.

Result:

```text
STEP78_PARITY: PASS
```

## Clean Eligibility

Main denominator frames:

| Frame | Clean status | Clean gripper token |
| ---: | --- | ---: |
| 70 | `CLEAN_ALREADY_TARGET` | 31744 |
| 72 | `CLEAN_ALREADY_TARGET` | 31744 |
| 74 | `CLEAN_ALREADY_TARGET` | 31744 |
| 76 | `CLEAN_ALREADY_TARGET` | 31744 |
| 80 | `CLEAN_ELIGIBLE` | 31872 |
| 82 | `CLEAN_ALREADY_TARGET` | 31744 |
| 84 | `CLEAN_ALREADY_TARGET` | 31744 |
| 86 | `CLEAN_ELIGIBLE` | 31872 |

Positive control:

| Frame | Clean status | Clean gripper token |
| ---: | --- | ---: |
| 78 | `CLEAN_ELIGIBLE` | 31872 |

Summary:

```text
MAIN_CLEAN_ELIGIBLE: 2/8
MAIN_CLEAN_INELIGIBLE: 6/8
```

The preregistered panel frame set is therefore not executable as a valid
CLOSE-context panel denominator. No frame was replaced.

## Result Class

```text
PANEL_CAPTURE_ONLY_PREFLIGHT_PASS_WITH_PANEL_DENOMINATOR_BLOCKED
```

## Allowed Claim

On the exact f41ab1a clean single-replay capture, step78 input parity was
reproduced, but the preregistered main panel frame set contains only two clean
CLOSE contexts.

## Forbidden Claims

Do not claim:

- seed85 attack was run;
- panel robustness was tested;
- arm-v4 multi-frame robustness is established;
- LIBERO closed-loop Layer3 is established;
- clean-ineligible frames may be replaced.

## Next Gate

Seed85 panel attack remains unauthorized. The current preregistered frame set
should stop at capture-only audit unless the responsible reviewer explicitly
changes the frame set or denominator definition.
