# M3 GPU45 Fixed-Frame Qualification - 2026-06-16

## Result

```text
STAGE: M3_GPU45_FIXED_FRAME_INFRA_QUALIFICATION
RESULT_CLASS: GPU45_FIXED_FRAME_REPEATABILITY_FAIL
GPU_MAPPING: physical 4,5
COMMIT: b8a6b6505fd8c9507e75ea3db97b4fb7086571a0
INPUT: /data/liuyu/outputs/m3_arm_v4_panel_capture_f41ab1a_r2/step78
NO_PGD: true
NO_RAND21: true
NO_SHUFFLED_GRAD21: true
NO_PANEL_CAPTURE: true
NO_LIBERO_ROLLOUT: true
```

The authorized GPU45 infra qualification was run only on the development
step78 fixed-frame input.  It did not run a scientific attack, final panel
frame, V5.1 20-state capture, or LIBERO rollout.

## GPU Binding

```text
CUDA_VISIBLE_DEVICES=4,5
GPU4: GPU-d0a54f5d-938c-a148-fff9-c135201e3f61
GPU5: GPU-9794d733-042f-46a2-fc86-5a3fe32a158a
```

The runner hard-checks the physical index to UUID order before loading the
model.  GPU 4/5 were idle before and after the final run.

## Runs

| Output dir | Commit | Profile | Result |
| --- | --- | --- | --- |
| `/data/liuyu/outputs/m3_gpu45_fixed_frame_qualification_b029273_20260616_020341` | `b029273` | initial | OOM during first backward |
| `/data/liuyu/outputs/m3_gpu45_fixed_frame_qualification_b029273_20260616_020440_mm7500` | `b029273` | max_memory=7500MiB | OOM during first backward |
| `/data/liuyu/outputs/m3_gpu45_fixed_frame_qualification_b029273_20260616_020521_mm6000` | `b029273` | max_memory=6000MiB | OOM before/at backward |
| `/data/liuyu/outputs/m3_gpu45_fixed_frame_qualification_d8b8c25_20260616_020708` | `d8b8c25` | model params frozen | completed, repeatability fail |
| `/data/liuyu/outputs/m3_gpu45_fixed_frame_qualification_b8a6b65_20260616_020856` | `b8a6b65` | deterministic profile | completed, repeatability fail |

The OOM was fixed by freezing model parameters before the pixel-gradient
check, matching the production attacker's model-freeze behavior.  A subsequent
deterministic profile disabled TF32 and cudnn benchmarking, but the fixed-frame
repeatability gate still failed.

## Final Run Details

Final output:

```text
/data/liuyu/outputs/m3_gpu45_fixed_frame_qualification_b8a6b65_20260616_020856
```

Summary:

```text
tokens_stable: false
gripper_stable: false
score_hash_stable: false
gradient_hash_stable: false
gradient_all_finite: true
```

Repeat rows:

| repeat | official gripper | official target margin | target score | best competitor | gradient finite |
| ---: | ---: | ---: | ---: | ---: | --- |
| 0 | 31872 | -0.25 | 41.00 | 31872 @ 41.25 | true |
| 1 | 31872 | -0.25 | 40.75 | 31872 @ 41.00 | true |
| 2 | 31744 | 0.00 | 41.25 | 31872 @ 41.25 | true |

The fixed clean input is near the 31744/31872 boundary, and the official
generation row varied enough across repeated clean forwards to flip the emitted
gripper token on repeat 2.  Pixel gradients were finite, but the gradient hash
was not stable across repeats.

## Claim Matrix

Allowed:

```text
GPU45 physical index to UUID binding was enforced.
Model-load/OOM profiling was performed on the development step78 input.
The runner can complete the fixed-frame gradient check after freezing model parameters.
The final deterministic-profile run failed fixed-frame repeatability.
```

Forbidden:

```text
Do not claim TRUE_PGD > RAND.
Do not claim fixed-frame official-token effect.
Do not claim multi-frame robustness.
Do not claim closed-loop Layer3.
Do not proceed to panel GPU execution from this result.
```

## Next Gate

```text
NEXT_ACTION: stop for audit
BLOCKER: clean fixed-frame official generation and gradient repeatability failed on GPU45
```

