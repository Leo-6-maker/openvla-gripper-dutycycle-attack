# M3 Step78 Session Handoff

## Current State

`M3_STEP78_TRUE_PGD_CANARY = RANDOM_NOT_BEATEN`

The route and official transfer worked, but the matched `RAND20` control tied TRUE_PGD on official target margin and emitted the same target token. A follow-up audit computed that the selected `RAND20` best only matched the clean arm prefix `4/6`, while TRUE_PGD matched `6/6`; this does not change the preregistered primary result.

## Key Commits

- Integration base after PR #14 merge: `a47f0a9ddd00ad61b47a16e439aea4c9c3f8d7e7`
- Experiment commit: `af545e1c5eb1012ad5dc8b8872e50596315bd4d5`
- Initial result report commit: `6d5abca43dd6dae3487d547c20ff617a242e1abe`

## Pull Request

Draft PR: `#15`

## Server Artifacts

- Clean capture: `/data/liuyu/outputs/m3_step78_true_pgd_20260614/capture_step78_f18537d_r2`
- Preflight: `/data/liuyu/outputs/m3_step78_true_pgd_20260614/preflight_step78_912dfff_seed80`
- Canary: `/data/liuyu/outputs/m3_step78_true_pgd_20260614/canary_step78_af545e1_seed80`

## Commands Run

CPU:

```text
/data/aviary/envs/openvla_official_libero_20260525/bin/python -m py_compile src/gripper_attack/attack_adapter.py scripts/stageb/run_m3_step78_true_pgd_fixed_frame.py tests/stageb/test_m3_true_pgd_route_contract.py tests/stageb/test_m3_step78_fixed_frame_runner.py
manual_cpu_harness_passed 32 tests
```

Capture:

```text
CUDA_VISIBLE_DEVICES=2,6 OPENVLA_CUDA_MAX_MEMORY=10000MiB \
/data/aviary/envs/openvla_official_libero_20260525/bin/python \
scripts/stageb/run_m3_step78_true_pgd_fixed_frame.py \
  --mode capture_input \
  --config configs/m3_step78_true_pgd_31744.yaml \
  --output_dir /data/liuyu/outputs/m3_step78_true_pgd_20260614/capture_step78_f18537d_r2 \
  --model_gpu_device_id -1 \
  --render_gpu_device_id 2
```

Preflight:

```text
CUDA_VISIBLE_DEVICES=2,6 OPENVLA_CUDA_MAX_MEMORY=10000MiB \
/data/aviary/envs/openvla_official_libero_20260525/bin/python \
scripts/stageb/run_m3_step78_true_pgd_fixed_frame.py \
  --mode preflight_zero_step \
  --config configs/m3_step78_true_pgd_31744.yaml \
  --input_dir /data/liuyu/outputs/m3_step78_true_pgd_20260614/capture_step78_f18537d_r2 \
  --output_dir /data/liuyu/outputs/m3_step78_true_pgd_20260614/preflight_step78_912dfff_seed80 \
  --attack_seed 80 \
  --model_gpu_device_id -1
```

Canary:

```text
CUDA_VISIBLE_DEVICES=2,6 OPENVLA_CUDA_MAX_MEMORY=10000MiB \
/data/aviary/envs/openvla_official_libero_20260525/bin/python \
scripts/stageb/run_m3_step78_true_pgd_fixed_frame.py \
  --mode canary \
  --config configs/m3_step78_true_pgd_31744.yaml \
  --input_dir /data/liuyu/outputs/m3_step78_true_pgd_20260614/capture_step78_f18537d_r2 \
  --output_dir /data/liuyu/outputs/m3_step78_true_pgd_20260614/canary_step78_af545e1_seed80 \
  --attack_seed 80 \
  --model_gpu_device_id -1
```

## Stop Reason

The fixed-frame random superiority gate failed. Do not run the fixed-frame panel, full-window Tomato rollout, rescue, or held-out stages from this configuration.

## Next Action

Phase A follow-up has tightened PR #15 artifacts. The next authorized work is Phase C: audit all 20 frozen RAND candidates for official token and arm-selectivity behavior, then Phase D CPU-only implementation of the non-saturating target-logratio v2 objective. Changing epsilon, target token, frame, random candidate count, or primary success criteria requires owner approval.
