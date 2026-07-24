# Upstream Clean30 Final Report

## Gate

`UPSTREAM_CLEAN30_COMPLETION_AND_GITHUB_EVIDENCE_FREEZE`

## Date

2026-06-21

## Profiles

| Profile | dtype | attn | backend | Success | Rate |
|---|---|---|---|---|---|
| FP32-Eager | float32 | eager | upstream_tf_jpeg | 24/30 | 80.0% |
| BF16-Flash2 | bfloat16 | flash_attention_2 | upstream_tf_jpeg | 22/30 | 73.3% |
| BF16-Eager | bfloat16 | eager | upstream_tf_jpeg | DEFERRED | - |

## Paired Analysis

| Class | Count |
|---|---|
| both_success | 19 |
| fp32_only | 5 |
| flash2_only | 3 |
| neither | 3 |
| invalid | 0 |

## Common Success Set (19 episodes, 10 tasks)

`task0_init0`, `task1_init0`, `task1_init1`, `task1_init2`, `task2_init1`, `task2_init2`,
`task3_init0`, `task4_init0`, `task5_init0`, `task5_init2`, `task6_init0`, `task6_init1`,
`task6_init2`, `task7_init1`, `task7_init2`, `task8_init0`, `task8_init2`, `task9_init0`,
`task9_init2`

### Task Coverage

| Task | Common Success Count |
|---|---|
| 0 | 1/3 |
| 1 | 3/3 |
| 2 | 2/3 |
| 3 | 1/3 |
| 4 | 1/3 |
| 5 | 2/3 |
| 6 | 3/3 |
| 7 | 2/3 |
| 8 | 2/3 |
| 9 | 2/3 |

## Divergent Episodes

### FP32 success but Flash2 failure (FP32-only)
- task3_init2, task4_init2, task7_init0, task8_init1, task9_init1

### Flash2 success but FP32 failure (Flash2-only)
- task0_init1, task0_init2, task2_init0

### Both fail
- task3_init1, task4_init1, task5_init1

## Running Code SHA

- runner: `ef0b2c606b1beafc75410e149f43e85bbf9b42659829c79da35fdffe77567f20`
- preprocess: `6de76cb53e9b0acdb5b8d877c988fa69aca10a30519270556c6569f3b5f9fdcc`
- plan: `9549ee9a53b23bdc06329b2b8b12a4ed73278bdc7f97a732f2ef00653655219e`

## Contract

- seed=42, max_steps=220, wait_steps=10
- unnorm_key=libero_spatial, resize=224
- center_crop=True, crop_scale=0.9
- Model: spatial_c8f03f4_20260620
- GPU4 UUID: NVIDIA A800-SXM4-80GB (FP32)
- GPU6 UUID: NVIDIA A800-SXM4-80GB (Flash2)

## Next Gate

`UPSTREAM_DETECTOR_TRANSFER_AND_THRESHOLD_AUDIT`
