# S20d Prior Clean Config Recovery

**Date:** 2026-06-11
**Source repos:** `scripts/v4_run_eval_openvla.py` + shell launchers under `scripts/`

## Recovered V4 Clean Baseline Configuration

### Python environment
- `/data/aviary/envs/openvla_official_libero_20260525/bin/python` (Conda env `openvla_official_libero_20260525`)

### Model
- Path: `/data/aviary/models/openvla/openvla-7b-finetuned-libero-object`
- dtype: bfloat16
- low_cpu_mem_usage: True
- local_files_only: True
- device_map: auto (model_gpu_device_id=-1)
- max_memory per GPU: 10000MiB (OPENVLA_CUDA_MAX_MEMORY default)
- attn_implementation: eager (2080 Ti Turing, no FlashAttention)
- trust_remote_code: True

### Processor
- use_fast: False
- local_files_only: True
- trust_remote_code: True

### Image Preprocessing
- Function: `prepare_openvla_image()`
- Backend: official_pil_lanczos
- center_crop: True
- resize_size: 224
- rotate_180: True
- No JPEG round-trip

### Action Decode
- Function: `decode_with_scores()`
- EOS token: 29871 inserted if missing
- drop_attention_mask: True
- max_new_tokens: action_dim (from model.get_action_dim)
- do_sample: False

### Action Postprocess
- Function: `postprocess_openvla_action_for_libero()`
- normalize_gripper_action: binarize=True
- invert_gripper_action: sign flip

### Environment
- render_gpu_device_id: 0, 2, or 4 (depends on GPU pair)
- image_size: 256x256
- camera_obs_key: agentview_image
- num_steps_wait: 10
- wait action: [0,0,0,0,0,0,-1]
- env.seed(0): hardcoded
- deterministic_init_states: True
- state_ids: explicit per-state

### Clean-only Parameters
- trigger: clean
- rho: 0.0
- epsilon: 0.0
- step_size: 0.0
- attack_steps: 0
- attack_objective: "" (empty)

### Success Definition
- primary: done (LIBERO done flag) — `success = bool(done) if args.success_metric == "done"`
- logged: check_success (env.check_success())
- success_metric flag: "done"

### Task Config
- tasks_config: `configs/v4_tasks_libero_full4_official_eval_20260525.yaml`
- unnorm_key: libero_object (resolved from task suite)
- suite: libero_object

### Max Steps
- 280 (from --max_steps_override) for Object suite
- 10 wait steps not counted in max_steps

## Known Data Sources (Server-Side)

| Source | Seed | Max Steps | States | Notes |
|--------|------|-----------|--------|-------|
| milestone_r1_official_eval_20260526 | 1 | 280? | 0-9 | ketchup s0=158, s1=127 claimed |
| milestone_r2_official_v4_object_alignment_20260526 | seed 0? | 280? | 0-9 | May have different results |
| milestone_1d / object_full_10x10 | seed 0 | 280 | 0-9 | 10 tasks × 10 states |
| full4 Table1 (local mirror) | seed 1 | 400 | 0-9 | ketchup s1=FAIL, s3=FAIL |

## Source Confidence

- High: parameters confirmed from shell scripts + Python code
- Medium: attn_implementation (env var, not explicitly in shell scripts but deduced from 2080Ti hw)
- Low: exact per-state success from specific server runs (not accessible from local machine)

## Action

S20d smoke will establish ground truth using the exact recovered config.
