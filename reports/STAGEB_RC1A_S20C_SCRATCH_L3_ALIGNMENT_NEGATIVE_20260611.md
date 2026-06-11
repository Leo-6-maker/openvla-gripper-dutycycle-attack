# S20c Scratch L3 Alignment — Negative Result

**Date:** 2026-06-11
**Branch:** exp/vis-prefix-margin-repair-20260603
**Runner:** `scripts/stageb/run_s20c_official_l3_runner.py`

## Result: FAILED

The S20c scratch official-aligned L3 runner could not reproduce clean baseline success.

### Smoke Test Results

| Episode | State | Steps | Success | Done |
|---------|-------|-------|---------|------|
| ketchup (+center_crop) | s0 | 400 | False | False |
| ketchup (+center_crop +env.seed(0)) | s0 | 400 | False | False |
| tomato_sauce (+center_crop) | s3 | 400 | False | False |

### What Was Tried

| Dimension | Setting | Status |
|-----------|---------|--------|
| Image preprocessing | `prepare_openvla_image` + `official_pil_lanczos` + `center_crop=True` + `resize_size=224` | Aligned |
| Gripper postprocess | normalize + invert | Aligned |
| Success detection | `env.check_success()` | V4 uses `--success_metric done` |
| Wait steps | 10 | Aligned |
| Env seed | 0 | Aligned |
| Model loading | `device_map='auto'`, `torch.bfloat16` | V4 uses additional `max_memory`, `attn_implementation` |
| Action decode | `model.generate()` + manual bin_centers decode | V4 uses `decode_with_scores()` + EOS token 29871 insertion |
| Max steps | 400 | V4 uses 280 |
| Processor | `use_fast=False` | Aligned |
| Attention mask | Not handled | V4 uses `drop_attention_mask=True` |
| EOS token | Not inserted | V4 inserts token 29871 if missing |

### Root Cause

Multiple residual gaps vs V4 official runner prevented clean reproduction:
1. Missing EOS token 29871 insertion in `model.generate()` input
2. Missing `drop_attention_mask=True`
3. Different decode path (manual vs `decode_with_scores()`)
4. `max_steps=400` vs V4's `max_steps_override=280`
5. `success_metric=check_success` vs V4's `success_metric=done`
6. Missing `attn_implementation` in model loading

### Decision

**Scratch S20c runner is not accepted for Level 3.** All future L3 work will use the V4 runner (`scripts/v4_run_eval_openvla.py`) as the base.
