# S20d V4 Clean Clone Smoke — PASSED

**Date:** 2026-06-11
**Runner:** `scripts/stageb/run_s20d_v4_fixed_window_l3_runner.py`
**Commit:** a11ce18 + hotfixes (use_fast=True, render_gpu_device_id fix)

## Result: 3/3 PASS

| Task | State | Steps | Success Done | Success Check | GPU |
|------|-------|-------|-------------|---------------|-----|
| ketchup | s1 | 180 | True | True | 1,0 |
| tomato_sauce | s3 | 141 | True | True | 2,6 |
| tomato_sauce | s5 | 137 | True | True | 4,5 |

- `done` and `check_success` agree on all 3 states — no metric conflict
- Steps close to expected prior baseline (ketchup ~157, tomato_sauce ~135/136)
- No timeouts, no infra errors

## Hotfixes Applied During Smoke

1. **`use_fast=True`** — conda env lacks protobuf for LlamaTokenizer slow path; fast tokenizer available (model has tokenizer.json)
2. **`render_gpu_device_id` must be physical GPU ID** — EGL renderer does not respect CUDA_VISIBLE_DEVICES remapping
3. **TF/gym imports removed** — not needed for V4-aligned path, TF not in conda env

## Conclusion

S20d V4-based clean clone reproduces baseline success on all tested clean-success states. Safe to proceed to fixed-window RAND-veto and VIS attack on these states.
