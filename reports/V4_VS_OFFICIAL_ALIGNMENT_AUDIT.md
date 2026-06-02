# v4 Runner vs Official OpenVLA LIBERO Eval — Alignment Audit

**Generated**: 2026-05-26

## Summary

| Aspect | Official | v4 Runner | Aligned? |
|--------|----------|-----------|----------|
| Image preprocessing | 180° rotate + resize to 256 | Same (`prepare_openvla_image`) | ALIGNED |
| Gripper postprocess | normalize + invert | Same (`postprocess_openvla_action_for_libero`) | ALIGNED |
| Wait steps | dummy_action [0,0,0,0,0,0,-1], num_steps_wait | Same | ALIGNED |
| env.seed(0) | Called after env creation | Called (line 971) | ALIGNED |
| Step loop | `while t < max_steps + num_steps_wait` | `for t in range(max_steps)` | STYLE ONLY |
| Model norm_stats check | Assert unnorm_key in model.norm_stats | Same fallback logic | ALIGNED |
| BDDL path | `get_libero_path("bddl_files") + task.problem_folder + task.bddl_file` | `bench.get_task_bddl_file_path(idx)` | EQUIVALENT |
| Env creation: camera | resolution=256 → camera_heights=256, camera_widths=256 | `--image_size 256` → same | ALIGNED |
| **Env creation: horizon** | **NOT PASSED** (default 1000) | **`horizon=280`** | **MISMATCH** |
| **Env creation: render_gpu_device_id** | **NOT PASSED** | **`render_gpu_device_id=0`** | **EXTRA** |

## Root Cause: horizon = 280 + wait steps = early termination

The `horizon` parameter in robosuite 1.4.1 sets `self.done = (self.timestep >= self.horizon)`.
With `horizon=280`:

```
10 wait steps  → env timestep = 10
270 main steps → env timestep = 280 → done=True
Step 271      → ValueError("executing action in terminated episode")
```

The 10 wait steps consume horizon budget! The model only gets 270 steps instead of 280.

The official code does NOT pass horizon — it uses the default (1000). The episode length
is controlled entirely by the caller's loop condition, not the env's internal horizon.

## Fix Required

Remove `horizon=int(args.max_steps_override or task["max_steps"])` from both `OffScreenRenderEnv`
calls (lines 858 and 969).

Before:
```python
env=OffScreenRenderEnv(..., horizon=int(args.max_steps_override or task["max_steps"]))
```

After:
```python
env=OffScreenRenderEnv(...)
```

This aligns with the official code and gives the model the full 280 steps.
