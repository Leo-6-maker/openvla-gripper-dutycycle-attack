# V4 Runner vs Corrected Official Script — Implementation Diff

## 1. Prompt Format

| Aspect | Official (corrected) | v4 Runner |
|--------|---------------------|-----------|
| Format | `"In: What action should the robot take to {task.lower()}?\nOut:"` | `f"In: What action should the robot take to {instruction}?\nOut:"` |
| Task text | `task_desc.lower()` from LIBERO task.language | `base_instruction` from `bench.get_task(idx).language` |
| Same? | **YES** | Same format and source |

## 2. Image Preprocessing — **MAJOR DIFFERENCE**

### Official (corrected) — PIL path

```python
img = obs["agentview_image"]
img = img[::-1, ::-1]  # rotate 180
img = Image.fromarray(img).convert("RGB")
img = img.resize((224, 224), Image.LANCZOS)        # PIL Lanczos
# center_crop:
scale = 0.9 ** 0.5
cw, ch = max(1, int(w * scale)), max(1, int(h * scale))
left, top = (w - cw) // 2, (h - ch) // 2
img = img.crop((left, top, left + cw, top + ch))
img = img.resize((224, 224), Image.LANCZOS)        # PIL Lanczos again
```

### v4 Runner (`--libero_official_preprocess`) — TF path

```python
arr = np.asarray(image_np)
arr = arr[::-1, ::-1]                              # rotate 180
tensor = tf.convert_to_tensor(arr)
tensor = tf.io.decode_image(                       # JPEG encode → decode (LOSSY!)
    tf.io.encode_jpeg(tensor), expand_animations=False, dtype=tf.uint8)
tensor = tf.image.resize(tensor, [224, 224],
    method="lanczos3", antialias=True)              # TF Lanczos3
# center_crop:
tensor = tf.image.crop_and_resize(                  # TF bilinear crop+resize
    tf.expand_dims(tensor, axis=0),
    boxes=[[(1-scale)/2, (1-scale)/2, (1+scale)/2, (1+scale)/2]],
    box_indices=[0], crop_size=[224, 224], method="bilinear")[0]
tensor = tf.cast(tf.clip_by_value(tf.round(tensor), 0, 255), tf.uint8)
```

### Key Differences

| Step | Official PIL | v4 TF |
|------|-------------|-------|
| Resize algorithm | PIL LANCZOS | TF lanczos3 |
| JPEG round-trip | **NO** | **YES** — lossy compression |
| Center crop resize | PIL LANCZOS | TF bilinear |
| Rounding | None (float → uint8 implicit) | `tf.round()` + clip |

**The JPEG encode/decode in the TF path introduces lossy compression artifacts.** This is the most significant difference and could explain why v4 produces different actions.

## 3. Model Inference Path

| Aspect | Official (corrected) | v4 Runner |
|--------|---------------------|-----------|
| Method | `model.generate()` | `model.generate()` (via `decode_with_scores()`) |
| Input construction | `processor(prompt, image)` | `processor(prompt, image)` |
| Attention mask | Dropped | Dropped (`drop_attention_mask=True`) |
| EOS token | Added if missing | Not explicitly added in code path |
| dtype | bfloat16 | bfloat16 |
| do_sample | False | False |
| max_new_tokens | `action_dim` (7) | `K_trigger` (from config, typically 7) |

**Potential difference:** The v4 runner does NOT explicitly add the EOS token (29871) before generation. The official script checks `if not torch.all(input_ids[:, -1] == 29871)` and appends it. This could affect token generation behavior.

## 4. Action Decoding

Both use identical v4 decoding logic:
```python
token_ids = gen.sequences[0, -action_dim:]
vocab_size = model.config.text_config.vocab_size - model.config.pad_to_multiple_of
discretized = np.clip(vocab_size - token_ids - 1, 0, model.bin_centers.shape[0] - 1)
norm_actions = model.bin_centers[discretized]
stats = model.get_action_stats(unnorm_key)
mask = stats.get("mask", np.ones_like(stats["q01"], dtype=bool))
high, low = np.array(stats["q99"]), np.array(stats["q01"])
action = np.where(mask, 0.5 * (norm_actions + 1) * (high - low) + low, norm_actions)
```

| Aspect | Official | v4 |
|--------|----------|-----|
| unnorm_key | `libero_object` | `libero_object` (resolved via `resolve_unnorm_key`) |
| Decoding logic | Manual v4 decode | Manual v4 decode in `decode_with_scores()` |
| Same? | **YES** | Same logic |

## 5. Gripper Postprocess

Both use identical logic:
```python
action[..., -1] = 2.0 * action[..., -1] - 1.0       # [0,1] → [-1,1]
action[..., -1] = np.sign(action[..., -1])            # binarize
action[..., -1] = 1.0 if action[..., -1] == 0 else action[..., -1]
action[..., -1] = -1.0 * action[..., -1]              # invert sign
```

| Aspect | Official | v4 |
|--------|----------|-----|
| normalize_gripper_action | Yes | Yes (`postprocess_gripper=True`) |
| invert_gripper_action | Yes | Yes |
| Same? | **YES** | Same logic |

## 6. Environment Setup

| Aspect | Official (corrected) | v4 Runner |
|--------|---------------------|-----------|
| Env class | `OffScreenRenderEnv` | `OffScreenRenderEnv` |
| camera_heights | 256 | `args.image_size` (likely 256) |
| camera_widths | 256 | `args.image_size` (likely 256) |
| horizon | **NOT passed** (uses default 1000) | **`max_steps + num_steps_wait`** (e.g. 290) |
| render_gpu_device_id | **NOT passed** | Passed |
| env.seed(0) | Yes | Yes (try/except) |

**The horizon parameter DOES NOT affect the 71/100 result** because the main loop uses `for t in range(max_steps)` where `max_steps=280`, so the episode ends at 280 model steps regardless of horizon value (as long as horizon >= max_steps + wait_steps = 290).

## 7. Episode Flow

### Official
```python
while t < max_steps + num_steps_wait:  # 290
    if t < num_steps_wait:              # first 10 are wait
        env.step(dummy_action)
        continue
    action = get_vla_action(...)        # inference
    action = normalize_gripper(action)
    action = invert_gripper(action)
    obs, reward, done, info = env.step(action)
    if done: success = True; break
```

### v4
```python
# Wait steps BEFORE main loop
for _ in range(num_steps_wait):        # 10 wait steps
    env.step(dummy_action)
# Main loop
for t in range(max_steps):             # 280 model steps
    clean, ... = decode_with_scores(...)
    clean_env = postprocess_openvla_action_for_libero(clean)
    # ... trigger/budget/attack logic (no-op for clean) ...
    obs, reward, done, info = env.step(executed)
    if done: success = True; break
```

Both do 10 wait steps + up to 280 model steps. The total budget is identical.

## 8. Success Metric

Both use `done` from `env.step()`:
```python
if done:
    success = True
    break
```

## Summary of Differences

| # | Category | Official | v4 | Impact |
|---|----------|----------|-----|--------|
| 1 | Prompt | `In:...\nOut:` | `In:...\nOut:` | **Same** |
| 2 | Image resize | PIL LANCZOS | TF lanczos3 | **Different** |
| 2a | JPEG round-trip | No | Yes (lossy) | **Different** |
| 2b | Center crop resize | PIL LANCZOS | TF bilinear | **Different** |
| 2c | Rounding | Implicit | `tf.round()` | **Different** |
| 3 | EOS token | Added if missing | Not explicitly added | **Potentially different** |
| 4 | Action decode | v4 logic | v4 logic | **Same** |
| 5 | Gripper postprocess | normalize+invert | normalize+invert | **Same** |
| 6 | Env horizon | Not set (1000 default) | 290 | **Same effective** |
| 7 | Model steps | 280 | 280 | **Same** |
| 8 | Wait steps | 10 (inside loop) | 10 (before loop) | **Same effective** |

## Primary Suspect

**Image preprocessing (Category B)** — specifically the JPEG round-trip in the TF path. This introduces lossy compression that degrades image quality, potentially causing the model to produce worse actions on borderline episodes.

Secondary suspect: **EOS token (Category C)** — if v4 doesn't add the EOS token but the official script does, this could shift token generation slightly.
