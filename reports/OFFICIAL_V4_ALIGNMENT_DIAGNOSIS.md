# Official vs v4 Object Alignment Diagnosis

## 1. Why does corrected official achieve 80/100 while v4 gets 71/100?

The +9 gap comes from 5 tasks, primarily cream_cheese (+3), salad_dressing (+2), butter (+2):

| Task | Official | v4 | Gap | % of Total Gap |
|------|----------|-----|-----|----------------|
| cream_cheese | 0.80 | 0.50 | +3 | 33% |
| salad_dressing | 0.90 | 0.70 | +2 | 22% |
| butter | 0.80 | 0.60 | +2 | 22% |
| BBQ sauce | 0.50 | 0.40 | +1 | 11% |
| chocolate_pudding | 0.70 | 0.60 | +1 | 11% |
| alphabet_soup | 0.80 | 0.80 | 0 | — |
| ketchup | 1.00 | 1.00 | 0 | — |
| tomato_sauce | 0.90 | 0.90 | 0 | — |
| milk | 0.80 | 0.80 | 0 | — |
| orange_juice | 0.80 | 0.80 | 0 | — |

The gap is not uniform — 5 tasks match perfectly, 5 tasks differ. This pattern is consistent with a perceptual difference: tasks where visual precision matters more (cream_cheese, salad_dressing, butter — small objects, precise placement) show gaps, while tasks with robust visual features (ketchup, tomato_sauce) match perfectly.

## 2. Root Cause Classification

After comparing all implementation paths:

| Category | Description | Verdict |
|----------|-------------|---------|
| A | Prompt mismatch | **EXCLUDED** — Both use exact same format |
| B | **Image preprocessing mismatch** | **PRIMARY SUSPECT** |
| C | Model inference path mismatch | **SECONDARY** — EOS token handling differs |
| D | Action decoding mismatch | **EXCLUDED** — Identical v4 decode logic |
| E | Gripper postprocess mismatch | **EXCLUDED** — Identical normalize+invert |
| F | Env/state reset mismatch | **MINOR** — horizon differs but effective steps same |
| G | Success/done loop mismatch | **EXCLUDED** — Same done-based check |
| H | Runtime/GPU instability | **EXCLUDED** — GPUs stable, no Xid during runs |
| I | Unresolved | TBD |

### Category B: Image Preprocessing — PRIMARY

The v4 runner with `--libero_official_preprocess` uses a **TensorFlow-based pipeline** that differs from the official OpenVLA eval in three ways:

1. **JPEG round-trip (`tf.io.encode_jpeg` → `tf.io.decode_image`)**: The official OpenVLA eval script does NOT apply lossy JPEG compression to the observation. The v4 TF pipeline does. This introduces compression artifacts that degrade fine visual details — critical for distinguishing small objects like cream_cheese vs alphabet_soup.

2. **TF lanczos3 vs PIL LANCZOS**: While both are Lanczos-based, the implementations differ. TF's `lanczos3` uses a 6-tap filter while PIL's `LANCZOS` uses a 3-tap filter. This produces slightly different pixel values.

3. **TF bilinear center crop vs PIL LANCZOS center crop**: The v4 TF path uses `tf.image.crop_and_resize(method="bilinear")` for the center crop step, while the official PIL path uses `Image.LANCZOS`. Bilinear is softer than Lanczos and loses more high-frequency detail.

### Category C: EOS Token — SECONDARY

The v4 `decode_with_scores()` function does NOT explicitly add the EOS token (29871) before `model.generate()`. The official script does:

```python
if not torch.all(input_ids[:, -1] == 29871):
    inputs["input_ids"] = torch.cat(
        (input_ids, torch.tensor([[29871]]).long().to(input_ids.device)), dim=1)
```

If the processor's output already ends with 29871 (which varies by tokenizer behavior), this is a no-op. But if it doesn't, the model sees a different input context, potentially shifting token generation.

### Category F: Env Horizon — MINOR

The v4 runner passes `horizon=290` to OffScreenRenderEnv while the official script does not (default 1000). Effective model steps are identical (280), so this should not affect results. However, OffScreenRenderEnv's internal behavior with different horizon values has not been tested. Minor concern only.

## 3. Why do some tasks match perfectly?

ketchup (10/10 both), tomato_sauce (9/10 both), alphabet_soup (8/10 both), milk (8/10 both), orange_juice (8/10 both) — these tasks have **large, visually distinctive objects** (bright red ketchup, red tomato sauce can, white milk carton, orange juice carton). The slight image degradation from JPEG compression does not affect model perception enough to change actions.

cream_cheese (+3 gap), salad_dressing (+2), butter (+2) — these involve **smaller, less distinctive objects** (cream cheese tub, salad dressing bottle, butter stick). The JPEG compression artifacts are more likely to degrade fine visual features that distinguish these objects, leading to slightly worse actions that cause grasping or placement failures.

## 4. Can v4 be patched to match corrected official?

**Yes.** The fix is straightforward:

Replace the TF preprocessing path with the PIL path in `prepare_openvla_image()`:

```python
def prepare_openvla_image(image_np, *, libero_official_preprocess=False, center_crop=False, resize_size=224):
    # Use PIL path (matching corrected official script)
    arr = np.asarray(image_np)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    arr = arr[::-1, ::-1]  # rotate 180
    image = Image.fromarray(arr).convert("RGB")
    image = image.resize((resize_size, resize_size), Image.LANCZOS)
    if center_crop:
        scale = 0.9 ** 0.5
        w, h = image.size
        cw, ch = max(1, int(w * scale)), max(1, int(h * scale))
        left, top = (w - cw) // 2, (h - ch) // 2
        image = image.crop((left, top, left + cw, top + ch))
        image = image.resize((resize_size, resize_size), Image.LANCZOS)
    return image
```

Also add EOS token handling to `decode_with_scores()`.

## 5. Recommended Next Action

**Option A (Recommended): Patch v4 image preprocessing to PIL path + add EOS token**

- Minimal change: swap TF preprocessing for PIL preprocessing
- Add EOS token check before model.generate()
- Run 12-episode validation (cream_cheese s0-2, salad_dressing s0-2, butter s0-2, ketchup s0-2)
- If validation shows alignment, re-run full Object 100

**Option B: Migrate attack runner to official-compatible eval path**

- Replace v4 runner's preprocess/inference/postprocess with corrected official script's functions
- Keep v4 runner's attack, trigger, budget, logging infrastructure
- More work but more maintainable long-term

## 6. Final Diagnosis

The +9 Object gap between corrected official (80/100) and v4 runner (71/100) is primarily caused by **different image preprocessing pipelines** (Category B). The v4 TF pipeline's JPEG round-trip and bilinear center crop degrade image quality relative to the official PIL Lanczos pipeline, affecting tasks that require fine visual discrimination. A secondary factor may be missing EOS token handling (Category C). Gripper postprocessing, action decoding, prompt format, and episode flow are identical and can be ruled out.

The official script's PIL preprocessing path should be the reference for all future clean evaluations.
