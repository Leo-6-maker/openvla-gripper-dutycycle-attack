# Upstream Preprocessing Alignment Protocol

## Gate

`UPSTREAM_PREPROCESS_ALIGNMENT_AUDIT`

## Status

**PASSED** — 2026-06-21

## Summary

The project's default PIL preprocessing (`official_pil_lanczos`) does NOT match the current
OpenVLA upstream main branch. The upstream uses JPEG round-trip, TensorFlow Lanczos3, and
`tf.image.crop_and_resize`. The project PIL path produces completely different tokens and actions
on ALL tested frames.

## Audit Results

| Metric | Value |
|---|---|
| Static frames compared | 10 |
| Token match (PIL vs upstream) | 0/10 |
| Action match | 0/10 |
| Gripper direction match | 5/10 |
| Max action diff | 0.996 |
| Upstream canary (4-ep) | 3/4 success |

## Canonical Backends

| Backend | Description | Matches upstream |
|---|---|---|
| `upstream_tf_jpeg` | Official path: JPEG, TF Lanczos3, TF crop_and_resize | Yes |
| `project_pil_lanczos` | Project PIL approximation | No |
| `none` | Pass-through | No |

## Clean30 Profiles

| Profile | dtype | attn | backend | Status |
|---|---|---|---|---|
| FP32-Eager | float32 | eager | upstream_tf_jpeg | COMPLETE |
| BF16-Flash2 | bfloat16 | flash_attention_2 | upstream_tf_jpeg | RUNNING |
| BF16-Eager | bfloat16 | eager | upstream_tf_jpeg | DEFERRED |

## Contract

- seed=42, max_steps=220, wait_steps=10
- unnorm_key=libero_spatial, resize=224
- center_crop=True, crop_scale=0.9
- Rotation: img[::-1, ::-1] (180 degrees)
- JPEG round-trip: tf.image.encode_jpeg + tf.io.decode_image
- Resize: tf.image.resize(method="lanczos3", antialias=True)
- Crop: tf.image.crop_and_resize with sqrt(0.9) -> 224x224
- Model: spatial_c8f03f4_20260620

## Running Code SHA

- runner: ef0b2c606b1beafc75410e149f43e85bbf9b42659829c79da35fdffe77567f20
- preprocess: 6de76cb53e9b0acdb5b8d877c988fa69aca10a30519270556c6569f3b5f9fdcc
- plan: 9549ee9a53b23bdc06329b2b8b12a4ed73278bdc7f97a732f2ef00653655219e
- audit: 307b83e343d1e6204bf6b38e84034f5453610923643a2d7d634a3a9462dc07c1

## Next Gate

`UPSTREAM_DETECTOR_TRANSFER_AND_THRESHOLD_AUDIT`
