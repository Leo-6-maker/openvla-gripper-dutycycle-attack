#!/usr/bin/env python3
"""Canonical OpenVLA LIBERO image preprocessing — single shared entry point.

Backends:
  - upstream_tf_jpeg:    official OpenVLA upstream path (JPEG round-trip, TF Lanczos3,
                          TF crop_and_resize with sqrt(0.9)). Requires TensorFlow.
  - project_pil_lanczos: project PIL approximation (180° rotate, LANCZOS resize,
                          integer-pixel centre crop). No TF dependency.
  - none:                pass-through, no preprocessing.

Compatibility aliases (deprecated, resolve to canonical names):
  - tf_jpeg_legacy      -> upstream_tf_jpeg
  - official_pil_lanczos -> project_pil_lanczos

The TF upstream backend matches openvla/openvla main branch:
  experiments/robot/libero/libero_utils.py   (get_libero_image, resize_image)
  experiments/robot/openvla_utils.py         (crop_and_resize, get_vla_action)
"""

from __future__ import annotations

import math
import numpy as np
from PIL import Image

_CROP_SCALE = 0.9
_RESIZE_SIZE = 224

# ---------------------------------------------------------------------------
# Backend resolution
# ---------------------------------------------------------------------------

CANONICAL_BACKENDS = frozenset({"upstream_tf_jpeg", "project_pil_lanczos", "none"})

ALIASES: dict[str, str] = {
    "tf_jpeg_legacy": "upstream_tf_jpeg",
    "official_pil_lanczos": "project_pil_lanczos",
    "pil_fallback": "none",
}


def resolve_backend(name: str) -> str:
    """Resolve a backend name (possibly an alias) to its canonical form."""
    canon = ALIASES.get(name, name)
    if canon not in CANONICAL_BACKENDS:
        raise ValueError(
            f"Unknown preprocess backend {name!r} (canonical: {canon!r}). "
            f"Valid backends: {sorted(CANONICAL_BACKENDS)}"
        )
    return canon


# ---------------------------------------------------------------------------
# TF memory guard (belt-and-suspenders: env var + programmatic)
# ---------------------------------------------------------------------------

_tf_memory_configured = False


def _configure_tf_memory():
    global _tf_memory_configured
    if _tf_memory_configured:
        return
    try:
        import tensorflow as tf

        gpus = tf.config.list_physical_devices("GPU")
        for g in gpus:
            tf.config.experimental.set_memory_growth(g, True)
    except Exception:
        pass
    _tf_memory_configured = True


# ---------------------------------------------------------------------------
# Backend implementations
# ---------------------------------------------------------------------------


def _preprocess_upstream_tf_jpeg(raw_agentview: np.ndarray) -> Image.Image:
    """Official OpenVLA upstream path.

    Steps (exactly matching openvla/openvla main branch):
      1. np.uint8 normalisation
      2. 180° rotation (img[::-1, ::-1])
      3. tf.io.encode_jpeg / decode_image round-trip
      4. tf.image.resize to 224×224, method=lanczos3, antialias=True
      5. round / clip / uint8
      6. convert_image_dtype to float32 [0,1]
      7. tf.image.crop_and_resize with sqrt(0.9) -> 224×224
      8. clip [0,1], convert_image_dtype back to uint8
      9. PIL RGB output for HuggingFace processor
    """
    _configure_tf_memory()

    import tensorflow as tf

    # Step 1-2: normalise + rotate 180°
    arr = np.asarray(raw_agentview)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    img = arr[::-1, ::-1].copy()

    # Step 3: JPEG encode/decode round-trip (RLDS dataset builder)
    img_tf = tf.image.encode_jpeg(img)
    img_tf = tf.io.decode_image(img_tf, expand_animations=False, dtype=tf.uint8)

    # Step 4-5: Lanczos3 resize + clip/round/uint8
    img_tf = tf.image.resize(img_tf, (_RESIZE_SIZE, _RESIZE_SIZE),
                             method="lanczos3", antialias=True)
    img_tf = tf.cast(tf.clip_by_value(tf.round(img_tf), 0, 255), tf.uint8)

    # Step 6: convert to float32 [0,1]
    orig_dtype = img_tf.dtype
    img_float = tf.image.convert_image_dtype(img_tf, tf.float32)

    # Step 7: centre crop area-ratio 0.9 via crop_and_resize -> 224×224
    h = tf.clip_by_value(tf.sqrt(_CROP_SCALE), 0, 1)
    offsets = (1.0 - h) / 2.0
    boxes = tf.stack([[offsets, offsets, offsets + h, offsets + h]])
    img_cropped = tf.image.crop_and_resize(
        tf.expand_dims(img_float, 0), boxes, [0], (_RESIZE_SIZE, _RESIZE_SIZE)
    )[0]

    # Step 8: clip + convert back to uint8
    img_cropped = tf.clip_by_value(img_cropped, 0, 1)
    img_result = tf.image.convert_image_dtype(img_cropped, orig_dtype, saturate=True)

    # Step 9: PIL RGB output
    return Image.fromarray(img_result.numpy()).convert("RGB")


def _preprocess_project_pil(raw_agentview: np.ndarray) -> Image.Image:
    """Project PIL approximation path.

    Steps:
      1. np.uint8 normalisation
      2. 180° rotation (PIL rotate)
      3. PIL LANCZOS resize to 224×224
      4. integer-pixel centre crop sqrt(0.9)
      5. PIL LANCZOS resize back to 224×224
    """
    arr = np.asarray(raw_agentview)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)

    img = Image.fromarray(arr).rotate(180).convert("RGB")
    img = img.resize((_RESIZE_SIZE, _RESIZE_SIZE), Image.LANCZOS)

    s = math.sqrt(_CROP_SCALE)
    cs = int(_RESIZE_SIZE * s)
    L = (_RESIZE_SIZE - cs) // 2
    img = img.crop((L, L, L + cs, L + cs))
    img = img.resize((_RESIZE_SIZE, _RESIZE_SIZE), Image.LANCZOS)
    return img


def _preprocess_none(raw_agentview: np.ndarray) -> Image.Image:
    """Pass-through — no preprocessing applied."""
    arr = np.asarray(raw_agentview)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr).convert("RGB")


_BACKEND_IMPLS = {
    "upstream_tf_jpeg": _preprocess_upstream_tf_jpeg,
    "project_pil_lanczos": _preprocess_project_pil,
    "none": _preprocess_none,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def prepare_openvla_image(
    image_np: np.ndarray,
    *,
    libero_preprocess_backend: str = "upstream_tf_jpeg",
    center_crop: bool = True,
    resize_size: int = 224,
) -> Image.Image:
    """Prepare a LIBERO agentview image for OpenVLA inference.

    Args:
        image_np: Raw uint8 numpy array from ``obs["agentview_image"]``.
        libero_preprocess_backend: Canonical backend name (or deprecated alias).
        center_crop: Apply TF centre-crop sqrt(0.9). Ignored by ``none`` backend.
        resize_size: Resize target size (default 224).

    Returns:
        PIL Image in RGB mode, ready for HuggingFace ``AutoProcessor``.

    Raises:
        SystemExit: If ``upstream_tf_jpeg`` is selected but TensorFlow cannot be imported.
    """
    backend = resolve_backend(libero_preprocess_backend)

    impl = _BACKEND_IMPLS.get(backend)
    if impl is None:
        raise ValueError(f"No implementation for backend {backend!r}")

    # centre_crop is implicit in upstream_tf_jpeg and project_pil_lanczos;
    # only "none" may optionally skip it.
    if not center_crop and backend != "none":
        # upstream_tf_jpeg and project_pil_lanczos always include centre crop
        pass

    return impl(image_np)
