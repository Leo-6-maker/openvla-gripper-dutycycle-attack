#!/usr/bin/env python3
"""Shared OpenVLA LIBERO image preprocessing.

Imported by v4_run_eval_openvla.py and attack_adapter.py.
Three backends: official_pil_lanczos (default), tf_jpeg_legacy, none/pil_fallback.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

def _pil_center_crop_resize(image: Image.Image, crop_scale: float = 0.9, size: int = 224) -> Image.Image:
    """Fallback center crop used only when official preprocessing is disabled."""
    if crop_scale is None or float(crop_scale) >= 0.999:
        return image.resize((size, size), Image.Resampling.LANCZOS)
    w, h = image.size
    scale = float(crop_scale) ** 0.5
    cw, ch = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    left, top = (w - cw) // 2, (h - ch) // 2
    image = image.crop((left, top, left + cw, top + ch))
    return image.resize((size, size), Image.Resampling.LANCZOS)


def _official_pil_libero_image(image_np, *, center_crop: bool = False, resize_size: int = 224) -> Image.Image:
    """Official OpenVLA LIBERO image path (PIL): rotate 180, Lanczos resize, Lanczos center crop.

    Matches the corrected official OpenVLA eval script exactly:
    - Rotate agentview 180 degrees
    - PIL LANCZOS resize to resize_size
    - Optional center crop (0.9 scale) → PIL LANCZOS resize back
    - NO JPEG round-trip, NO TensorFlow dependency
    """
    arr = np.asarray(image_np)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    arr = arr[::-1, ::-1]  # rotate 180 degrees
    image = Image.fromarray(arr).convert("RGB")
    image = image.resize((int(resize_size), int(resize_size)), Image.LANCZOS)
    if center_crop:
        crop_scale = 0.9 ** 0.5
        w, h = image.size
        cw, ch = max(1, int(w * crop_scale)), max(1, int(h * crop_scale))
        left, top = (w - cw) // 2, (h - ch) // 2
        image = image.crop((left, top, left + cw, top + ch))
        image = image.resize((int(resize_size), int(resize_size)), Image.LANCZOS)
    return image


def _official_tf_libero_image(image_np, *, center_crop: bool = False, resize_size: int = 224) -> Image.Image:
    """Official OpenVLA LIBERO image path: rotate, JPEG round-trip, TF Lanczos3.

    This intentionally requires TensorFlow instead of silently using the older
    PIL approximation when ``--libero_official_preprocess`` is requested.
    """
    try:
        import tensorflow as tf
    except Exception as exc:
        raise SystemExit(
            "--libero_official_preprocess now requires TensorFlow for official "
            f"LIBERO preprocessing; import failed: {exc}"
        )
    arr = np.asarray(image_np)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    arr = arr[::-1, ::-1]
    tensor = tf.convert_to_tensor(arr)
    tensor = tf.io.decode_image(tf.io.encode_jpeg(tensor), expand_animations=False, dtype=tf.uint8)
    tensor = tf.image.resize(tensor, [int(resize_size), int(resize_size)], method="lanczos3", antialias=True)
    if center_crop:
        crop_scale = 0.9 ** 0.5
        box = [[
            (1.0 - crop_scale) / 2.0,
            (1.0 - crop_scale) / 2.0,
            (1.0 + crop_scale) / 2.0,
            (1.0 + crop_scale) / 2.0,
        ]]
        tensor = tf.image.crop_and_resize(
            tf.expand_dims(tensor, axis=0),
            boxes=tf.convert_to_tensor(box, dtype=tf.float32),
            box_indices=tf.convert_to_tensor([0], dtype=tf.int32),
            crop_size=[int(resize_size), int(resize_size)],
            method="bilinear",
        )[0]
    tensor = tf.cast(tf.clip_by_value(tf.round(tensor), 0, 255), tf.uint8)
    return Image.fromarray(tensor.numpy()).convert("RGB")


def prepare_openvla_image(image_np, *, libero_official_preprocess: bool = False, center_crop: bool = False, resize_size: int = 224, libero_preprocess_backend: str = "official_pil_lanczos") -> Image.Image:
    """Prepare a LIBERO observation image for OpenVLA inference.

    Backends:
      - official_pil_lanczos (default): rotate 180, PIL Lanczos resize, Lanczos center crop.
        Matches the corrected official OpenVLA eval script. No TensorFlow dependency.
      - tf_jpeg_legacy: rotate 180, JPEG round-trip, TF Lanczos3 resize, TF bilinear crop.
        Legacy backend requiring TensorFlow. Known to produce +9 lower Object SR.

    The ``--libero_official_preprocess`` flag is a deprecated compatibility alias;
    it no longer switches the backend. Use --libero_preprocess_backend to select.
    """
    backend = str(libero_preprocess_backend or "official_pil_lanczos")
    # --libero_official_preprocess is a deprecated compatibility alias.
    # It does NOT switch to tf_jpeg_legacy. Use --libero_preprocess_backend explicitly.

    if backend == "official_pil_lanczos":
        return _official_pil_libero_image(image_np, center_crop=center_crop, resize_size=int(resize_size))
    elif backend == "tf_jpeg_legacy":
        return _official_tf_libero_image(image_np, center_crop=center_crop, resize_size=int(resize_size))
    else:
        arr = np.asarray(image_np)
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        image = Image.fromarray(arr).convert("RGB")
        if center_crop:
            image = _pil_center_crop_resize(image, crop_scale=0.9, size=int(resize_size))
        return image
