#!/usr/bin/env python3
"""Official OpenVLA upstream preprocessing: 180° rotation → JPEG round-trip →
TF Lanczos3 resize → TF center crop sqrt(0.9) → PIL Image ready for HF processor.

Matches openvla/openvla main branch (liber_utils.py + openvla_utils.py) exactly.
"""
import numpy as np
from PIL import Image
import tensorflow as tf

# Prevent TF from eagerly allocating all GPU memory — must be called before any TF ops.
# The env var TF_FORCE_GPU_ALLOW_GROWTH=true provides a belt-and-suspenders approach.
_gpus = tf.config.list_physical_devices("GPU")
for _g in _gpus:
    tf.config.experimental.set_memory_growth(_g, True)

_CROP_SCALE = 0.9


def preprocess_upstream_tf_jpeg(raw_agentview):
    """Apply official OpenVLA preprocessing to a raw agentview image.

    Args:
        raw_agentview: numpy uint8 array (256, 256, 3) from LIBERO observation.

    Returns:
        PIL Image (224, 224) in RGB mode, ready for HuggingFace processor.
    """
    # Step 1: 180° rotation (official uses img[::-1, ::-1])
    img = raw_agentview[::-1, ::-1].copy()

    # Step 2: JPEG encode/decode round-trip (as in RLDS dataset builder)
    img_tf = tf.image.encode_jpeg(img)
    img_tf = tf.io.decode_image(img_tf, expand_animations=False, dtype=tf.uint8)

    # Step 3: Lanczos3 resize with antialiasing to 224x224
    img_tf = tf.image.resize(img_tf, (224, 224), method="lanczos3", antialias=True)
    img_tf = tf.cast(tf.clip_by_value(tf.round(img_tf), 0, 255), tf.uint8)

    # Step 4: Center crop area-ratio 0.9, resize back to 224x224 via crop_and_resize
    orig_dtype = img_tf.dtype  # tf.uint8
    img_float = tf.image.convert_image_dtype(img_tf, tf.float32)

    h = tf.clip_by_value(tf.sqrt(_CROP_SCALE), 0, 1)
    offsets = (1.0 - h) / 2.0
    boxes = tf.stack([[offsets, offsets, offsets + h, offsets + h]])

    img_cropped = tf.image.crop_and_resize(
        tf.expand_dims(img_float, 0), boxes, [0], (224, 224)
    )[0]

    # Convert back to uint8 and then to PIL
    img_cropped = tf.clip_by_value(img_cropped, 0, 1)
    img_result = tf.image.convert_image_dtype(img_cropped, orig_dtype, saturate=True)

    return Image.fromarray(img_result.numpy()).convert("RGB")


def preprocess_project_pil(raw_agentview):
    """Current project PIL preprocessing (for comparison during audit).

    Legacy path: 180° rotate → RGB → LANCZOS 224 → integer center crop → LANCZOS 224.
    """
    import math
    img = Image.fromarray(raw_agentview).rotate(180).convert("RGB")
    img = img.resize((224, 224), Image.LANCZOS)
    s = math.sqrt(0.9)
    cs = int(224 * s)
    L = (224 - cs) // 2
    img = img.crop((L, L, L + cs, L + cs)).resize((224, 224), Image.LANCZOS)
    return img
