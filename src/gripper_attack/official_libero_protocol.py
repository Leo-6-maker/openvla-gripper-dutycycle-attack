"""Small, exact helpers for the upstream OpenVLA LIBERO protocol.

The execution path intentionally calls ``model.predict_action``.  The score
path uses the same prepared inputs and the same token de-tokenization only so
that logits can be inspected and differentiated for later attacks.
"""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import torch
from PIL import Image


OFFICIAL_HORIZONS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
}
NUM_STEPS_WAIT = 10
NUM_TRIALS_PER_TASK = 50
ACTION_DIM = 7
EOS_TOKEN_ID = 29871


def official_prompt(task_label: str, base_vla_name: str = "") -> str:
    if "openvla-v01" in str(base_vla_name).lower():
        return (
            "A chat between a curious user and an artificial intelligence assistant. "
            "The assistant gives helpful, detailed, and polite answers to the user's questions. "
            f"USER: What action should the robot take to {task_label.lower()}? ASSISTANT:"
        )
    return f"In: What action should the robot take to {task_label.lower()}?\nOut:"


def official_center_crop(image: Image.Image) -> Image.Image:
    """Copy OpenVLA's TensorFlow crop-and-resize path exactly."""
    import tensorflow as tf

    image = image.convert("RGB")
    tensor = tf.convert_to_tensor(np.array(image))
    orig_dtype = tensor.dtype
    tensor = tf.image.convert_image_dtype(tensor, tf.float32)
    # The official implementation uses crop_scale=0.9.
    scale = float(0.9 ** 0.5)
    offset = (1.0 - scale) / 2.0
    tensor = tf.image.crop_and_resize(
        tensor[None, ...],
        tf.constant([[offset, offset, offset + scale, offset + scale]], dtype=tf.float32),
        tf.constant([0], dtype=tf.int32),
        (224, 224),
    )
    tensor = tf.clip_by_value(tensor, 0.0, 1.0)
    tensor = tf.image.convert_image_dtype(tensor, orig_dtype, saturate=True)
    return Image.fromarray(tensor.numpy()[0]).convert("RGB")


def prepare_official_inputs(
    processor: Any,
    image_np: np.ndarray,
    task_label: str,
    device: torch.device | str,
    *,
    center_crop: bool = True,
    base_vla_name: str = "",
) -> tuple[dict[str, Any], str, Image.Image]:
    """Prepare the exact inputs used by upstream ``get_vla_action``."""
    image = Image.fromarray(np.asarray(image_np)).convert("RGB")
    if center_crop:
        image = official_center_crop(image)
    prompt = official_prompt(task_label, base_vla_name)
    inputs = processor(prompt, image)
    # BatchFeature.to is the upstream call. It moves integer fields and casts
    # floating fields to BF16 without dropping attention_mask or adding EOS.
    inputs = inputs.to(device, dtype=torch.bfloat16)
    return dict(inputs), prompt, image


def _with_official_eos(inputs: dict[str, Any]) -> dict[str, Any]:
    out = dict(inputs)
    input_ids = out["input_ids"]
    if not torch.all(input_ids[:, -1] == EOS_TOKEN_ID):
        out["input_ids"] = torch.cat(
            (
                input_ids,
                torch.tensor([[EOS_TOKEN_ID]], dtype=torch.long, device=input_ids.device),
            ),
            dim=1,
        )
    return out


def decode_official_generated_action(model: Any, generated_ids: torch.Tensor, unnorm_key: str) -> np.ndarray:
    action_dim = int(model.get_action_dim(unnorm_key))
    token_ids = generated_ids[0, -action_dim:].detach().cpu().numpy()
    vocab_size = int(
        getattr(
            model,
            "vocab_size",
            model.config.text_config.vocab_size - model.config.pad_to_multiple_of,
        )
    )
    bin_centers = np.asarray(model.bin_centers)
    discretized = np.clip(vocab_size - token_ids - 1, 0, bin_centers.shape[0] - 1)
    normalized = bin_centers[discretized]
    stats = model.get_action_stats(unnorm_key)
    mask = stats.get("mask", np.ones_like(stats["q01"], dtype=bool))
    high, low = np.asarray(stats["q99"]), np.asarray(stats["q01"])
    return np.where(mask, 0.5 * (normalized + 1.0) * (high - low) + low, normalized).astype(np.float32)


def official_predict_action(
    model: Any,
    processor: Any,
    image_np: np.ndarray,
    task_label: str,
    unnorm_key: str,
    device: torch.device | str,
    *,
    center_crop: bool = True,
    base_vla_name: str = "",
) -> tuple[np.ndarray, dict[str, Any]]:
    inputs, prompt, image = prepare_official_inputs(
        processor,
        image_np,
        task_label,
        device,
        center_crop=center_crop,
        base_vla_name=base_vla_name,
    )
    action = model.predict_action(**inputs, unnorm_key=unnorm_key, do_sample=False)
    return np.asarray(action, dtype=np.float32), {
        "inputs": inputs,
        "prompt": prompt,
        "processed_image": image,
    }


def score_official_action(
    model: Any,
    processor: Any,
    image_np: np.ndarray,
    task_label: str,
    unnorm_key: str,
    device: torch.device | str,
    *,
    center_crop: bool = True,
    base_vla_name: str = "",
) -> tuple[np.ndarray, Any, dict[str, Any]]:
    """Return official-compatible action plus generation scores.

    This is not the execution path. It exists to expose the same generated
    action tokens for logit/gradient instrumentation, then parity-checks them
    against ``model.predict_action`` before formal collection.
    """
    inputs, prompt, image = prepare_official_inputs(
        processor,
        image_np,
        task_label,
        device,
        center_crop=center_crop,
        base_vla_name=base_vla_name,
    )
    generation_inputs = _with_official_eos(inputs)
    action_dim = int(model.get_action_dim(unnorm_key))
    generation = model.generate(
        **generation_inputs,
        max_new_tokens=action_dim,
        do_sample=False,
        return_dict_in_generate=True,
        output_scores=True,
    )
    action = decode_official_generated_action(model, generation.sequences, unnorm_key)
    return action, generation, {
        "inputs": inputs,
        "generation_inputs": generation_inputs,
        "prompt": prompt,
        "processed_image": image,
    }


def postprocess_official_action(action: np.ndarray) -> np.ndarray:
    """Official LIBERO action postprocess: normalize then invert gripper."""
    out = np.asarray(action, dtype=np.float32).copy()
    out[-1] = 2.0 * out[-1] - 1.0
    out[-1] = np.sign(out[-1])
    out[-1] = -out[-1]
    return out


def tensor_sha256(value: Any) -> str:
    tensor = value.detach().cpu().contiguous()
    # NumPy cannot directly expose BF16 tensors.  Hash the raw storage bytes
    # through a byte view and include dtype/shape so the audit is unambiguous.
    raw = tensor.view(torch.uint8).numpy().tobytes()
    header = f"{tensor.dtype}|{tuple(tensor.shape)}|".encode("utf-8")
    return hashlib.sha256(header + raw).hexdigest()


def generated_action_tokens(model: Any, generation: Any, unnorm_key: str) -> list[int]:
    dim = int(model.get_action_dim(unnorm_key))
    sequences = generation.sequences if hasattr(generation, "sequences") else generation
    return [int(x) for x in sequences[0, -dim:].detach().cpu().tolist()]
