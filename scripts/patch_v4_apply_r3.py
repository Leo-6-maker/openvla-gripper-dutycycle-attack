#!/usr/bin/env python3
"""Apply R3 patches to the v4 runner: PIL preprocessing + backend flag + logging."""
import sys
from pathlib import Path

TARGET = Path("/data/liuyu/repos/openvla-gripper-dutycycle-attack-clean-main-20260524/scripts/v4_run_eval_openvla.py")
BACKUP = Path(str(TARGET) + ".r3_backup")

def patch():
    original = TARGET.read_text(encoding="utf-8")

    # Save backup
    if not BACKUP.exists():
        BACKUP.write_text(original, encoding="utf-8")
        print(f"Backup: {BACKUP}")

    patched = original

    # --- Patch 1: Add _official_pil_libero_image function ---
    old_func = '''def _official_tf_libero_image(image_np, *, center_crop: bool = False, resize_size: int = 224) -> Image.Image:'''
    new_pil_func = '''def _official_pil_libero_image(image_np, *, center_crop: bool = False, resize_size: int = 224) -> Image.Image:
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


def _official_tf_libero_image(image_np, *, center_crop: bool = False, resize_size: int = 224) -> Image.Image:'''

    if new_pil_func not in patched:
        patched = patched.replace(old_func, new_pil_func)
        print("Patch 1: Added _official_pil_libero_image()")
    else:
        print("Patch 1: Already applied")

    # --- Patch 2: Modify prepare_openvla_image to accept backend ---
    old_prepare = '''def prepare_openvla_image(image_np, *, libero_official_preprocess: bool = False, center_crop: bool = False, resize_size: int = 224) -> Image.Image:
    """Prepare a LIBERO observation image for OpenVLA inference.

    Matches the official LIBERO/OpenVLA eval path when
    ``libero_official_preprocess`` is enabled: rotate the agentview image by
    180 degrees, resize to the OpenVLA input size, then optionally center-crop
    with crop_scale=0.9 and resize back.
    """
    if libero_official_preprocess:
        return _official_tf_libero_image(image_np, center_crop=center_crop, resize_size=int(resize_size))
    arr = np.asarray(image_np)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    image = Image.fromarray(arr).convert("RGB")
    if center_crop:
        image = _pil_center_crop_resize(image, crop_scale=0.9, size=int(resize_size))
    return image'''

    new_prepare = '''def prepare_openvla_image(image_np, *, libero_official_preprocess: bool = False, center_crop: bool = False, resize_size: int = 224, libero_preprocess_backend: str = "official_pil_lanczos") -> Image.Image:
    """Prepare a LIBERO observation image for OpenVLA inference.

    Backends:
      - official_pil_lanczos (default): rotate 180, PIL Lanczos resize, Lanczos center crop.
        Matches the corrected official OpenVLA eval script. No TensorFlow dependency.
      - tf_jpeg_legacy: rotate 180, JPEG round-trip, TF Lanczos3 resize, TF bilinear crop.
        Legacy backend requiring TensorFlow. Known to produce +9 lower Object SR.

    When ``libero_official_preprocess`` is True (legacy flag), uses tf_jpeg_legacy.
    The ``libero_preprocess_backend`` flag takes precedence over the legacy boolean.
    """
    backend = str(libero_preprocess_backend or "official_pil_lanczos")
    # Legacy boolean flag overrides to tf_jpeg_legacy for backward compat
    if libero_official_preprocess and backend == "official_pil_lanczos":
        backend = "tf_jpeg_legacy"

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
        return image'''

    if new_prepare not in patched:
        patched = patched.replace(old_prepare, new_prepare)
        print("Patch 2: Updated prepare_openvla_image() with backend parameter")
    else:
        print("Patch 2: Already applied")

    # --- Patch 3: Update decode_with_scores to pass backend ---
    old_decode = '''def decode_with_scores(model, processor, device, image_np, instruction, unnorm_key, k, *, libero_official_preprocess=False, center_crop=False, resize_size=224, drop_attention_mask=True):
    image=prepare_openvla_image(image_np, libero_official_preprocess=libero_official_preprocess, center_crop=center_crop, resize_size=resize_size)'''

    new_decode = '''def decode_with_scores(model, processor, device, image_np, instruction, unnorm_key, k, *, libero_official_preprocess=False, center_crop=False, resize_size=224, drop_attention_mask=True, libero_preprocess_backend="official_pil_lanczos"):
    image=prepare_openvla_image(image_np, libero_official_preprocess=libero_official_preprocess, center_crop=center_crop, resize_size=resize_size, libero_preprocess_backend=libero_preprocess_backend)'''

    if new_decode not in patched:
        patched = patched.replace(old_decode, new_decode)
        print("Patch 3: Updated decode_with_scores() signature")
    else:
        print("Patch 3: Already applied")

    # --- Patch 4: Add --libero_preprocess_backend argument ---
    old_arg = '''ap.add_argument("--libero_official_preprocess",action="store_true",help="Use official OpenVLA-LIBERO image preprocessing: 256 render + 180-degree rotate + resize before processor")'''

    new_arg = '''ap.add_argument("--libero_official_preprocess",action="store_true",help="(Legacy) Use TF JPEG preprocessing. Prefer --libero_preprocess_backend official_pil_lanczos for official-aligned runs.")
    ap.add_argument("--libero_preprocess_backend",choices=["official_pil_lanczos","tf_jpeg_legacy","none"],default="official_pil_lanczos",help="Image preprocessing backend. official_pil_lanczos matches corrected official eval (no JPG, PIL Lanczos). tf_jpeg_legacy is old TF path with JPEG round-trip.")'''

    if new_arg not in patched:
        patched = patched.replace(old_arg, new_arg)
        print("Patch 4: Added --libero_preprocess_backend argument")
    else:
        print("Patch 4: Already applied")

    # --- Patch 5: Add preprocess metadata to run_manifest ---
    # Find build_run_manifest and add fields
    old_manifest_line = '''"attack_objective":effective_attack_objective(args,cfg),"force_open_raw_gripper":float(getattr(args,"force_open_raw_gripper",0.0)),"grasp_gate_dist_threshold":float(getattr(args,"grasp_gate_dist",0.10)),**matched_provenance(args)}'''

    new_manifest_fields = '''"attack_objective":effective_attack_objective(args,cfg),"force_open_raw_gripper":float(getattr(args,"force_open_raw_gripper",0.0)),"grasp_gate_dist_threshold":float(getattr(args,"grasp_gate_dist",0.10)),
        "preprocess_backend": str(getattr(args, "libero_preprocess_backend", "official_pil_lanczos")),
        "resize_interpolation": "LANCZOS" if str(getattr(args, "libero_preprocess_backend", "official_pil_lanczos")) == "official_pil_lanczos" else "lanczos3" if str(getattr(args, "libero_preprocess_backend", "official_pil_lanczos")) == "tf_jpeg_legacy" else "LANCZOS",
        "uses_jpeg_roundtrip": str(getattr(args, "libero_preprocess_backend", "official_pil_lanczos")) == "tf_jpeg_legacy",
        "center_crop": bool(args.center_crop),
        "rotate_180": True,
        "prompt_format": "In: What action should the robot take to {task}?\\nOut:",
        "eos_token_handling": "add_if_missing_29871",
        "model_inference_path": "model.generate()",
        "python_executable": sys.executable,
        "transformers_version": str(transformers.__version__) if hasattr(transformers, "__version__") else "unknown",
        "tokenizers_version": str(tokenizers.__version__) if hasattr(tokenizers, "__version__") else "unknown",
        "torch_version": str(torch.__version__) if hasattr(torch, "__version__") else "unknown",
        "mujoco_version": "",
        "robosuite_version": "",
        **matched_provenance(args)}'''

    try:
        import transformers as _tf
    except Exception:
        import transformers as _tf
    _tf_version = getattr(_tf, "__version__", "unknown")

    if new_manifest_fields not in patched:
        patched = patched.replace(old_manifest_line, new_manifest_fields)
        print(f"Patch 5: Added preprocess metadata to run_manifest (transformers={_tf_version})")
    else:
        print("Patch 5: Already applied")

    # Verify patches applied
    checks = [
        ("_official_pil_libero_image", "Patch 1 verify"),
        ("libero_preprocess_backend", "Patch 2/3/4 verify"),
        ('"preprocess_backend"', "Patch 5 verify"),
    ]
    for token, desc in checks:
        if token in patched:
            print(f"  {desc}: OK")
        else:
            print(f"  {desc}: MISSING!")

    TARGET.write_text(patched, encoding="utf-8")
    print(f"\nPatched: {TARGET}")
    return 0

if __name__ == "__main__":
    raise SystemExit(patch())
