#!/usr/bin/env python3
"""Audit: compare project PIL preprocessing vs upstream TF/JPEG preprocessing.
Loads frozen raw frames, runs both paths through model, reports per-dimension diffs.

Gate: UPSTREAM_PREPROCESS_ALIGNMENT — static 10-frame audit.
"""
import os, json, hashlib, sys, argparse, time, math
import numpy as np
from PIL import Image

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
assert "MUJOCO_GL" in os.environ
assert "CUDA_VISIBLE_DEVICES" in os.environ

import torch
import tensorflow as tf
from transformers import AutoProcessor, AutoModelForVision2Seq

# Import the two preprocessing backends
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocess_upstream import preprocess_upstream_tf_jpeg, preprocess_project_pil, _CROP_SCALE


MODEL = "/mnt/sdc/dty_user/openvla_attack/models/libero-spatial/spatial_c8f03f4_20260620"
FRAMES_DIR = "/mnt/sdc/dty_user/openvla_attack/migration_audit/parity/frozen_frames_v2_crop"
FRAMES = [
    "libero_spatial_t00_s10_step000", "libero_spatial_t00_s10_step026",
    "libero_spatial_t00_s15_step000", "libero_spatial_t00_s15_step021",
    "libero_spatial_t01_s10_step000", "libero_spatial_t01_s10_step041",
    "libero_spatial_t02_s10_step000", "libero_spatial_t02_s10_step034",
    "libero_spatial_t02_s15_step000", "libero_spatial_t02_s15_step036",
]


def sha256_hex(data):
    if isinstance(data, torch.Tensor):
        data = data.float().cpu().numpy().tobytes()
    elif isinstance(data, np.ndarray):
        data = data.tobytes()
    elif isinstance(data, Image.Image):
        data = np.array(data).tobytes()
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    parser.add_argument("--attn", default="eager", choices=["eager", "flash_attention_2"])
    args = parser.parse_args()

    DTYPE = torch.float32 if args.dtype == "float32" else torch.bfloat16
    DEV = torch.device("cuda:0")

    print("=" * 70)
    print("UPSTREAM PREPROCESS ALIGNMENT AUDIT")
    print("dtype=%s  attn=%s  crop_scale=%.4f" % (args.dtype, args.attn, _CROP_SCALE))
    print("=" * 70)

    # Load model
    model = AutoModelForVision2Seq.from_pretrained(
        MODEL, torch_dtype=DTYPE, attn_implementation=args.attn,
        device_map="cuda:0", local_files_only=True, trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    proc = AutoProcessor.from_pretrained(MODEL, local_files_only=True, trust_remote_code=True)
    actual_attn = getattr(model.config, "_attn_implementation", "unknown")
    print("Model loaded: attn=%s  VRAM=%.1f GiB" % (
        actual_attn, torch.cuda.max_memory_allocated() / 1024**3))

    # Load manifest for reference SHAs
    with open(os.path.join(FRAMES_DIR, "bundle_manifest.json")) as f:
        manifest = json.load(f)

    rows = []
    token_match = 0
    action_match = 0
    gripper_match = 0

    for fname in FRAMES:
        npy_path = os.path.join(FRAMES_DIR, fname + "_crop.npy")
        raw = np.load(npy_path)
        raw_sha = sha256_hex(raw)

        # Verify raw frame integrity (SHA stored in manifest is for PNG format, .npy differs)
        manifest_entry = next((m for m in manifest["frames"] if m["file"].startswith(fname)), None)

        task_name = manifest_entry["task_name"] if manifest_entry else "unknown"
        prompt = manifest_entry["prompt"] if manifest_entry else (
            "In: What action should the robot take to %s?\nOut:" % task_name.lower()
        )
        prompt_ids_ref = manifest_entry["prompt_ids"] if manifest_entry else None

        # === Path A: Project PIL ===
        img_pil = preprocess_project_pil(raw)
        sha_img_pil = sha256_hex(img_pil)

        inputs_pil = proc(prompt, img_pil, return_tensors="pt")
        ids_pil = inputs_pil["input_ids"].to(device=DEV)
        px_pil = inputs_pil["pixel_values"].to(dtype=DTYPE, device=DEV)
        sha_px_pil = sha256_hex(px_pil)
        sha_ids_pil = sha256_hex(ids_pil)

        result_pil = model.predict_action(
            input_ids=ids_pil, pixel_values=px_pil,
            unnorm_key="libero_spatial", do_sample=False,
        )
        act_pil = np.array(result_pil).flatten()

        # === Path B: Upstream TF/JPEG ===
        img_tf = preprocess_upstream_tf_jpeg(raw)
        sha_img_tf = sha256_hex(img_tf)

        inputs_tf = proc(prompt, img_tf, return_tensors="pt")
        ids_tf = inputs_tf["input_ids"].to(device=DEV)
        px_tf = inputs_tf["pixel_values"].to(dtype=DTYPE, device=DEV)
        sha_px_tf = sha256_hex(px_tf)
        sha_ids_tf = sha256_hex(ids_tf)

        result_tf = model.predict_action(
            input_ids=ids_tf, pixel_values=px_tf,
            unnorm_key="libero_spatial", do_sample=False,
        )
        act_tf = np.array(result_tf).flatten()

        # === Comparison ===
        img_match = sha_img_pil == sha_img_tf
        px_match = sha_px_pil == sha_px_tf
        ids_match = sha_ids_pil == sha_ids_tf

        max_act_diff = float(np.max(np.abs(act_pil - act_tf)))
        gripper_match_frame = float(act_pil[6]) == float(act_tf[6])
        all_action_match = max_act_diff == 0.0

        # Lane M token audit
        ids_audit = ids_tf.clone()
        if ids_audit[0, -1].item() != 29871:
            ids_audit = torch.cat((ids_audit, torch.tensor([[29871]], dtype=ids_audit.dtype, device=ids_audit.device)), dim=1)
        gen = model.generate(input_ids=ids_audit, pixel_values=px_tf,
                            max_new_tokens=model.get_action_dim("libero_spatial"),
                            do_sample=False, pad_token_id=model.pad_token_id)
        adim = model.get_action_dim("libero_spatial")
        tokens_tf = gen[0, -adim:].cpu().numpy()

        # Same for PIL
        ids_audit_pil = ids_pil.clone()
        if ids_audit_pil[0, -1].item() != 29871:
            ids_audit_pil = torch.cat((ids_audit_pil, torch.tensor([[29871]], dtype=ids_audit_pil.dtype, device=ids_audit_pil.device)), dim=1)
        gen_pil = model.generate(input_ids=ids_audit_pil, pixel_values=px_pil,
                                max_new_tokens=adim, do_sample=False,
                                pad_token_id=model.pad_token_id)
        tokens_pil = gen_pil[0, -adim:].cpu().numpy()

        token_match_frame = bool(np.array_equal(tokens_tf, tokens_pil))

        row = {
            "frame": fname,
            "task": task_name[:60],
            "img_sha_pil": sha_img_pil[:16],
            "img_sha_tf": sha_img_tf[:16],
            "img_match": img_match,
            "px_sha_pil": sha_px_pil[:16],
            "px_sha_tf": sha_px_tf[:16],
            "px_match": px_match,
            "ids_match": ids_match,
            "tokens_pil": " ".join("%d" % x for x in tokens_pil),
            "tokens_tf": " ".join("%d" % x for x in tokens_tf),
            "token_match": token_match_frame,
            "act_pil": " ".join("%.6f" % x for x in act_pil),
            "act_tf": " ".join("%.6f" % x for x in act_tf),
            "max_act_diff": max_act_diff,
            "all_action_match": all_action_match,
            "gripper_match": gripper_match_frame,
        }
        rows.append(row)

        if token_match_frame:
            token_match += 1
        if all_action_match:
            action_match += 1
        if gripper_match_frame:
            gripper_match += 1

        status = "IDENTICAL" if (all_action_match and token_match_frame) else ("PIXEL_DIFF" if not px_match else "DIFF")
        print("  %s: img=%s px=%s token=%s action=%s max_diff=%.2e  [%s]" % (
            fname, "M" if img_match else "X", "M" if px_match else "X",
            "M" if token_match_frame else "X", "M" if all_action_match else "X",
            max_act_diff, status,
        ))

    print("\n" + "=" * 70)
    print("SUMMARY: token_match=%d/%d  action_match=%d/%d  gripper_match=%d/%d" % (
        token_match, len(FRAMES), action_match, len(FRAMES), gripper_match, len(FRAMES)))
    print("=" * 70)

    # Output manifest
    output = {
        "gate": "UPSTREAM_PREPROCESS_ALIGNMENT",
        "dtype": args.dtype, "attn": args.attn, "actual_attn": actual_attn,
        "n_frames": len(FRAMES),
        "token_match": token_match, "action_match": action_match, "gripper_match": gripper_match,
        "all_identical": token_match == len(FRAMES) and action_match == len(FRAMES),
        "rows": rows,
    }

    out_path = os.path.join(FRAMES_DIR, "..", "audit_upstream_preprocess.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print("Saved: %s" % out_path)

    return 0 if output["all_identical"] else 1


if __name__ == "__main__":
    sys.exit(main())
