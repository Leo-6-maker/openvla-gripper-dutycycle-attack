#!/usr/bin/env python3
"""Phase 3: Run paired official-vs-v4 traces for key mismatch episodes."""
import os, sys, json, time, math, csv, argparse
import numpy as np
import torch
from pathlib import Path
from PIL import Image

OUT = Path("/data/liuyu/outputs/milestone_r2_official_v4_object_alignment_20260526")
(OUT / "traces").mkdir(parents=True, exist_ok=True)

os.environ["MUJOCO_GL"] = "egl"

# Config
MODEL_PATH = "/data/aviary/models/openvla/openvla-7b-finetuned-libero-object"
UNNORM_KEY = "libero_object"
TASK_SUITE = "libero_object"
NUM_STEPS_WAIT = 10
MAX_STEPS = 280
SEED = 0

# Task selection: cream_cheese (gap=+3), salad_dressing (gap=+2), butter (gap=+2), ketchup (control, gap=0)
TARGETS = [
    ("cream_cheese", 0, 0),  # state 0 — known to differ
    ("salad_dressing", 0, 0),
    ("butter", 0, 0),
    ("ketchup", 0, 0),
]

def resolve_task(task_key):
    from libero.libero import benchmark, get_libero_path
    bm = benchmark.get_benchmark_dict()
    suite = bm[TASK_SUITE]()
    for i in range(suite.n_tasks):
        name = suite.get_task_names()[i]
        if task_key in name.replace("pick_up_the_", "").replace("_and_place_it_in_the_basket", ""):
            return {"idx": i, "name": name, "desc": suite.get_task(i).language,
                    "init_states": suite.get_task_init_states(i)}
    raise ValueError(f"Task {task_key} not found")

PIL_CROP_SCALE = 0.9

def official_preprocess_pil(obs_image, center_crop=True):
    """Official script preprocessing: rotate 180, PIL Lanczos resize, PIL center crop, PIL Lanczos resize."""
    img = obs_image[::-1, ::-1].copy()  # rotate 180
    img_pil = Image.fromarray(img).convert("RGB")
    img_pil = img_pil.resize((224, 224), Image.LANCZOS)
    if center_crop:
        w, h = img_pil.size
        scale = PIL_CROP_SCALE ** 0.5
        cw, ch = max(1, int(w * scale)), max(1, int(h * scale))
        left, top = (w - cw) // 2, (h - ch) // 2
        img_pil = img_pil.crop((left, top, left + cw, top + ch))
        img_pil = img_pil.resize((224, 224), Image.LANCZOS)
    return np.array(img_pil), img_pil

def v4_preprocess_tf(obs_image, center_crop=True):
    """v4 runner TF preprocessing: rotate 180, JPEG round-trip, TF Lanczos3, TF bilinear crop."""
    import tensorflow as tf
    arr = np.asarray(obs_image)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    arr = arr[::-1, ::-1]
    tensor = tf.convert_to_tensor(arr)
    tensor = tf.io.decode_image(tf.io.encode_jpeg(tensor), expand_animations=False, dtype=tf.uint8)
    tensor = tf.image.resize(tensor, [224, 224], method="lanczos3", antialias=True)
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
            crop_size=[224, 224],
            method="bilinear",
        )[0]
    tensor = tf.cast(tf.clip_by_value(tf.round(tensor), 0, 255), tf.uint8)
    return tensor.numpy(), Image.fromarray(tensor.numpy()).convert("RGB")


def main():
    from transformers import AutoModelForVision2Seq, AutoProcessor
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    print("=" * 60)
    print("Phase 3: Paired Trace Alignment")
    print("=" * 60)

    # Load model once
    print("\nLoading model...")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "2,6")
    n_gpu = len(visible.split(","))
    max_memory = {i: "10000MiB" for i in range(n_gpu)}
    max_memory["cpu"] = "128GiB"

    model = AutoModelForVision2Seq.from_pretrained(
        MODEL_PATH, attn_implementation="eager", torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True, trust_remote_code=True,
        device_map="auto", max_memory=max_memory,
    )
    device = "cuda:0"
    if hasattr(model, 'hf_device_map') and model.hf_device_map:
        for v in model.hf_device_map.values():
            if isinstance(v, str) and v.startswith("cuda"): device = v; break
            if isinstance(v, int): device = f"cuda:{v}"; break
    print(f"Model loaded, device={device}")

    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)

    stats_path = os.path.join(MODEL_PATH, "dataset_statistics.json")
    with open(stats_path) as f:
        model.norm_stats = json.load(f)

    action_dim = int(model.get_action_dim(UNNORM_KEY))
    vocab_size = model.config.text_config.vocab_size - model.config.pad_to_multiple_of

    # Get init states
    bm = benchmark.get_benchmark_dict()
    suite = bm[TASK_SUITE]()

    paired_rows = []

    for task_key, state_id, ep_idx in TARGETS:
        task_info = resolve_task(task_key)
        task_idx = task_info["idx"]
        task_name = task_info["name"]
        task_desc = task_info["desc"]
        init_states = task_info["init_states"]

        bddl = os.path.join(get_libero_path("bddl_files"),
                           suite.get_task(task_idx).problem_folder,
                           suite.get_task(task_idx).bddl_file)
        prompt_str = f"In: What action should the robot take to {task_desc.lower()}?\nOut:"

        print(f"\n--- {task_key} s{state_id} ---")
        print(f"  Task: {task_name}")
        print(f"  Prompt: {prompt_str.strip()}")

        row = {
            "task_key": task_key, "state_id": state_id,
            "task_name": task_name, "prompt": prompt_str,
        }

        # --- Official PIL path ---
        print("  [Official PIL] Running...")
        env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256)
        env.seed(0)
        obs = env.reset()
        obs = env.set_init_state(init_states[state_id])

        # Wait steps
        dummy = np.array([0, 0, 0, 0, 0, 0, -1], dtype=np.float32)
        for _ in range(NUM_STEPS_WAIT):
            obs, _, _, _ = env.step(dummy)

        success_pil = False
        steps_pil = 0
        first_action_pil = None

        for t in range(MAX_STEPS):
            # PIL preprocessing
            img_pil_np, img_pil_obj = official_preprocess_pil(obs["agentview_image"], center_crop=True)

            # Record first step preprocess
            if t == 0:
                row["pil_img_mean"] = float(img_pil_np.mean())
                row["pil_img_std"] = float(img_pil_np.std())
                row["pil_img_min"] = int(img_pil_np.min())
                row["pil_img_max"] = int(img_pil_np.max())
                img_pil_obj.save(OUT / "traces" / f"{task_key}_s{state_id}_official_pil.png")

            # Build observation
            observation = {
                "full_image": img_pil_np,
                "state": np.concatenate([
                    obs["robot0_eef_pos"],
                    quat2axisangle(obs["robot0_eef_quat"]),
                    obs["robot0_gripper_qpos"],
                ]),
            }

            # Model inference
            inputs = processor(prompt_str, img_pil_obj, return_tensors="pt")
            inputs.pop("attention_mask", None)
            for k, v in list(inputs.items()):
                inputs[k] = v.to(device=device, dtype=torch.bfloat16 if torch.is_floating_point(v) else None)

            # EOS token
            input_ids = inputs["input_ids"]
            if not torch.all(input_ids[:, -1] == 29871):
                inputs["input_ids"] = torch.cat(
                    (input_ids, torch.tensor([[29871]]).long().to(input_ids.device)), dim=1)

            with torch.inference_mode():
                gen = model.generate(**inputs, max_new_tokens=action_dim, do_sample=False,
                                    return_dict_in_generate=True, output_scores=True)

            token_ids = gen.sequences[0, -action_dim:].detach().cpu().numpy()
            discretized = np.clip(vocab_size - token_ids - 1, 0, model.bin_centers.shape[0] - 1)
            norm_actions = model.bin_centers[discretized]
            stats = model.get_action_stats(UNNORM_KEY)
            mask = stats.get("mask", np.ones_like(stats["q01"], dtype=bool))
            high, low = np.array(stats["q99"]), np.array(stats["q01"])
            action = np.where(mask, 0.5 * (norm_actions + 1) * (high - low) + low, norm_actions).astype(np.float32)

            if t == 0:
                first_action_pil = action.copy()

            # Postprocess
            action_np = action.copy()
            action_np[..., -1] = 2.0 * action_np[..., -1] - 1.0
            action_np[..., -1] = np.sign(action_np[..., -1])
            action_np[..., -1] = 1.0 if action_np[..., -1] == 0 else action_np[..., -1]
            action_np[..., -1] = -1.0 * action_np[..., -1]

            obs, reward, done, info = env.step(action_np.tolist())
            steps_pil = t + 1
            if done:
                success_pil = True
                break

        env.close()
        row["official_pil_success"] = success_pil
        row["official_pil_steps"] = steps_pil
        row["official_pil_first_action"] = json.dumps(first_action_pil.tolist()) if first_action_pil is not None else ""

        # --- v4 TF path ---
        print("  [v4 TF] Running...")
        env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256)
        env.seed(0)
        obs = env.reset()
        obs = env.set_init_state(init_states[state_id])

        for _ in range(NUM_STEPS_WAIT):
            obs, _, _, _ = env.step(dummy)

        success_tf = False
        steps_tf = 0
        first_action_tf = None

        for t in range(MAX_STEPS):
            # TF preprocessing
            img_tf_np, img_tf_obj = v4_preprocess_tf(obs["agentview_image"], center_crop=True)

            if t == 0:
                row["tf_img_mean"] = float(img_tf_np.mean())
                row["tf_img_std"] = float(img_tf_np.std())
                row["tf_img_min"] = int(img_tf_np.min())
                row["tf_img_max"] = int(img_tf_np.max())
                img_tf_obj.save(OUT / "traces" / f"{task_key}_s{state_id}_v4_tf.png")

            observation = {
                "full_image": img_tf_np,
                "state": np.concatenate([
                    obs["robot0_eef_pos"],
                    quat2axisangle(obs["robot0_eef_quat"]),
                    obs["robot0_gripper_qpos"],
                ]),
            }

            inputs = processor(prompt_str, img_tf_obj, return_tensors="pt")
            inputs.pop("attention_mask", None)
            for k, v in list(inputs.items()):
                inputs[k] = v.to(device=device, dtype=torch.bfloat16 if torch.is_floating_point(v) else None)

            input_ids = inputs["input_ids"]
            if not torch.all(input_ids[:, -1] == 29871):
                inputs["input_ids"] = torch.cat(
                    (input_ids, torch.tensor([[29871]]).long().to(input_ids.device)), dim=1)

            with torch.inference_mode():
                gen = model.generate(**inputs, max_new_tokens=action_dim, do_sample=False,
                                    return_dict_in_generate=True, output_scores=True)

            token_ids = gen.sequences[0, -action_dim:].detach().cpu().numpy()
            discretized = np.clip(vocab_size - token_ids - 1, 0, model.bin_centers.shape[0] - 1)
            norm_actions = model.bin_centers[discretized]
            action = np.where(mask, 0.5 * (norm_actions + 1) * (high - low) + low, norm_actions).astype(np.float32)

            if t == 0:
                first_action_tf = action.copy()

            action_np = action.copy()
            action_np[..., -1] = 2.0 * action_np[..., -1] - 1.0
            action_np[..., -1] = np.sign(action_np[..., -1])
            action_np[..., -1] = 1.0 if action_np[..., -1] == 0 else action_np[..., -1]
            action_np[..., -1] = -1.0 * action_np[..., -1]

            obs, reward, done, info = env.step(action_np.tolist())
            steps_tf = t + 1
            if done:
                success_tf = True
                break

        env.close()
        row["v4_tf_success"] = success_tf
        row["v4_tf_steps"] = steps_tf
        row["v4_tf_first_action"] = json.dumps(first_action_tf.tolist()) if first_action_tf is not None else ""

        # Image diff
        if first_action_pil is not None and first_action_tf is not None:
            img_pil_np_check, _ = official_preprocess_pil(
                obs_raw.copy() if 'obs_raw' in dir() else np.zeros((256,256,3), dtype=np.uint8), center_crop=True)
            # Compute pixel diff from first observation
            row["pil_vs_tf_first_action_l2"] = float(np.linalg.norm(first_action_pil - first_action_tf))

        print(f"  Result: official_pil={success_pil} ({steps_pil}s), v4_tf={success_tf} ({steps_tf}s)")
        paired_rows.append(row)

    # Write results
    with open(OUT / "tables" / "paired_trace_selected_cases.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(paired_rows[0].keys()))
        w.writeheader()
        w.writerows(paired_rows)

    print(f"\nPaired trace results → {OUT}/tables/paired_trace_selected_cases.csv")

    # Summary
    for row in paired_rows:
        if row["official_pil_success"] != row["v4_tf_success"]:
            print(f"  MISMATCH: {row['task_key']}_s{row['state_id']} "
                  f"PIL={row['official_pil_success']} TF={row['v4_tf_success']}")

    # If both still success or both fail, the preprocessing might NOT explain everything
    pil_successes = sum(1 for r in paired_rows if r["official_pil_success"])
    tf_successes = sum(1 for r in paired_rows if r["v4_tf_success"])
    print(f"PIL successes: {pil_successes}/{len(paired_rows)}, TF successes: {tf_successes}/{len(paired_rows)}")


def quat2axisangle(quat):
    quat = np.asarray(quat).copy()
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if abs(den) < 1e-8:
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den

if __name__ == "__main__":
    main()
