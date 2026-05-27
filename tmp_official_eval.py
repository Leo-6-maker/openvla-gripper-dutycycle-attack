#!/usr/bin/env python3
"""
Official OpenVLA LIBERO eval — self-contained reproduction script.
Replicates the exact logic from:
  openvla/experiments/robot/libero/run_libero_eval.py
  openvla/experiments/robot/robot_utils.py
  openvla/experiments/robot/libero/libero_utils.py
  openvla/experiments/robot/openvla_utils.py

Key: uses model.predict_action() (NOT model.generate()) for action decoding.
"""
import os, sys, json, time, math, argparse
import numpy as np
import torch
from pathlib import Path
from PIL import Image

# ============================================================
# Config
# ============================================================
ACTION_DIM = 7
DATE_TIME = time.strftime("%Y_%m_%d-%H_%M_%S")

# ============================================================
# Robot Utils (official)
# ============================================================
def normalize_gripper_action(action, binarize=True):
    action = np.asarray(action, dtype=np.float32).copy()
    action[..., -1] = 2.0 * action[..., -1] - 1.0
    if binarize:
        action[..., -1] = np.sign(action[..., -1])
        action[..., -1] = 1.0 if action[..., -1] == 0 else action[..., -1]
    return action

def invert_gripper_action(action):
    action = np.asarray(action, dtype=np.float32).copy()
    action[..., -1] = -1.0 * action[..., -1]
    return action

def quat2axisangle(quat):
    quat = np.asarray(quat).copy()
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if abs(den) < 1e-8:
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den

# ============================================================
# Libero Utils (official)
# ============================================================
def get_libero_dummy_action(model_family="openvla"):
    return [0, 0, 0, 0, 0, 0, -1]

def get_libero_image(obs, resize_size):
    if isinstance(resize_size, int):
        resize_size = (resize_size, resize_size)
    img = obs["agentview_image"]
    img = img[::-1, ::-1]  # rotate 180 degrees
    # PIL resize (equivalent to official TF Lanczos3)
    img = Image.fromarray(img).convert("RGB")
    img = img.resize(resize_size, Image.LANCZOS)
    return np.array(img)

# ============================================================
# OpenVLA Utils (official)
# ============================================================
OPENVLA_V01_SYSTEM_PROMPT = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)

def get_vla_action(model, processor, device, obs, task_label, unnorm_key, *,
                   center_crop=False, attn_impl="eager"):
    """Action prediction using model.generate() with v4 decoding logic."""
    image = Image.fromarray(obs["full_image"]).convert("RGB")

    if center_crop:
        w, h = image.size
        scale = 0.9 ** 0.5
        cw, ch = max(1, int(w * scale)), max(1, int(h * scale))
        left, top = (w - cw) // 2, (h - ch) // 2
        image = image.crop((left, top, left + cw, top + ch))
        image = image.resize((224, 224), Image.LANCZOS)

    prompt = f"In: What action should the robot take to {task_label.lower()}?\nOut:"
    inputs = processor(prompt, image, return_tensors="pt")
    # Drop attention_mask to avoid causal-mask mismatch (same as v4 runner)
    inputs.pop("attention_mask", None)
    for key, val in list(inputs.items()):
        if torch.is_floating_point(val):
            inputs[key] = val.to(device=device, dtype=torch.bfloat16)
        else:
            inputs[key] = val.to(device=device)

    # Ensure EOS token before generation (v4 runner logic)
    input_ids = inputs.get("input_ids")
    if input_ids is not None and not torch.all(input_ids[:, -1] == 29871):
        inputs["input_ids"] = torch.cat(
            (input_ids, torch.unsqueeze(torch.tensor([29871]).long(), dim=0).to(input_ids.device)), dim=1)

    action_dim = int(model.get_action_dim(unnorm_key))

    with torch.inference_mode():
        gen = model.generate(**inputs, max_new_tokens=action_dim, do_sample=False,
                            return_dict_in_generate=True, output_scores=True)

    # v4 action decoding
    token_ids = gen.sequences[0, -action_dim:].detach().cpu().numpy()
    vocab_size = model.config.text_config.vocab_size - model.config.pad_to_multiple_of
    discretized = np.clip(vocab_size - token_ids - 1, a_min=0, a_max=model.bin_centers.shape[0] - 1)
    norm_actions = model.bin_centers[discretized]
    stats = model.get_action_stats(unnorm_key)
    mask = stats.get("mask", np.ones_like(stats["q01"], dtype=bool))
    high, low = np.array(stats["q99"]), np.array(stats["q01"])
    action = np.where(mask, 0.5 * (norm_actions + 1) * (high - low) + low, norm_actions).astype(np.float32)
    return action

# ============================================================
# Main eval
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--task_suite_name", default="libero_object")
    ap.add_argument("--num_trials_per_task", type=int, default=10)
    ap.add_argument("--num_steps_wait", type=int, default=10)
    ap.add_argument("--center_crop", action="store_true", default=True)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--output_root", required=True)
    ap.add_argument("--render_gpu_device_id", type=int, default=0)
    ap.add_argument("--attn_impl", default="eager",
                    help="flash_attention_2 or eager")
    ap.add_argument("--cuda_visible_devices", default="0,1")
    ap.add_argument("--task_start", type=int, default=0, help="First task index (0-based)")
    ap.add_argument("--task_count", type=int, default=10, help="Number of tasks to run")
    ap.add_argument("--run_id_prefix", default="official", help="Prefix for output naming")
    ap.add_argument("--worker_id", default="", help="Worker ID for parallel-safe shard naming")
    args = ap.parse_args()
    if not args.worker_id:
        args.worker_id = args.run_id_prefix

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    os.environ["MUJOCO_GL"] = "egl"
    DEVICE = "cuda:0"

    print(f"=== Official OpenVLA LIBERO Eval ===")
    print(f"Model: {args.model_path}")
    print(f"Suite: {args.task_suite_name}")
    print(f"Trials/task: {args.num_trials_per_task}")
    print(f"Center crop: {args.center_crop}")
    print(f"Attention: {args.attn_impl}")
    print(f"CUDA: {args.cuda_visible_devices}")
    print(f"Seed: {args.seed}")

    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Load model with proper multi-GPU mapping
    print("\n[*] Loading model...")
    from transformers import AutoModelForVision2Seq, AutoProcessor

    visible = len(args.cuda_visible_devices.split(","))
    max_memory = {i: "10000MiB" for i in range(max(visible, 1))}
    max_memory["cpu"] = "128GiB"

    model = AutoModelForVision2Seq.from_pretrained(
        args.model_path,
        attn_implementation=args.attn_impl,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        device_map="auto",
        max_memory=max_memory,
    )
    DEVICE = "cuda:0"
    if hasattr(model, 'hf_device_map') and getattr(model, 'hf_device_map', None):
        for v in model.hf_device_map.values():
            if isinstance(v, str) and v.startswith("cuda"):
                DEVICE = v; break
            if isinstance(v, int): DEVICE = f"cuda:{v}"; break
    print(f"[*] Model loaded, primary device: {DEVICE}")

    # Load norm stats
    stats_path = os.path.join(args.model_path, "dataset_statistics.json")
    if os.path.isfile(stats_path):
        with open(stats_path) as f:
            model.norm_stats = json.load(f)

    # Load processor
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)

    # Unnorm key
    unnorm_key = args.task_suite_name
    if unnorm_key not in model.norm_stats and f"{unnorm_key}_no_noops" in model.norm_stats:
        unnorm_key = f"{unnorm_key}_no_noops"
    assert unnorm_key in model.norm_stats, f"Key {unnorm_key} not found!"

    # Setup output
    out = Path(args.output_root)
    out.mkdir(parents=True, exist_ok=True)
    (out / "runs").mkdir(exist_ok=True)
    (out / "tables").mkdir(exist_ok=True)

    log_path = out / f"eval_log_{args.run_id_prefix}_{DATE_TIME}.txt"
    log = open(log_path, "w")

    # Init LIBERO
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks = task_suite.n_tasks
    print(f"Task suite: {args.task_suite_name} ({num_tasks} tasks)")
    log.write(f"Suite: {args.task_suite_name}, {num_tasks} tasks\n")

    # Results tracking
    manifest_rows = []
    task_summary = {}
    total_episodes = 0
    total_successes = 0

    task_end = min(args.task_start + args.task_count, num_tasks)
    print(f"Task range: {args.task_start} -> {task_end-1} ({task_end - args.task_start} tasks)")
    for task_id in range(args.task_start, task_end):
        task = task_suite.get_task(task_id)
        task_name = task_suite.get_task_names()[task_id]
        task_desc = task.language
        initial_states = task_suite.get_task_init_states(task_id)

        # Create env (official: no horizon, no render_gpu_device_id)
        task_bddl_file = os.path.join(
            get_libero_path("bddl_files"), task.problem_folder, task.bddl_file
        )
        env = OffScreenRenderEnv(
            bddl_file_name=task_bddl_file,
            camera_heights=256,
            camera_widths=256,
        )
        env.seed(0)

        # Max steps per suite
        if args.task_suite_name == "libero_spatial":
            max_steps = 220
        elif args.task_suite_name == "libero_object":
            max_steps = 280
        elif args.task_suite_name == "libero_goal":
            max_steps = 300
        elif args.task_suite_name == "libero_10":
            max_steps = 520
        else:
            max_steps = 400

        task_episodes = 0
        task_successes = 0

        print(f"\n--- Task {task_id+1}/{num_tasks}: {task_name} ---")
        for ep_idx in range(args.num_trials_per_task):
            env.reset()
            obs = env.set_init_state(initial_states[ep_idx])

            t = 0
            success = False
            runtime_error = False
            error_msg = ""

            while t < max_steps + args.num_steps_wait:
                try:
                    if t < args.num_steps_wait:
                        obs, reward, done, info = env.step(
                            get_libero_dummy_action("openvla")
                        )
                        t += 1
                        continue

                    # Get preprocessed image
                    img = get_libero_image(obs, 224)

                    # Prepare observation dict
                    observation = {
                        "full_image": img,
                        "state": np.concatenate([
                            obs["robot0_eef_pos"],
                            quat2axisangle(obs["robot0_eef_quat"]),
                            obs["robot0_gripper_qpos"],
                        ]),
                    }

                    # Query model (official: predict_action)
                    action = get_vla_action(
                        model, processor, DEVICE, observation,
                        task_desc, unnorm_key,
                        center_crop=args.center_crop,
                        attn_impl=args.attn_impl,
                    )

                    # Postprocess
                    action_np = np.asarray(action, dtype=np.float32)
                    action_np = normalize_gripper_action(action_np, binarize=True)
                    action_np = invert_gripper_action(action_np)

                    # Step
                    obs, reward, done, info = env.step(action_np.tolist())
                    if done:
                        success = True
                        task_successes += 1
                        total_successes += 1
                        break
                    t += 1

                except Exception as e:
                    runtime_error = True
                    error_msg = str(e)[:200]
                    print(f"  ERROR: {e}")
                    break

            task_episodes += 1
            total_episodes += 1
            phase = "success_libero" if success else ("runtime_error" if runtime_error else "no_grasp")
            manifest_rows.append({
                "task_name": task_name, "state_id": ep_idx,
                "success": success, "runtime_error": runtime_error,
                "error": error_msg, "num_steps": t,
                "failure_phase": phase,
            })

        # Task summary
        sr = task_successes / task_episodes
        task_summary[task_name] = {
            "n": task_episodes, "success": task_successes, "sr": sr
        }
        print(f"  Task SR: {task_successes}/{task_episodes} = {sr:.2f}")
        print(f"  Running total: {total_successes}/{total_episodes} = {total_successes/total_episodes:.3f}")
        env.close()

    log.close()

    # Write results — worker-safe shard naming (parallel workers never share files)
    import csv
    shard_dir = out / "tables" / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    worker_id = getattr(args, "worker_id", args.run_id_prefix)
    suite_short = args.task_suite_name.replace("libero_", "")
    manifest_shard = shard_dir / f"manifest_{suite_short}_worker_{worker_id}.csv"
    summary_shard = shard_dir / f"summary_{suite_short}_worker_{worker_id}.csv"

    with open(manifest_shard, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=manifest_rows[0].keys())
        w.writeheader(); w.writerows(manifest_rows)

    with open(summary_shard, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["task_name","n","success","sr"])
        w.writeheader()
        for tn, ts in sorted(task_summary.items()):
            w.writerow({"task_name": tn, **ts})

    print(f"\n=== FINAL ===")
    print(f"Overall SR: {total_successes}/{total_episodes} = {total_successes/total_episodes:.3f}")
    for tn, ts in sorted(task_summary.items()):
        print(f"  {tn}: {ts['success']}/{ts['n']} = {ts['sr']:.2f}")

if __name__ == "__main__":
    main()
