#!/usr/bin/env python3
"""MIG3B: A800-F32-S closed-loop clean canary for LIBERO-Spatial."""
import os, json, hashlib, time, csv, numpy as np
from PIL import Image
import torch

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["MUJOCO_GL"] = "egl"

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from transformers import AutoProcessor, AutoModelForVision2Seq
import imageio

MODEL = "/mnt/sdc/dty_user/openvla_attack/models/libero-spatial/spatial_c8f03f4_20260620"
OUT = "/mnt/sdc/dty_user/openvla_attack/evidence/mig3b_spatial_canary"
os.makedirs(OUT, exist_ok=True)
os.makedirs(f"{OUT}/videos", exist_ok=True)
os.makedirs(f"{OUT}/traces", exist_ok=True)

CANARY = [
    {"task_idx": 0, "init_state": 0, "label": "taskA_init0"},
    {"task_idx": 0, "init_state": 1, "label": "taskA_init1"},
    {"task_idx": 1, "init_state": 0, "label": "taskB_init0"},
    {"task_idx": 1, "init_state": 1, "label": "taskB_init1"},
]

print("Loading model...")
model = AutoModelForVision2Seq.from_pretrained(
    MODEL, torch_dtype=torch.float32, attn_implementation="eager",
    local_files_only=True, trust_remote_code=True, low_cpu_mem_usage=True,
    device_map="cuda:0",
)
proc = AutoProcessor.from_pretrained(MODEL, local_files_only=True, trust_remote_code=True)
DEV = next(model.parameters()).device
devices = sorted(set(str(p.device) for p in model.parameters()))
print(f"Devices: {devices}")

bench_dict = benchmark.get_benchmark_dict()
task_suite = bench_dict["libero_spatial"]()
print(f"Tasks: {task_suite.n_tasks}")

action_dim = model.get_action_dim("libero_spatial")
stats = model.get_action_stats("libero_spatial")
mask = stats.get("mask", np.ones_like(stats["q01"], dtype=bool))
hi, lo = np.array(stats["q99"]), np.array(stats["q01"])

W = H = 256
WAIT_ACTION = [0, 0, 0, 0, 0, 0, -1]
MAX_STEPS = 220
WAIT_STEPS = 10

all_results = []

for ep_idx, plan in enumerate(CANARY):
    task_idx = plan["task_idx"]
    init_idx = plan["init_state"]
    label = plan["label"]

    print(f"\n=== EPISODE {ep_idx}: {label} (task={task_idx}, init={init_idx}) ===")

    task = task_suite.get_task(task_idx)
    init_states = task_suite.get_task_init_states(task_idx)
    bddl_path = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    task_name = task.language if hasattr(task, "language") else task.name

    env = OffScreenRenderEnv(
        bddl_file_name=bddl_path,
        camera_heights=H, camera_widths=W,
        has_renderer=True, has_offscreen_renderer=True,
        render_gpu_device_id=0, use_camera_obs=True,
    )
    env.seed(42)
    obs = env.reset()
    obs = env.set_init_state(init_states[init_idx])

    for w in range(WAIT_STEPS):
        obs, rew, done, info = env.step(WAIT_ACTION)

    frames = []
    step_trace = []
    episode_done = False
    episode_success = False
    error = None

    t0 = time.time()

    for step in range(MAX_STEPS):
        img = Image.fromarray(obs["agentview_image"])
        prompt = f"In: What action should the robot take to {task_name.lower()}?\nOut:"

        inputs = proc(prompt, img, return_tensors="pt")
        ids = inputs["input_ids"]
        px = inputs["pixel_values"].to(dtype=torch.float32, device=DEV)

        if ids[0, -1].item() != 29871:
            ids = torch.cat((ids, torch.tensor([[29871]], dtype=ids.dtype)), dim=1)
        ids = ids.to(device=DEV)

        gen = model.generate(
            input_ids=ids, pixel_values=px,
            max_new_tokens=action_dim, do_sample=False,
            pad_token_id=model.pad_token_id,
        )
        act_tokens = gen[0, -action_dim:].cpu().numpy()

        disc = model.vocab_size - act_tokens
        disc = np.clip(disc - 1, 0, model.bin_centers.shape[0] - 1)
        norm = model.bin_centers[disc]
        act = np.where(mask, 0.5 * (norm + 1) * (hi - lo) + lo, norm)

        raw_grip = act[6]
        norm_grip = (raw_grip * 2) - 1
        bin_grip = 1.0 if norm_grip >= 0 else -1.0
        env_grip = -bin_grip

        env_action = np.zeros(7)
        env_action[:6] = act[:6]
        env_action[6] = env_grip

        frame_sha = hashlib.sha256(np.array(img).tobytes()).hexdigest()

        try:
            obs, rew, done, info = env.step(env_action.tolist())
        except Exception as e:
            error = str(e)
            break

        frames.append(np.array(img))
        trace_row = {
            "episode": label, "policy_step": step,
            "raw_image_sha": frame_sha,
            "generated_tokens": " ".join(str(x) for x in act_tokens.tolist()),
            "final_action": " ".join(f"{x:.8f}" for x in act.tolist()),
            "raw_gripper": f"{raw_grip:.8f}",
            "norm_gripper": f"{norm_grip:.8f}",
            "env_gripper": f"{env_grip:.8f}",
            "done": str(done),
        }
        step_trace.append(trace_row)

        if done:
            episode_done = True
            episode_success = rew > 0
            break

    dt = time.time() - t0
    env.close()

    video_path = f"{OUT}/videos/{label}.mp4"
    imageio.mimsave(video_path, frames, fps=10)

    trace_path = f"{OUT}/traces/{label}_trace.csv"
    with open(trace_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=trace_row.keys())
        w.writeheader()
        w.writerows(step_trace)

    valid_grip = all(abs(float(row["env_gripper"])) == 1.0 for row in step_trace)
    result = {
        "episode": ep_idx, "label": label,
        "task_idx": task_idx, "init_state": init_idx,
        "task_name": task_name,
        "steps": len(step_trace),
        "done": episode_done, "success": episode_success,
        "error": error,
        "duration_s": round(dt, 1),
        "gripper_valid": valid_grip,
        "video": f"videos/{label}.mp4",
        "trace": f"traces/{label}_trace.csv",
    }
    all_results.append(result)

    print(f"  steps={len(step_trace)} done={episode_done} success={episode_success}")
    print(f"  gripper_valid={valid_grip} error={error}")

    if ep_idx == 0:
        print("\n=== C0 COMPLETE ===")
        break

with open(f"{OUT}/canary_results.json", "w") as f:
    json.dump(all_results, f, indent=2)

print(f"\nResults: {OUT}")
