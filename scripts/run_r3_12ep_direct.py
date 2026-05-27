#!/usr/bin/env python3
"""R3: 12-episode direct validation using patched PIL preprocessing."""
import os, sys, json, time, math, csv
import numpy as np
import torch
from pathlib import Path
from PIL import Image

OUT = Path("/data/liuyu/outputs/milestone_r3_v4_official_preprocess_patch_20260526")
for d in ["tables", "runs"]:
    OUT.joinpath(d).mkdir(parents=True, exist_ok=True)

os.environ["CUDA_VISIBLE_DEVICES"] = "2,6"
os.environ["MUJOCO_GL"] = "egl"

MODEL_PATH = "/data/aviary/models/openvla/openvla-7b-finetuned-libero-object"
TASK_SUITE = "libero_object"
UNNORM_KEY = "libero_object"
NUM_STEPS_WAIT = 10

# cream_cheese(idx=1), salad_dressing(idx=2), butter(idx=6), ketchup(idx=4)
TARGETS = [
    ("cream_cheese", 1, [0, 1, 2]),
    ("salad_dressing", 2, [0, 1, 2]),
    ("butter", 6, [0, 1, 2]),
    ("ketchup", 4, [0, 1, 2]),
]


def quat2axisangle(quat):
    quat = np.asarray(quat).copy()
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if abs(den) < 1e-8:
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def main():
    from transformers import AutoModelForVision2Seq, AutoProcessor
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    print("=" * 60)
    print("R3: 12-Episode Validation (PIL Lanczos preprocessing)")
    print("=" * 60)

    bm = benchmark.get_benchmark_dict()
    suite = bm[TASK_SUITE]()
    print(f"Suite: {TASK_SUITE}, {suite.n_tasks} tasks")

    print("\nLoading model...")
    max_memory = {0: "10000MiB", 1: "10000MiB", "cpu": "128GiB"}
    model = AutoModelForVision2Seq.from_pretrained(
        MODEL_PATH, attn_implementation="eager", torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True, trust_remote_code=True,
        device_map="auto", max_memory=max_memory,
    )
    device = "cuda:0"
    if hasattr(model, "hf_device_map") and model.hf_device_map:
        for v in model.hf_device_map.values():
            if isinstance(v, str) and v.startswith("cuda"):
                device = v
                break
            if isinstance(v, int):
                device = f"cuda:{v}"
                break
    print(f"Model ready, device={device}")

    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    with open(os.path.join(MODEL_PATH, "dataset_statistics.json")) as f:
        model.norm_stats = json.load(f)

    action_dim = int(model.get_action_dim(UNNORM_KEY))
    vocab_size = model.config.text_config.vocab_size - model.config.pad_to_multiple_of
    stats_data = model.get_action_stats(UNNORM_KEY)
    mask = stats_data.get("mask", np.ones_like(stats_data["q01"], dtype=bool))
    high, low = np.array(stats_data["q99"]), np.array(stats_data["q01"])

    results = []

    for task_name, task_idx, state_ids in TARGETS:
        task = suite.get_task(task_idx)
        task_desc = task.language
        init_states = suite.get_task_init_states(task_idx)
        bddl = os.path.join(
            get_libero_path("bddl_files"), task.problem_folder, task.bddl_file
        )
        prompt_str = f"In: What action should the robot take to {task_desc.lower()}?\nOut:"
        max_steps = 280

        for state_id in state_ids:
            print(f"\n--- {task_name} s{state_id} ---")
            env = OffScreenRenderEnv(
                bddl_file_name=bddl, camera_heights=256, camera_widths=256
            )
            env.seed(0)
            obs = env.reset()
            obs = env.set_init_state(init_states[state_id])

            dummy = np.array([0, 0, 0, 0, 0, 0, -1], dtype=np.float32)
            for _ in range(NUM_STEPS_WAIT):
                obs, _, _, _ = env.step(dummy)

            t = 0
            success = False
            runtime_error = False
            error_msg = ""

            while t < max_steps:
                try:
                    # PIL preprocessing (matching corrected official exactly)
                    img = obs["agentview_image"]
                    img = img[::-1, ::-1].copy()
                    img_pil = Image.fromarray(img).convert("RGB")
                    img_pil = img_pil.resize((224, 224), Image.LANCZOS)
                    scale = 0.9**0.5
                    w, h = img_pil.size
                    cw, ch = max(1, int(w * scale)), max(1, int(h * scale))
                    left, top = (w - cw) // 2, (h - ch) // 2
                    img_pil = img_pil.crop((left, top, left + cw, top + ch))
                    img_pil = img_pil.resize((224, 224), Image.LANCZOS)

                    # Model inference (matching corrected official)
                    inputs = processor(prompt_str, img_pil, return_tensors="pt")
                    inputs.pop("attention_mask", None)
                    for k, v in list(inputs.items()):
                        if torch.is_floating_point(v):
                            inputs[k] = v.to(device=device, dtype=torch.bfloat16)
                        else:
                            inputs[k] = v.to(device=device)

                    input_ids = inputs["input_ids"]
                    if not torch.all(input_ids[:, -1] == 29871):
                        inputs["input_ids"] = torch.cat(
                            (
                                input_ids,
                                torch.tensor([[29871]]).long().to(input_ids.device),
                            ),
                            dim=1,
                        )

                    with torch.inference_mode():
                        gen = model.generate(
                            **inputs,
                            max_new_tokens=action_dim,
                            do_sample=False,
                            return_dict_in_generate=True,
                            output_scores=True,
                        )

                    token_ids = gen.sequences[0, -action_dim:].detach().cpu().numpy()
                    discretized = np.clip(
                        vocab_size - token_ids - 1,
                        0,
                        model.bin_centers.shape[0] - 1,
                    )
                    norm_actions = model.bin_centers[discretized]
                    action = np.where(
                        mask,
                        0.5 * (norm_actions + 1) * (high - low) + low,
                        norm_actions,
                    ).astype(np.float32)

                    action_np = action.copy()
                    action_np[..., -1] = 2.0 * action_np[..., -1] - 1.0
                    action_np[..., -1] = np.sign(action_np[..., -1])
                    action_np[..., -1] = (
                        1.0 if action_np[..., -1] == 0 else action_np[..., -1]
                    )
                    action_np[..., -1] = -1.0 * action_np[..., -1]

                    obs, reward, done, info = env.step(action_np.tolist())
                    if done:
                        success = True
                        break
                    t += 1
                except Exception as e:
                    runtime_error = True
                    error_msg = str(e)[:200]
                    print(f"  ERROR: {e}")
                    break

            env.close()
            results.append(
                {
                    "task_name": task_name,
                    "state_id": state_id,
                    "patched_v4_success": success,
                    "num_steps": t + 1,
                    "runtime_error": runtime_error,
                    "error": error_msg,
                    "preprocess_backend": "official_pil_lanczos",
                    "resize_interpolation": "LANCZOS",
                    "uses_jpeg_roundtrip": False,
                    "eos_handling": "add_if_missing_29871",
                }
            )
            print(f"  success={success} steps={t+1}")

    # Write results
    fields = list(results[0].keys())
    with open(OUT / "tables" / "patched_v4_12ep_validation.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(results)

    # Summary
    total = len(results)
    successes = sum(1 for r in results if r["patched_v4_success"])
    print(f"\n{'='*60}")
    print(f"Patched v4 (PIL) 12ep SR: {successes}/{total} = {successes/total:.3f}")

    for tn in ["cream_cheese", "salad_dressing", "butter", "ketchup"]:
        tr = [r for r in results if r["task_name"] == tn]
        if tr:
            sr = sum(1 for r in tr if r["patched_v4_success"]) / len(tr)
            print(f"  {tn}: {sr:.2f}")

    print(f"\nResults: {OUT}/tables/patched_v4_12ep_validation.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
